"""Event detectors: one Vision Transformer per modality, 14 event outputs per 30-s window.

Backbone: timm vit_small_patch16_224.augreg_in21k (ViT-S/16, ImageNet-21k), 12 blocks, the CLS token -> linear
layer -> one logit per event.
  video  input 3 x 30 x 64 x 64 (fx, fy, |f| in the quantised log scale x 30 one-second bins), upsampled to
         128 x 128 (bilinear); tubelets of 6 s x 16 x 16 pixels (the 2-D patch embedding repeated over 6 bins
         and divided by 6) -> 5 x 8 x 8 = 320 tokens; spatial position embedding interpolated to 8 x 8 plus a
         temporal one (5 positions, zero-initialised)
  audio  input 1 x 128 x 480 (the 469 pooled log-Mel frames, frames after the valid ones set to 0, zero-padded
         to 480); 16 x 16 patches -> 8 x 30 = 240 tokens (timm sums the RGB patch weights for one channel and
         interpolates the position embedding)
Inputs are standardised with the mean and SD of 20,000 training windows of the fold (video: per channel).
Training: per-event binary cross-entropy on windows with a definite label for that event (-1 = ignored),
pos_weight = (n_negative / n_positive)^0.25; AdamW (lr 1e-4, weight decay 0.05 except biases, norms, position
embeddings and the CLS token), warm-up over 0.1 epoch then cosine decay over a 5-epoch schedule, of which 3 epochs
are run; batch 64 (or 32 x 2 accumulated); gradient clipping 1.0; drop path 0.1; mixed precision (fp16).
Augmentation: video left-right flip (with the sign of fx) and shifts of up to 4 pixels; audio two frequency masks
(<= 16 bands) and two time masks (<= 32 frames).
Folds (model.folds): training on the train exams, the epoch with the highest mean validation AUC over the 13
events (all but anychinemg activity) is kept; its predictions for every window of every centre are saved.

    python -m psgscreen.model.vit --modality video --fold 0 [--device cuda:0]
    python -m psgscreen.model.vit --modality video --fold 0 --predict_only
        -> models/vit/<modality>_f<fold>/{best.pt, epoch<e>.pt, last.pt, log.jsonl, report.json, preds/<centre>.npy}
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from psgscreen import paths
from psgscreen.dataset import vit_inputs as vi
from psgscreen.events import EXCLUDED
from psgscreen.model.folds import exam_roles

TIMM_NAME = "vit_small_patch16_224.augreg_in21k"
EFF_BATCH = 64


# ---------------------------------------------------------------- data
class InputStore:
    """Per-centre input arrays (opened lazily in every process) and row metadata."""

    def __init__(self, modality):
        self.mod, self.files, self.rows, self._mm = modality, {}, {}, {}
        for c in paths.CENTERS:
            d = f"{paths.VIT_INPUTS}/{c}"
            if not os.path.exists(f"{d}/rows.npz"):
                continue
            r = vi.load_rows(c)
            self.rows[c] = dict(uuid=r["uuid"], y=r["y_multi"], ev=[str(e) for e in r["event_names"]])
            xs = ("video_u8.npy", "video_status.npy") if modality == "video" else ("audio_f16.npy", "audio_valid.npy")
            if all(os.path.exists(f"{d}/{x}") for x in xs):
                self.files[c] = tuple(f"{d}/{x}" for x in xs)
        evs = {tuple(v["ev"]) for v in self.rows.values()}
        assert len(evs) == 1, "event names differ between centres"
        self.events = list(evs.pop())

    def __getstate__(self):
        st = dict(self.__dict__); st["_mm"] = {}           # memory maps are reopened in the loader workers
        return st

    def read(self, c, idx):
        key = (os.getpid(), c)
        if key not in self._mm:
            self._mm[key] = tuple(np.load(p, mmap_mode="r") for p in self.files[c])
        X, V = self._mm[key]
        return np.asarray(X[idx]), np.asarray(V[idx]).astype(np.int16)

    def complete(self, c):
        """Every exam of the centre has been written by dataset.vit_inputs."""
        p = f"{paths.VIT_INPUTS}/{c}/{self.mod}_done.txt"
        if c not in self.files or not os.path.exists(p):
            return False
        return set(np.unique(self.rows[c]["uuid"])) <= {l.split("\t")[0] for l in open(p) if l.strip()}


class TrainStream(torch.utils.data.IterableDataset):
    """Chunk shuffling: runs of 64 consecutive rows in random order, 32 runs per buffer, shuffled within the
    buffer (sequential reads from the input arrays, ~16 recording segments per batch)."""

    def __init__(self, store, items, epoch, seed, chunk=64, buf_chunks=32):
        super().__init__()
        self.store, self.items, self.epoch, self.seed, self.chunk, self.buf = store, items, epoch, seed, chunk, buf_chunks

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = np.random.default_rng(self.seed * 1000 + self.epoch)
        chunks = [(c, idx[s:s + self.chunk]) for c, idx in self.items for s in range(0, len(idx), self.chunk)]
        mine = rng.permutation(len(chunks))[wid::nw]
        r2 = np.random.default_rng(self.seed * 7919 + self.epoch * 31 + wid)
        for b in range(0, len(mine), self.buf):
            xs, vs, ys = [], [], []
            for ci in mine[b:b + self.buf]:
                c, idx = chunks[ci]
                x, v = self.store.read(c, idx)
                xs.append(x); vs.append(v); ys.append(self.store.rows[c]["y"][idx])
            x, v, y = np.concatenate(xs), np.concatenate(vs), np.concatenate(ys)
            for i in r2.permutation(len(x)):
                yield x[i], v[i], y[i]


class SeqStream(torch.utils.data.IterableDataset):
    """Rows of one centre in order, in blocks of bs."""

    def __init__(self, store, c, idx, bs):
        super().__init__()
        self.store, self.c, self.idx, self.bs = store, c, idx, bs

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        blocks = [self.idx[s:s + self.bs] for s in range(0, len(self.idx), self.bs)]
        for bi in range(wid, len(blocks), nw):
            x, v = self.store.read(self.c, blocks[bi])
            yield bi, torch.from_numpy(x), torch.from_numpy(v)


# ---------------------------------------------------------------- models
class VideoViT(nn.Module):
    def __init__(self, n_out, drop_path=0.1, tub=6, nbin=30, img=128, pretrained=True):
        super().__init__()
        import timm
        base = timm.create_model(TIMM_NAME, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path, img_size=img)
        D = base.embed_dim
        w2d = base.patch_embed.proj.weight.data
        p = w2d.shape[-1]
        self.proj = nn.Conv3d(3, D, kernel_size=(tub, p, p), stride=(tub, p, p))
        self.proj.weight.data.copy_(w2d.unsqueeze(2).repeat(1, 1, tub, 1, 1) / tub)
        self.proj.bias.data.copy_(base.patch_embed.proj.bias.data)
        pe = base.pos_embed.data
        self.cls_token = base.cls_token
        self.pos_cls = nn.Parameter(pe[:, :1].clone())
        self.pos_sp = nn.Parameter(pe[:, 1:].clone())
        self.pos_t = nn.Parameter(torch.zeros(1, nbin // tub, 1, D))
        self.blocks, self.norm = base.blocks, base.norm
        self.head = nn.Linear(D, n_out)
        self.img = img

    def forward(self, x):                                         # (B, 3, T, 64, 64)
        B, C, T, H, W = x.shape
        x = F.interpolate(x.reshape(B, C * T, H, W), size=(self.img, self.img),
                          mode="bilinear", align_corners=False).reshape(B, C, T, self.img, self.img)
        x = self.proj(x).flatten(3).permute(0, 2, 3, 1)           # (B, T/6, 64, D)
        x = (x + self.pos_sp.unsqueeze(1) + self.pos_t).flatten(1, 2)
        cls = self.cls_token.expand(B, -1, -1) + self.pos_cls
        x = self.norm(self.blocks(torch.cat([cls, x], 1)))
        return self.head(x[:, 0])


def AudioViT(n_out, drop_path=0.1, pretrained=True):
    import timm
    return timm.create_model(TIMM_NAME, pretrained=pretrained, num_classes=n_out, in_chans=1, img_size=(128, 480),
                             drop_path_rate=drop_path)


def n_outputs(m):
    return m.head.out_features if hasattr(m.head, "out_features") else m.num_classes


class Prep:
    """Input arrays -> standardised model input on the device (and the training augmentation)."""

    def __init__(self, modality, stats, device):
        self.mod, self.dev = modality, device
        if modality == "video":
            self.mu = torch.tensor(stats["mu"], device=device).view(1, 3, 1, 1, 1)
            self.sd = torch.tensor(stats["sd"], device=device).view(1, 3, 1, 1, 1)
        else:
            self.mu, self.sd = float(stats["mu"]), float(stats["sd"])

    def __call__(self, x, v, train):
        x = x.to(self.dev, non_blocking=True)
        if self.mod == "video":
            f = torch.empty(x.shape, dtype=torch.float32, device=self.dev)
            f[:, :2] = x[:, :2].view(torch.int8).float() / 127.0
            f[:, 2] = x[:, 2].float() / 255.0
            if not train:
                return (f - self.mu) / self.sd
            B = f.shape[0]
            flip = torch.rand(B, device=self.dev) < 0.5
            if flip.any():
                g = f[flip].flip(-1)
                g[:, 0] = -g[:, 0]                                # left-right flip reverses horizontal motion
                f[flip] = g
            f = (f - self.mu) / self.sd
            pad = F.pad(f, (4, 4, 4, 4))
            ox, oy = torch.randint(0, 9, (B,)).tolist(), torch.randint(0, 9, (B,)).tolist()
            return torch.stack([pad[i, :, :, oy[i]:oy[i] + vi.S, ox[i]:ox[i] + vi.S] for i in range(B)])
        v = v.to(self.dev, non_blocking=True)
        f = (x.float() - self.mu) / self.sd
        tt = torch.arange(f.shape[-1], device=self.dev)
        f = F.pad(f.masked_fill(tt[None, None, :] >= v[:, None, None], 0.0), (0, 480 - f.shape[-1]))
        if train:
            B, Fq, Tm = f.shape
            fr, tr = torch.arange(Fq, device=self.dev)[None, :], torch.arange(Tm, device=self.dev)[None, :]
            m = torch.zeros(B, Fq, Tm, dtype=torch.bool, device=self.dev)
            for _ in range(2):
                w = torch.randint(0, 17, (B, 1), device=self.dev)
                s = (torch.rand(B, 1, device=self.dev) * (Fq - w + 1).float()).long()
                m |= ((fr >= s) & (fr < s + w))[:, :, None]
                w = torch.randint(0, 33, (B, 1), device=self.dev)
                s = (torch.rand(B, 1, device=self.dev) * (Tm - w + 1).float()).long()
                m |= ((tr >= s) & (tr < s + w))[:, None, :]
            f = f.masked_fill(m, 0.0)
        return f.unsqueeze(1)


# ---------------------------------------------------------------- helpers
def fold_items(store, k, role):
    """[(centre, row indices)] of the exams with this role in fold k; training and validation rows need at least
    one definite label."""
    roles = exam_roles(k)
    want = [u for u, r in roles.items() if r == role]
    items = []
    for c, r in store.rows.items():
        if c not in store.files:
            continue
        m = np.isin(r["uuid"], want) & (r["y"] >= 0).any(1)
        idx = np.flatnonzero(m)
        if len(idx):
            items.append((c, idx))
    return items


def norm_stats(store, items, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    allr = [(c, i) for c, idx in items for i in idx]
    byc = {}
    for j in sorted(rng.choice(len(allr), size=min(n, len(allr)), replace=False)):
        c, i = allr[j]
        byc.setdefault(c, []).append(i)
    if store.mod == "video":
        s1, s2, cnt = np.zeros(3), np.zeros(3), 0
        for c, idx in byc.items():
            f = vi.to_model_scale(store.read(c, np.array(idx))[0])
            s1 += f.mean(axis=(0, 2, 3, 4)) * len(f); s2 += (f ** 2).mean(axis=(0, 2, 3, 4)) * len(f)
            cnt += len(f)
        mu = s1 / cnt
        return dict(mu=mu.tolist(), sd=np.sqrt(np.maximum(s2 / cnt - mu ** 2, 1e-8)).tolist())
    vals = []
    for c, idx in byc.items():
        x, v = store.read(c, np.array(idx))
        vals += [a[:, :n_].astype(np.float32).ravel()[::7] for a, n_ in zip(x, v) if n_ > 0]
    a = np.concatenate(vals)
    return dict(mu=float(a.mean()), sd=float(a.std()))


def macro_auc(y, p, events):
    from sklearn.metrics import roc_auc_score
    out = {}
    for e, name in enumerate(events):
        m = y[:, e] >= 0
        if name == EXCLUDED or y[m, e].min() == y[m, e].max():
            continue
        out[name] = float(roc_auc_score(y[m, e], p[m, e]))
    return (float(np.mean(list(out.values()))) if out else float("nan")), out


@torch.no_grad()
def predict(model, prep, store, c, idx, bs, device, workers):
    model.eval()
    dl = torch.utils.data.DataLoader(SeqStream(store, c, idx, bs), batch_size=None, num_workers=workers, pin_memory=True)
    out = np.zeros((len(idx), n_outputs(model)), np.float16)
    for bi, x, v in dl:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            lo = model(prep(x, v, train=False))
        out[bi * bs: bi * bs + len(x)] = torch.sigmoid(lo.float()).cpu().numpy().astype(np.float16)
    return out


def param_groups(model, wd):
    dec, nodec = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (nodec if p.ndim < 2 or "pos" in n or "cls_token" in n or n.endswith(".bias") else dec).append(p)
    return [dict(params=dec, weight_decay=wd), dict(params=nodec, weight_decay=0.0)]


# ---------------------------------------------------------------- training
def train(a, store, model, prep, out, dev, log):
    tr, va = fold_items(store, a.fold, "train"), fold_items(store, a.fold, "val")
    n_tr = sum(len(i) for _, i in tr)
    ytr = np.concatenate([store.rows[c]["y"][i] for c, i in tr])
    npos, nneg = (ytr == 1).sum(0), (ytr == 0).sum(0)
    pw = torch.tensor(np.where(npos > 0, (nneg / np.maximum(npos, 1)) ** a.pw_pow, 1.0).astype(np.float32), device=dev)
    log(stage="setup", modality=a.modality, fold=a.fold, n_train=n_tr, n_val=sum(len(i) for _, i in va),
        events=store.events, n_pos=npos.tolist(), pos_weight=[round(float(x), 3) for x in pw.tolist()])
    opt = torch.optim.AdamW(param_groups(model, a.wd), lr=a.lr, betas=(0.9, 0.999))
    steps = n_tr // EFF_BATCH
    total, warm = steps * a.sched_epochs, max(1, int(a.warmup_epochs * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    scaler = torch.cuda.amp.GradScaler()
    ck = f"{out}/last.pt"
    ep0, best, best_ep, hist = 0, -1.0, -1, []
    if os.path.exists(ck):                                         # resume from the last finished epoch
        st = torch.load(ck, map_location="cpu")
        if st["sched_epochs"] != a.sched_epochs or st["n_train"] != n_tr:
            raise SystemExit("last.pt was written with another schedule or training set")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        torch.set_rng_state(st["rng_cpu"]); torch.cuda.set_rng_state(st["rng_cuda"], dev)
        ep0, best, best_ep, hist = st["epoch"] + 1, st["best"], st["best_ep"], st["hist"]
    for ep in range(ep0, a.epochs):
        model.train()
        dl = torch.utils.data.DataLoader(TrainStream(store, tr, ep, paths.SEED + a.fold), batch_size=a.batch,
                                         num_workers=a.workers, pin_memory=True, drop_last=True)
        t0, run, step, it = time.time(), 0.0, 0, -1
        opt.zero_grad(set_to_none=True)
        try:
            for it, (x, v, y) in enumerate(dl):
                y = y.to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    lo = model(prep(x, v, train=True))
                m = (y >= 0).float()
                lm = F.binary_cross_entropy_with_logits(lo.float(), y.clamp(min=0).float(), pos_weight=pw,
                                                        reduction="none") * m
                loss = lm.sum() / m.sum().clamp(min=1.0)
                scaler.scale(loss / a.accum).backward()
                run += float(loss)
                if (it + 1) % a.accum == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                    step += 1
                    if step % 200 == 0:
                        log(stage="train", epoch=ep, step=step, of=steps, loss=round(run / (it + 1), 4),
                            lr=sched.get_last_lr()[0])
                    if step >= steps:
                        break
        except torch.cuda.OutOfMemoryError:
            log(stage="oom", epoch=ep, batch=a.batch, accum=a.accum)
            sys.exit(3)                                            # rerun with --batch 32 --accum 2
        yv = np.concatenate([store.rows[c]["y"][i] for c, i in va])
        pv = np.concatenate([predict(model, prep, store, c, i, 256, dev, a.workers) for c, i in va]).astype(np.float32)
        mac, per = macro_auc(yv, pv, store.events)
        hist.append(dict(epoch=ep, val_macro=mac, train_loss=run / max(1, it + 1), minutes=(time.time() - t0) / 60))
        log(stage="val", epoch=ep, macro_auc=round(mac, 4), per_event={k: round(v, 4) for k, v in per.items()})
        torch.save({"model": model.state_dict(), "epoch": ep, "val_macro": mac}, f"{out}/epoch{ep}.pt")
        if mac > best:
            best, best_ep = mac, ep
            torch.save({"model": model.state_dict(), "epoch": ep, "val_macro": mac}, f"{out}/best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(dev),
                    "epoch": ep, "best": best, "best_ep": best_ep, "hist": hist, "sched_epochs": a.sched_epochs,
                    "n_train": n_tr}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
    json.dump(dict(modality=a.modality, fold=a.fold, epochs_run=len(hist), sched_epochs=a.sched_epochs,
                   best_epoch=best_ep, best_val_macro=best, hist=hist, n_train=n_tr, events=store.events),
              open(f"{out}/report.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=["video", "audio"])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=3, help="epochs run")
    ap.add_argument("--sched_epochs", type=int, default=5, help="length of the learning-rate schedule (epochs)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup_epochs", type=float, default=0.1)
    ap.add_argument("--pw_pow", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--predict_only", action="store_true", help="only (re)write the predictions of best.pt")
    a = ap.parse_args()
    assert a.batch * a.accum == EFF_BATCH and a.epochs <= a.sched_epochs
    out = f"{paths.VIT_MODELS}/{a.modality}_f{a.fold}"
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(paths.SEED + a.fold); np.random.seed(paths.SEED + a.fold)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(a.device)
    store = InputStore(a.modality)
    model = (VideoViT(len(store.events)) if a.modality == "video" else AudioViT(len(store.events))).to(dev)
    sp = f"{out}/norm_stats.json"
    if not os.path.exists(sp):
        json.dump(norm_stats(store, fold_items(store, a.fold, "train"), seed=a.fold), open(sp, "w"))
    prep = Prep(a.modality, json.load(open(sp)), dev)
    logf = open(f"{out}/log.jsonl", "a")

    def log(**kw):
        kw["t"] = time.strftime("%F %T")
        print(json.dumps(kw), flush=True)
        logf.write(json.dumps(kw) + "\n"); logf.flush()

    if not a.predict_only:
        train(a, store, model, prep, out, dev, log)
    st = torch.load(f"{out}/best.pt", map_location="cpu")
    model.load_state_dict(st["model"])
    os.makedirs(f"{out}/preds", exist_ok=True)
    for c in paths.CENTERS:
        if not store.complete(c):
            log(stage="predict_skip", center=c)
            continue
        p = predict(model, prep, store, c, np.arange(len(store.rows[c]["uuid"])), 256, dev, a.workers)
        np.save(f"{out}/preds/{c}.npy", p)
        log(stage="predict", center=c, n=len(p), epoch=int(st["epoch"]))


if __name__ == "__main__":
    main()
