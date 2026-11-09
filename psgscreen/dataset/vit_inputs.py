"""Per-window input arrays of the ViT event detectors, in the row order of the window table.

Rows: the windows of dataset.windows (annotated and no-annotation exams) of the study roster, sorted by exam uuid
and window index.
Video (3 x 30 x 64 x 64 uint8 per window), from the raw optical flow of the window's video:
  1) per pixel fx, fy and |f| (px per frame pair; |f| at full resolution, before any averaging)
  2) 30 one-second bins (5 flow fields each), averaged over the fields whose sample time falls in the bin
  3) area-averaged to 64 x 64 (cv2.INTER_AREA)
  4) mu-law quantisation q = sign(v) ln(1 + |v| / a) / ln(1 + 20 / a), a = 0.1 px: fx, fy -> int8 round(127 q)
     (stored as uint8), |f| -> uint8 round(255 q). Empty bins are zero motion.
  Timing as in dataset.windows: Hospital A by clip index; Hospitals B and C on the recording clock from the
  clips that overlap the window (all their samples, shorter last clips included; magnitudes rescaled to the
  0.2-s frame interval) when the clip-time table agrees with the motion database, by clip index otherwise.
Audio (128 x 469 float16 per window): the clip's log-Mel spectrogram (dB), frames [0, 938) at 31.25 Hz (missing
  frames = 0), averaged over pairs of frames (64 ms); the number of valid pooled frames is stored with it.

    python -m psgscreen.dataset.vit_inputs rows --center SC
    python -m psgscreen.dataset.vit_inputs video --center SC [--workers 4]
    python -m psgscreen.dataset.vit_inputs audio --center SC
        -> vit_inputs/<centre>/{rows.npz, video_u8.npy, video_status.npy, audio_f16.npy, audio_valid.npy}
"""
import argparse
import io
import json
import multiprocessing as mp
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from psgscreen import paths
from psgscreen.dataset.store import CLIP_SEC, N_FLOW, Store, key_variants, split_key, uuid_of
from psgscreen.dataset.windows import clip_tables, overlapping_clips

S, NB = 64, 30                       # spatial size, one-second bins
CANON_DT = 0.2                       # flow sample interval (s) the magnitudes refer to
MEL_T = 938                          # 30 s at 31.25 frames / s
MEL_TP = MEL_T // 2                  # 469 pooled frames
MU_A = 0.1                           # px: knee of the logarithmic quantisation
MU_DEN = float(np.log1p(20.0 / MU_A))
_V = np.arange(256, dtype=np.float32) * (40.0 / 255.0) - 20.0          # flow code -> px
_MAGLUT = np.sqrt(_V[:, None] ** 2 + _V[None, :] ** 2).astype(np.float32)


def out_dir(center):
    d = f"{paths.VIT_INPUTS}/{center}"
    os.makedirs(d, exist_ok=True)
    return d


def load_rows(center):
    return np.load(f"{paths.VIT_INPUTS}/{center}/rows.npz", allow_pickle=False)


def exam_blocks(uu):
    """Sorted uuid array -> [(uuid, first row, end row)]."""
    b = np.flatnonzero(np.r_[True, uu[1:] != uu[:-1], True])
    return [(uu[b[i]], int(b[i]), int(b[i + 1])) for i in range(len(b) - 1)]


# ---------------------------------------------------------------- rows
def build_rows(center):
    ros = pd.read_csv(paths.ROSTER, encoding="utf-8-sig")
    want = set(ros.loc[ros.center == center, "uuid"].str.lower())
    parts, ev = [], None
    for src, stem in enumerate(("windows", "windows_noann")):
        p = f"{paths.WINDOWS}/{center}/{stem}_meta.npz"
        if not os.path.exists(p):
            continue
        m = np.load(p, allow_pickle=False)
        sel = np.flatnonzero(np.isin(m["exam_id"], list(want)))
        assert ev is None or (m["event_names"] == ev).all(), "event names differ between window tables"
        ev = m["event_names"]
        parts.append(dict(src=np.full(len(sel), src, np.int8), meta_row=sel, uuid=m["exam_id"][sel],
                          sidx=m["win_sidx"][sel].astype(np.int32), y=m["y_multi"][sel].astype(np.int8),
                          key=m["key"][sel]))
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    o = np.lexsort((cat["sidx"], cat["uuid"]))
    cat = {k: v[o] for k, v in cat.items()}
    assert not pd.Series(list(zip(cat["uuid"], cat["sidx"]))).duplicated().any(), "duplicate (exam, window) rows"
    np.savez(f"{out_dir(center)}/rows.npz", src=cat["src"], meta_row=cat["meta_row"], uuid=cat["uuid"],
             sidx=cat["sidx"], y_multi=cat["y"], key=cat["key"], event_names=ev)
    print(f"{center}: {len(o):,} rows, {len(set(cat['uuid']))} exams of {len(want)} in the roster")


# ---------------------------------------------------------------- video
class FlowSource:
    """Raw flow fields of a window: [(fields (n, H, W, 2) uint8, sample times (n,) s from the window start)]."""

    def __init__(self, center, flow_lmdb, tables):
        import lmdb
        import zstandard as zstd
        self.env = lmdb.open(flow_lmdb, readonly=True, lock=False, readahead=False, max_readers=256,
                             subdir=os.path.isdir(flow_lmdb))
        self.tables = {} if center == "SC" else tables
        self.zstd = zstd
        self.local = threading.local()
        self.cache = OrderedDict()                   # neighbouring windows share clips when the clocks drift
        self.pending = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(1)            # decompresses the next window's clips meanwhile

    def _read(self, k):
        dec = getattr(self.local, "dec", None) or self.zstd.ZstdDecompressor()
        self.local.dec = dec
        with self.env.begin(buffers=True) as txn:
            buf = txn.get(k.encode("utf-8"))
            raw = bytes(buf) if buf is not None else None
        if raw is None:
            return None
        a = np.load(io.BytesIO(dec.decompress(raw, max_output_size=400_000_000)), allow_pickle=False)
        return a if a.ndim == 4 and a.shape[-1] == 2 and a.shape[0] >= 1 else None

    def _get(self, k):
        with self.lock:
            if k in self.cache:
                return self.cache[k]
            fut = self.pending.pop(k, None)
        a = fut.result() if fut is not None else self._read(k)
        with self.lock:
            self.cache[k] = a
            while len(self.cache) > 3:
                self.cache.popitem(last=False)
        return a

    def _clip_keys(self, key):
        base, idx = split_key(key)
        tab = self.tables.get(uuid_of(key))
        if tab is None:
            return key_variants(key)
        return [k for j in overlapping_clips(tab, idx * CLIP_SEC, idx * CLIP_SEC + CLIP_SEC)
                for k in (f"{base}_{int(j):06d}", f"{base}_{int(j):04d}")]

    def prefetch(self, key):
        with self.lock:
            for k in self._clip_keys(key):
                if k not in self.cache and k not in self.pending:
                    self.pending[k] = self.pool.submit(self._read, k)

    def reset(self):
        with self.lock:
            for f in self.pending.values():
                f.cancel()
            self.pending.clear(); self.cache.clear()

    def window(self, key):
        """(parts, magnitude scale) or (None, None)."""
        base, idx = split_key(key)
        tab = self.tables.get(uuid_of(key))
        if tab is None:
            for k in key_variants(key):
                a = self._get(k)
                if a is not None:
                    n = min(a.shape[0], N_FLOW)
                    return [(a[:n], np.arange(n) * CANON_DT)], 1.0
            return None, None
        t0 = idx * CLIP_SEC
        parts, dts = [], []
        for j in overlapping_clips(tab, t0, t0 + CLIP_SEC):
            for k in (f"{base}_{int(j):06d}", f"{base}_{int(j):04d}"):
                a = self._get(k)
                if a is not None:
                    break
            if a is None:
                continue
            n = int(min(tab[j, 2], a.shape[0]))
            if n > 0:
                parts.append((a[:n], tab[j, 0] + np.arange(n) * tab[j, 1] - t0))
                dts.append(float(tab[j, 1]))
        if not parts:
            return None, None
        return parts, CANON_DT / float(np.median(dts))


_KERNEL = []


def _kernel():
    """One pass over the pixels: per bin, sums of the fx and fy codes (exact integers) and of |f| at full
    resolution."""
    if not _KERNEL:
        import numba

        @numba.njit(cache=True, nogil=True, parallel=True)
        def k(a, b, maglut, sfx, sfy, smg, cnt):
            n, H, W, _ = a.shape
            for t in range(n):
                bb = b[t]
                if bb < 0:
                    continue
                cnt[bb] += 1
                for i in numba.prange(H):
                    for j in range(W):
                        cx = a[t, i, j, 0]
                        cy = a[t, i, j, 1]
                        sfx[bb, i, j] += cx
                        sfy[bb, i, j] += cy
                        smg[bb, i, j] += maglut[cx, cy]
        _KERNEL.append(k)
    return _KERNEL[0]


def video_bins(parts, scale):
    """(3, 30, 64, 64) float32 bin means in px of fx, fy and |f|. Area averaging and bin averaging are linear,
    so the full-resolution bin sums are resized first and divided by the field count afterwards."""
    import cv2
    kern = _kernel()
    acc = np.zeros((3, NB, S, S), np.float32)
    cnt_t = np.zeros(NB, np.int64)
    for a, ts in parts:
        b = np.floor(ts + 1e-6).astype(np.int64)
        b[(b < 0) | (b >= NB)] = -1
        if (b < 0).all():
            continue
        n, H, W, _ = a.shape
        sfx, sfy = np.zeros((NB, H, W), np.int32), np.zeros((NB, H, W), np.int32)
        smg, cnt = np.zeros((NB, H, W), np.float32), np.zeros(NB, np.int64)
        kern(np.ascontiguousarray(a), b, _MAGLUT, sfx, sfy, smg, cnt)
        for bb in np.flatnonzero(cnt):
            acc[0, bb] += cv2.resize(sfx[bb].astype(np.float32), (S, S), interpolation=cv2.INTER_AREA)
            acc[1, bb] += cv2.resize(sfy[bb].astype(np.float32), (S, S), interpolation=cv2.INTER_AREA)
            acc[2, bb] += cv2.resize(smg[bb], (S, S), interpolation=cv2.INTER_AREA)
        cnt_t += cnt
    nz = cnt_t > 0
    c = cnt_t[nz].astype(np.float32)[:, None, None]
    out = np.zeros((3, NB, S, S), np.float32)
    out[0, nz] = acc[0, nz] / c * np.float32(40.0 / 255.0) - np.float32(20.0)
    out[1, nz] = acc[1, nz] / c * np.float32(40.0 / 255.0) - np.float32(20.0)
    out[2, nz] = acc[2, nz] / c
    return out * np.float32(scale)


def quantize(acc):
    q = np.empty((3, NB, S, S), np.uint8)
    for c in (0, 1):
        v = acc[c]
        z = np.sign(v) * np.log1p(np.abs(v) / MU_A) / MU_DEN
        q[c] = np.clip(np.rint(127.0 * z), -127, 127).astype(np.int8).view(np.uint8)
    z = np.log1p(np.maximum(acc[2], 0.0) / MU_A) / MU_DEN
    q[2] = np.clip(np.rint(255.0 * z), 0, 255).astype(np.uint8)
    return q


def to_model_scale(q):
    """uint8 video input -> log-scale values the model sees (fx, fy in [-1, 1], |f| in [0, 1]), float32."""
    q = np.asarray(q)
    x = np.empty(q.shape, np.float32)
    x[..., 0, :, :, :] = q[..., 0, :, :, :].view(np.int8) / 127.0
    x[..., 1, :, :, :] = q[..., 1, :, :, :].view(np.int8) / 127.0
    x[..., 2, :, :, :] = q[..., 2, :, :, :] / 255.0
    return x


_W = {}


def _video_init(center, flow_lmdb, tables):
    import cv2
    cv2.setNumThreads(1)
    d = out_dir(center)
    _W.update(src=FlowSource(center, flow_lmdb, tables), keys=load_rows(center)["key"],
              X=np.load(f"{d}/video_u8.npy", mmap_mode="r+"), st=np.load(f"{d}/video_status.npy", mmap_mode="r+"))


def _video_exam(task):
    u, s0, s1 = task
    src, X, ST, keys = _W["src"], _W["X"], _W["st"], _W["keys"]
    src.reset()
    n_miss = 0
    for r in range(s0, s1):
        if r + 1 < s1:
            src.prefetch(str(keys[r + 1]))
        parts, scale = src.window(str(keys[r]))
        if parts is None:
            X[r], ST[r] = 0, -1
            n_miss += 1
        else:
            X[r], ST[r] = quantize(video_bins(parts, scale)), 1
    X.flush(); ST.flush()
    return u, s1 - s0 - n_miss, n_miss


# ---------------------------------------------------------------- audio
def audio_window(mel):
    """(128, 469) float32 pooled log-Mel and the number of valid pooled frames."""
    win = np.zeros((mel.shape[0], MEL_T), np.float32)
    n = min(mel.shape[1], MEL_T)
    win[:, :n] = mel[:, :n]
    return win.reshape(mel.shape[0], MEL_TP, 2).mean(-1), min(n // 2, MEL_TP)


def _audio_init(center):
    d = out_dir(center)
    _W.update(store=Store(center), keys=load_rows(center)["key"],
              A=np.load(f"{d}/audio_f16.npy", mmap_mode="r+"), V=np.load(f"{d}/audio_valid.npy", mmap_mode="r+"))


def _audio_exam(task):
    u, s0, s1 = task
    st, A, V, keys = _W["store"], _W["A"], _W["V"], _W["keys"]
    n_miss = 0
    for r in range(s0, s1):
        mel, _ = st.audio(str(keys[r]))
        if mel is None:
            A[r], V[r] = 0, 0
            n_miss += 1
            continue
        w, nv = audio_window(mel)
        A[r], V[r] = w.astype(np.float16), nv
    A.flush(); V.flush()
    return u, s1 - s0 - n_miss, n_miss


# ---------------------------------------------------------------- driver
def run(kind, center, workers, flow_lmdb=None, exams=None):
    rows = load_rows(center)
    N, d = len(rows["uuid"]), out_dir(center)
    files = {"video": [("video_u8.npy", np.uint8, (N, 3, NB, S, S)), ("video_status.npy", np.int8, (N,))],
             "audio": [("audio_f16.npy", np.float16, (N, 128, MEL_TP)), ("audio_valid.npy", np.int16, (N,))]}[kind]
    for name, dt, shape in files:
        if not os.path.exists(f"{d}/{name}"):
            np.lib.format.open_memmap(f"{d}/{name}", mode="w+", dtype=dt, shape=shape).flush()
    done_p = f"{d}/{kind}_done.txt"
    done = {l.split("\t")[0] for l in open(done_p)} if os.path.exists(done_p) else set()
    todo = [b for b in exam_blocks(rows["uuid"]) if b[0] not in done and (exams is None or b[0] in exams)]
    todo.sort(key=lambda b: -(b[2] - b[1]))
    if kind == "video":
        init, work = _video_init, _video_exam
        args = (center, flow_lmdb or paths.lmdb("flow", center), clip_tables(center, Store(center)))
    else:
        init, work, args = _audio_init, _audio_exam, (center,)
    print(f"{kind} {center}: {len(todo)} exams to do ({sum(b[2] - b[1] for b in todo):,} windows)", flush=True)
    t0 = time.time()
    with mp.Pool(workers, initializer=init, initargs=args) as pool, open(done_p, "a") as fo:
        for u, n_ok, n_miss in pool.imap_unordered(work, todo, chunksize=1):
            fo.write(f"{u}\t{n_ok}\t{n_miss}\n"); fo.flush()
            print(f"  {u[:8]}  {n_ok} windows, {n_miss} missing  ({time.time() - t0:.0f} s)", flush=True)
    if kind == "video":
        json.dump(dict(encoding="mu-law", a_px=MU_A, max_px=20.0, bin_s=1, size=S),
                  open(f"{d}/video_encoding.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["rows", "video", "audio"])
    ap.add_argument("--center", required=True, choices=list(paths.CENTERS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--flow_lmdb", default=None, help="default: the centre's flow database")
    ap.add_argument("--exams", default=None, help="file with exam uuids to process (default: all)")
    a = ap.parse_args()
    if a.step == "rows":
        build_rows(a.center)
    else:
        exams = {uuid_of(l) for l in open(a.exams) if l.strip()} if a.exams else None
        run(a.step, a.center, a.workers, a.flow_lmdb, exams)


if __name__ == "__main__":
    main()
