"""Window table: one 30-s window per audio clip of every exam, with multi-label event targets.

These windows are the rows of the event detectors (model.vit) and of the detection tables.
Targets (per event; 14 events, the RBD-related labels p, t, m1 and rbd merged into "rbd"): a window is positive
when an annotated event overlaps it by at least half of min(event duration, 30 s) (instantaneous events: the
window containing the time point); windows overlapped less, or within 1 s outside an event, are indeterminate
(-1, ignored in training and evaluation); all others are negative.
A window is kept when its audio clip has at least 10 valid frames and the motion records (p95 flow magnitude,
flow-magnitude histogram, mean flow) of the video covering it exist.
Clip timing: optical flow restarts its 30-s clips at every video file, so clip indices drift ahead of the audio
clock. Hospital A: the video clip with the same index is used. Hospitals B and C: when the exam's clip-time table
(alignment/clip_times_<centre>.npz) agrees with the motion database, the video covering [30 i, 30 i + 30) s is
taken from the clips that overlap it on the recording clock; otherwise the clip with the same index is used.
Agreement: Hospital B, table rows within 2 of the number of motion records; Hospital C, table rows within 2 of the
highest motion clip index + 1, and the shorter last clip of each video file counts with its own samples.

    python -m psgscreen.dataset.windows --center SC [--no_annotation] [--workers 6]
        -> windows/<centre>/windows_meta.npz   (--no_annotation: windows_noann_meta.npz, all targets -1)
"""
import argparse
import math
import multiprocessing as mp
import os
import re
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd

from psgscreen import paths
from psgscreen.dataset.store import AUDIO_HZ, CLIP_SEC, FLOW_HZ, N_FLOW, Store, split_key, uuid_of

EVENTS = ["anychinemg activity", "arousal", "candidate signal", "central apnea", "hypopnea", "limb movement",
          "mixed apnea", "obstructive apnea", "obstructive hypopnea", "oximeter event", "oxygen desaturation", "rbd",
          "snoring", "spo2 threshold event"]
RBD_LABELS = {"rbd", "p", "t", "m1"}
MS_EDGES = np.linspace(0.0, np.sqrt(20.0), 65) ** 2
MIN_VALID = 10
GATE = 2                                         # max disagreement between a clip-time table and the motion records
GATE_SPAN = {"YI"}                               # centres checked against the motion clip-index span
SHORT_OK = {"YI"}                                # centres whose shorter last clip per video file is used
POS_FRAC, GUARD = 0.5, 1.0


# ---------------------------------------------------------------- annotations and targets
def canonical_event(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    raw = str(s).strip()
    if not raw:
        return ""
    if re.fullmatch(r"^[\W_]+$", raw):           # symbol-only marks are scored as "candidate signal"
        return "candidate signal"
    s = re.sub(r"^[?!\.,;:~'\"`>]+|[\)\]\}]+$|[?!\.,;:~'\"`>]+$", "", raw)
    s = re.sub(r"^[\(\[\{]+", "", s)
    s = re.sub(r"\s+", " ", s.replace("_", " ").replace("-", " ")).strip().casefold()
    return "rbd" if s in RBD_LABELS else s


def hms_to_sec(t):
    try:
        h, m, s = str(t).split(":")[:3]
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return None


def read_annotations(path, events=EVENTS):
    """{event: [(start s, duration s), ...]} of one exam, in file order."""
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, encoding="cp949", on_bad_lines="skip")
    df.columns = [c.strip().lower() for c in df.columns]
    out = defaultdict(list)
    for row in df.itertuples(index=False):
        ev = canonical_event(getattr(row, "event", ""))
        t = hms_to_sec(getattr(row, "time", "")) if ev in events else None
        if t is None:
            continue
        d = float(getattr(row, "duration", 0)) if "duration" in df.columns else 0.0
        out[ev].append((float(t), max(d, 0.0) if np.isfinite(d) else 0.0))
    return dict(out)


def label_windows(events, W=CLIP_SEC):
    """{window index: int8 targets} for windows touched by an event; untouched windows are all-negative."""
    lab = {}

    def row(g):
        return lab.setdefault(g, np.zeros(len(EVENTS), np.int8))

    def ignore(g, i):
        r = row(g)
        if r[i] == 0:
            r[i] = -1

    for ev, lst in events.items():
        i = EVENTS.index(ev)
        for t0, d in lst:
            if d <= 0:
                row(int(math.floor(t0 / W)))[i] = 1
                ignore(int(math.floor((t0 - GUARD) / W)), i)
                ignore(int(math.floor((t0 + GUARD) / W)), i)
                continue
            t1 = t0 + d
            thr = POS_FRAC * min(d, W) - 1e-9
            for g in range(int(math.floor(t0 / W)), int(math.ceil(t1 / W))):
                ov = min(t1, (g + 1) * W) - max(t0, g * W)
                if ov <= 0:
                    continue
                if ov >= thr:
                    row(g)[i] = 1
                else:
                    ignore(g, i)
            ignore(int(math.floor((t0 - GUARD) / W)), i)
            ignore(int(math.floor((t1 + GUARD - 1e-9) / W)), i)
    return lab


# ---------------------------------------------------------------- video clips on the recording clock
_CENTERS = (MS_EDGES[:-1] + MS_EDGES[1:]) / 2.0


def overlapping_clips(table, t0, t1):
    """Rows of a clip-time table (start s, sample interval s, valid samples) that overlap [t0, t1)."""
    s, dt, nv = table[:, 0], table[:, 1], table[:, 2]
    return np.where((s < t1) & (s + np.maximum(nv, 1) * dt > t0))[0]


def aligned(store, table, key, read, kind):
    """150-sample series of [30 i, 30 i + 30) s from the flow clips that overlap it (linear interpolation to
    5 Hz; flow magnitudes rescaled to the 0.2-s frame interval)."""
    base, idx = split_key(key)
    t0 = idx * CLIP_SEC
    ts, vs, dts = [], [], []
    for j in overlapping_clips(table, t0, t0 + CLIP_SEC):
        arr = next((a for a in (read(f"{base}_{int(j):06d}"), read(f"{base}_{int(j):04d}")) if a is not None), None)
        if arr is None:
            continue
        n = int(min(table[j, 2], arr.shape[0]))
        if n > 0:
            ts.append(table[j, 0] + np.arange(n) * table[j, 1]); vs.append(arr[:n]); dts.append(float(table[j, 1]))
    if not ts:
        return None
    T, V = np.concatenate(ts), np.concatenate(vs, axis=0)
    o = np.argsort(T, kind="stable")
    T, V = T[o], V[o]
    u = np.concatenate(([True], np.diff(T) > 1e-9))
    T, V = T[u], V[u]
    if len(T) < 2:
        return None
    grid = t0 + np.arange(N_FLOW) * (1.0 / FLOW_HZ)
    if V.ndim == 2:
        out = np.empty((N_FLOW, V.shape[1]), dtype=np.float32)
        for b in range(V.shape[1]):
            out[:, b] = np.interp(grid, T, V[:, b])
    else:
        out = np.interp(grid, T, V).astype(np.float32)
    scale = (1.0 / FLOW_HZ) / float(np.median(dts))
    if abs(scale - 1.0) > 1e-6:
        if kind == "magspec":                     # histogram over magnitude: stretch the bin axis
            src = _CENTERS / scale
            for i in range(out.shape[0]):
                out[i] = np.interp(_CENTERS, src, out[i]) / scale
        else:
            out = out * scale
    return out


def clip_tables(center, store):
    """Clip-time tables that agree with the motion database (see the module docstring)."""
    p = paths.clip_times(center)
    if center == "SC" or not os.path.exists(p):
        return {}
    z, counts = np.load(p), store.motion_clip_counts(index_span=center in GATE_SPAN)
    return {u: z[u] for u in z.files if u in counts and abs(len(z[u]) - counts[u]) <= GATE}


# ---------------------------------------------------------------- window availability
_G = {}


def _init(center, tables):
    _G["store"], _G["table"], _G["short"] = Store(center), tables, center in SHORT_OK


@lru_cache(maxsize=512)
def _clip(key):
    st, tab, short = _G["store"], _G["table"].get(uuid_of(key)), _G["short"]
    mel, cqt = st.audio(key)
    if tab is None:
        mag, ms, rep = st.first(st.motion_record, key), st.first(st.magspec_record, key), st.first(st.flow_rep_record, key)
    else:
        mag = aligned(st, tab, key, lambda k: st.motion_record(k, short), "linear")
        ms = aligned(st, tab, key, lambda k: st.magspec_record(k, short), "magspec")
        rep = aligned(st, tab, key, st.flow_rep_record, "linear")
    return mel, cqt, mag, ms, rep


def _worker(batch):
    rows, n_drop = [], 0
    for key, ri in batch:
        mel, cqt, mag, ms, rep = _clip(key)
        if mel is None or cqt is None or mag is None or ms is None or rep is None:
            n_drop += 1
            continue
        n_audio = min(mel.shape[1], cqt.shape[1], int(np.round(CLIP_SEC * AUDIO_HZ)))
        n_video = min(len(mag), ms.shape[0], len(rep), N_FLOW)
        if n_audio < MIN_VALID or n_video < MIN_VALID:
            n_drop += 1
            continue
        rows.append(ri)
    return np.asarray(rows, np.int64), n_drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", required=True, choices=list(paths.CENTERS))
    ap.add_argument("--no_annotation", action="store_true", help="exams without annotations (targets all -1)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    inv = pd.read_csv(paths.INVENTORY, encoding="utf-8-sig")
    inv = inv[inv.center == a.center]
    # exams are processed in the order of their name in the feature databases (A: exam name, B, C: uuid)
    name = dict(zip(inv.uuid.str.lower(), inv.exam_id.astype(str) if a.center == "SC" else inv.uuid.str.lower()))

    events, d = {}, os.path.join(paths.ANNOTATIONS, a.center)
    for f in sorted(os.listdir(d)):
        if f.endswith(".csv") and uuid_of(f) in name:
            events[uuid_of(f)] = read_annotations(os.path.join(d, f))
    store = Store(a.center)
    clips = defaultdict(dict)
    for k in store.audio_keys():
        u, (_, idx) = uuid_of(k), split_key(k)
        if u in name and idx is not None:
            clips[u][idx] = k
    # annotated = at least one target event; the others form the no-annotation set
    exams = sorted((u for u in clips if bool(events.get(u)) != a.no_annotation), key=lambda u: name[u])
    tables = clip_tables(a.center, store)
    print(f"{a.center}: {len(exams)} exams, clip-time tables for {len(set(tables) & set(exams))}")

    keys, Y, U, S = [], [], [], []
    for u in exams:
        lab = {} if a.no_annotation else label_windows(events[u])
        for sidx in sorted(clips[u]):
            y = np.full(len(EVENTS), -1, np.int8) if a.no_annotation else lab.get(sidx, np.zeros(len(EVENTS), np.int8))
            keys.append((clips[u][sidx], len(Y)))
            Y.append(y); U.append(u); S.append(sidx)

    ok = np.zeros(len(Y), bool)
    batches = [keys[i:i + a.batch] for i in range(0, len(keys), a.batch)]
    n_drop = 0
    with mp.Pool(a.workers, initializer=_init, initargs=(a.center, tables)) as pool:
        for rows, nd in pool.imap_unordered(_worker, batches, chunksize=2):
            ok[rows] = True; n_drop += nd
    keep = np.flatnonzero(ok)
    out = f"{paths.WINDOWS}/{a.center}"
    os.makedirs(out, exist_ok=True)
    stem = f"{out}/windows{'_noann' if a.no_annotation else ''}"
    np.savez(f"{stem}_meta.npz", y_multi=np.stack(Y)[keep].astype(np.int8),
             exam_id=np.array(U, dtype="<U40")[keep], win_sidx=np.array(S, np.int32)[keep],
             key=np.array([k for k, _ in keys], dtype="<U200")[keep], event_names=np.array(EVENTS, dtype="<U64"))
    print(f"{len(keep):,} windows ({n_drop:,} dropped) -> {stem}_meta.npz")


if __name__ == "__main__":
    main()
