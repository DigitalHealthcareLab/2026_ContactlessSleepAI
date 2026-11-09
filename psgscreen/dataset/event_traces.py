"""Event-locked raw signal around annotated events (event-detection figure c, d), development exams of
Hospitals A and B.

Exams: development exams whose clip-time table agrees with the motion database (the video series are taken on
the recording clock). Per event type, every exam contributes at most 30 instances (random subset), and 2,000
instances are drawn from the pool (Hospital B: the same fraction of its pool as drawn in Hospital A). Each
instance gets a control time of the same exam, drawn uniformly and at least 30 s away from any instance of that
event. Traces span -15 to +15 s around the event midpoint (control: the drawn time):
  aud  sound level, 10 log10 of the summed Mel-band power per audio frame (31.25 Hz)
  vid  p95 optical-flow magnitude (5 Hz)
Traces need >= 90 % coverage by audio, motion and flow-histogram data. Exams are visited in the listing order
of the annotation folder; random draws: numpy default_rng(42) for A, (43) for B.
Two draws are used by the figure: the default event list and, separately (its own random stream), hypopnea.

    python -m psgscreen.dataset.event_traces --center INT                          -> traces/traces_<centre>.npz
    python -m psgscreen.dataset.event_traces --center INT --events hypopnea --name hypopnea
                                                                                    -> traces/traces_hypopnea_<centre>.npz
"""
import argparse
import bisect
import json
import multiprocessing as mp
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from psgscreen import paths
from psgscreen.dataset.store import AUDIO_HZ, CLIP_SEC, FLOW_HZ, Store, split_key, uuid_of
from psgscreen.dataset.windows import GATE, aligned, read_annotations

EVENTS = ["oximeter event", "snoring", "obstructive hypopnea", "rbd", "limb movement", "central apnea",
          "obstructive apnea"]
N_INST, PER_EXAM, PRE, POST, MIN_GAP, MIN_VALID, SEED = 2000, 30, 15.0, 15.0, 30.0, 0.9, 42
T_AUD = np.arange(-PRE, POST, 1.0 / AUDIO_HZ).astype(np.float32)
T_VID = np.arange(-PRE, POST, 1.0 / FLOW_HZ).astype(np.float32)
_G = {}


def _init(center, tables):
    _G["store"], _G["tables"] = Store(center), tables


def _clip_series(cache, key):
    if key not in cache:
        st, tab = _G["store"], _G["tables"][uuid_of(key)]
        mel, _ = st.audio(key)
        mag = aligned(st, tab, key, st.motion_record, "linear")
        ms = aligned(st, tab, key, st.magspec_record, "magspec")
        lvl = None
        if mel is not None and mel.ndim == 2 and mel.shape[1] >= 900:
            lvl = (10.0 * np.log10(np.power(10.0, mel.astype(np.float64) / 10.0).sum(axis=0) + 1e-20)).astype(np.float32)
        mag = None if mag is None or np.asarray(mag).size != 150 else np.asarray(mag, np.float32).reshape(-1)
        cov = np.ones(150, np.float32) if ms is not None and ms.ndim == 2 and ms.shape == (150, 64) else None
        cache[key] = (lvl, mag, cov)
    return cache[key]


def _worker(task):
    clips, items = task
    cache, out = {}, []
    for tid, t0 in items:
        src = {"aud": ([], []), "vid": ([], []), "cov": ([], [])}
        for s in range(max(int(np.floor((t0 - PRE) / CLIP_SEC)), 0), int(np.floor((t0 + POST - 1e-6) / CLIP_SEC)) + 1):
            key = clips.get(s)
            if key is None:
                continue
            for name, x, hz in zip(("aud", "vid", "cov"), _clip_series(cache, key), (AUDIO_HZ, FLOW_HZ, FLOW_HZ)):
                if x is not None:
                    src[name][0].append(s * CLIP_SEC + np.arange(len(x)) / hz); src[name][1].append(x)
        res = {}
        for name, grid in (("aud", T_AUD), ("vid", T_VID), ("cov", T_VID)):
            dst = np.full(len(grid), np.nan, np.float32)
            tt, xx = src[name]
            if tt:
                tt, xx = np.concatenate(tt), np.concatenate(xx)
                o = np.argsort(tt)
                tt, xx = tt[o], xx[o]
                q = t0 + grid
                m = (q >= tt[0]) & (q <= tt[-1])
                if m.any():
                    val = np.interp(q[m], tt, xx)
                    for b in np.flatnonzero(np.diff(tt) > 2.5 * np.median(np.diff(tt))):   # no values across gaps
                        val[(q[m] > tt[b]) & (q[m] < tt[b + 1])] = np.nan
                    dst[m] = val
            res[name] = dst
        out.append((tid, res["aud"], res["vid"], res["cov"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", required=True, choices=list(paths.INTERNAL))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--events", nargs="+", default=EVENTS)
    ap.add_argument("--name", default="", help="output tag: traces/traces_<name>_<centre>.npz")
    a = ap.parse_args()
    c, events, tag = a.center, a.events, f"_{a.name}" if a.name else ""
    ros = pd.read_csv(paths.ROSTER, encoding="utf-8-sig")
    dev = set(ros.loc[(ros.center == c) & (ros.split == "dev"), "uuid"].str.lower())
    store = Store(c)
    z, counts = np.load(paths.clip_times(c)), store.motion_clip_counts()
    tables = {u: z[u] for u in z.files if u in counts and abs(len(z[u]) - counts[u]) <= GATE}
    clips = defaultdict(dict)
    for k in store.audio_keys():
        u, (_, idx) = uuid_of(k), split_key(k)
        if u in dev and u in tables and idx is not None:
            clips[u][idx] = k
    d = f"{paths.ANNOTATIONS}/{c}"
    ann = {}
    for f in os.listdir(d):                                   # listing order (fixes the order of the random draws)
        u = uuid_of(f)
        if f.endswith(".csv") and u in clips:
            e = read_annotations(os.path.join(d, f), events)
            if e:
                ann[u] = e
    print(f"{c}: {len(clips)} development exams with aligned video, {len(ann)} with these events")

    frac = {}
    if c != "INT":                                            # the fractions drawn from the Hospital A pools
        frac = json.loads(str(np.load(f"{paths.TRACES}/traces{tag}_INT.npz")["cfg"]))["fraction"]
    rng = np.random.default_rng(SEED + paths.INTERNAL.index(c))
    recs, drawn = [], {}
    for e in events:
        pool = []
        for u, evm in ann.items():
            lst = evm.get(e)
            if not lst:
                continue
            idx = np.arange(len(lst))
            if len(lst) > PER_EXAM:
                idx = rng.choice(idx, size=PER_EXAM, replace=False)
            pool += [(u, float(lst[i][0]), float(lst[i][1])) for i in idx]
        n_take = min(int(round(frac[e] * len(pool))) if e in frac else min(N_INST, len(pool)), len(pool))
        drawn[e] = round(n_take / len(pool), 6) if pool else 1.0
        for u, t0, dur in ([pool[i] for i in rng.choice(len(pool), size=n_take, replace=False)] if pool else []):
            recs.append(dict(event=e, group="event", uuid=u, t0=t0, dur=dur))
            times = np.sort([t for t, _ in ann[u][e]])
            lo, hi = PRE, (max(clips[u]) + 1) * CLIP_SEC - POST
            for _ in range(200):
                if hi <= lo:
                    break
                tc = float(rng.uniform(lo, hi))
                j = bisect.bisect_left(times, tc)
                near = [abs(tc - times[k]) for k in (j - 1, j) if 0 <= k < len(times)]
                if not near or min(near) >= MIN_GAP:
                    recs.append(dict(event=e, group="control", uuid=u, t0=tc, dur=np.nan))
                    break
    R = pd.DataFrame(recs)
    d_ = R["dur"].astype(float).fillna(0.0).clip(lower=0.0)
    R["anchor"] = R["t0"] + np.where(R["group"].values == "event", d_.values / 2.0, 0.0)
    tasks = sorted(((clips[u], [(int(i), float(t)) for i, t in zip(g.index, g["anchor"])]) for u, g in R.groupby("uuid")),
                   key=lambda t: next(iter(t[0].values())))
    AUD = np.full((len(R), len(T_AUD)), np.nan, np.float32)
    VID = np.full((len(R), len(T_VID)), np.nan, np.float32)
    COV = np.full((len(R), len(T_VID)), np.nan, np.float32)
    with mp.Pool(a.workers, initializer=_init, initargs=(c, tables)) as pool:
        for res in pool.imap_unordered(_worker, tasks, chunksize=1):
            for tid, aud, vid, cov in res:
                AUD[tid], VID[tid], COV[tid] = aud, vid, cov
    ok = ((np.isfinite(AUD).mean(1) >= MIN_VALID) & (np.isfinite(VID).mean(1) >= MIN_VALID)
          & (np.isfinite(COV).mean(1) >= MIN_VALID))
    R, AUD, VID = R[ok].reset_index(drop=True), AUD[ok], VID[ok]
    os.makedirs(paths.TRACES, exist_ok=True)
    np.savez_compressed(f"{paths.TRACES}/traces{tag}_{c}.npz", event=R["event"].values.astype("<U32"),
                        group=R["group"].values.astype("<U16"), uuid=R["uuid"].values.astype("<U40"),
                        t0=R["t0"].values.astype(np.float32), dur=R["dur"].values.astype(np.float32),
                        aud=AUD, vid=VID, t_aud=T_AUD, t_vid=T_VID, cfg=json.dumps(dict(fraction=drawn)))
    print(R.groupby(["event", "group"]).size().to_string())


if __name__ == "__main__":
    main()
