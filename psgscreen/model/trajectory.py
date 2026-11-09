"""Cumulative predictions over the recording: scores after the first 5, 10, 15, ... minutes.

For each locked internal test exam, the exam features are recomputed from the window probabilities of the
first t minutes only (same 39 statistics) and scored by the 10 fold models of the disorder classifier
(mean over folds), giving p_abnormal(t), q_OSA(t) and q_RBD(t).
Input: window probabilities written by `exam_features --dump_probs`.

    python -m psgscreen.model.trajectory
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from psgscreen import paths
from psgscreen.model import disorder as dm

BLOCK_MIN = 5.0


def block_features(probs, events):
    out = {}
    for i, ev in enumerate(events):
        p = probs[:, i]
        out[f"{ev}__mean_prob"] = float(p.mean())
        out[f"{ev}__max_prob"] = float(p.max())
        out[f"{ev}__std_prob"] = float(p.std())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default=paths.WINDOW_PROBS)
    ap.add_argument("--nthread", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(paths.RESULTS, exist_ok=True)

    mg, feats, Xs = dm.load_fold_tables()
    y, di, _, _ = dm.cohort_index(mg)
    fold = mg["fold"].values
    cfg1, cfg2 = dm.load_params()
    heads = [dm.fit_fold(Xs[k], y, di, fold, k, cfg1, cfg2, nthread=a.nthread) for k in range(dm.NF)]
    is_dev = np.zeros(len(mg), bool); is_dev[di] = True
    info = {str(r.exam_id): r for r in mg[~is_dev].itertuples()}

    rows = []
    for f0 in sorted(glob.glob(f"{a.probs}/f0/*.npz")):
        name = os.path.basename(f0)
        eid = str(np.load(f0, allow_pickle=True)["exam_id"])
        m = info.get(eid)
        if m is None:
            continue
        Z = [np.load(f"{a.probs}/f{k}/{name}", allow_pickle=True) for k in range(dm.NF)]
        stride = float(Z[0]["stride_sec"]); nfull = len(Z[0]["probs"])
        bounds = list(range(int(round(BLOCK_MIN * 60 / stride)), nfull + 1, int(round(BLOCK_MIN * 60 / stride))))
        if not bounds:
            continue
        acc = np.zeros((len(bounds), 3), np.float64)
        for k, z in enumerate(Z):
            P, ev = z["probs"], [str(x) for x in z["events"]]
            M = np.empty((len(bounds), len(feats)), np.float32)
            for j, b1 in enumerate(bounds):
                fe = block_features(P[:b1], ev)
                M[j] = [fe.get(c, 0.0) for c in feats]
            D = xgb.DMatrix(np.nan_to_num(M))
            for h, b in enumerate(heads[k]):
                acc[:, h] += b.predict(D, iteration_range=(0, b.best_iteration + 1))
        acc /= dm.NF
        for j, b1 in enumerate(bounds):
            rows.append(dict(exam_id=eid, center=m.center, label=m.label, t_min=(b1 * stride) / 60.0,
                             p_abnormal=float(acc[j, 0]), p_normal=float(1.0 - acc[j, 0]),
                             q_OSA=float(acc[j, 1]), q_RBD=float(acc[j, 2])))
    out = f"{paths.RESULTS}/trajectory_5min.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{len(rows):,} rows -> {out}")


if __name__ == "__main__":
    main()
