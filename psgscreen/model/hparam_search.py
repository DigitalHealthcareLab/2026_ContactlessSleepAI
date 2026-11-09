"""Hyper-parameter search for the stage-2 heads (OSA, RBD), on development out-of-fold predictions only.

The locked internal test set and the external set are never used. For every configuration of the grid
(6,912 = max_depth 6 x eta 4 x min_child_weight 4 x subsample 3 x colsample_bytree 4 x reg_lambda 3 x gamma 2,
class weighting spw = "full") each head is trained on the abnormal exams of every fold and scored out of fold.
`pick` keeps, per head, the configuration with the highest out-of-fold AUC (the first in grid order on ties).
--variant audio_only / video_only: the same search on the single-modality exam features.

    python -m psgscreen.model.hparam_search grid --shard 0/16 [--variant ...]
    python -m psgscreen.model.hparam_search pick [--variant ...]     -> hparam/[<variant>/]stage2_best.json
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from psgscreen import paths
from psgscreen.model import disorder as dm

GRID = dict(
    max_depth=[2, 3, 4, 5, 6, 8],
    eta=[0.02, 0.05, 0.10, 0.20],
    min_child_weight=[1, 3, 10, 30],
    subsample=[0.6, 0.8, 1.0],
    colsample_bytree=[0.3, 0.5, 0.8, 1.0],
    reg_lambda=[1.0, 5.0, 30.0],
    gamma=[0.0, 1.0],
    spw=["full"],
)
KEYS = sorted(GRID)


def enumerate_grid():
    return [dict(zip(KEYS, v)) for v in itertools.product(*[GRID[k] for k in KEYS])]


def build_cache(Xs, y, di, fold):
    """Per-fold DMatrices of the abnormal development exams (train / validation / held-out)."""
    cache = {}
    for k in range(dm.NF):
        X = Xs[k]
        te, va, tr = dm.fold_rows(fold, di, k)
        atr, ava, ate = tr[y[tr] != 0], va[y[va] != 0], te[y[te] != 0]
        cache[k] = dict(dtr=xgb.DMatrix(X[atr]), dva=xgb.DMatrix(X[ava]), dte=xgb.DMatrix(X[ate]),
                        ytr=y[atr], yva=y[ava], yte=y[ate])
    return cache


def eval_cfg(cache, cfg, head, nthread, seed=dm.SEED):
    """Out-of-fold AUC of one configuration for one head."""
    cs = dm.HEAD_CLASSES[head]
    P, Y = [], []
    for k in range(dm.NF):
        c = cache[k]
        ytr = np.isin(c["ytr"], cs).astype(np.float32)
        c["dtr"].set_label(ytr)
        c["dva"].set_label(np.isin(c["yva"], cs).astype(np.float32))
        par = {**dm.make_params(cfg, ytr, seed), "nthread": nthread}
        b = xgb.train(par, c["dtr"], num_boost_round=dm.ROUNDS, evals=[(c["dva"], "v")],
                      early_stopping_rounds=dm.EARLY_STOP, verbose_eval=False)
        P.append(b.predict(c["dte"], iteration_range=(0, b.best_iteration + 1)))
        Y.append(np.isin(c["yte"], cs).astype(int))
    return dm.auc(np.concatenate(Y), np.concatenate(P))


def load_cache(variant):
    mg, _, Xs = dm.load_fold_tables(dm.VARIANT_FEATURES.get(variant))
    y, di, _, _ = dm.cohort_index(mg)
    return build_cache(Xs, y, di, mg["fold"].values)


def out_root(variant):
    return f"{paths.HPARAM}/{variant}" if variant else paths.HPARAM


def grid(shard, nthread, variant):
    i_sh, n_sh = (int(x) for x in shard.split("/"))
    out_dir = f"{out_root(variant)}/stage2_grid"
    os.makedirs(out_dir, exist_ok=True)
    cache = load_cache(variant)
    rows = []
    for n, cfg in enumerate(enumerate_grid()[i_sh::n_sh]):
        gid = i_sh + n * n_sh
        for h in ("OSA", "RBD"):
            rows.append(dict(gid=gid, head=h, **cfg, AUC=eval_cfg(cache, cfg, h, nthread)))
        if (n + 1) % 25 == 0:
            pd.DataFrame(rows).to_csv(f"{out_dir}/shard_{i_sh:03d}.csv", index=False, encoding="utf-8-sig")
            print(f"  {n + 1} configurations", flush=True)
    pd.DataFrame(rows).to_csv(f"{out_dir}/shard_{i_sh:03d}.csv", index=False, encoding="utf-8-sig")


def pick(variant):
    D = pd.concat([pd.read_csv(f, encoding="utf-8-sig")
                   for f in sorted(glob.glob(f"{out_root(variant)}/stage2_grid/shard_*.csv"))], ignore_index=True)
    D.columns = [c.lstrip("\ufeff") for c in D.columns]
    D = D.drop_duplicates(subset=["gid", "head"], keep="last").sort_values("gid")
    assert len(D) == 2 * len(enumerate_grid()), "grid incomplete"
    best = {}
    for h in ("OSA", "RBD"):
        r = D[D["head"] == h].sort_values("AUC", ascending=False, kind="stable").iloc[0]
        best[h] = {k: (int(r[k]) if k in ("max_depth", "min_child_weight") else (r[k] if k == "spw" else float(r[k])))
                   for k in KEYS}
        print(f"  {h}: gid {int(r.gid)}, out-of-fold AUC {r.AUC:.4f}", flush=True)
    out = f"{out_root(variant)}/stage2_best.json"
    json.dump(best, open(out, "w"), indent=1)
    print(json.dumps(best, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid"); g.add_argument("--shard", default="0/1"); g.add_argument("--nthread", type=int, default=2)
    p = sub.add_parser("pick")
    for q in (g, p):
        q.add_argument("--variant", default=None, choices=list(dm.VARIANT_FEATURES))
    a = ap.parse_args()
    grid(a.shard, a.nthread, a.variant) if a.cmd == "grid" else pick(a.variant)


if __name__ == "__main__":
    main()
