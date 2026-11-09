"""Two-stage disorder classifier on the 39 exam features.

Stage 1  Abnormal vs Normal (all development exams).
Stage 2  two independent binary heads trained on abnormal exams only: OSA (OSA-only or Both) and
         RBD (RBD-only or Both).
Nested 10-fold cross-validation on the internal development set: fold k uses the fold-k feature table,
trains on folds other than k and k+1 and early-stops on fold k+1. Out-of-fold predictions are reported for
the development set; locked internal test and external exams get the mean of the 10 fold models.
Class weighting: scale_pos_weight = n_negative / n_positive within each training fold (spw = "full").
Hyper-parameters: configs/stage1.json (abnormal stage, defaults) and configs/stage2.json (OSA and RBD heads).
--variant audio_only / video_only: the same classifier on the single-modality exam features, with the head
hyper-parameters selected on those features (configs/<variant>/).

    python -m psgscreen.model.disorder [--variant audio_only|video_only]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from psgscreen import paths

CLASSES = ["Normal", "OSA-only", "RBD-only", "Both"]
HEAD_CLASSES = {"OSA": [1, 3], "RBD": [2, 3]}
META = {"exam_id", "uuid", "center", "label", "split", "fold", "n_windows", "duration_sec"}
SEED = paths.SEED
BASE = dict(max_depth=4, eta=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", seed=SEED)
ROUNDS, EARLY_STOP = 1000, 40
NF = paths.N_FOLDS
VARIANT_FEATURES = {"audio_only": f"{paths.FEATURES}_audio", "video_only": f"{paths.FEATURES}_video"}


def auc(y, s):
    """Rank-based ROC AUC (ties get the mean rank); NaN if only one class is present."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return np.nan
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); r = ((cs - cnt + cs + 1) / 2.0)[inv]
    p = int((y == 1).sum()); n = int((y == 0).sum())
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def load_features(feat_dir):
    """Roster rows joined with one fold's feature tables; returns the table and the feature names."""
    ros = pd.read_csv(paths.ROSTER, encoding="utf-8-sig")
    ros["exam_id"] = ros["exam_id"].astype(str)
    dfs = []
    for c in paths.CENTERS:
        d = pd.read_csv(f"{feat_dir}/{c}_features.csv", encoding="utf-8-sig")
        d.columns = [x.lstrip("﻿") for x in d.columns]
        d["exam_id"] = d["exam_id"].astype(str)
        dfs.append(d.drop(columns=[x for x in ("label",) if x in d.columns]))
    mg = ros.merge(pd.concat(dfs, ignore_index=True), on="exam_id", how="inner")
    feats = [c for c in mg.columns if c not in META]
    if mg[feats].isna().any().any():
        mg[feats] = mg[feats].fillna(0.0)
    return mg, feats


def load_fold_tables(feat_root=None):
    """The 10 fold feature tables aligned on the same exams: (table, feature names, {k: X_k})."""
    feat_root = feat_root or paths.FEATURES
    base, feats = load_features(f"{feat_root}/f0")
    order = base["exam_id"].tolist()
    Xs = {0: base[feats].values.astype(np.float32)}
    for k in range(1, NF):
        mg, ft = load_features(f"{feat_root}/f{k}")
        if ft != feats:
            raise ValueError(f"fold {k}: feature names differ")
        Xs[k] = mg.set_index("exam_id").reindex(order)[feats].values.astype(np.float32)
    Xs = {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in Xs.items()}
    return base.reset_index(drop=True), feats, Xs


def cohort_index(mg):
    """Class index per exam and the development / locked-test / external row indices."""
    y = mg["label"].map({c: i for i, c in enumerate(CLASSES)}).values
    ext = mg.center.isin(paths.EXTERNAL).values
    dev = (mg.split == "dev").values & ~ext
    return y, np.where(dev)[0], np.where(~dev & ~ext)[0], np.where(ext)[0]


def make_params(cfg, ytr, seed=SEED):
    """XGBoost parameters: BASE, overridden by a config dict; 'spw' sets scale_pos_weight."""
    par = {**BASE, "objective": "binary:logistic", "eval_metric": "auc", "seed": seed}
    if cfg:
        for k, v in cfg.items():
            if k != "spw":
                par[k] = v
        mode = str(cfg.get("spw", "none"))
        if mode.startswith("x"):
            par["scale_pos_weight"] = float(mode[1:])
        elif mode != "none":
            pos = max(int(np.asarray(ytr).sum()), 1); neg = max(len(ytr) - pos, 1)
            par["scale_pos_weight"] = float(neg / pos if mode == "full" else np.sqrt(neg / pos))
    return par


def train(params, X, y, Xv, yv, nthread=2, rounds=ROUNDS, early_stop=EARLY_STOP):
    ev = [(xgb.DMatrix(Xv, label=yv), "val")] if Xv is not None and len(Xv) else None
    return xgb.train({**params, "nthread": nthread}, xgb.DMatrix(X, label=y), num_boost_round=rounds,
                     evals=ev, early_stopping_rounds=(early_stop if ev else None), verbose_eval=False)


def predict(b, X):
    """Prediction with the trees up to the early-stopping iteration."""
    bi = getattr(b, "best_iteration", None)
    d = xgb.DMatrix(X)
    return b.predict(d, iteration_range=(0, bi + 1)) if bi is not None and bi >= 0 else b.predict(d)


def fold_rows(fold, di, k):
    """(held-out, validation, training) development rows of fold k."""
    te = di[fold[di] == k]
    va = di[fold[di] == (k + 1) % NF]
    tr = di[(fold[di] != k) & (fold[di] != (k + 1) % NF)]
    return te, va, tr


def head_config(cfg, head):
    return cfg[head] if isinstance(cfg, dict) and set(cfg) >= {"OSA", "RBD"} else cfg


def fit_fold(X, y, di, fold, k, cfg1, cfg2, seed=SEED, nthread=2):
    """Stage-1, OSA and RBD boosters of fold k."""
    _, va, tr = fold_rows(fold, di, k)

    def fit(rows_tr, rows_va, target, cfg):
        ytr, yva = target(y[rows_tr]).astype(int), target(y[rows_va]).astype(int)
        return train(make_params(cfg, ytr, seed), X[rows_tr], ytr.astype(np.float32),
                     X[rows_va], yva.astype(np.float32), nthread=nthread)

    b1 = fit(tr, va, lambda v: v != 0, cfg1)
    atr, ava = tr[y[tr] != 0], va[y[va] != 0]
    bO = fit(atr, ava, lambda v: np.isin(v, HEAD_CLASSES["OSA"]), head_config(cfg2, "OSA"))
    bR = fit(atr, ava, lambda v: np.isin(v, HEAD_CLASSES["RBD"]), head_config(cfg2, "RBD"))
    return b1, bO, bR


def run(Xs, y, di, ti, xi, fold, cfg1, cfg2, seed=SEED, nthread=2):
    """Out-of-fold (development) and fold-averaged (test, external) scores of the three heads."""
    oof = np.zeros((len(di), 3), np.float32)
    test, ext = [], []
    for k in range(NF):
        X = Xs[k]
        te, _, _ = fold_rows(fold, di, k)
        heads = fit_fold(X, y, di, fold, k, cfg1, cfg2, seed, nthread)
        oof[np.searchsorted(di, te)] = np.column_stack([predict(b, X[te]) for b in heads])
        test.append(np.column_stack([predict(b, X[ti]) for b in heads]))
        if len(xi):
            ext.append(np.column_stack([predict(b, X[xi]) for b in heads]))
        print(f"  fold {k} done", flush=True)
    out = {"OOF": oof, "TEST": np.mean(test, 0)}
    if len(xi):
        out["EXT"] = np.mean(ext, 0)
    return out


def load_params(variant=None):
    d = paths.configs(variant)
    return json.load(open(f"{d}/stage1.json")), json.load(open(f"{d}/stage2.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None, choices=list(VARIANT_FEATURES))
    ap.add_argument("--nthread", type=int, default=2)
    a = ap.parse_args()
    out = paths.disorder_predictions(a.variant)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    mg, feats, Xs = load_fold_tables(VARIANT_FEATURES.get(a.variant))
    y, di, ti, xi = cohort_index(mg)
    fold = mg["fold"].values
    print(f"{len(feats)} features  development {len(di)} / locked test {len(ti)} / external {len(xi)}")
    cfg1, cfg2 = load_params(a.variant)
    res = run(Xs, y, di, ti, xi, fold, cfg1, cfg2, SEED, a.nthread)

    rows = []
    for bucket, ix in (("OOF", di), ("TEST", ti), ("EXT", xi)):
        if bucket not in res:
            continue
        P = res[bucket]
        for j, gi in enumerate(ix):
            rows.append(dict(bucket=bucket, uuid=mg.uuid.values[gi], center=mg.center.values[gi],
                             label=mg.label.values[gi], p_abnormal=float(P[j, 0]),
                             q_OSA=float(P[j, 1]), q_RBD=float(P[j, 2])))
    D = pd.DataFrame(rows)
    D.to_csv(out, index=False, encoding="utf-8-sig")
    for bucket in ("OOF", "TEST", "EXT"):
        g = D[D.bucket == bucket]
        ab = g[g.label != "Normal"]
        print(f"  {bucket:4s} AUC Abnormal {auc(g.label != 'Normal', g.p_abnormal):.4f}  "
              f"OSA {auc(ab.label.isin(['OSA-only', 'Both']), ab.q_OSA):.4f}  "
              f"RBD {auc(ab.label.isin(['RBD-only', 'Both']), ab.q_RBD):.4f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
