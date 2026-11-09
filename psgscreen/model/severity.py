"""Severity classifiers on the same 39 exam features, trained on internal development exams.

  osa_severity  AHI >= 30 versus 5 <= AHI < 30, among exams with AHI >= 5
  rbd_severity  RBDSQ >= development-set median versus below, among RBD-positive exams with an RBDSQ score;
                one exam per patient (the earliest)
Same nested folds as the disorder classifier (train on folds other than k and k+1, early stop on k+1).
Hyper-parameters: configs/severity_*.json; scale_pos_weight = sqrt(n_negative / n_positive).
Locked internal test and external exams get the mean of the fold models.

    python -m psgscreen.model.severity --target osa_severity
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from psgscreen import paths
from psgscreen.model import disorder as dm

ROUNDS, EARLY_STOP = 600, 40


def add_target(b, target):
    """Continuous severity value ('_y') and study population ('_inpop') per exam."""
    t = pd.read_csv(paths.REPORT_TABLE, encoding="utf-8-sig")
    t["exam_id"] = t["exam_id"].astype(str)
    t["exam_date"] = pd.to_datetime(t["exam_date"], errors="coerce")
    b = b.merge(t[["exam_id", "ahi", "rbdsq", "exam_date"]], on="exam_id", how="left")
    if target == "osa_severity":
        b["_y"] = b["ahi"]
        b["_inpop"] = b["ahi"] >= 5.0
        return b
    b["_y"] = b["rbdsq"]
    b["_inpop"] = b.label.isin(["RBD-only", "Both"]) & b["_y"].notna()
    b["patient"] = b["exam_id"].str.extract(r"^(\d+)")[0]
    first = set(b[b["_inpop"]].sort_values(["patient", "exam_date"]).drop_duplicates("patient").exam_id)
    b["_inpop"] = b["_inpop"] & b.exam_id.isin(first)
    return b


def population(base, target):
    """Exam table with the target, development / locked-test / external rows, binary target and its cut."""
    b = add_target(base, target)
    pop = b["_inpop"].values
    internal = b.center.isin(paths.INTERNAL).values
    is_dev = (b.split == "dev").values
    fold = pd.to_numeric(b.fold, errors="coerce").values
    di = np.where(pop & internal & is_dev & np.isfinite(fold))[0]
    ti = np.where(pop & internal & ~is_dev)[0]
    xi = np.where(pop & ~internal)[0]
    y_raw = b["_y"].values.astype(float)
    cut = 30.0 if target == "osa_severity" else float(np.median(y_raw[di]))
    return b, fold, di, ti, xi, (y_raw >= cut).astype(int), y_raw, cut


def fold_models(Xs, y, fold, di, target, nthread=2):
    """[(fold, booster)] for the folds with both classes in training and a validation fold."""
    cfg = json.load(open(paths.SEVERITY_PARAMS[target]))
    par = {**dm.BASE, "objective": "binary:logistic", "eval_metric": "logloss", "seed": dm.SEED, "nthread": nthread}
    for k, v in cfg.items():
        if k != "spw":
            par[k] = v
    out = []
    for k in range(dm.NF):
        X = Xs[k]
        te, va, tr = dm.fold_rows(fold, di, k)
        if len(np.unique(y[tr])) < 2 or not len(va):
            continue
        r = (len(tr) - y[tr].sum()) / max(int(y[tr].sum()), 1)
        par["scale_pos_weight"] = np.sqrt(r) if cfg.get("spw") == "sqrt" else r
        out.append((k, xgb.train(par, xgb.DMatrix(X[tr], label=y[tr]), ROUNDS,
                                 evals=[(xgb.DMatrix(X[va], label=y[va]), "v")],
                                 early_stopping_rounds=EARLY_STOP, verbose_eval=False)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(paths.SEVERITY_PARAMS))
    ap.add_argument("--nthread", type=int, default=2)
    a = ap.parse_args()

    base, feats, Xs = dm.load_fold_tables()
    b, fold, di, ti, xi, y, y_raw, cut = population(base, a.target)
    print(f"{a.target}: cut {cut:.2f}  development {len(di)} ({y[di].sum()} positive) / "
          f"test {len(ti)} / external {len(xi)}")
    oof, test, ext = np.zeros(len(di)), [], []
    for k, bst in fold_models(Xs, y, fold, di, a.target, a.nthread):
        X = Xs[k]
        te = dm.fold_rows(fold, di, k)[0]
        rng = {"iteration_range": (0, bst.best_iteration + 1)}
        if len(te):
            oof[np.searchsorted(di, te)] = bst.predict(xgb.DMatrix(X[te]), **rng)
        test.append(bst.predict(xgb.DMatrix(X[ti]), **rng) if len(ti) else np.zeros(0))
        ext.append(bst.predict(xgb.DMatrix(X[xi]), **rng) if len(xi) else np.zeros(0))

    rows = []
    for bucket, idx, p in (("OOF", di, oof),
                           ("TEST", ti, np.mean(test, 0) if test and len(ti) else np.zeros(len(ti))),
                           ("EXT", xi, np.mean(ext, 0) if ext and len(xi) else np.zeros(len(xi)))):
        for j, i in enumerate(idx):
            rows.append(dict(bucket=bucket, uuid=b.uuid.values[i], exam_id=b.exam_id.values[i],
                             center=b.center.values[i], label=b.label.values[i], y_raw=y_raw[i],
                             y=int(y[i]), p=float(p[j]), cut=cut))
        if len(idx) and len(np.unique(y[idx])) > 1:
            print(f"  {bucket:4s} n={len(idx):4d}  AUC {roc_auc_score(y[idx], p):.4f}")
    out = paths.severity_predictions(a.target)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
