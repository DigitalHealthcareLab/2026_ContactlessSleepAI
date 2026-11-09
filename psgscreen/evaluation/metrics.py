"""Performance tables of the disorder and severity classifiers (point estimates with bootstrap 95 % CIs).

Disorder (threshold 0.5): Abnormal = all exams, score p_abnormal; OSA and RBD = abnormal exams only
(stage 1 assumed correct), scores q_OSA and q_RBD; Average = mean of OSA and RBD.
CIs: exam-level bootstrap, B = 4000, RandomState(42) for every table.

    python -m psgscreen.evaluation.metrics disorder
    python -m psgscreen.evaluation.metrics severity
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from psgscreen import paths
from psgscreen.model.disorder import CLASSES, auc

METRICS = ["AUC", "Sensitivity", "Specificity", "Precision", "F1-Score"]
HEADS = {"OSA": [1, 3], "RBD": [2, 3]}


def binary(yt, yp, score):
    tp = int(((yt == 1) & (yp == 1)).sum()); fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum()); fp = int(((yt == 0) & (yp == 1)).sum())
    se = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1)
    pr = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * pr * se / (pr + se) if (pr and not np.isnan(pr) and pr + se > 0) else 0.0
    return {"AUC": auc(yt, score), "Sensitivity": se, "Specificity": sp, "Precision": pr, "F1-Score": f1}


def disease_scores(Q, y4, thr=0.5):
    """OSA, RBD and Average rows for abnormal exams (Q = [q_OSA, q_RBD], y4 = class index)."""
    out = {}
    for i, h in enumerate(HEADS):
        out[h] = binary(np.isin(y4, HEADS[h]).astype(int), (Q[:, i] > thr).astype(int), Q[:, i])
    out["Average"] = {m: float(np.nanmean([out[h][m] for h in HEADS])) for m in METRICS}
    return out


def disorder_scores(pa, Q, y4, thr=0.5):
    """{(task, metric): value} for Abnormal (all exams) and OSA / RBD / Average (abnormal exams)."""
    yb = (y4 != 0).astype(int)
    res = {"Abnormal": binary(yb, (pa > thr).astype(int), pa)}
    res.update(disease_scores(Q[yb == 1], y4[yb == 1], thr))
    return {(t, m): v for t, d in res.items() for m, v in d.items()}


def boot_ci(fn, n, B, seed=42):
    rng = np.random.RandomState(seed)
    acc = {}
    for _ in range(B):
        s = rng.choice(n, n, replace=True)
        try:
            r = fn(s)
        except Exception:
            continue
        for k, d in r.items():
            for m, v in d.items():
                acc.setdefault((k, m), []).append(v)
    ci = {}
    for k, v in acc.items():
        v = np.array(v, float)
        ci[k] = ((np.percentile(v[~np.isnan(v)], 2.5), np.percentile(v[~np.isnan(v)], 97.5))
                 if np.isfinite(v).any() else (np.nan, np.nan))
    return ci


def disorder_table(pred, B):
    D = pd.read_csv(pred, encoding="utf-8-sig")
    D.columns = [c.lstrip("﻿") for c in D.columns]
    rows = []
    for bucket in ("TEST", "EXT", "OOF"):
        g = D[D.bucket == bucket]
        if not len(g):
            continue
        y4 = g.label.map({c: i for i, c in enumerate(CLASSES)}).values
        pa = g.p_abnormal.values.astype(float); yb = (y4 != 0).astype(int)
        m = yb == 1
        Q = g.loc[m, ["q_OSA", "q_RBD"]].values.astype(float); yd = y4[m]
        blocks = [({"Abnormal": binary(yb, (pa > 0.5).astype(int), pa)},
                   boot_ci(lambda s: {"Abnormal": binary(yb[s], (pa[s] > 0.5).astype(int), pa[s])}, len(pa), B),
                   len(pa), {"Abnormal": int(yb.sum())}),
                  (disease_scores(Q, yd), boot_ci(lambda s: disease_scores(Q[s], yd[s]), len(yd), B),
                   int(m.sum()), {"OSA": int(np.isin(yd, [1, 3]).sum()), "RBD": int(np.isin(yd, [2, 3]).sum()),
                                  "Average": int(m.sum())})]
        for point, ci, n, counts in blocks:
            for task, d in point.items():
                for metric, v in d.items():
                    lo, hi = ci.get((task, metric), (np.nan, np.nan))
                    rows.append(dict(bucket=bucket, task=task, n=n, n_class=counts.get(task, np.nan),
                                     metric=metric, value=v, CI_low=lo, CI_high=hi,
                                     display=f"{v:.4f} [{lo:.4f}, {hi:.4f}]"))
    return pd.DataFrame(rows)


SEV_METRICS = ["AUC", "AP", "Sensitivity", "Specificity", "Precision", "NPV", "F1-Score", "Accuracy",
               "BalancedAcc", "MCC"]


def severity_metrics(y, p, s):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    se = tp / (tp + fn) if tp + fn else np.nan
    sp = tn / (tn + fp) if tn + fp else np.nan
    pr = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * pr * se / (pr + se) if (pr and se and pr + se) else np.nan
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else np.nan
    au = roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan
    ap = average_precision_score(y, s) if len(np.unique(y)) > 1 else np.nan
    return dict(zip(SEV_METRICS, [au, ap, se, sp, pr, npv, f1, (tp + tn) / len(y), (se + sp) / 2, mcc])), \
        (tn, fp, fn, tp)


def severity_table(B):
    rows = []
    for target, title in (("osa_severity", "OSA severity"), ("rbd_severity", "RBD severity")):
        D = pd.read_csv(paths.severity_predictions(target), encoding="utf-8-sig")
        rng = np.random.RandomState(42)
        for bucket, name in (("TEST", "Internal test"), ("EXT", "External")):
            g = D[D.bucket == bucket]
            y = g.y.values.astype(int); s = g.p.values; p = (s > 0.5).astype(int)
            point, cm = severity_metrics(y, p, s)
            acc = {k: [] for k in SEV_METRICS}
            for _ in range(B):
                i = rng.randint(0, len(y), len(y))
                if len(np.unique(y[i])) < 2:
                    continue
                r, _ = severity_metrics(y[i], p[i], s[i])
                for k in SEV_METRICS:
                    acc[k].append(r[k])
            rows.append(dict(task=title, bucket=name, n=len(y), n_pos=int(y.sum()),
                             cm=f"TN{cm[0]}/FP{cm[1]}/FN{cm[2]}/TP{cm[3]}",
                             **{k: f"{point[k]:.4f} [{np.nanpercentile(acc[k], 2.5):.4f}, "
                                   f"{np.nanpercentile(acc[k], 97.5):.4f}]" for k in SEV_METRICS}))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table", choices=["disorder", "severity"])
    ap.add_argument("--B", type=int, default=4000)
    a = ap.parse_args()
    os.makedirs(paths.RESULTS, exist_ok=True)
    if a.table == "disorder":
        T = disorder_table(paths.disorder_predictions(), a.B)
        T.to_csv(f"{paths.RESULTS}/table_disorder.csv", index=False, encoding="utf-8-sig")
    else:
        T = severity_table(a.B)
        T.to_csv(f"{paths.RESULTS}/table_severity.csv", index=False, encoding="utf-8-sig")
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
