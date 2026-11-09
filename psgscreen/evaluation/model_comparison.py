"""Supplementary comparisons of disorder classifiers evaluated on the same exams.

Per model: AUC with exam-bootstrap 95 % CI (B = 4000), sensitivity, specificity and macro-F1 at 0.5. Against the
first model of a family: paired ΔAUC on the same resamples, two-sided bootstrap P = 2 min(P(Δ <= 0), P(Δ >= 0)),
Holm-adjusted over the internal-test and external comparisons of the family (P floored at the bootstrap
resolution 2 / B). Every bootstrap starts from RandomState(42). Abnormal: all exams; OSA and RBD: abnormal exams.

  alt      XGBoost (main model) vs logistic regression, random forest, SVM, MLP
                                                                  -> results/comparison_alt.csv
  fusion   weighted late fusion (main model) vs audio only and video only; video only vs audio only; classifier
           output averaging (mean of the audio-only and video-only classifier probabilities) vs main
                                                                  -> results/comparison_fusion_<family>.csv,
                                                                     results/comparison_fusion.csv (all models)

    python -m psgscreen.evaluation.model_comparison alt|fusion
"""
import argparse
import os

import numpy as np
import pandas as pd

from psgscreen import paths
from psgscreen.model.disorder import CLASSES, auc

BUCKETS = ["TEST", "EXT", "OOF"]
HEADS = ["Abnormal", "OSA", "RBD"]
B = 4000
MAIN = "Audio + video (main model)"
ALT = [("XGBoost", paths.disorder_predictions()), ("Logistic regression", paths.alt_predictions("lr")),
       ("Random forest", paths.alt_predictions("rf")), ("SVM (RBF)", paths.alt_predictions("svm")),
       ("MLP", paths.alt_predictions("mlp"))]
VARIANT = {MAIN: None, "Audio only": "audio_only", "Video only": "video_only"}
FUSION_FAMILIES = {"modality": [MAIN, "Audio only", "Video only"], "video_vs_audio": ["Audio only", "Video only"],
                   "output_averaging": [MAIN, "Output averaging"]}


def read(path):
    if not os.path.exists(path):
        return None
    P = pd.read_csv(path, encoding="utf-8-sig")
    P["y4"] = P.label.map({c: i for i, c in enumerate(CLASSES)})
    return P


def output_average(A, V):
    """Mean of the audio-only and video-only classifier probabilities (no training)."""
    m = A.merge(V, on=["bucket", "uuid"], suffixes=("", "_v"), validate="one_to_one")
    out = m[list(A.columns)].copy()
    for c in ("p_abnormal", "q_OSA", "q_RBD"):
        out[c] = 0.5 * m[c].values + 0.5 * m[f"{c}_v"].values
    return out


def load_variant(name):
    if name == "Output averaging":
        A, V = (load_variant(n) for n in ("Audio only", "Video only"))
        return None if A is None or V is None else output_average(A, V)
    return read(paths.disorder_predictions(VARIANT[name]))


def yscore(P, bucket, head):
    S = P[P.bucket == bucket]
    if head == "Abnormal":
        return (S.y4.values > 0).astype(int), S.p_abnormal.values.astype(float), S.uuid.values
    D = S[S.y4 > 0]
    pos, col = ([1, 3], "q_OSA") if head == "OSA" else ([2, 3], "q_RBD")
    return np.isin(D.y4.values, pos).astype(int), D[col].values.astype(float), D.uuid.values


def operating_point(y, s):
    p = (s > 0.5).astype(int)
    tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    f1p = 2 * tp / max(2 * tp + fp + fn, 1); f1n = 2 * tn / max(2 * tn + fn + fp, 1)
    return tp / max(tp + fn, 1), tn / max(tn + fp, 1), (f1p + f1n) / 2


def boot_auc(y, s):
    rng = np.random.RandomState(42); v = []
    for _ in range(B):
        i = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[i])) > 1:
            v.append(auc(y[i], s[i]))
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def paired(ya, sa, ua, sb, ub):
    """ΔAUC (b - a) with its paired-bootstrap CI and two-sided P."""
    o = pd.Index(ub).get_indexer(pd.Index(ua))
    assert (o >= 0).all(), "models were evaluated on different exams"
    sb = sb[o]
    rng = np.random.RandomState(42); ds = []
    for _ in range(B):
        i = rng.choice(len(ya), len(ya), replace=True)
        if len(np.unique(ya[i])) > 1:
            ds.append(auc(ya[i], sb[i]) - auc(ya[i], sa[i]))
    ds = np.array(ds)
    p = min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean()))
    return auc(ya, sb) - auc(ya, sa), np.percentile(ds, 2.5), np.percentile(ds, 97.5), p


def holm(p):
    p = np.maximum(np.asarray(p, float), 2.0 / B)
    m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(np.argsort(p, kind="stable")):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def compare(models, load):
    """Rows for every model, bucket and head; ΔAUC against models[0]."""
    ref, rows = load(models[0]), []
    for name in models:
        P = load(name)
        for b in BUCKETS:
            for h in HEADS:
                y, s, u = yscore(P, b, h)
                lo, hi = boot_auc(y, s)
                se, sp, mf1 = operating_point(y, s)
                d = dict(model=name, bucket=b, head=h, n=len(y), n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
                         AUC=auc(y, s), CI_lo=lo, CI_hi=hi, Sens=se, Spec=sp, macroF1=mf1)
                if name != models[0]:
                    ya, sx, ux = yscore(ref, b, h)
                    d["dAUC"], d["d_lo"], d["d_hi"], d["p"] = paired(ya, sx, ux, s, u)
                rows.append(d)
        print(f"  {name} done", flush=True)
    T = pd.DataFrame(rows)
    fam = (T.model != models[0]) & T.bucket.isin(["TEST", "EXT"])
    T.loc[fam, "p_holm"] = holm(T.loc[fam, "p"].values)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis", choices=["alt", "fusion"])
    a = ap.parse_args()
    os.makedirs(paths.RESULTS, exist_ok=True)
    if a.analysis == "alt":
        files = dict(ALT)
        compare([n for n, _ in ALT], lambda name: read(files[name])).to_csv(
            f"{paths.RESULTS}/comparison_alt.csv", index=False, encoding="utf-8-sig")
        return
    T = {}
    for fam, models in FUSION_FAMILIES.items():
        T[fam] = compare(models, load_variant)
        T[fam].to_csv(f"{paths.RESULTS}/comparison_fusion_{fam}.csv", index=False, encoding="utf-8-sig")
    M = T["modality"]
    C = pd.concat([M[M.model == MAIN], T["output_averaging"][T["output_averaging"].model == "Output averaging"],
                   M[M.model != MAIN]], ignore_index=True)
    C.to_csv(f"{paths.RESULTS}/comparison_fusion.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
