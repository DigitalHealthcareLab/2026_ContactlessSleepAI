"""TreeSHAP values of the disorder classifier (XGBoost pred_contribs, log-odds) on the locked internal test set,
averaged over the 10 fold models.

The fold models are re-trained exactly as in disorder.py and their fold-averaged probabilities are checked against
the saved predictions. Abnormal: all locked-test exams; OSA and RBD: abnormal locked-test exams.

    python -m psgscreen.model.shap_values          -> results/shap_disorder.npz
"""
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from psgscreen import paths
from psgscreen.model import disorder as dm

HEADS = ["Abnormal", "OSA", "RBD"]
CACHE = f"{paths.RESULTS}/shap_disorder.npz"


def contribs(b, X):
    return np.asarray(b.predict(xgb.DMatrix(X), pred_contribs=True, iteration_range=(0, b.best_iteration + 1)))


def main():
    os.makedirs(paths.RESULTS, exist_ok=True)
    mg, feats, Xs = dm.load_fold_tables()
    y, di, ti, _ = dm.cohort_index(mg)
    fold = mg["fold"].values
    ta = ti[y[ti] != 0]
    rows_for = {"Abnormal": ti, "OSA": ta, "RBD": ta}
    cfg1, cfg2 = dm.load_params()

    C = {h: np.zeros((len(rows_for[h]), len(feats) + 1)) for h in HEADS}
    Xc = {h: np.zeros((len(rows_for[h]), len(feats))) for h in HEADS}
    P = np.zeros((len(ti), 3))
    for k in range(dm.NF):
        heads = dm.fit_fold(Xs[k], y, di, fold, k, cfg1, cfg2)
        for h, b in zip(HEADS, heads):
            C[h] += contribs(b, Xs[k][rows_for[h]])
            Xc[h] += Xs[k][rows_for[h]]
        for j, b in enumerate(heads):
            P[:, j] += dm.predict(b, Xs[k][ti])
    for h in HEADS:
        C[h] /= dm.NF; Xc[h] /= dm.NF
    P /= dm.NF

    S = pd.read_csv(paths.disorder_predictions(), encoding="utf-8-sig")
    S = S[S.bucket == "TEST"].set_index("uuid").loc[mg.uuid.values[ti]]
    print("max |difference| to saved predictions:", np.abs(P - S[["p_abnormal", "q_OSA", "q_RBD"]].values).max(0))
    np.savez_compressed(CACHE, **{f"C_{h}": C[h] for h in HEADS}, **{f"X_{h}": Xc[h] for h in HEADS},
                        feats=np.array(feats, dtype=object))
    print(f"saved -> {CACHE}")


if __name__ == "__main__":
    main()
