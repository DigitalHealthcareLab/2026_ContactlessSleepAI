"""Alternative disorder classifiers (supplement).

The XGBoost models of the two-stage disorder classifier are replaced by logistic regression (L2), random forest,
SVM (RBF) or a multilayer perceptron. Features, folds and evaluation sets are those of the main model; each fold
model is trained on the training folds only (the early-stopping fold of XGBoost is not used). Class weights are
balanced (positive:negative = n_negative / n_positive, as scale_pos_weight in XGBoost). Seed 42.

  grid    pre-specified grid, scored by development out-of-fold AUC   -> hparam/alt/grid_<model>.csv
  select  per task the highest AUC; ties within 1e-4 go to the earlier (simpler) configuration
                                                                   -> hparam/alt/<model>.json
  run     predictions with the selected configuration (configs/alt/<model>.json)
                                                                   -> predictions/alt/<model>.csv

    python -m psgscreen.model.alt_classifiers grid|select|run --model lr
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

from psgscreen import paths
from psgscreen.model import disorder as dm

GRIDS = {
    "lr": [dict(C=c) for c in (0.001, 0.01, 0.1, 1.0, 10.0)],
    "rf": [dict(min_samples_leaf=l, max_features=f) for l in (20, 5, 1) for f in ("sqrt", 0.5)],
    "svm": [dict(C=c, gamma_mult=g) for c in (0.1, 1.0, 10.0, 100.0) for g in (0.1, 1.0, 10.0)],
    "mlp": [dict(hidden=list(h), alpha=a) for h in ((32,), (64, 32)) for a in (1e-1, 1e-2, 1e-3, 1e-4)],
}
TASKS = ("Abnormal", "OSA", "RBD")
TIE = 1e-4
HP_DIR = f"{paths.HPARAM}/alt"
OUT_DIR = f"{paths.PREDICTIONS}/alt"


def balanced_weights(y):
    n = len(y); n1 = max(int(y.sum()), 1); n0 = max(n - n1, 1)
    return np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0))


def fit_predict(cfg, Xtr, ytr, X_list, seed, nthread):
    """Fit one binary model and return its positive-class probabilities for each matrix in X_list."""
    from sklearn.preprocessing import StandardScaler
    model, ytr = cfg["model"], np.asarray(ytr).astype(int)
    if model == "lr":
        from sklearn.linear_model import LogisticRegression
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=float(cfg["C"]), solver="lbfgs", max_iter=5000, class_weight="balanced")
        m.fit(sc.transform(Xtr), ytr)
        prob = lambda X: m.predict_proba(sc.transform(X))[:, 1]
    elif model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        mf = cfg["max_features"]
        m = RandomForestClassifier(n_estimators=500, min_samples_leaf=int(cfg["min_samples_leaf"]),
                                   max_features=mf if isinstance(mf, str) else float(mf),
                                   class_weight="balanced", n_jobs=nthread, random_state=seed)
        m.fit(Xtr, ytr)
        prob = lambda X: m.predict_proba(X)[:, 1]
    elif model == "svm":
        from sklearn.svm import SVC
        sc = StandardScaler().fit(Xtr)
        m = SVC(kernel="rbf", C=float(cfg["C"]), gamma=float(cfg["gamma_mult"]) / Xtr.shape[1],
                class_weight="balanced")
        m.fit(sc.transform(Xtr), ytr)
        prob = lambda X: 1.0 / (1.0 + np.exp(-m.decision_function(sc.transform(X))))
    elif model == "mlp":
        from sklearn.neural_network import MLPClassifier
        sc = StandardScaler().fit(Xtr)
        m = MLPClassifier(hidden_layer_sizes=tuple(int(h) for h in cfg["hidden"]), alpha=float(cfg["alpha"]),
                          activation="relu", solver="adam", learning_rate_init=1e-3, batch_size=64, max_iter=500,
                          early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=seed)
        m.fit(sc.transform(Xtr), ytr, sample_weight=balanced_weights(ytr))
        prob = lambda X: m.predict_proba(sc.transform(X))[:, 1]
    else:
        raise ValueError(model)
    return [np.asarray(prob(X) if len(X) else np.zeros(0), np.float32) for X in X_list]


def run(cfg, Xs, y, di, ti, xi, fold, seed, nthread, oof_only=False):
    """Same fold scheme as disorder.run, with the configured model in place of XGBoost."""
    oof = np.zeros((len(di), 3), np.float32)
    test, ext = [], []
    for k in range(dm.NF):
        X = Xs[k]
        te, _, tr = dm.fold_rows(fold, di, k)
        X_list = [X[te]] if oof_only else [X[te], X[ti], X[xi]]
        atr = tr[y[tr] != 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P = [fit_predict(cfg["Abnormal"], X[tr], y[tr] != 0, X_list, seed, nthread),
                 fit_predict(cfg["OSA"], X[atr], np.isin(y[atr], dm.HEAD_CLASSES["OSA"]), X_list, seed, nthread),
                 fit_predict(cfg["RBD"], X[atr], np.isin(y[atr], dm.HEAD_CLASSES["RBD"]), X_list, seed, nthread)]
        oof[np.searchsorted(di, te)] = np.column_stack([p[0] for p in P])
        if not oof_only:
            test.append(np.column_stack([p[1] for p in P]))
            ext.append(np.column_stack([p[2] for p in P]))
        print(f"  fold {k} done", flush=True)
    return oof, (np.mean(test, 0) if test else None), (np.mean(ext, 0) if ext else None)


def task_auc(y, scores):
    """Development OOF AUC per task (OSA and RBD among abnormal exams)."""
    ab = y != 0
    return {"Abnormal": dm.auc(ab.astype(int), scores[:, 0]),
            "OSA": dm.auc(np.isin(y[ab], dm.HEAD_CLASSES["OSA"]).astype(int), scores[ab, 1]),
            "RBD": dm.auc(np.isin(y[ab], dm.HEAD_CLASSES["RBD"]).astype(int), scores[ab, 2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["grid", "select", "run"])
    ap.add_argument("--model", required=True, choices=list(GRIDS))
    ap.add_argument("--nthread", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(HP_DIR, exist_ok=True)
    grid_csv = f"{HP_DIR}/grid_{a.model}.csv"

    if a.step == "select":
        G = pd.read_csv(grid_csv).sort_values("cfg_id").reset_index(drop=True)
        assert set(G.cfg_id) == set(range(len(GRIDS[a.model]))), "grid incomplete"
        pick = {}
        for t in TASKS:
            r = G[G[f"AUC_{t}"] >= G[f"AUC_{t}"].max() - TIE].iloc[0]
            pick[t] = {"model": a.model, **json.loads(r.hp)}
        json.dump(pick, open(f"{HP_DIR}/{a.model}.json", "w"), indent=1)
        print(pick)
        return

    mg, feats, Xs = dm.load_fold_tables()
    y, di, ti, xi = dm.cohort_index(mg)
    fold = mg["fold"].values
    if a.step == "grid":
        done = set(pd.read_csv(grid_csv).cfg_id) if os.path.exists(grid_csv) else set()
        for gi, hp in enumerate(GRIDS[a.model]):
            if gi in done:
                continue
            cfg = {t: {"model": a.model, **hp} for t in TASKS}
            oof, _, _ = run(cfg, Xs, y, di, ti, xi, fold, paths.SEED, a.nthread, oof_only=True)
            row = dict(cfg_id=gi, hp=json.dumps(hp), **{f"AUC_{t}": v for t, v in task_auc(y[di], oof).items()})
            pd.DataFrame([row]).to_csv(grid_csv, mode="a", header=not os.path.exists(grid_csv), index=False)
            print(row, flush=True)
        return

    cfg = json.load(open(f"{paths.CONFIGS}/alt/{a.model}.json"))
    oof, test, ext = run(cfg, Xs, y, di, ti, xi, fold, paths.SEED, a.nthread)
    rows = []
    for bucket, ix, P in (("OOF", di, oof), ("TEST", ti, test), ("EXT", xi, ext)):
        for j, gi in enumerate(ix):
            rows.append(dict(bucket=bucket, uuid=mg.uuid.values[gi], center=mg.center.values[gi],
                             label=mg.label.values[gi], p_abnormal=float(P[j, 0]), q_OSA=float(P[j, 1]),
                             q_RBD=float(P[j, 2])))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = paths.alt_predictions(a.model)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
