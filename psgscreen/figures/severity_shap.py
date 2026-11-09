"""Supplementary severity SHAP figure: event-level SHAP of the OSA and RBD severity classifiers on the locked
internal test set (fold-mean TreeSHAP, log-odds).

The fold models are re-trained as in model.severity and their fold-mean probabilities are checked against the
saved predictions. Event SHAP = sum over the event's mean / max / SD features; colour = event level (mean of the
three features, each scaled to its 5th-95th percentile).

    python -m psgscreen.figures.severity_shap      -> figures/severity_shap/fig_severity_shap.{png,pdf,csv}
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.lines import Line2D

from psgscreen import paths
from psgscreen.events import name
from psgscreen.figures import shap as sh
from psgscreen.figures.style import use_fonts
from psgscreen.model import disorder as dm, severity as sv

TARGETS = ("osa_severity", "rbd_severity")
TITLE = {"osa_severity": "OSA severity classification", "rbd_severity": "RBD severity classification"}
FS = dict(title=17, label=15, tick=13, legend=13)
OUT = f"{paths.FIGURES}/severity_shap"


def test_shap(target, base, feats, Xs):
    b, fold, di, ti, _, y, _, _ = sv.population(base, target)
    F = len(feats)
    C, X, P = np.zeros((len(ti), F + 1)), np.zeros((len(ti), F)), np.zeros(len(ti))
    models = sv.fold_models(Xs, y, fold, di, target)
    for k, bst in models:
        rng = {"iteration_range": (0, bst.best_iteration + 1)}
        d = xgb.DMatrix(Xs[k][ti])
        P += bst.predict(d, **rng)
        C += bst.predict(d, pred_contribs=True, **rng)
        X += Xs[k][ti]
    P /= len(models); C /= len(models); X /= len(models)
    S = pd.read_csv(paths.severity_predictions(target), encoding="utf-8-sig")
    S = S[S.bucket == "TEST"].set_index("uuid").loc[b.uuid.values[ti]]
    print(f"[{target}] test n={len(ti)}, {len(models)} fold models, max |difference| to saved predictions "
          f"{np.abs(P - S.p.values).max():.2e}")
    return C, X


def main():
    os.makedirs(OUT, exist_ok=True)
    base, feats, Xs = dm.load_fold_tables()
    R = {t: test_shap(t, base, feats, Xs) for t in TARGETS}
    evs = list(dict.fromkeys(f.split("__", 1)[0] for f in feats))
    M = np.zeros((len(feats), len(evs)))
    for i, f in enumerate(feats):
        M[i, evs.index(f.split("__", 1)[0])] = 1.0

    use_fonts()
    sh.FS = FS
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(16.0, 5.6), constrained_layout=True)
    rows = []
    for ax, t in zip(axes, TARGETS):
        C, X = R[t]
        order, imp = sh.beeswarm(ax, C[:, :len(feats)] @ M, sh.event_level(X, feats, evs), [name(e) for e in evs])
        ax.set_xlabel("Event SHAP value (log-odds)", fontsize=FS["label"])
        ax.set_title(TITLE[t], fontsize=FS["title"], pad=8)
        for tl, j in zip(ax.get_yticklabels(), order):
            tl.set_color(sh.GROUP_COLOR[sh.group_of(evs[j])])
        rows += [dict(head=TITLE[t], rank=r, event=name(evs[j]), group=sh.GROUP_LABEL[sh.group_of(evs[j])],
                      mean_abs_shap=float(imp[j])) for r, j in enumerate(np.argsort(-imp), 1)]
    axes[-1].legend(handles=[Line2D([], [], marker="s", ls="none", ms=8.0, color=sh.GROUP_COLOR[g],
                                    label=sh.GROUP_LABEL[g]) for g in ("OSA", "RBD")],
                    loc="lower right", frameon=False, fontsize=FS["legend"], handletextpad=0.4, borderaxespad=0.4)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1)), ax=axes,
                      fraction=0.02, pad=0.01, aspect=30)
    cb.set_label("Event level (low → high)", fontsize=FS["label"])
    cb.ax.tick_params(labelsize=FS["tick"])
    fig.savefig(f"{OUT}/fig_severity_shap.png", dpi=200)
    fig.savefig(f"{OUT}/fig_severity_shap.pdf")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(f"{OUT}/fig_severity_shap.csv", index=False, encoding="utf-8-sig")
    print("saved", f"{OUT}/fig_severity_shap.png")


if __name__ == "__main__":
    main()
