"""Disorder detection figure: ROC (row a), calibration (row b) and decision curves (row c) for Abnormal, OSA and
RBD on the internal test and external sets. Also the supplementary severity ROC panels.

ROC: AUC CI and pointwise 95 % band from B = 4000 exam bootstrap resamples (RandomState(42) per cohort; the
severity panels share one generator for internal and external). Markers: development OOF Youden threshold
applied to each cohort. Calibration and decision curves come from evaluation/clinical.py (curves.pkl).

    python -m psgscreen.figures.disorder_detection
    python -m psgscreen.figures.disorder_detection --severity
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, roc_curve

from psgscreen import paths
from psgscreen.evaluation import clinical as cm
from psgscreen.figures.style import COHORT, COHORT_LS, use_fonts

HEADS = ["Abnormal", "OSA", "RBD"]
SEVERITY = {"OSA severity": "osa_severity", "RBD severity": "rbd_severity"}
LABEL = {"Internal": "Internal test", "External": "External"}
MARKER = {"Internal": "o", "External": "s"}
FS_TITLE, FS_LABEL, FS_TICK, FS_LEG, FS_NOTE = 16, 14, 12.5, 12, 12.5
LW, BAND = 2.2, .11
# page layout in inches: left margin, panel, gap, right margin; title / plain top; bottom; DCA legend
L_, S_, G_, R_ = 1.0, 5.3, 1.25, 0.25
T_TITLE, T_PLAIN, BOT, LEG_H = 0.55, 0.25, 0.85, 0.55
W_ = L_ + 3 * S_ + 2 * G_ + R_
OUT = f"{paths.FIGURES}/disorder_detection"


def read(p):
    D = pd.read_csv(p, encoding="utf-8-sig"); D.columns = [c.lstrip("﻿") for c in D.columns]
    return D


def series(P, head):
    if head == "Abnormal":
        return (P.label != "Normal").astype(int).values, P.p_abnormal.values
    D = P[P.label != "Normal"]
    pos = ["OSA-only", "Both"] if head == "OSA" else ["RBD-only", "Both"]
    return D.label.isin(pos).astype(int).values, D[f"q_{head}"].values


def boot_auc(y, s, B, rng):
    o = []
    for _ in range(B):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            o.append(roc_auc_score(y[i], s[i]))
    return (np.percentile(o, 2.5), np.percentile(o, 97.5)) if o else (np.nan, np.nan)


def roc_band(y, s, B, ngrid=201):
    g = np.linspace(0, 1, ngrid)
    rng = np.random.RandomState(42)
    T = []
    for _ in range(B):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            f, t, _ = roc_curve(y[i], s[i])
            T.append(np.interp(g, f, t))
    T = np.asarray(T)
    return g, np.percentile(T, 2.5, 0), np.percentile(T, 97.5, 0)


def compute(items, B, shared):
    rs = np.random.RandomState(42) if shared else None
    out = []
    for nm, y, s in items:
        rng = rs if (rs is not None and nm in ("Internal", "External")) else np.random.RandomState(42)
        lo, hi = boot_auc(y, s, B, rng)
        g, b1, b2 = roc_band(y, s, B)
        f, t, _ = roc_curve(y, s)
        out.append((nm, roc_auc_score(y, s), lo, hi, g, b1, b2, f, t))
    return out


def roc_items():
    P = read(paths.disorder_predictions())
    items = {h: [("Internal",) + series(P[P.bucket == "TEST"], h), ("External",) + series(P[P.bucket == "EXT"], h)]
             for h in HEADS}
    for title, target in SEVERITY.items():
        S = read(paths.severity_predictions(target))
        ti, te = S[S.bucket == "TEST"], S[S.bucket == "EXT"]
        items[title] = [("Internal", ti.y.astype(int).values, ti.p.values),
                        ("External", te.y.astype(int).values, te.p.values)]
    return items


def roc_panel(ax, title, res):
    for nm, a, lo, hi, g, b1, b2, f, t in res:
        ax.fill_between(g, b1, b2, color=COHORT[nm], alpha=BAND, lw=0)
        ax.plot(f, t, color=COHORT[nm], lw=LW, ls=COHORT_LS[nm], label=f"{LABEL[nm]}  {a:.4f} ({lo:.4f}–{hi:.4f})")
    ax.plot([0, 1], [0, 1], color="#999", ls=":", lw=1)
    ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.02); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("1 − Specificity", fontsize=FS_LABEL); ax.set_ylabel("Sensitivity", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.legend(fontsize=FS_LEG, loc="lower right", framealpha=.92, handlelength=2.8, borderpad=.35,
              labelspacing=.3, handletextpad=.5)


def cal_panel(ax, task, CUR, title):
    ax.plot([0, 1], [0, 1], color="#999", ls=":", lw=1)
    for c in cm.FIG_COHORTS:
        q = CUR[(task, c)]["ref"]
        ax.fill_between(q["g"], q["lo"], q["hi"], color=COHORT[c], alpha=BAND, lw=0)
        ax.plot(q["g"], q["f"], color=COHORT[c], ls=COHORT_LS[c], lw=LW,
                label=f"{LABEL[c]}  slope {q['cal']['slope']:.4f} · CITL {q['cal']['CITL']:+.4f}")
        ax.plot(q["bins"].p, q["bins"].y, MARKER[c], color=COHORT[c], ms=6.5, mfc="white", mew=1.5, ls="none")
    ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.02); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted probability", fontsize=FS_LABEL); ax.set_ylabel("Observed proportion", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    if title:
        ax.set_title(title, fontsize=FS_TITLE)
    ax.legend(fontsize=FS_LEG, loc="lower right", framealpha=.92, handlelength=2.8, borderpad=.35,
              labelspacing=.3, handletextpad=.5)


def dca_panel(ax, task, CUR, title):
    pi = CUR[(task, "Internal")]["dca"]["pi"]
    ax.plot(cm.PT, np.zeros_like(cm.PT), color="#b5b5b5", lw=1.0, zorder=1)
    for c in cm.FIG_COHORTS:
        q = CUR[(task, c)]["dca"]
        ax.fill_between(cm.PT, q["lo"], q["hi"], color=COHORT[c], alpha=BAND, lw=0, zorder=2)
    ax.plot(cm.PT, CUR[(task, "Internal")]["dca"]["all"], color="#6f6f6f", lw=1.5, zorder=3)
    for c in cm.FIG_COHORTS:
        ax.plot(cm.PT, CUR[(task, c)]["dca"]["model"], color=COHORT[c], lw=LW, ls=COHORT_LS[c], zorder=4)
    ax.set_xlim(0, 1); ax.set_ylim(-0.12 * pi, pi * 1.08)
    high = pi > 0.6
    ax.text(.03 if high else .97, .30 if high else .95, f"Reference prevalence {pi:.4f}", transform=ax.transAxes,
            ha="left" if high else "right", va="top", fontsize=FS_NOTE, color="#333333")
    ax.set_xlabel("Threshold probability", fontsize=FS_LABEL); ax.set_ylabel("Net benefit", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    if title:
        ax.set_title(title, fontsize=FS_TITLE)


def operating_points():
    thr = read(f"{cm.OUT}/recalibration.csv").set_index("task")["youden_dev_OOF"].to_dict()
    P = read(paths.disorder_predictions())
    D = {"Internal": P[P.bucket == "TEST"], "External": P[P.bucket == "EXT"]}
    pts = {}
    for h in HEADS:
        for nm, d in D.items():
            y, s = series(d, h)
            pr = s >= thr[h]
            pts[(h, nm)] = (1 - (~pr & (y == 0)).sum() / (y == 0).sum(), (pr & (y == 1)).sum() / (y == 1).sum(), thr[h])
    return pts


def build(rows, path, RES, CUR, PTS):
    bots = [BOT + (LEG_H if r == "dca" else 0) for r in rows]
    H = sum(T_TITLE + S_ + b for b in bots)
    fig = plt.figure(figsize=(W_, H))
    y_top = H
    for r, bot in zip(rows, bots):
        y0 = y_top - T_TITLE - S_
        axes = [fig.add_axes([(L_ + j * (S_ + G_)) / W_, y0 / H, S_ / W_, S_ / H]) for j in range(3)]
        for ax, h in zip(axes, HEADS):
            if r == "roc":
                roc_panel(ax, h, RES[h])
                cohorts = [res[0] for res in RES[h]]
                for nm in cohorts:
                    fx, ty, _ = PTS[(h, nm)]
                    ax.plot(fx, ty, MARKER[nm], color=COHORT[nm], ms=10, mec="white", mew=1.6, zorder=6, clip_on=False)
                hh, ll = ax.get_legend_handles_labels()
                mk = tuple(Line2D([], [], ls="none", marker=MARKER[nm], color=COHORT[nm], ms=9, mec="white", mew=1.4)
                           for nm in cohorts)
                ax.legend(hh + [mk], ll + [f"Optimal threshold  {PTS[(h, 'Internal')][2]:.4f}"],
                          handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)}, fontsize=FS_LEG,
                          loc="lower right", framealpha=.92, handlelength=2.8, borderpad=.35, labelspacing=.3,
                          handletextpad=.5)
            elif r == "cal":
                cal_panel(ax, h, CUR, h)
            else:
                dca_panel(ax, h, CUR, h)
        if r == "dca":
            hh = [Line2D([], [], color=COHORT["Internal"], lw=LW, ls=COHORT_LS["Internal"]),
                  Line2D([], [], color=COHORT["External"], lw=LW, ls=COHORT_LS["External"]),
                  Line2D([], [], color="#6f6f6f", lw=1.5), Line2D([], [], color="#b5b5b5", lw=1.0)]
            fig.legend(hh, ["Internal test", "External", "Treat all", "Treat none"], loc="center",
                       bbox_to_anchor=((L_ + (3 * S_ + 2 * G_) / 2) / W_, (y0 - BOT - LEG_H / 2) / H), ncol=4,
                       fontsize=13.5, frameon=False, handlelength=3.2, columnspacing=2.2)
        y_top -= T_TITLE + S_ + bot
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=4000)
    ap.add_argument("--severity", action="store_true", help="supplementary severity ROC panels instead")
    a = ap.parse_args()
    use_fonts()
    os.makedirs(OUT, exist_ok=True)
    items = roc_items()
    if a.severity:
        for title in SEVERITY:
            res = compute(items[title], a.B, shared=True)
            fig, ax = plt.subplots(figsize=(6.4, 6.4))
            roc_panel(ax, f"{title} classification", res)
            fig.tight_layout()
            fig.savefig(f"{OUT}/roc_{SEVERITY[title]}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
        return
    RES = {h: compute(items[h], a.B, shared=False) for h in HEADS}
    CUR = pickle.load(open(cm.CURVES, "rb"))["CUR"]
    PTS = operating_points()
    build(["roc", "cal", "dca"], f"{OUT}/fig_disorder_detection", RES, CUR, PTS)
    for name, r in (("row_a_roc", "roc"), ("row_b_calibration", "cal"), ("row_c_dca", "dca")):
        build([r], f"{OUT}/{name}", RES, CUR, PTS)


if __name__ == "__main__":
    main()
