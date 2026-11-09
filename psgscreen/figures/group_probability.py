"""Supplementary group-probability figure: mean cumulative predicted probability over the recording per group
(Normal, Abnormal, OSA, RBD; locked internal test exams of Hospital A).

Input: results/trajectory_5min.csv (model.trajectory). Normal panel: 1 - p_abnormal. Each curve is cut at the
last block where >= 90 % of the group's exams are still recording; band = 95 % CI of the mean (exam bootstrap,
B = 2000, RandomState(42), one resample shared by all time points); dashed line = development OOF Youden
threshold (results/clinical/table_optimal_threshold.csv; Normal panel: 1 - the Abnormal threshold); grey =
30-min windows whose share of the group's patients in REM sleep is in the top quartile (figures.time_course).

    python -m psgscreen.figures.group_probability   -> figures/group_probability/fig_group_probability.{png,csv}
"""
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from psgscreen import paths
from psgscreen.figures.style import HEAD, use_fonts
from psgscreen.figures.time_course import gmask, rem_zones

OUT = f"{paths.FIGURES}/group_probability"
GROUPS = ["Normal", "Abnormal", "OSA", "RBD"]
GCOL = {"Normal": "p_normal", "Abnormal": "p_abnormal", "OSA": "q_OSA", "RBD": "q_RBD"}
TASK = {"Abnormal": "Abnormal classification", "OSA": "OSA classification", "RBD": "RBD classification"}
FS = dict(title=17, label=15, tick=13, thr=13, legend=15)
B, SEED, COV = 2000, 42, 0.9
LABEL_RIGHT = ("Abnormal", "RBD")             # panels whose threshold label sits at the right end


def mean_band(sub, col):
    """Mean over the exams still recording at each block, and the exam-bootstrap 95 % CI of that mean."""
    mid = sub.groupby("t_min")[col].mean()
    M = sub.pivot_table(index="exam_id", columns="t_min", values=col).reindex(columns=mid.index).values.astype(np.float64)
    n = len(M)
    rng = np.random.RandomState(SEED)
    boot = np.empty((B, M.shape[1]))
    for s in range(0, B, 250):
        R = rng.randint(0, n, size=(min(250, B - s), n))
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            boot[s:s + len(R)] = np.nanmean(M[R], axis=1)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return mid, lo, hi


def main():
    os.makedirs(OUT, exist_ok=True)
    use_fonts()
    T = pd.read_csv(f"{paths.RESULTS}/trajectory_5min.csv", encoding="utf-8-sig")
    T = T[T.center == "INT"].copy()
    T["uuid"] = T.exam_id.astype(str).str.split("_").str[-1].str.lower()
    C = pd.read_csv(f"{paths.RESULTS}/clinical/table_optimal_threshold.csv", encoding="utf-8-sig").set_index(["Task", "Cohort"])
    thr = {g: float(C.loc[(t, "Internal test"), "Threshold"]) for g, t in TASK.items()}
    thr["Normal"] = 1.0 - thr["Abnormal"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 9.4))
    rows = []
    for k, g in enumerate(GROUPS):
        ax, col = axes[k // 2][k % 2], GCOL[g]
        sub = T[gmask(T["label"].values, g)]
        cov = sub.groupby("t_min").exam_id.nunique() / sub.exam_id.nunique()
        keep = cov[cov >= COV].index.max()
        n_group = sub.exam_id.nunique()
        sub = sub[sub.t_min <= keep]
        mid, lo, hi = mean_band(sub, col)
        rows.append(pd.DataFrame({"group": g, "n": n_group, "t_min": mid.index, "mean": mid.values, "ci_lo": lo,
                                  "ci_hi": hi}))
        x = mid.index / 60.0
        ax.plot(x, mid.values, color=HEAD[g], lw=2.2, zorder=4)
        ax.fill_between(x, lo, hi, color=HEAD[g], alpha=.16, linewidth=0, zorder=1)
        for r in rem_zones(set(sub.uuid)).itertuples():
            ax.axvspan(r.zone_lo_h, r.zone_hi_h, color="#888", alpha=.16, linewidth=0, zorder=0)
        ax.set_xlim(0, keep / 60.0)
        ax.set_ylim(0, 1.0)
        t_ = thr[g]
        ax.axhline(t_, color="#8A8A8A", ls="--", lw=1.1, zorder=2)
        ax.text(.988 if g in LABEL_RIGHT else .012, t_, f"optimal threshold {t_:.2f}", fontsize=FS["thr"], color="#666",
                va="bottom", ha="right" if g in LABEL_RIGHT else "left", transform=ax.get_yaxis_transform())
        ax.set_title(g, fontsize=FS["title"])
        ax.set_xlabel("Time from recording start (hours)", fontsize=FS["label"])
        ax.tick_params(labelsize=FS["tick"])
        ax.set_ylabel("Predicted probability", fontsize=FS["label"])
    fig.legend(handles=[Patch(facecolor="#888", alpha=.16, edgecolor="#999", linewidth=.6,
                              label="30-min windows in which the proportion of patients in REM "
                                    "sleep is in the top quartile for that group")],
               loc="lower center", bbox_to_anchor=(.5, .003), frameon=False, fontsize=FS["legend"], handlelength=2.2,
               handleheight=1.2)
    plt.tight_layout(rect=[0, .045, .99, 1], w_pad=3.0)
    plt.savefig(f"{OUT}/fig_group_probability.png", dpi=200)
    plt.close()
    pd.concat(rows).to_csv(f"{OUT}/fig_group_probability.csv", index=False, encoding="utf-8-sig")
    print("saved", f"{OUT}/fig_group_probability.png")


if __name__ == "__main__":
    main()
