"""Cumulative recording-time figure (locked internal test exams of Hospital A).

Input: results/trajectory_5min.csv (model.trajectory) = each exam scored on its first t minutes, t = 5, 10, ...

  AUC(t)      per task over exams; an exam that has stopped recording keeps its last score; the curve is cut at
              the last block where >= 90 % of the task's exams are still recording. 95 % CI: exam bootstrap
              (B = 2000, RandomState(42), one stream over Abnormal, OSA, RBD).
  T_NI        earliest block from which the lower 2.5 % bound of AUC(t) - AUC(end) stays above -0.05
              (non-inferiority to the full night, margin 0.05); dotted line in the figure.
  examples    individual cumulative probabilities (right axis) of correctly classified exams (final score at
              or above the development OOF Youden threshold): Abnormal panel = the Normal woman of at least the
              median Normal age closest to the median age of that subgroup, and the RBD-only and OSA-only exams
              closest to their subgroup's median AHI; OSA / RBD panels = an illustrative pair near the first and
              third quartiles of AHI / RBDSQ (6 nearest candidates each) whose curves separate most cleanly. The RBD
              pair must also rise gradually from a flat low curve; if no pair meets these conditions they are dropped
              in turn, and the last fallback uses only the exams closest to each quartile.
  grey       30-min windows whose share of exams in REM sleep is in the top quartile (panel population).

    python -m psgscreen.figures.time_course
"""
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from scipy.stats import rankdata

from psgscreen import paths
from psgscreen.evaluation.clinical import youden
from psgscreen.figures.style import HEAD, use_fonts

OUT = f"{paths.FIGURES}/time_course"
AUC_TABLE = f"{paths.RESULTS}/time_course_auc.csv"
TIMEPOINT_TABLE = f"{paths.RESULTS}/time_course_timepoints.csv"
TASKS = [("Abnormal", "p_abnormal", "Abnormal classification"),
         ("OSA", "q_OSA", "OSA classification"),
         ("RBD", "q_RBD", "RBD classification")]
FS = {k: v * 1.15 for k, v in dict(title=17, label=15, tick=13, legend=15, ind_legend=12.5).items()}
B, COV_MIN, MARGIN = 2000, 0.9, 0.05


def task_rows(labels, task):
    """Exams evaluated for a task (Abnormal: all; OSA, RBD: abnormal exams) and their binary labels."""
    if task == "Abnormal":
        return np.ones(len(labels), bool), (labels != "Normal").astype(int)
    pos = np.isin(labels, ["OSA-only", "Both"] if task == "OSA" else ["RBD-only", "Both"])
    return labels != "Normal", pos.astype(int)


def auc_cols(S, y):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.full(S.shape[1], np.nan)
    R = rankdata(S, axis=0)
    return (R[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def sustained_first(ok, tt):
    """First time from which ok holds until the end (NaN if never)."""
    tail = np.flip(np.logical_and.accumulate(np.flip(ok, -1), axis=-1), -1)
    idx = np.where(tail.any(-1), tail.argmax(-1), -1)
    return np.where(idx >= 0, tt[np.clip(idx, 0, None)], np.nan)


def load_trajectories():
    T = pd.read_csv(f"{paths.RESULTS}/trajectory_5min.csv", encoding="utf-8-sig")
    T = T[T.center == "SC"].copy()
    T["exam_id"] = T["exam_id"].astype(str)
    T["uuid"] = T.exam_id.str.split("_").str[-1].str.lower()
    return T


def cumulative_auc(T):
    ex = T.drop_duplicates("uuid").set_index("uuid")
    uu, labels = ex.index.values, ex.label.values
    times = np.sort(T.t_min.unique())
    rng = np.random.RandomState(42)
    rows, points = [], []
    for task, col, _ in TASKS:
        m, y = task_rows(labels, task)
        u_t, y_t = uu[m], y[m]
        P = (T[T.uuid.isin(set(u_t))].pivot_table(index="uuid", columns="t_min", values=col)
             .reindex(index=u_t, columns=times))
        cov = P.notna().values.mean(0)
        kk = times <= times[cov >= COV_MIN].max()
        S, tt, cov = P.ffill(axis=1).values[:, kk], times[kk], cov[kk]
        auc = auc_cols(S, y_t)
        boot = []
        for _ in range(B):
            i = rng.randint(0, len(y_t), len(y_t))
            if 0 < y_t[i].sum() < len(i):
                boot.append(auc_cols(S[i], y_t[i]))
        boot = np.array(boot)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        rows += [dict(task=task, t_min=t, AUC=v, CI_low=l, CI_high=h, coverage=c, n=len(y_t),
                      n_pos=int(y_t.sum()), n_neg=int((1 - y_t).sum()))
                 for t, v, l, h, c in zip(tt, auc, lo, hi, cov)]

        ref, bref = auc[-1], boot[:, -1:]
        d_lo = np.percentile(boot - bref, 2.5, axis=0)
        t_ni = float(sustained_first(d_lo > -MARGIN, tt))
        j = np.searchsorted(tt, t_ni)
        r = dict(task=task, n=len(y_t), n_pos=int(y_t.sum()), n_neg=int((1 - y_t).sum()), t_end_h=tt[-1] / 60,
                 AUC_end=ref, AUC_end_lo=np.percentile(bref, 2.5), AUC_end_hi=np.percentile(bref, 97.5),
                 T_NI_h=t_ni / 60, dAUC_at_TNI=auc[j] - ref, dAUC_at_TNI_lo=d_lo[j],
                 dAUC_at_TNI_hi=np.percentile(boot[:, j] - bref[:, 0], 97.5))
        for h in (1, 2, 3, 4):
            j = np.searchsorted(tt, h * 60)
            dd = boot[:, j] - bref[:, 0]
            r.update({f"AUC_{h}h": auc[j], f"dAUC_{h}h": auc[j] - ref, f"dAUC_{h}h_lo": np.percentile(dd, 2.5),
                      f"dAUC_{h}h_hi": np.percentile(dd, 97.5)})
        points.append(r)
    return pd.DataFrame(rows), pd.DataFrame(points)


# ---------------------------------------------------------------- example exams
GROUP_COL = {"Normal": "p_normal", "Abnormal": "p_abnormal", "OSA": "q_OSA", "RBD": "q_RBD"}


def gmask(labels, g):
    if g == "Normal":
        return labels == "Normal"
    if g == "Abnormal":
        return labels != "Normal"
    return np.isin(labels, ["OSA-only", "Both"] if g == "OSA" else ["RBD-only", "Both"])


def visible_end(T, g):
    sub = T[gmask(T["label"].values, g)]
    cov = sub.groupby("t_min").exam_id.nunique() / sub.exam_id.nunique()
    return cov[cov >= COV_MIN].index.max()


def nearest_median(s, col):
    return s.iloc[(s[col] - s[col].median()).abs().values.argsort()[0]]


def quartile_pair(D, T, g, col, gradual=False, flat_low=False, nearest_only=False):
    """Exams near Q1 and Q3 of col whose curves separate most cleanly (see module docstring).
    gradual: the Q3 curve must lie above the Q1 curve in >= 98 % of blocks and start within one SD above the
    group's mean first-block score, and steady rises are preferred; flat_low: the Q1 curve must not exceed its
    final value by more than 0.02 in the first two hours; nearest_only: only the exams closest to each quartile."""
    pc = GROUP_COL[g]
    keep = visible_end(T, g)
    sub = T[T.t_min <= keep]
    C = {e: v.sort_values("t_min") for e, v in sub.groupby("exam_id")}
    gsub = sub[gmask(sub["label"].values, g)]
    m0 = gsub[gsub.t_min == gsub.t_min.min()][pc]
    band = (float(m0.mean()), float(m0.mean() + m0.std(ddof=1)))
    q1, q3 = D[col].quantile(.25), D[col].quantile(.75)
    lows = D.assign(_d=(D[col] - q1).abs()).nsmallest(6, "_d")
    highs = D.assign(_d=(D[col] - q3).abs()).nsmallest(6, "_d")
    if nearest_only:
        lows, highs = lows[lows._d == lows._d.min()], highs[highs._d == highs._d.min()]
    best, best_score = None, -np.inf
    for (_, a), (_, b) in itertools.product(lows.iterrows(), highs.iterrows()):
        if a.exam_id == b.exam_id:
            continue
        ca, cb = C.get(a.exam_id), C.get(b.exam_id)
        if ca is None or cb is None:
            continue
        m = pd.merge(ca[["t_min", pc]], cb[["t_min", pc]], on="t_min", suffixes=("_a", "_b"))
        if len(m) < 10:
            continue
        gap = m[pc + "_b"] - m[pc + "_a"]
        rough = float(ca[pc].diff().abs().mean() + cb[pc].diff().abs().mean())
        dev = abs(a[col] - q1) + abs(b[col] - q3)
        if flat_low:
            va, ta = ca[pc].values, ca["t_min"].values
            early = va[ta <= 120]
            if len(early) and float(early.max() - va[-1]) > 0.02:
                continue
        if gradual:
            vb = cb[pc].values
            if float((gap > 0).mean()) < 0.98 or not (band[0] <= float(vb[0]) <= band[1]):
                continue
            score = (2 * float((gap > 0).mean()) + 2 * float(vb[-1] - vb[0]) - 1.5 * float(vb[0])
                     - 2 * rough - 0.02 * dev)
            if flat_low:
                score -= 1.0 * abs(float(ca[pc].values[-1] - ca[pc].values[0]))
        else:
            score = 2 * float((gap > 0).mean()) + 3 * float(gap.mean()) - 2 * rough - 0.02 * dev
        if score > best_score:
            best_score, best = score, (a, b)
    return best


def select_examples(T):
    """{task: [(exam_id, style, legend label)]} for the three panels."""
    pred = pd.read_csv(paths.disorder_predictions(), encoding="utf-8-sig")
    oof = pred[pred.bucket == "OOF"]
    dis = oof[oof.label != "Normal"]
    thr = {"Normal": 1.0 - youden((oof.label != "Normal").astype(int), oof.p_abnormal),
           "Abnormal": youden((oof.label != "Normal").astype(int), oof.p_abnormal),
           "OSA": youden(dis.label.isin(["OSA-only", "Both"]).astype(int), dis.q_OSA),
           "RBD": youden(dis.label.isin(["RBD-only", "Both"]).astype(int), dis.q_RBD)}
    last = T.sort_values("t_min").groupby("exam_id").tail(1)
    correct = {g: set(last.exam_id[gmask(last.label.values, g) & (last[GROUP_COL[g]].values >= thr[g])])
               for g in GROUP_COL}

    ros = pd.read_csv(paths.ROSTER, encoding="utf-8-sig")
    ros["exam_id"] = ros["exam_id"].astype(str)
    D = ros[ros.exam_id.isin(set(T.exam_id))]
    D = D.merge(pd.read_csv(paths.DEMOGRAPHICS, encoding="utf-8-sig")[["uuid", "age", "sex", "ahi"]],
                on="uuid", how="left")
    D = D.merge(pd.read_csv(paths.REPORT_TABLE, encoding="utf-8-sig")[["exam_id", "rbdsq"]].astype({"exam_id": str}),
                on="exam_id", how="left")

    def pool(g, col):
        return D[gmask(D.label.values, g) & D.exam_id.isin(correct[g])].dropna(subset=[col])

    s = pool("Normal", "age")
    normal = nearest_median(s[(s.sex == "F") & (s.age >= s.age.median())], "age")
    s = pool("Abnormal", "ahi")
    rbd_only, osa_only = (nearest_median(s[s.label == lb], "ahi") for lb in ("RBD-only", "OSA-only"))
    o1, o3 = quartile_pair(pool("OSA", "ahi"), T, "OSA", "ahi")
    r1, r3 = (quartile_pair(pool("RBD", "rbdsq"), T, "RBD", "rbdsq", gradual=True, flat_low=True)
              or quartile_pair(pool("RBD", "rbdsq"), T, "RBD", "rbdsq", gradual=True)
              or quartile_pair(pool("RBD", "rbdsq"), T, "RBD", "rbdsq", nearest_only=True))
    return {"Abnormal": [(normal.exam_id, "normal", "Normal"), (osa_only.exam_id, "q3", "OSA-only"),
                         (rbd_only.exam_id, "q1", "RBD-only")],
            "OSA": [(o1.exam_id, "q1", f"AHI {o1.ahi:.1f} (Q1)"), (o3.exam_id, "q3", f"AHI {o3.ahi:.1f} (Q3)")],
            "RBD": [(r1.exam_id, "q1", f"RBDSQ {r1.rbdsq:.0f} (Q1)"), (r3.exam_id, "q3", f"RBDSQ {r3.rbdsq:.0f} (Q3)")]}


def rem_zones(uuids, zone_min=30.0, min_cov=0.5):
    """30-min windows whose share of exams in REM (>= half of the window) is in the top quartile among windows
    with data for >= half of the exams."""
    H = pd.read_csv(paths.HYPNOGRAM, encoding="utf-8-sig")
    H = H[H.uuid.str.lower().isin(set(uuids))]
    n_tot = H.uuid.nunique()
    df = pd.DataFrame({"b": ((H.t_sec / 60.0) // zone_min).astype(int).values,
                       "u": H.uuid.values, "r": (H.stage.values == "R").astype(float)})
    per = df.groupby(["b", "u"]).r.mean().reset_index()
    Z = per.groupby("b").agg(n_rem=("r", lambda s: int((s >= .5).sum())), n_data=("r", "size")).reset_index()
    Z["frac"] = Z.n_rem / Z.n_data
    Z["coverage"] = Z.n_data / n_tot
    ok = Z[Z.coverage >= min_cov]
    Z = Z[(Z.frac >= ok.frac.quantile(0.75)) & (Z.coverage >= min_cov)]
    return pd.DataFrame({"zone_lo_h": Z.b * zone_min / 60.0, "zone_hi_h": (Z.b + 1) * zone_min / 60.0})


STYLE = {"q1": dict(ls="-", marker="o"), "q3": dict(ls=(0, (6, 2.5)), marker="s"),
         "normal": dict(ls=(0, (7, 2, 1.5, 2)), marker="D")}


def draw(T, examples, out):
    A = pd.read_csv(AUC_TABLE, encoding="utf-8-sig")
    TP = pd.read_csv(TIMEPOINT_TABLE, encoding="utf-8-sig").set_index("task")
    ex = T.drop_duplicates("uuid").set_index("uuid")
    uu, labels = ex.index.values, ex.label.values
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 6.6), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.08, wspace=0.06)
    ymin = min(0.5, np.floor(A.CI_low.min() / .05) * .05)
    for ax, (task, col, title) in zip(axes, TASKS):
        d = A[A.task == task].sort_values("t_min")
        end = d.t_min.max()
        x = d.t_min.values / 60.0
        c = HEAD[task]
        for r in rem_zones(set(uu[task_rows(labels, task)[0]])).itertuples():
            ax.axvspan(r.zone_lo_h, r.zone_hi_h, color="#888", alpha=.16, linewidth=0, zorder=0)
        ax.fill_between(x, d.CI_low, d.CI_high, color=c, alpha=.16, linewidth=0, zorder=1)
        ax.plot(x, d.AUC, color=c, lw=2.2, zorder=4)
        ax.set_xlim(0, end / 60.0)
        th = TP.loc[task, "T_NI_h"]
        ax.axvline(th, color=c, ls=(0, (2, 2)), lw=1.4, zorder=3)
        ax.plot([th], [np.interp(th, x, d.AUC.values)], "o", color=c, markersize=7, markeredgecolor="white",
                markeredgewidth=1.2, zorder=8)
        mins = int(round(th * 60))
        ax.text(th + .08, ymin + .008, f"{mins // 60} h {mins % 60:02d} min" if th >= 1 else f"{mins} min",
                color=c, fontsize=FS["tick"] * .9, ha="left", va="bottom", zorder=8)
        ax.set_ylim(ymin, 1.0)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.set_title(title, fontsize=FS["title"])
        ax.set_xlabel("Cumulative recording time (hours)", fontsize=FS["label"])
        ax.set_ylabel("AUROC", fontsize=FS["label"], color=c)
        ax.tick_params(labelsize=FS["tick"])
        ax.tick_params(axis="y", labelcolor=c)
        ax2 = ax.twinx()
        handles = []
        for eid, style, lab in examples[task]:
            q = T[(T.exam_id == eid) & (T.t_min <= end)].sort_values("t_min")
            ax2.plot(q.t_min / 60.0, q[col].values, color="#111111", lw=1.4, alpha=.95, zorder=6,
                     markevery=12, markersize=3.4, markerfacecolor="white", markeredgewidth=.9, **STYLE[style])
            handles.append(Line2D([], [], color="#111111", lw=1.4, markersize=3.4 * 1.3, markerfacecolor="white",
                                  markeredgewidth=.9, label=lab, **STYLE[style]))
        ax2.set_ylim(0, 1.0)
        ax2.yaxis.set_major_locator(MultipleLocator(0.2))
        ax2.set_ylabel("Predicted probability", fontsize=FS["label"])
        ax2.tick_params(labelsize=FS["tick"])
        ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=len(handles),
                   frameon=False, fontsize=FS["ind_legend"], handlelength=2.6, columnspacing=1.1,
                   handletextpad=0.5, borderaxespad=0.0)
    fig.legend(handles=[Patch(facecolor="#888", alpha=.16, edgecolor="#999", linewidth=.6,
                              label="30-min windows in which the proportion of exams in REM sleep "
                                    "is in the top quartile for the exams evaluated in that panel")],
               loc="outside lower center", frameon=False, fontsize=FS["legend"], handlelength=2.2, handleheight=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/fig_time_course.{ext}", dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    use_fonts()
    T = load_trajectories()
    A, TP = cumulative_auc(T)
    A.to_csv(AUC_TABLE, index=False, encoding="utf-8-sig")
    TP.to_csv(TIMEPOINT_TABLE, index=False, encoding="utf-8-sig")
    examples = select_examples(T)
    pd.DataFrame([dict(task=t, style=s, label=lab, uuid=e.split("_")[-1]) for t, v in examples.items()
                  for e, s, lab in v]).to_csv(f"{OUT}/examples.csv", index=False, encoding="utf-8-sig")
    draw(T, examples, OUT)
    print("saved", f"{OUT}/fig_time_course.png")


if __name__ == "__main__":
    main()
