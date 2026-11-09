"""Explainability figure of the disorder classifier (locked internal test set, fold-mean TreeSHAP).

  a  event-level SHAP (sum over the event's mean / max / SD features) of the Abnormal, OSA and RBD
     classifiers; colour = event level (mean of the three features, each scaled to its 5th-95th percentile)
  b  SHAP waterfall of one illustrative correctly classified exam per class. Pool: end-to-end correct, patient
     without a development exam, sign of the fold-mean logit agreeing with the reported probability; exams ranked
     by the distance of their event-SHAP vector to the pool mean. Normal: the closest exam aged 60-69; OSA and RBD:
     the closest exam whose most influential event (largest |event SHAP|) is at or above the 90th percentile of
     event level. The right axis gives the exam's event-level percentile among locked-test exams.

Input: results/shap_disorder.npz (model.shap_values).

    python -m psgscreen.figures.shap [a b]
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import MultipleLocator

from psgscreen import paths
from psgscreen.events import RBD_RELATED, name
from psgscreen.figures.style import HEAD, use_fonts
from psgscreen.model import disorder as dm
from psgscreen.model.shap_values import CACHE, HEADS

OUT = f"{paths.FIGURES}/shap"
FS = {k: v * 1.15 for k, v in dict(title=17, label=15, tick=13, legend=15).items()}
TITLE = {"Abnormal": "Abnormal classification", "OSA": "OSA classification", "RBD": "RBD classification"}
GROUP_COLOR = {"OSA": HEAD["OSA"], "RBD": HEAD["RBD"]}
GROUP_LABEL = {"OSA": "OSA-related", "RBD": "RBD-related"}


def group_of(e):
    return "RBD" if e in RBD_RELATED else "OSA"


def load_event_shap():
    """Events (feature order), event-level SHAP per head, bias, and fold-mean features per head."""
    z = np.load(CACHE, allow_pickle=True)
    feats = [str(f) for f in z["feats"]]
    evs = list(dict.fromkeys(f.split("__", 1)[0] for f in feats))
    M = np.zeros((len(feats), len(evs)))
    for i, f in enumerate(feats):
        M[i, evs.index(f.split("__", 1)[0])] = 1.0
    C = {h: z[f"C_{h}"] for h in HEADS}
    return feats, evs, M, C, {h: z[f"X_{h}"] for h in HEADS}


def norm01(x):
    lo, hi = np.nanpercentile(x, 5), np.nanpercentile(x, 95)
    return np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1)


def event_level(X, feats, evs):
    return np.stack([np.mean([norm01(X[:, feats.index(f"{e}__{s}")]) for s in ("mean_prob", "max_prob", "std_prob")],
                             0) for e in evs], 1)


# ---------------------------------------------------------------- a: event-level beeswarms
def beeswarm(ax, sv, X, labels):
    imp = np.abs(sv).mean(0)
    order = np.argsort(imp)
    rng = np.random.RandomState(0)
    for row, j in enumerate(order):
        x = X[:, j]
        lo, hi = np.nanpercentile(x, 5), np.nanpercentile(x, 95)
        ax.scatter(sv[:, j], np.full(len(sv), row) + (rng.rand(len(sv)) - 0.5) * 0.30,
                   c=np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1), cmap="coolwarm", s=12, alpha=0.75, linewidths=0)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[j] for j in order], fontsize=FS["label"])
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.tick_params(axis="y", pad=3)
    ax.tick_params(axis="x", labelsize=FS["tick"])
    return order, imp


def panel_a(out):
    feats, evs, M, C, X = load_event_shap()
    fig, axes = plt.subplots(1, len(HEADS), figsize=(16.0, 6.0), constrained_layout=True)
    rows = []
    for ax, h in zip(axes, HEADS):
        E = C[h][:, :len(feats)] @ M
        order, imp = beeswarm(ax, E, event_level(X[h], feats, evs), [name(e) for e in evs])
        ax.set_xlabel("Event SHAP value (log-odds)", fontsize=FS["label"])
        ax.set_title(TITLE[h], fontsize=FS["title"], pad=8)
        for tl, j in zip(ax.get_yticklabels(), order):
            tl.set_color(GROUP_COLOR[group_of(evs[j])])
        rows += [dict(head=h, rank=r, event=name(evs[j]), group=GROUP_LABEL[group_of(evs[j])],
                      mean_abs_shap=float(imp[j])) for r, j in enumerate(np.argsort(-imp), 1)]
    fig.legend(handles=[Line2D([], [], marker="s", ls="none", ms=8.0, color=GROUP_COLOR[g], label=GROUP_LABEL[g])
                        for g in ("OSA", "RBD")],
               loc="outside lower center", ncol=2, frameon=False, fontsize=FS["legend"], handletextpad=0.4,
               columnspacing=2.0)
    for ax in axes:
        lo, hi = ax.get_xlim()
        ax.xaxis.set_major_locator(MultipleLocator(1.0 if hi - lo > 2.5 else 0.5))
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1)), ax=axes,
                      fraction=0.02, pad=0.01, aspect=30)
    cb.set_label("Event level (low → high)", fontsize=FS["label"])
    cb.ax.tick_params(labelsize=FS["tick"])
    fig.savefig(f"{out}/panel_a.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(f"{out}/panel_a.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------- b: individual waterfalls
GROUPS = ["Normal", "OSA-only", "RBD-only"]
HEAD_OF = {"Normal": "Abnormal", "OSA-only": "OSA", "RBD-only": "RBD"}
PCOL = {"Abnormal": "p_abnormal", "OSA": "q_OSA", "RBD": "q_RBD"}
PANEL_TITLE = {"Normal": "Normal", "OSA-only": "OSA", "RBD-only": "RBD"}
XLABEL = {h: f"{h} classifier (log-odds)" for h in HEADS}
NORMAL_AGE = (60, 69)
MIN_TOP_PCT = 90
POS_COL, NEG_COL = "#3b3b3b", "#b9b7b1"
WF = {k: v * 1.15 for k, v in dict(title=17, label=15, tick=13, val=12.5, small=12, hdr=11.5).items()}


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def prob_ticks(lo, hi, min_frac=0.17):
    """Probability ticks for the top axis that are at least min_frac of the axis apart."""
    acc = []
    for p in [0.5, 0.1, 0.9, 0.25, 0.75, 0.05, 0.95, 0.01, 0.99, 0.02, 0.98, 0.2, 0.8, 0.3, 0.7]:
        x = logit(p)
        if lo <= x <= hi and all(abs(x - logit(q)) >= min_frac * (hi - lo) for q in acc):
            acc.append(p)
    return sorted(acc)


def fmt_p(p):
    s = f"{p:.2f}".rstrip("0").rstrip(".")
    return s if s not in ("0", "1") else f"{p:.2f}"


def waterfall(ax, phi, base, names, gcols, pct, title, xlabel):
    """Arrows from E[f(X)] to f(x), smallest |phi| at the bottom; top axis = probability."""
    n = len(phi)
    order = np.argsort(np.abs(phi), kind="stable")
    starts, ends = np.empty(n), np.empty(n)
    cur = base
    for r, j in enumerate(order):
        starts[r], ends[r] = cur, cur + phi[j]
        cur = ends[r]
    fx = cur
    lo = min(starts.min(), ends.min(), base, fx)
    hi = max(starts.max(), ends.max(), base, fx)
    span = max(hi - lo, 1e-6)
    h = 0.66
    for xv in (base, fx):
        ax.plot([xv, xv], [-1.35, n - 1 + 0.95], color="#9a9a9a", lw=0.9, ls=(0, (2, 2)), zorder=1)
    texts, bend, ws = [], [], []
    for r, j in enumerate(order):
        s, e = starts[r], ends[r]
        w = e - s
        hl = min(0.45 * abs(w), 0.035 * span)
        body_end = e - (np.sign(w) if w != 0 else 1.0) * hl
        ax.add_patch(Polygon([(s, r - h / 2), (body_end, r - h / 2), (e, r), (body_end, r + h / 2), (s, r + h / 2)],
                             closed=True, facecolor=POS_COL if w > 0 else NEG_COL, edgecolor="none", zorder=3))
        texts.append(ax.text(e, r, f"{phi[j]:+.2f}".replace("-", "−"), ha="left", va="center", zorder=4,
                             fontsize=WF["val"], color="#222",
                             bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none")))
        bend.append(body_end)
        ws.append(w)
    ax.set_xlim(lo - 0.25 * span, hi + 0.25 * span)
    ax.set_ylim(-1.6, n - 1 + 1.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([names[j] for j in order], fontsize=WF["label"])
    for t, j in zip(ax.get_yticklabels(), order):
        t.set_color(gcols[j])
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", labelsize=WF["tick"])
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=WF["label"])
    ax.set_title(title, fontsize=WF["title"], pad=12)
    box = dict(boxstyle="square,pad=0.08", fc="white", ec="none")
    t_base = ax.text(base, -1.0, f"E[f(X)] = {base:+.2f}".replace("-", "−").replace("+", ""), va="center",
                     fontsize=WF["val"], color="#555", zorder=5, bbox=box)
    t_fx = ax.text(fx, n - 1 + 0.82, f"f(x) = {fx:.2f}".replace("-", "−") + f"  (p = {sigmoid(fx):.2f})",
                   va="center", fontsize=WF["val"], color="#222", zorder=5, bbox=box)
    top = ax.secondary_xaxis("top", functions=(sigmoid, logit))
    top.tick_params(axis="x", colors="#777", length=3, labelsize=WF["small"])
    top.spines["top"].set_color("#bbb")
    rax = ax.secondary_yaxis("right")
    rax.set_yticks(range(n))
    rax.set_yticklabels([f"{int(round(pct[j]))}%" for j in order], fontsize=WF["small"], color="#777")
    rax.tick_params(axis="y", length=0, pad=6)
    rax.spines["right"].set_visible(False)
    ax.annotate("pct", xy=(1, n - 1 + 0.82), xycoords=("axes fraction", "data"), xytext=(6, 0),
                textcoords="offset points", ha="left", va="center", fontsize=WF["hdr"], color="#777")
    return dict(ax=ax, starts=starts, ends=ends, bend=np.array(bend), w=np.array(ws), texts=texts,
                lo=lo, hi=hi, base=base, fx=fx, t_base=t_base, t_fx=t_fx, top=top)


def fit_panels(fig, states, iters=4):
    """Put each value label inside its arrow when it fits (else beside it), widen the x range to the labels,
    keep the E[f(X)] / f(x) labels and the titles inside the figure, and set the tick spacing."""
    for _ in range(iters):
        fig.canvas.draw()
        rnd = fig.canvas.get_renderer()
        for st in states:
            ax = st["ax"]
            x0, x1 = ax.get_xlim()
            ppd = ax.bbox.width / (x1 - x0)
            gap = 5.0 / ppd
            L, R = st["lo"], st["hi"]
            for k, (s, e, be, w, t) in enumerate(zip(st["starts"], st["ends"], st["bend"], st["w"], st["texts"])):
                tw = t.get_window_extent(renderer=rnd).width / ppd
                if abs(be - s) >= tw + 2 * gap:
                    t.set_position(((s + be) / 2, k)); t.set_ha("center")
                    t.set_color("white" if w > 0 else "#222")
                    t.get_bbox_patch().set_visible(False)
                else:
                    sg = 1.0 if w >= 0 else -1.0
                    x = e + sg * gap
                    t.set_position((x, k)); t.set_ha("left" if sg > 0 else "right"); t.set_color("#222")
                    t.get_bbox_patch().set_visible(True)
                    if sg > 0:
                        R = max(R, x + tw)
                    else:
                        L = min(L, x - tw)
            mg_ = 6.0 / ppd
            ax.set_xlim(L - mg_, R + mg_)
            xl0, xl1 = ax.get_xlim()
            for t, xv in ((st["t_base"], st["base"]), (st["t_fx"], st["fx"])):
                right = xv > (xl0 + xl1) / 2
                t.set_ha("right" if right else "left")
                x = xv + (-gap if right else gap)
                tw = t.get_window_extent(renderer=rnd).width / ppd
                xa, xb = (x - tw, x) if right else (x, x + tw)
                if xb > xl1 - mg_:
                    x -= xb - (xl1 - mg_)
                elif xa < xl0 + mg_:
                    x += (xl0 + mg_) - xa
                t.set_position((x, t.get_position()[1]))
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    fw = fig.bbox.width
    for st in states:
        ax = st["ax"]
        tb = ax.title.get_window_extent(renderer=rnd)
        over = max(0.0, tb.x1 - (fw - 6)) - max(0.0, 6 - tb.x0)
        if over != 0:
            ax.title.set_x(0.5 - over / ax.bbox.width)
    for st in states:
        xl0, xl1 = st["ax"].get_xlim()
        st["ax"].xaxis.set_major_locator(MultipleLocator(1.0 if xl1 - xl0 <= 6 else 2.0))
        tk = prob_ticks(xl0, xl1)
        st["top"].set_xticks(tk)
        st["top"].set_xticklabels([fmt_p(p) for p in tk])


def panel_b(out):
    feats, evs, M, C, _ = load_event_shap()
    mg, ft, Xs = dm.load_fold_tables()
    assert ft == feats
    y, _, ti, _ = dm.cohort_index(mg)
    rows_for = {"Abnormal": ti, "OSA": ti[y[ti] != 0], "RBD": ti[y[ti] != 0]}
    xbar = np.mean([Xs[k] for k in range(dm.NF)], 0)
    names = [name(e) for e in evs]
    gcols = [GROUP_COLOR[group_of(e)] for e in evs]

    comp = event_level(xbar[ti], feats, evs)          # event-level percentile among locked-test exams
    pct_of = dict(zip(mg.uuid.values[ti], np.stack([[100.0 * np.mean(comp[:, k] <= comp[i, k])
                                                     for k in range(len(evs))] for i in range(len(ti))])))
    PHI, BASE, FX = {}, {}, {}
    for h, r in rows_for.items():
        uu = mg.uuid.values[r]
        PHI[h] = dict(zip(uu, C[h][:, :len(feats)] @ M))
        BASE[h] = dict(zip(uu, C[h][:, -1]))
        FX[h] = dict(zip(uu, C[h].sum(1)))

    S = pd.read_csv(paths.disorder_predictions(), encoding="utf-8-sig")
    S = S[S.bucket == "TEST"].set_index("uuid")
    T = pd.DataFrame({"uuid": mg.uuid.values[ti], "label": mg.label.values[ti]})
    T = T.join(S[["p_abnormal", "q_OSA", "q_RBD"]], on="uuid")
    D = pd.read_csv(paths.DEMOGRAPHICS, encoding="utf-8-sig")
    ros = pd.read_csv(paths.ROSTER, encoding="utf-8-sig")
    D = D.merge(ros[["uuid", "split"]], on="uuid")
    D = D[D.center.isin(paths.INTERNAL)]
    T = T.merge(D[["uuid", "patient_id", "age", "sex"]], on="uuid", how="left")
    T["pid_in_dev"] = T.patient_id.isin(set(D.loc[D.split == "dev", "patient_id"]))
    a_, o_, r_ = T.p_abnormal > 0.5, T.q_OSA > 0.5, T.q_RBD > 0.5
    T["correct"] = np.select([T.label == "Normal", T.label == "OSA-only", T.label == "RBD-only"],
                             [~a_, a_ & o_ & ~r_, a_ & ~o_ & r_], default=False)
    T["head"] = T.label.map(HEAD_OF)
    ok = T["head"].notna()
    T.loc[ok, "p_head"] = [getattr(r, PCOL[r.head]) for r in T[ok].itertuples()]
    T.loc[ok, "fx_head"] = [FX[r.head][r.uuid] for r in T[ok].itertuples()]
    T["side_ok"] = (sigmoid(T.fx_head) > 0.5) == (T.p_head > 0.5)

    picks, phi_rows = [], []
    for g in GROUPS:
        h = HEAD_OF[g]
        P = T[(T.label == g) & T.correct & ~T.pid_in_dev & T.side_ok].copy()
        Phi = np.stack([PHI[h][u] for u in P.uuid])
        P["dist"] = np.linalg.norm(Phi - Phi.mean(0), axis=1)
        P["typ_rank"] = P.dist.rank(method="min").astype(int)
        P = P.sort_values(["dist"])
        if g == "Normal":
            s = P[(P.age >= NORMAL_AGE[0]) & (P.age <= NORMAL_AGE[1])].iloc[0]
        else:
            top = [int(np.argmax(np.abs(PHI[h][u]))) for u in P.uuid]
            s = P[[pct_of[u][j] >= MIN_TOP_PCT for u, j in zip(P.uuid, top)]].iloc[0]
        picks.append(dict(group=g, head=h, uuid=s.uuid, age=s.age, sex=s.sex, p=s.p_head, fx=FX[h][s.uuid],
                          base=BASE[h][s.uuid], dist=s.dist, typ_rank=int(s.typ_rank)))
        phi_rows += [dict(group=g, head=h, event=names[k], phi=PHI[h][s.uuid][k], activity_pct=pct_of[s.uuid][k])
                     for k in range(len(evs))]
    pd.DataFrame(picks).to_csv(f"{out}/panel_b_picks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(phi_rows).to_csv(f"{out}/panel_b.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 6.8), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.06, wspace=0.04)
    states = []
    for ax, rec in zip(axes, picks):
        sex = {"M": "Male", "F": "Female"}.get(str(rec["sex"]), str(rec["sex"]))
        title = f"{PANEL_TITLE[rec['group']]} ({sex}, {float(rec['age']):.0f} y)"
        u, h = rec["uuid"], rec["head"]
        states.append(waterfall(ax, PHI[h][u], BASE[h][u], names, gcols, pct_of[u], title, XLABEL[h]))
    fit_panels(fig, states)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/panel_b.{ext}", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panels", nargs="*", help="any of a b (default: both)")
    panels = ap.parse_args().panels or list("ab")
    os.makedirs(OUT, exist_ok=True)
    use_fonts()
    for p, f in (("a", panel_a), ("b", panel_b)):
        if p in panels:
            f(OUT)
            print("saved", f"{OUT}/panel_{p}.png")


if __name__ == "__main__":
    main()
