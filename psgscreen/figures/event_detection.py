"""Event-detection figure, one PNG per panel.

  a  window AUC of the audio-only, video-only and fused detectors on the locked internal test set, per fold
     (bar = mean, error bar = SD over the 10 fold detectors; the best of the three per event in bold)
  b  late-fusion audio / video weight per event (10-fold mean), drawn under a on the same event axis
  c  event-locked audio and video signal of the three most audio-weighted events
  d  the same for the three most video-weighted events

Events are ordered by their fold-mean audio fusion weight in a and b. Each panel is drawn at twice its size in the
final layout, saved at 300 dpi and tagged 600 dpi; letters and placement were added by hand.
Inputs: results/detection_perfold_summary.csv (evaluation.detection), models/fusion/f*/fusion_report.csv,
traces/traces_{INT,GN}.npz and traces/traces_hypopnea_{INT,GN}.npz (dataset.event_traces).

    python -m psgscreen.figures.event_detection [a b c d]
"""
import argparse
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from psgscreen import paths
from psgscreen.events import EXCLUDED, RBD_RELATED, name, raw
from psgscreen.figures.style import FUSED, MODALITY, NEUTRAL, tag_dpi, use_fonts

OUT = f"{paths.FIGURES}/event_detection"
WRAP = {"central apnea": "central\napnea", "mixed apnea": "mixed\napnea", "obstructive apnea": "obstructive\napnea",
        "obstructive hypopnea": "obstructive\nhypopnea", "oximeter event": "oximeter\nevent",
        "oxygen desaturation": "oxygen\ndesaturation", "SpO2 threshold event": "SpO2\nthreshold\nevent",
        "indeterminate event": "indeterminate\nevent", "limb movement": "limb\nmovement", "RBD symptom": "RBD\nsymptom"}
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"


def fusion_weights():
    """Fold-mean audio fusion weight per event (events weighted in every fold)."""
    W = {}
    for k in range(paths.N_FOLDS):
        for e, v in json.load(open(f"{paths.FUSION}/f{k}/fusion_weights.json")).items():
            W.setdefault(e, []).append(float(v[0]))
    return {e: float(np.mean(v)) for e, v in W.items() if len(v) == paths.N_FOLDS}


def event_order(display_names):
    """Display names ordered by fold-mean audio fusion weight (descending); ties by group, then name."""
    W = fusion_weights()
    ev = sorted(display_names, key=lambda e: (raw(e) in RBD_RELATED, e.lower()))
    return sorted(ev, key=lambda e: -W[raw(e)])


# ---------------------------------------------------------------- a: per-fold detection AUC
def draw_a():
    """Figure, axes and per-event table of panel a."""
    rows = [("Audio only", MODALITY["Audio"]), ("Video only", MODALITY["Video"]), ("Fused", FUSED)]
    S = pd.read_csv(f"{paths.RESULTS}/detection_perfold_summary.csv", encoding="utf-8-sig").set_index(["row", "event"])
    events = event_order([e for e in S.loc["Fused"].index if e != "Mean"]) + ["Mean"]
    val, lo, hi, table = {}, {}, {}, []
    for r, _ in rows:
        g = S.loc[r].loc[events]
        assert (g["n"] == paths.N_FOLDS).all(), r
        val[r], lo[r], hi[r] = g["mean"].values, g["mean"].values - g["sd"].values, g["mean"].values + g["sd"].values
        table += [dict(row=r, event=e, AUC_mean=g.loc[e, "mean"], AUC_sd=g.loc[e, "sd"], AUC_min=g.loc[e, "min"],
                       AUC_max=g.loc[e, "max"], n_folds=int(g.loc[e, "n"])) for e in events]

    gap, gw = 0.55, 0.84
    x = np.arange(len(events), dtype=float); x[-1] += gap
    bw = gw / len(rows)
    fig, ax = plt.subplots(figsize=(15.830, 4.097), dpi=300)
    best = np.argmax(np.vstack([val[r] for r, _ in rows]), axis=0)
    texts = []
    for j, (r, col) in enumerate(rows):
        xs = x - gw / 2 + bw * (j + 0.5)
        v = val[r]
        ax.bar(xs, v, width=bw, color=col, edgecolor="white", linewidth=1.0, label=r, zorder=2)
        ax.errorbar(xs, v, yerr=[v - lo[r], hi[r] - v], fmt="none", ecolor="#222222",
                    elinewidth=1.0, capsize=2.2, capthick=1.0, zorder=3)
        for i in range(len(v)):
            top = best[i] == j
            texts.append(ax.text(xs[i], hi[r][i] + 0.012, f"{v[i]:.4f}", rotation=90, ha="center", va="bottom",
                                 fontsize=13, fontweight="bold" if top else "normal",
                                 color="#000000" if top else "#6b6b6b"))
    ax.set_xticks(x)
    ax.set_xticklabels([WRAP.get(e, e) for e in events], fontsize=14)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.round(np.arange(0.0, 1.0001, 0.2), 1))
    ax.tick_params(axis="y", labelsize=13)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("AUROC", fontsize=14)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    hd, lb = ax.get_legend_handles_labels()
    ax.legend(hd, lb, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(rows), frameon=False,
              fontsize=14, handlelength=1.2, columnspacing=1.8)
    fig.tight_layout()
    for _ in range(8):          # raise the upper y limit until the value labels fit; the axis line stays 0-1
        fig.canvas.draw()
        rnd = fig.canvas.get_renderer(); inv = ax.transData.inverted()
        top_ = max(1.0, max(inv.transform((0, t.get_window_extent(rnd).y1))[1] for t in texts) + 0.01)
        if abs(top_ - ax.get_ylim()[1]) < 1e-4:
            break
        ax.set_ylim(0.0, top_)
    ax.spines["left"].set_bounds(0.0, 1.0)
    return fig, ax, table


def panel_a(out):
    fig, _, table = draw_a()
    fig.savefig(f"{out}/panel_a.png", dpi=300)
    fig.savefig(f"{out}/panel_a.pdf")
    plt.close(fig)
    pd.DataFrame(table).to_csv(f"{out}/panel_a.csv", index=False, encoding="utf-8-sig")
    return f"{out}/panel_a.png"


# ---------------------------------------------------------------- b: fusion weights
def weights_table():
    """Fold-mean fusion weight and validation AUC per event (display names), sorted by fused AUC."""
    D = pd.concat([pd.read_csv(f"{paths.FUSION}/f{k}/fusion_report.csv", encoding="utf-8-sig").assign(fold=k)
                   for k in range(paths.N_FOLDS)], ignore_index=True)
    D = D[D.event != EXCLUDED]
    n_folds = D.groupby("event").fold.nunique()
    D = D[D.event.isin(n_folds[n_folds == paths.N_FOLDS].index)]
    D["event"] = D.event.map(name)
    g = D.groupby("event").agg(auc=("auc_fused", "mean"), auc_sd=("auc_fused", "std"),
                               w=("w_audio", "mean"), w_sd=("w_audio", "std"),
                               a_auc=("auc_audio", "mean"), v_auc=("auc_video", "mean")).sort_values("w")
    T = g.reset_index().rename(columns=dict(auc="fused_AUC", auc_sd="fused_AUC_sd", w="w_audio", w_sd="w_audio_sd",
                                            a_auc="AUC_audio_only", v_auc="AUC_video_only"))
    return T.sort_values("fused_AUC", ascending=False)


def panel_b(out):
    """Stacked audio / video weight per event, on the x positions, column width and fonts of panel a."""
    figa, axa, _ = draw_a()
    pos, xlim, xt = axa.get_position(), axa.get_xlim(), axa.get_xticks()
    labels = [t.get_text().replace("\n", " ") for t in axa.get_xticklabels()]
    plt.close(figa)
    T = weights_table()
    T.to_csv(f"{out}/panel_b.csv", index=False, encoding="utf-8-sig")
    B = T.set_index("event")
    W, bottom_in, axes_in, top_in = 15.830, 0.78, 1.804, 0.12
    H = bottom_in + axes_in + top_in
    fig = plt.figure(figsize=(W, H), dpi=300)
    ax = fig.add_axes([pos.x0, bottom_in / H, pos.width, axes_in / H])
    gw = 0.84
    for x, e in zip(xt[:-1], labels[:-1]):
        w = float(B.loc[e, "w_audio"])
        ax.bar(x, w, width=gw, color=MODALITY["Audio"], edgecolor="white", linewidth=1.0, zorder=2)
        ax.bar(x, 1 - w, bottom=w, width=gw, color=MODALITY["Video"], edgecolor="white", linewidth=1.0, zorder=2)
        ax.text(x, w / 2, f"{w * 100:.0f}%", ha="center", va="center", fontsize=13, color="white", zorder=3)
        ax.text(x, w + (1 - w) / 2, f"{(1 - w) * 100:.0f}%", ha="center", va="center", fontsize=13, color="white", zorder=3)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .5, 1])
    ax.set_yticklabels(["0", "50", "100"], fontsize=13)
    ax.set_xticks(xt[:-1])
    ax.set_xticklabels([WRAP.get(e, e) for e in labels[:-1]], fontsize=14)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Late-fusion weight (%)", fontsize=14, labelpad=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_bounds(0, 1)
    hd = [plt.Rectangle((0, 0), 1, 1, color=MODALITY["Video"]), plt.Rectangle((0, 0), 1, 1, color=MODALITY["Audio"])]
    ax.legend(hd, ["Video", "Audio"], loc="center", bbox_to_anchor=(xt[-1], 0.58), bbox_transform=ax.get_xaxis_transform(),
              frameon=False, fontsize=14, handlelength=1.2)
    ax.text(xt[-1], 0.20, "10-fold mean", ha="center", va="center", fontsize=13, color="#52514e")
    fig.savefig(f"{out}/panel_b.png", dpi=300, facecolor="#ffffff")
    plt.close(fig)
    return f"{out}/panel_b.png"


# ---------------------------------------------------------------- c, d: event-locked raw signal
TRACE_WIDTH = {"c": (8.133, None), "d": (8.127, (0.88, 1.2))}     # inches; fixed video y range of panel d
TRACE_TITLE = {"oximeter event": "Oximeter event", "snoring": "Snoring", "obstructive apnea": "Obstructive apnea",
               "rbd": "RBD symptom", "limb movement": "Limb movement", "central apnea": "Central apnea",
               "hypopnea": "Hypopnea"}
KEYS = ("event", "group", "uuid", "dur", "aud", "vid")
N_BOOT, SMOOTH = 300, 5         # bootstrap replicates; moving-average length (samples) of the plotted curves


def load_traces(tag=""):
    parts = [np.load(f"{paths.TRACES}/traces{tag}_{c}.npz") for c in paths.INTERNAL]
    Z = {k: np.concatenate([p[k] for p in parts]) for k in KEYS}
    Z["t_aud"], Z["t_vid"] = parts[0]["t_aud"], parts[0]["t_vid"]
    return Z


def traces():
    """Both trace draws, exam-centred. The default draw is centred on its own control traces, the hypopnea draw
    on the exam's control traces of both draws."""
    Z0, Z1 = load_traces(), load_traces("_hypopnea")
    A0, V0 = exam_centred(Z0)
    U = {k: np.concatenate([Z0[k], Z1[k]]) for k in KEYS}
    U["t_aud"], U["t_vid"] = Z0["t_aud"], Z0["t_vid"]
    AU, VU = exam_centred(U)
    n0 = len(Z0["event"])
    return U, np.vstack([A0, AU[n0:]]), np.vstack([V0, VU[n0:]])


def rolling_sd(X, win_s, hz):
    """Moving SD over win_s seconds (odd window, reflected edges)."""
    k = max(3, int(round(win_s * hz))); k += (k + 1) % 2
    P = np.pad(X.astype(np.float64), ((0, 0), (k // 2, k // 2)), mode="reflect")
    z = np.zeros((len(P), 1), P.dtype)
    c = np.concatenate([z, np.cumsum(P, axis=1)], axis=1)
    c2 = np.concatenate([z, np.cumsum(P ** 2, axis=1)], axis=1)
    m1, m2 = (c[:, k:] - c[:, :-k]) / k, (c2[:, k:] - c2[:, :-k]) / k
    return np.sqrt(np.maximum(m2 - m1 ** 2, 0))


def exam_centred(Z):
    """Log audio fluctuation (1-s moving SD of the sound level, dB) and log p95 flow magnitude, each minus the
    mean over the exam's no-event traces."""
    hz = len(Z["t_aud"]) / float(Z["t_aud"][-1] - Z["t_aud"][0])
    A = np.log(rolling_sd(Z["aud"], 1.0, hz) + 0.02)
    V = np.log(np.clip(Z["vid"].copy(), 1e-6, None))
    for u in np.unique(Z["uuid"]):
        m = Z["uuid"] == u
        mc = m & (Z["group"] == "control")
        ref = mc if mc.any() else m
        A[m] -= np.nanmean(A[ref])
        V[m] -= np.nanmean(V[ref])
    return A, V


def exam_mean(X, uu, B, seed=0):
    """Mean curve with each exam weighted equally, and its exam-bootstrap 95 % interval."""
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        E = np.vstack([np.nanmean(X[uu == u], axis=0) for u in np.unique(uu)])
        E = E[np.isfinite(E).any(axis=1)]
        mean = np.nanmean(E, axis=0)
        bs = np.empty((B, X.shape[1]), np.float32)
        for b in range(B):
            bs[b] = np.nanmean(E[rng.integers(0, len(E), len(E))], axis=0)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    return mean, lo, hi


def boot_diff(Xe, ue, Xc, uc, B, seed=1):
    """Event minus no-event mean curve with an exam-cluster bootstrap 95 % interval."""
    exams = np.unique(np.concatenate([ue, uc]))
    ie = {u: np.flatnonzero(ue == u) for u in exams}
    ic = {u: np.flatnonzero(uc == u) for u in exams}
    diff = np.nanmean(Xe, axis=0) - np.nanmean(Xc, axis=0)
    rng = np.random.default_rng(seed)
    bs = np.empty((B, Xe.shape[1]), np.float32)
    for b in range(B):
        samp = rng.choice(exams, size=len(exams), replace=True)
        bs[b] = (np.nanmean(Xe[np.concatenate([ie[u] for u in samp])], axis=0)
                 - np.nanmean(Xc[np.concatenate([ic[u] for u in samp])], axis=0))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    return diff, lo, hi


def smooth(y, k=SMOOTH):
    return np.convolve(np.nan_to_num(y, nan=np.nanmean(y)), np.ones(k) / k, mode="same")


def panel_traces(out, letter, events, Z, A, V):
    Wf, vid_ylim = TRACE_WIDTH[letter]
    H = 4.442
    mods = [("Audio", A, Z["t_aud"], "dB\nfluctuation"), ("Video", V, Z["t_vid"], "p95 flow\nmagnitude")]
    fig, axes = plt.subplots(2, len(events), figsize=(Wf, H), facecolor="#ffffff", squeeze=False)
    x0, x1 = float(Z["t_aud"][0]), float(Z["t_aud"][-1])
    stats = []
    for j, ev in enumerate(events):
        me = Z["event"] == ev
        dur_med = float(np.nanmedian(Z["dur"][me & (Z["group"] == "event")]))
        for i, (mod, X, t, ylab) in enumerate(mods):
            ax = axes[i, j]
            win = (t >= x0) & (t <= x1)
            for grp, color in (("control", NEUTRAL), ("event", MODALITY[mod])):
                m = me & (Z["group"] == grp)
                mean, lo, hi = (np.exp(smooth(v)) for v in exam_mean(X[m][:, win], Z["uuid"][m], N_BOOT))
                ax.fill_between(t[win], lo, hi, color=color, alpha=0.22, lw=0)
                ax.plot(t[win], mean, color=color, lw=1.6)
            ax.axvline(0, color=INK3, lw=0.8, ls=(0, (2, 2)))
            ax.axhline(1.0, color=INK3, lw=0.8, ls=(0, (2, 2)))

            mev, mct = me & (Z["group"] == "event"), me & (Z["group"] == "control")
            d, dlo, dhi = (smooth(v) for v in boot_diff(X[mev][:, win], Z["uuid"][mev], X[mct][:, win],
                                                       Z["uuid"][mct], N_BOOT))
            tw = t[win]
            w_ = min(max(dur_med, 2.0), x1)
            mm = (tw >= -w_ / 2) & (tw <= w_ / 2)
            kpk = np.nanargmax(np.abs(d[mm]))
            half = (dhi[mm][kpk] - dlo[mm][kpk]) / 2
            stats.append(dict(event=ev, modality=mod, peak_log_ratio=d[mm][kpk], ci_low=dlo[mm][kpk],
                              ci_high=dhi[mm][kpk], z=d[mm][kpk] / half if half > 0 else np.nan,
                              t_peak=tw[mm][kpk], frac_ci_excl0=float(np.mean((dlo[mm] > 0) | (dhi[mm] < 0))),
                              dur_median=dur_med, n_event=int(mev.sum()), n_control=int(mct.sum()),
                              n_exam=len(np.unique(Z["uuid"][me]))))

            ax.set_xlim(x0, x1)
            ax.set_xticks([v for v in (-10, 0, 10) if x0 <= v <= x1])
            ax.set_facecolor("#ffffff")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(INK3); ax.spines[s].set_linewidth(0.8)
            ax.tick_params(colors=INK2, labelsize=13, width=0.8, length=3)
            if i == 0:
                ax.set_title(TRACE_TITLE[ev], fontsize=14, color=INK, pad=12)
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=12.5, color=INK2, linespacing=1.15)
    # same y range within a row, and the same span (centred on each row's data) in both rows
    rl = []
    for i in range(2):
        lims = [axes[i, j].get_ylim() for j in range(len(events))]
        rl.append((min(l[0] for l in lims), max(l[1] for l in lims)))
    span = max(h - l for l, h in rl)
    for i, (l, h) in enumerate(rl):
        for j in range(len(events)):
            axes[i, j].set_ylim((l + h) / 2 - span / 2, (l + h) / 2 + span / 2)
    if vid_ylim:
        for j in range(len(events)):
            axes[1, j].set_ylim(*vid_ylim)
    for ax_ in axes.ravel():
        ax_.yaxis.set_major_locator(MultipleLocator(0.2))
        ax_.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax_.tick_params(axis="y", which="minor", length=2, width=0.6, colors=INK3)
    handles = [Line2D([], [], color=MODALITY["Audio"], lw=1.6, label="Event (audio)"),
               Line2D([], [], color=MODALITY["Video"], lw=1.6, label="Event (video)"),
               Line2D([], [], color=NEUTRAL, lw=1.6, label="No event")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=14,
               bbox_to_anchor=(0.5, 0.02 / H), handlelength=1.8, columnspacing=1.6)
    fig.text(0.5, 0.40 / H, "Time from event midpoint (s)", ha="center", va="bottom", fontsize=14, color=INK2)
    fig.subplots_adjust(left=0.90 / Wf, right=1 - 0.08 / Wf, top=1 - 0.42 / H, bottom=1.05 / H,
                        wspace=0.28, hspace=0.20)
    fig.savefig(f"{out}/panel_{letter}.png", dpi=300, facecolor="#ffffff")
    plt.close(fig)
    pd.DataFrame(stats).to_csv(f"{out}/panel_{letter}.csv", index=False, encoding="utf-8-sig")
    return f"{out}/panel_{letter}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panels", nargs="*", help="any of a b c d (default: all)")
    panels = ap.parse_args().panels or list("abcd")
    if set(panels) - set("abcd"):
        ap.error(f"unknown panel(s): {sorted(set(panels) - set('abcd'))}")
    os.makedirs(OUT, exist_ok=True)
    use_fonts()
    saved = [f(OUT) for p, f in (("a", panel_a), ("b", panel_b)) if p in panels]
    if {"c", "d"} & set(panels):
        w = weights_table().set_index("event").w_audio
        top = {"c": list(w.nlargest(3).index), "d": list(w.nsmallest(3).index)}     # audio- / video-weighted
        Z, A, V = traces()
        saved += [panel_traces(OUT, p, [raw(e) for e in top[p]], Z, A, V) for p in "cd" if p in panels]
    for p in saved:
        tag_dpi(p)
        print("saved", p)


if __name__ == "__main__":
    main()
