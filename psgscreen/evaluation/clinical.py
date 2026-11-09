"""Clinical performance: operating points, PPV / NPV at a reference prevalence, calibration and decision curves.

Tasks: Abnormal, OSA, RBD (disorder classifier) and OSA severity, RBD severity (severity classifiers).
Cohorts: internal test and external.

Reference prevalence pi*:
  Abnormal       0.2916 = P(OSA or RBD) in the general population (OSA 0.284, Benjafield 2019, Korea;
                 RBD 0.0106, Haba-Rubio 2018; independence assumed)
  other tasks    their distribution in the internal development set (these classifiers act on abnormal exams)
Recalibration: the development out-of-fold calibration-in-the-large intercept c, then a prior shift from the
development prevalence to pi*:  p_ref = expit(logit(p) + c - logit(pi_dev) + logit(pi*)).
Operating points: 0.5 and the development out-of-fold Youden threshold (applied unchanged to every cohort).
PPV / NPV at pi* by Bayes' rule. Calibration (CITL, slope, Brier, ICI; LOWESS curves) and net benefit at pi*.
CIs: exam bootstrap, B = 4000 (curves: first 1000 resamples).

    python -m psgscreen.evaluation.clinical
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from sklearn.metrics import roc_auc_score

from psgscreen import paths

TASKS = ["Abnormal", "OSA", "RBD", "OSA severity", "RBD severity"]
FIG_TASKS = TASKS[:4]
COHORTS = ["Internal", "External"]
FIG_COHORTS = ["Internal", "External"]
SPW = {"Abnormal": "full", "OSA": "full", "RBD": "full", "OSA severity": "sqrt", "RBD severity": "sqrt"}
TITLE = {"Abnormal": "Abnormal classification", "OSA": "OSA classification", "RBD": "RBD classification",
         "OSA severity": "OSA severity classification", "RBD severity": "RBD severity classification"}
LABEL = {"Internal": "Internal test", "External": "External"}
SEVERITY = {"OSA severity": "osa_severity", "RBD severity": "rbd_severity"}
LIT_OSA, LIT_RBD = 0.284, 0.0106
PT = np.round(np.arange(0.01, 1.0, 0.01), 2)      # threshold probabilities of the decision curves
EPS = 1e-6
OUT = f"{paths.RESULTS}/clinical"
CURVES = f"{OUT}/curves.pkl"


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def youden(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    o = np.argsort(-s); ys = y[o]; ss = s[o]
    P = max(int(ys.sum()), 1); N = max(len(ys) - int(ys.sum()), 1)
    return float(ss[int(np.argmax(np.cumsum(ys) / P - np.cumsum(1 - ys) / N))])


def logistic_fit(x, y, offset=None, slope=True, iters=60, w=None):
    """Weighted logistic regression by Newton-Raphson; returns NaN when it does not converge."""
    n = len(y)
    X = np.column_stack([np.ones(n), x]) if slope else np.ones((n, 1))
    off = np.zeros(n) if offset is None else offset
    sw = np.ones(n) if w is None else np.asarray(w, float)
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = expit(X @ beta + off)
        v = np.clip(mu * (1 - mu), 1e-12, None) * sw
        try:
            step = np.linalg.solve(X.T @ (X * v[:, None]), X.T @ (sw * (y - mu)))
        except np.linalg.LinAlgError:
            return np.full(X.shape[1], np.nan)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    else:
        return np.full(X.shape[1], np.nan)
    return np.full(X.shape[1], np.nan) if np.any(np.abs(beta) > 50) else beta


def lowess_at(y, p, xv):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    return np.clip(lowess(y.astype(float), p, frac=2 / 3, it=0, delta=0.0, xvals=xv), 0, 1)


def wlowess_at(y, p, w, xv, frac=2 / 3):
    """Weighted LOWESS (tricube kernel, local linear), evaluated at xv."""
    y = np.asarray(y, float); p = np.asarray(p, float); w = np.asarray(w, float); xv = np.asarray(xv, float)
    k = max(int(np.ceil(frac * len(p))), 2)
    d = np.abs(p[None, :] - xv[:, None])
    h = np.maximum(np.partition(d, k - 1, axis=1)[:, k - 1:k], 1e-12)
    K = np.where(d < h, (1 - np.clip(d / h, 0, 1) ** 3) ** 3, 0.0) * w[None, :]
    S0 = K.sum(1); xm = (K * p).sum(1) / S0; ym = (K * y).sum(1) / S0
    var = (K * p * p).sum(1) / S0 - xm ** 2; cov = (K * p * y).sum(1) / S0 - xm * ym
    b = np.where(var > 1e-12, cov / np.where(var > 1e-12, var, 1.0), 0.0)
    return np.clip(ym + b * (xv - xm), 0, 1)


def ref_weights(y, pi_ref):
    """Case weights that move the sample prevalence to pi_ref."""
    y = np.asarray(y); ps = y.mean()
    return np.where(y == 1, pi_ref / ps, (1 - pi_ref) / (1 - ps)).astype(float)


def calib(y, p, w=None):
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    ww = np.ones(len(y)) if w is None else np.asarray(w, float)
    sw = ww.sum(); o = (ww * y).sum() / sw
    out = {"mean_pred": (ww * p).sum() / sw, "observed": o, "Brier": (ww * (p - y) ** 2).sum() / sw}
    out["scaled_Brier"] = 1 - out["Brier"] / (o * (1 - o)) if 0 < o < 1 else np.nan
    if 0 < o < 1:
        lp = logit(p)
        out["CITL"] = logistic_fit(None, y, offset=lp, slope=False, w=w)[0]
        out["slope"] = logistic_fit(lp, y, w=w)[1]
        g = np.linspace(p.min(), p.max(), 101)
        fit = lowess_at(y, p, g) if w is None else wlowess_at(y, p, ww, g)
        out["ICI"] = float((ww * np.abs(np.interp(p, g, fit) - p)).sum() / sw)
    else:
        out["CITL"] = out["slope"] = out["ICI"] = np.nan
    return out


def operating_point(y, s, thr, strict):
    pr = (s > thr) if strict else (s >= thr)
    pos = y == 1
    tp = int((pr & pos).sum()); fn = int((~pr & pos).sum()); fp = int((pr & ~pos).sum()); tn = int((~pr & ~pos).sum())
    se = tp / (tp + fn) if tp + fn else np.nan
    sp = tn / (tn + fp) if tn + fp else np.nan
    return dict(TP=tp, FN=fn, FP=fp, TN=tn, Sens=se, Spec=sp,
                PPV=tp / (tp + fp) if tp + fp else np.nan, NPV=tn / (tn + fn) if tn + fn else np.nan,
                LRp=se / (1 - sp) if sp < 1 else np.inf, LRn=(1 - se) / sp if sp > 0 else np.nan)


def exact_ci(x, n, a=0.05):
    if n == 0:
        return np.nan, np.nan
    return (0.0 if x == 0 else beta_dist.ppf(a / 2, x, n - x + 1),
            1.0 if x == n else beta_dist.ppf(1 - a / 2, x + 1, n - x))


def bayes(se, sp, pi):
    d1 = se * pi + (1 - sp) * (1 - pi); d2 = sp * (1 - pi) + (1 - se) * pi
    return (se * pi / d1 if d1 > 0 else np.nan), (sp * (1 - pi) / d2 if d2 > 0 else np.nan)


def net_benefit(y, p, pi):
    """Net benefit at prevalence pi: Se * pi - (1 - Sp) * (1 - pi) * pt / (1 - pt)."""
    P = p[None, :] >= PT[:, None]
    pos = (y == 1)[None, :]
    se = (P & pos).sum(1) / pos.sum(); fpr = (P & ~pos).sum(1) / (~pos).sum()
    return se * pi - fpr * (1 - pi) * (PT / (1 - PT))


def net_benefit_all(pi):
    return pi - (1 - pi) * PT / (1 - PT)


def ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)


def read(p):
    D = pd.read_csv(p, encoding="utf-8-sig"); D.columns = [c.lstrip("﻿") for c in D.columns]
    return D


def disorder_cell(D, task):
    if task == "Abnormal":
        d = D; y = d.label != "Normal"; s = d.p_abnormal
    else:
        d = D[D.label != "Normal"]
        y = d.label.isin(["OSA-only", "Both"] if task == "OSA" else ["RBD-only", "Both"]); s = d["q_" + task]
    return pd.DataFrame({"uuid": d.uuid.astype(str).str.lower().values, "label": d.label.values,
                         "y": y.astype(int).values, "s": s.values.astype(float)})


def severity_cell(S, bucket):
    d = S[S.bucket == bucket]
    return pd.DataFrame({"uuid": d.uuid.astype(str).str.lower().values, "label": d.label.values,
                         "y": d.y.astype(int).values, "s": d.p.values.astype(float)})


def load_cells():
    P = read(paths.disorder_predictions())
    cells = {}
    for t in TASKS[:3]:
        cells[(t, "OOF")] = disorder_cell(P[P.bucket == "OOF"], t)
        cells[(t, "Internal")] = disorder_cell(P[P.bucket == "TEST"], t)
        cells[(t, "External")] = disorder_cell(P[P.bucket == "EXT"], t)
    for t, target in SEVERITY.items():
        S0 = read(paths.severity_predictions(target))
        cells[(t, "OOF")] = severity_cell(S0, "OOF")
        cells[(t, "Internal")] = severity_cell(S0, "TEST")
        cells[(t, "External")] = severity_cell(S0, "EXT")
    return cells


def bootstrap_indices(cells, B):
    """Same resamples as the ROC figure: severity tasks share one generator for internal and external."""
    IDX = {}
    for t in TASKS:
        n = {c: len(cells[(t, c)]) for c in COHORTS}
        if t in SEVERITY:
            rs = np.random.RandomState(42)
            IDX[(t, "Internal")] = [rs.randint(0, n["Internal"], n["Internal"]) for _ in range(B)]
            IDX[(t, "External")] = [rs.randint(0, n["External"], n["External"]) for _ in range(B)]
        else:
            for c in COHORTS:
                r = np.random.RandomState(42)
                IDX[(t, c)] = [r.randint(0, n[c], n[c]) for _ in range(B)]
    return IDX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=4000)
    ap.add_argument("--B_curve", type=int, default=1000)
    ap.add_argument("--digits", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cells = load_cells()

    PI = {t: dict(dev=(int(cells[(t, "OOF")].y.sum()), len(cells[(t, "OOF")])),
                  test=(int(cells[(t, "Internal")].y.sum()), len(cells[(t, "Internal")])),
                  ext=(int(cells[(t, "External")].y.sum()), len(cells[(t, "External")]))) for t in TASKS}
    prev = lambda t, k: PI[t][k][0] / PI[t][k][1]
    union = 1 - (1 - LIT_OSA) * (1 - LIT_RBD)
    REF, REF_SRC = {}, {}
    for t in TASKS:
        if t == "Abnormal":
            REF[t] = union
            REF_SRC[t] = (f"Literature: OSA {LIT_OSA:.3f} (Benjafield 2019, Korea) ∪ RBD {LIT_RBD:.4f} "
                          f"(Haba-Rubio 2018), independence → {union:.4f}")
        else:
            k, N = PI[t]["dev"]
            REF[t], REF_SRC[t] = k / N, f"Development (training) set, {k}/{N}"
            PI[t]["ref"] = (k, N)

    REC, THR = {}, {}
    for t in TASKS:
        o = cells[(t, "OOF")]
        lp = logit(o.s.values); yo = o.y.values.astype(float)
        REC[t] = dict(c=logistic_fit(None, yo, offset=lp, slope=False)[0], oof_slope=logistic_fit(lp, yo)[1])
        THR[t] = {"p>0.5": (0.5, True), "Youden(dev OOF)": (youden(o.y.values, o.s.values), False)}

    def to_ref(t):
        cc = REC[t]["c"]; shift = -logit(prev(t, "dev")) + logit(REF[t])
        return lambda s: expit(logit(s) + cc + shift)

    def versions(t, c):
        cc = REC[t]["c"]
        v = {"raw": lambda s: s}
        if c in FIG_COHORTS:
            v["recal"] = lambda s: expit(logit(s) + cc)
            v["ref"] = to_ref(t)
        return v

    IDX = bootstrap_indices(cells, a.B)
    OPS, CAL, DCA, CUR = [], [], [], {}
    for t in TASKS:
        for c in COHORTS:
            d = cells[(t, c)]
            y, s = d.y.values, d.s.values
            n, npos = len(y), int(y.sum())
            vers = versions(t, c)
            fig_cell = (t in FIG_TASKS) and (c in FIG_COHORTS)
            pi = REF[t]
            f_ref = to_ref(t) if c in FIG_COHORTS else None
            auc = roc_auc_score(y, s)
            pt_ops = {}
            for rule, (thr, strict) in THR[t].items():
                m = operating_point(y, s, thr, strict)
                m["PPV_ref"], m["NPV_ref"] = bayes(m["Sens"], m["Spec"], pi)
                pt_ops[rule] = m
            wref = ref_weights(y, pi)
            pt_cal = {v: calib(y, f(s), w=(wref if v == "ref" else None)) for v, f in vers.items()}
            bs_auc, bs_ops, bs_cal = [], {r: [] for r in pt_ops}, {v: [] for v in vers}
            curv = {v: [] for v in vers}
            nbB, dB = [], []
            gx = {v: np.linspace(*np.percentile(f(s), [1, 99]), 101) for v, f in vers.items()}
            for b, ix in enumerate(IDX[(t, c)]):
                yb, sb = y[ix], s[ix]
                two = 0 < yb.sum() < len(yb)
                wb_ref = ref_weights(yb, pi) if two else None
                bs_auc.append(roc_auc_score(yb, sb) if two else np.nan)
                for r in pt_ops:
                    thr, strict = THR[t][r]
                    m = operating_point(yb, sb, thr, strict)
                    m["PPV_ref"], m["NPV_ref"] = bayes(m["Sens"], m["Spec"], pi)
                    bs_ops[r].append(m)
                if not two:
                    continue
                for v, f in vers.items():
                    pb = f(sb)
                    bs_cal[v].append(calib(yb, pb, w=(wb_ref if v == "ref" else None)))
                    if fig_cell and b < a.B_curve:
                        curv[v].append(wlowess_at(yb, pb, wb_ref, gx[v]) if v == "ref" else lowess_at(yb, pb, gx[v]))
                if fig_cell:
                    nbm = net_benefit(yb, f_ref(sb), pi)
                    dB.append(nbm - np.maximum(net_benefit_all(pi), 0.0))
                    if b < a.B_curve:
                        nbB.append(nbm)
            lo, hi = ci(bs_auc)
            base = dict(task=t, cohort=c, n=n, n_pos=npos, prev_sample=y.mean(), prev_ref=pi)
            OPS.append({**base, "rule": "-", "threshold": np.nan, "metric": "AUC", "value": auc, "lo": lo, "hi": hi})
            for r, m in pt_ops.items():
                for k, v in m.items():
                    l2, h2 = ci([bb.get(k, np.nan) for bb in bs_ops[r]]) if k not in ("TP", "FN", "FP", "TN") \
                        else (np.nan, np.nan)
                    ex = {"Sens": ("TP", "FN"), "Spec": ("TN", "FP"), "PPV": ("TP", "FP"), "NPV": ("TN", "FN")}.get(k)
                    el, eh = exact_ci(m[ex[0]], m[ex[0]] + m[ex[1]]) if ex else (np.nan, np.nan)
                    OPS.append({**base, "rule": r, "threshold": THR[t][r][0], "metric": k, "value": v, "lo": l2,
                                "hi": h2, "lo_exact": el, "hi_exact": eh})
            for v, mm in pt_cal.items():
                for k, val in mm.items():
                    l2, h2 = ci([bb[k] for bb in bs_cal[v]])
                    CAL.append({**base, "version": v, "metric": k, "value": val, "lo": l2, "hi": h2})
            if fig_cell:
                cur = {}
                for v, f in vers.items():
                    p = f(s)
                    ww = wref if v == "ref" else np.ones(n)
                    D_ = pd.DataFrame({"p": p * ww, "y": y * ww, "w": ww}).groupby(
                        pd.qcut(p, 10 if n >= 200 else 5, duplicates="drop"), observed=True).sum()
                    B_ = np.array(curv[v])
                    cur[v] = dict(g=gx[v], f=wlowess_at(y, p, wref, gx[v]) if v == "ref" else lowess_at(y, p, gx[v]),
                                  lo=np.nanpercentile(B_, 2.5, 0), hi=np.nanpercentile(B_, 97.5, 0),
                                  bins=pd.DataFrame({"p": D_.p / D_.w, "y": D_.y / D_.w}), cal=pt_cal[v])
                nb_m, nbA = net_benefit(y, f_ref(s), pi), net_benefit_all(pi)
                Bm, Bd = np.array(nbB), np.array(dB)
                cur["dca"] = dict(model=nb_m, all=nbA, lo=np.percentile(Bm, 2.5, 0), hi=np.percentile(Bm, 97.5, 0),
                                  pi=pi, d=nb_m - np.maximum(nbA, 0.0), d_lo=np.percentile(Bd, 2.5, 0),
                                  d_hi=np.percentile(Bd, 97.5, 0))
                CUR[(t, c)] = cur
                for i, ptv in enumerate(PT):
                    DCA.append({**base, "pt": ptv, "NB_model": nb_m[i], "NB_model_lo": cur["dca"]["lo"][i],
                                "NB_model_hi": cur["dca"]["hi"][i], "NB_treat_all": nbA[i], "NB_treat_none": 0.0,
                                "dNB_vs_best_default": cur["dca"]["d"][i], "dNB_lo": cur["dca"]["d_lo"][i],
                                "dNB_hi": cur["dca"]["d_hi"][i]})
            print(f"  {t:13s} {c:9s} n {n} ({npos} positive)  AUC {auc:.4f} [{lo:.4f}, {hi:.4f}]", flush=True)

    pickle.dump({"CUR": CUR}, open(CURVES, "wb"))
    O, C, Dd = pd.DataFrame(OPS), pd.DataFrame(CAL), pd.DataFrame(DCA)
    O.to_csv(f"{OUT}/operating_points.csv", index=False, encoding="utf-8-sig")
    C.to_csv(f"{OUT}/calibration.csv", index=False, encoding="utf-8-sig")
    Dd.to_csv(f"{OUT}/decision_curves.csv", index=False, encoding="utf-8-sig")
    prow = []
    for t in TASKS:
        for k, nm in (("dev", "Internal development (OOF)"), ("test", "Internal test sample"), ("ext", "External")):
            prow.append(dict(task=t, source=nm, k=PI[t][k][0], N=PI[t][k][1], prevalence=prev(t, k)))
        kN = PI[t].get("ref", (np.nan, np.nan))
        prow.append(dict(task=t, source="Reference prevalence: " + REF_SRC[t], k=kN[0], N=kN[1], prevalence=REF[t]))
    PR = pd.DataFrame(prow)
    RC = pd.DataFrame([dict(task=t, spw=SPW[t], prev_dev=prev(t, "dev"), c_dev_OOF_CITL=REC[t]["c"],
                            OOF_raw_slope=REC[t]["oof_slope"], ref_prevalence=REF[t], ref_source=REF_SRC[t],
                            prior_shift_dev_to_ref=-logit(prev(t, "dev")) + logit(REF[t]),
                            youden_dev_OOF=THR[t]["Youden(dev OOF)"][0]) for t in TASKS])
    PR.to_csv(f"{OUT}/prevalence.csv", index=False, encoding="utf-8-sig")
    RC.to_csv(f"{OUT}/recalibration.csv", index=False, encoding="utf-8-sig")
    DS = dca_summary(CUR)
    DS.to_csv(f"{OUT}/dca_summary.csv", index=False, encoding="utf-8-sig")
    write_xlsx(O, C, PR, RC, DS, a.digits)


def threshold_ranges(mask):
    out, i = [], 0
    while i < len(PT):
        if mask[i]:
            j = i
            while j + 1 < len(PT) and mask[j + 1]:
                j += 1
            out.append(f"{PT[i]:.2f}–{PT[j]:.2f}" if j > i else f"{PT[i]:.2f}")
            i = j + 1
        else:
            i += 1
    return ", ".join(out) if out else "none"


def dca_summary(CUR):
    """Threshold ranges where the model's net benefit exceeds both treat-all and treat-none."""
    rows = []
    for (t, c), cur in CUR.items():
        q = cur["dca"]
        rows.append(dict(task=t, cohort=c, prevalence=float(q["pi"]), model_beats_defaults_point=threshold_ranges(q["d"] > 0),
                         model_beats_defaults_CI=threshold_ranges(q["d_lo"] > 0),
                         **{f"dNB@{p:.1f}": float(q["d"][np.argmin(np.abs(PT - p))]) for p in (.1, .3, .5, .7, .9)}))
    return pd.DataFrame(rows)


def fmt(v, lo, hi, d):
    if not np.isfinite(v):
        return "–"
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]" if np.isfinite(lo) else f"{v:.{d}f}"


def write_xlsx(O, C, PR, RC, DS, D):
    src = {"p>0.5": lambda c: "Fixed 0.5", "Youden(dev OOF)": lambda c: "Development OOF Youden"}
    rows = {r: [] for r in src}
    for (t, c), g in O.groupby(["task", "cohort"], sort=False):
        a = g[g.metric == "AUC"].iloc[0]
        for r in [x for x in g.rule.unique() if x != "-"]:
            q = g[g.rule == r].set_index("metric")
            f = lambda k: fmt(q.loc[k, "value"], q.loc[k, "lo"], q.loc[k, "hi"], D) if k in q.index else "–"
            fe = lambda k: f"[{q.loc[k, 'lo_exact']:.{D}f}, {q.loc[k, 'hi_exact']:.{D}f}]"
            rows[r].append({"Task": TITLE[t], "Cohort": LABEL[c], "n (pos)": f"{int(a.n)} ({int(a.n_pos)})",
                            "Prev. sample": round(a.prev_sample, D), "Reference prev.": round(a.prev_ref, D),
                            "AUC": fmt(a.value, a.lo, a.hi, D), "Threshold": round(q.threshold.iloc[0], D),
                            "Threshold source": src[r](c),
                            "TP/FN/FP/TN": "/".join(str(int(q.loc[k, "value"])) for k in ("TP", "FN", "FP", "TN")),
                            "Sensitivity": f("Sens"), "Specificity": f("Spec"),
                            "Sens exact CI": fe("Sens"), "Spec exact CI": fe("Spec"),
                            "PPV (sample)": f("PPV"), "NPV (sample)": f("NPV"),
                            "PPV (reference prev.)": f("PPV_ref"), "NPV (reference prev.)": f("NPV_ref"),
                            "LR+": f("LRp"), "LR−": f("LRn")})
    crow = []
    for (t, c, v), g in C.groupby(["task", "cohort", "version"], sort=False):
        q = g.set_index("metric")
        f = lambda k: fmt(q.loc[k, "value"], q.loc[k, "lo"], q.loc[k, "hi"], D)
        crow.append({"Task": TITLE[t], "Cohort": LABEL[c], "Version": v, "Mean predicted": f("mean_pred"),
                     "Observed": round(q.loc["observed", "value"], D), "CITL": f("CITL"), "Slope": f("slope"),
                     "Brier": f("Brier"), "Scaled Brier": f("scaled_Brier"), "ICI": round(q.loc["ICI", "value"], D)})
    sheets = {"Threshold 0.5": pd.DataFrame(rows["p>0.5"]), "Optimal threshold": pd.DataFrame(rows["Youden(dev OOF)"]),
              "calibration": pd.DataFrame(crow), "prevalence": PR.assign(prevalence=PR.prevalence.round(D)),
              "recalibration": RC.round({c: D for c in RC.columns if RC[c].dtype.kind == "f"}),
              "dca_summary": DS.round({c: D for c in DS.columns if DS[c].dtype.kind == "f"})}
    nf = "0." + "0" * D
    with pd.ExcelWriter(f"{OUT}/clinical_metrics.xlsx", engine="openpyxl") as xw:
        for nm, df in sheets.items():
            df.to_excel(xw, sheet_name=nm, index=False)
            ws = xw.sheets[nm]
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, float):
                        cell.number_format = nf
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(
                    max(len(str(c.value)) if c.value is not None else 0 for c in col) + 2, 60)
    for nm, fn in (("Threshold 0.5", "table_threshold_0.5"), ("Optimal threshold", "table_optimal_threshold"),
                   ("calibration", "table_calibration")):
        sheets[nm].to_csv(f"{OUT}/{fn}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
