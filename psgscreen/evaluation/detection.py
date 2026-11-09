"""Window-level performance of the event detectors.

Cohorts: locked internal test exams (Hospitals A and B) and all external exams (Hospital C); windows of exams with
event annotations, in the order of the window table. Windows with an indeterminate label for an event are excluded
for that event. "Ensemble" scores are the mean over the 10 fold detectors.

  table    per event: fused AUC with exam-cluster bootstrap 95 % CI (B = 400), audio-only and video-only AUC,
           sensitivity / specificity at the mean fused threshold, AP, PPV; macro averages over all events and
           over events with >= 50 positive windows                        -> results/detection_table.csv
  perfold  internal test: AUC of each fold's audio, video and fused probabilities
                                                                          -> results/detection_perfold*.csv

    python -m psgscreen.evaluation.detection
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from psgscreen import paths
from psgscreen.dataset import vit_inputs as vi
from psgscreen.events import EXCLUDED as EXCLUDE, name as display
from psgscreen.model.exam_features import common_events
from psgscreen.model.fusion import window_probs

MIN_POS = 50

def auc_fast(y, s):
    """AUC from the ranks of the scores (no tie correction, as used for all detection tables)."""
    ys = y[np.argsort(s, kind="quicksort")]
    n1 = int(ys.sum()); n0 = len(ys) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = np.arange(1, len(ys) + 1, dtype=np.float64)
    return float((r[ys == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_cohort(centers, keep_ids):
    """Detector rows ({centre: indices}), targets, exam uuids and event names of the annotated windows."""
    rows, Ys, Us, events = {}, [], [], None
    for c in centers:
        r = vi.load_rows(c)
        vi_ = np.flatnonzero(np.isin(r["uuid"], list(keep_ids)) & (r["src"] == 0))
        vi_ = vi_[np.argsort(r["meta_row"][vi_])]
        if len(vi_):
            rows[c] = vi_
            Ys.append(r["y_multi"][vi_]); Us.append(r["uuid"][vi_])
        events = [str(e) for e in r["event_names"]]
    return rows, np.concatenate(Ys), np.concatenate(Us), events


def fold_predictions(rows, events):
    """Yield (k, audio probs, video probs, fusion weights, fused thresholds) for every fold."""
    ev = common_events()
    for k in range(paths.N_FOLDS):
        wts = json.load(open(f"{paths.FUSION}/f{k}/fusion_weights.json"))
        ths = json.load(open(f"{paths.FUSION}/f{k}/fused_thresholds.json"))
        pa = np.concatenate([window_probs("audio", k, c)[i] for c, i in rows.items()])
        pv = np.concatenate([window_probs("video", k, c)[i] for c, i in rows.items()])
        yield (k, {e: pa[:, events.index(e)] for e in ev}, {e: pv[:, events.index(e)] for e in ev},
               {e: float(wts[e][0]) for e in ev}, {e: float(ths[e]) for e in ev})


def table_rows(cname, Yc, Uc, events, ens, rng, B=400):
    """Per-event rows of the detection table from ensemble sums."""
    acc, accA, accV, thr, nfold = ens
    ex_ids, ex_inv = np.unique(Uc, return_inverse=True)
    groups = [np.flatnonzero(ex_inv == i) for i in range(len(ex_ids))]
    rows = []
    for e in events:
        if e == EXCLUDE or e not in acc or nfold[e] < paths.N_FOLDS:
            continue
        i = events.index(e)
        m = Yc[:, i] >= 0
        y = (Yc[m, i] == 1).astype(np.int8)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        s = acc[e][m] / nfold[e]; sa = accA[e][m] / nfold[e]; sv = accV[e][m] / nfold[e]
        t = thr[e] / nfold[e]
        tp = int(((s >= t) & (y == 1)).sum()); fn = int(((s < t) & (y == 1)).sum())
        tn = int(((s < t) & (y == 0)).sum()); fp = int(((s >= t) & (y == 0)).sum())
        inv_pos = -np.ones(len(m), np.int64); inv_pos[np.flatnonzero(m)] = np.arange(int(m.sum()))
        by_exam = [inv_pos[g[m[g]]] for g in groups if m[g].any()]
        bs = []
        for _ in range(B):
            pick = rng.randint(0, len(by_exam), len(by_exam))
            ii = np.concatenate([by_exam[j] for j in pick])
            v = auc_fast(y[ii], s[ii])
            if np.isfinite(v):
                bs.append(v)
        lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
        rows.append(dict(cohort=cname, event=display(e), n_win=int(m.sum()), n_pos=int(y.sum()),
                         prevalence=float(y.mean()), AUC=auc_fast(y, s), CI_low=lo, CI_high=hi,
                         AUC_audio=auc_fast(y, sa), AUC_video=auc_fast(y, sv), thr=t,
                         Sens=tp / max(tp + fn, 1), Spec=tn / max(tn + fp, 1),
                         AP=float(average_precision_score(y, s)),
                         PPV=(tp / (tp + fp) if tp + fp else np.nan), n_exam=len(by_exam)))
    return rows


def main():
    os.makedirs(paths.RESULTS, exist_ok=True)
    ros = pd.read_csv(paths.ROSTER, encoding="utf-8-sig")
    cohorts = [("Internal (locked test)", list(paths.INTERNAL),
                set(ros.loc[(ros.split == "test") & ros.center.isin(paths.INTERNAL), "uuid"].str.lower())),
               ("External", list(paths.EXTERNAL), set(ros.loc[ros.center.isin(paths.EXTERNAL), "uuid"].str.lower()))]
    rng = np.random.RandomState(42)
    table, internal = [], None
    for cname, centers, keep in cohorts:
        rows, Yc, Uc, events = load_cohort(centers, keep)
        acc, accA, accV, thr, nfold, per = {}, {}, {}, {}, {}, []
        is_internal = cname.startswith("Internal")
        if is_internal:
            ev_ok = []
            for e in events:
                if e == EXCLUDE:
                    continue
                m = Yc[:, events.index(e)] >= 0
                yy = (Yc[m, events.index(e)] == 1).astype(np.int8)
                if 0 < yy.sum() < len(yy):
                    ev_ok.append((e, m, yy))
        for k, pa, pv, w, th in fold_predictions(rows, events):
            for e in pa:
                pf = w[e] * pa[e] + (1 - w[e]) * pv[e]
                acc[e] = pf if e not in acc else acc[e] + pf
                accA[e] = pa[e] if e not in accA else accA[e] + pa[e]
                accV[e] = pv[e] if e not in accV else accV[e] + pv[e]
                thr[e] = thr.get(e, 0.0) + th[e]
                nfold[e] = nfold.get(e, 0) + 1
            if is_internal:
                for e, m, yy in ev_ok:
                    if e not in pa:
                        continue
                    sc = {"Audio only": pa[e], "Video only": pv[e], "Fused": w[e] * pa[e] + (1 - w[e]) * pv[e]}
                    for r, s in sc.items():
                        per.append(dict(row=r, event=display(e), fold=k, AUC=auc_fast(yy, s[m]),
                                        n_pos=int(yy.sum()), n_win=int(m.sum()), w_audio=w[e]))
            print(f"  {cname}: fold {k} scored", flush=True)
        table += table_rows(cname, Yc, Uc, events, (acc, accA, accV, thr, nfold), rng)
        if is_internal:
            internal = per

    T = pd.DataFrame(table)

    def macro(D, label):
        return (D.groupby("cohort", sort=False)
                 .agg(event=("event", lambda v: label(len(v))), n_win=("n_win", "sum"), n_pos=("n_pos", "sum"),
                      AUC=("AUC", "mean"), AUC_audio=("AUC_audio", "mean"), AUC_video=("AUC_video", "mean"),
                      Sens=("Sens", "mean"), Spec=("Spec", "mean"), AP=("AP", "mean"), PPV=("PPV", "mean"))
                 .reset_index())
    T = pd.concat([T, macro(T, lambda k: f"MACRO ({k} events)"),
                   macro(T[T.n_pos >= MIN_POS], lambda k: f"MACRO (n_pos>={MIN_POS}, {k} events)")], ignore_index=True)
    T.to_csv(f"{paths.RESULTS}/detection_table.csv", index=False, encoding="utf-8-sig")

    perfold(internal)
    print(T[["cohort", "event", "n_pos", "AUC", "CI_low", "CI_high", "AUC_audio", "AUC_video"]].round(4).to_string(index=False))


def perfold(per):
    R = pd.DataFrame(per)
    first = R.groupby("event").n_pos.first()
    R = R[R.event.isin(set(first[first >= MIN_POS].index))]
    S = R.groupby(["row", "event"], sort=False).AUC.agg(mean="mean", min="min", max="max", sd="std",
                                                        n="count").reset_index()
    M = R.groupby(["row", "fold"]).AUC.mean().rename("MACRO").reset_index()
    Mm = M.groupby("row").MACRO.agg(mean="mean", min="min", max="max", sd="std", n="count").reset_index()
    Mm["event"] = "Mean"
    S = pd.concat([S, Mm[S.columns]], ignore_index=True)
    R.to_csv(f"{paths.RESULTS}/detection_perfold.csv", index=False, encoding="utf-8-sig")
    S.to_csv(f"{paths.RESULTS}/detection_perfold_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
