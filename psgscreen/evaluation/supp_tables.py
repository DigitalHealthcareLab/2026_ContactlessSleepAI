"""Supplementary tables (xlsx, values with 4 decimals).

  operating  Table S3: development OOF Youden threshold, sensitivity, specificity, PPV* and NPV* (reference
             prevalence) per task and cohort, from results/clinical/table_optimal_threshold.csv; sensitivities and
             specificities whose bootstrap interval is a single point get the exact Clopper-Pearson interval
                                                                      -> results/supp_table_operating.xlsx
  fusion     Table S4: disease classification by input modality and fusion strategy  -> results/supp_table_fusion.xlsx
  alt        Tables S5, S6: alternative disorder classifiers (internal test, external) -> results/supp_table_alt.xlsx

fusion / alt sheets: Table (AUC, 95 % CI), ΔAUC (paired bootstrap, Holm-adjusted P), Operating point (threshold 0.5),
Dev OOF (reference only), Hyperparameters (alt), Data (all values).

    python -m psgscreen.evaluation.supp_tables operating|fusion|alt
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, Side
from openpyxl.utils import get_column_letter

from psgscreen import paths
from psgscreen.evaluation.model_comparison import B, MAIN
from psgscreen.model import disorder as dm

FONT = "Arial"
THIN = Side(style="thin", color="000000")
HEADS = ["Abnormal", "OSA", "RBD"]
COH = [("TEST", "Internal test"), ("EXT", "External")]
MINUS = "−"
RES = 2 / B
DIG = 4

NOTES = {
    "common": [
        "AUC, area under the receiver operating characteristic curve; CI, confidence interval; OSA, obstructive sleep apnea; "
        "RBD, rapid eye movement sleep behavior disorder.",
        "Abnormal: all examinations (normal vs. any sleep disorder). OSA and RBD: examinations with a sleep disorder, "
        "evaluated label-wise. Average: mean of the abnormal, OSA and RBD AUCs.",
        "95% CIs from an examination-level bootstrap (4,000 resamples). Internal-test and external predictions are the "
        "average of the 10 cross-validation fold models.",
    ],
    "alt": [
        "SVM, support vector machine; RBF, radial basis function; MLP, multilayer perceptron.",
        "All classifiers used the same 39 examination-level features (mean, maximum and standard deviation of the nightly "
        "fused event-detector probability for 13 events), the same event detectors and the same cross-validation splits; "
        "only the disorder classifier (abnormal stage and the OSA/RBD heads) was replaced.",
        "Class weighting was matched across classifiers (positive:negative weight ratio = n_negative/n_positive, equivalent "
        "to XGBoost scale_pos_weight). Hyperparameters of the alternative classifiers were selected per task by "
        "development out-of-fold AUC from pre-specified grids (Hyperparameters sheet).",
    ],
    "fusion": [
        "Audio only / Video only: the examination-level features were computed from the audio (or video) event-detector "
        "probabilities alone, i.e. the per-event late-fusion weight was fixed to 1 (audio) or 0 (video). Event detectors, "
        "the 39-feature definition and cross-validation splits were identical to the main model; the disorder-classifier "
        "hyperparameters were selected for each input from the same grid.",
        "Classifier output averaging (decision-level fusion): the predicted probabilities of the audio-only and "
        "video-only disorder classifiers were averaged 1:1 (no additional training).",
        "ΔAUC sheets: audio only and video only versus the main model (Holm across 12 comparisons); classifier output "
        "averaging versus the main model (Holm across 6); video only versus audio only (Holm across 6).",
    ],
}
OOF_NOTE = {
    "alt": "Not a fair comparison: hyperparameters were selected on these out-of-fold predictions (XGBoost disorder heads "
           "from 6,912 configurations; alternatives from 5–12; the XGBoost abnormal stage was not tuned). Use the "
           "internal-test and external results for comparison.",
    "fusion": "Not a fair comparison: the OSA and RBD heads were tuned on these out-of-fold predictions, separately for "
              "the audio + video, audio-only and video-only features (classifier output averaging combines the latter "
              "two). The abnormal stage was not tuned. Use the internal-test and external results for comparison.",
}
FUSION_NAMES = [(MAIN, "Audio + video, weighted fusion (main model)"), ("Audio only", "Audio only"),
                ("Video only", "Video only"), ("Output averaging", "Audio + video, classifier output averaging")]
TABLES = {
    "alt": dict(
        csv="comparison_alt.csv",
        title="Discrimination of alternative disorder classifiers, AUC (95% CI)",
        models=[("XGBoost", "XGBoost (main model)"), ("Logistic regression", "Logistic regression (L2)"),
                ("Random forest", "Random forest"), ("SVM (RBF)", "SVM (RBF kernel)"), ("MLP", "Multilayer perceptron")],
        deltas=[("comparison_alt.csv", None, "ΔAUC", None)]),
    "fusion": dict(
        csv="comparison_fusion.csv",
        title="Disease classification by input modality and fusion strategy, AUC (95% CI)",
        models=FUSION_NAMES,
        deltas=[("comparison_fusion_modality.csv", None, "ΔAUC", None),
                ("comparison_fusion_output_averaging.csv", [FUSION_NAMES[0], FUSION_NAMES[3]],
                 "ΔAUC output averaging", "ΔAUC, classifier output averaging versus weighted fusion (95% CI)"),
                ("comparison_fusion_video_vs_audio.csv", [FUSION_NAMES[1], FUSION_NAMES[2]], "ΔAUC video vs audio",
                 "ΔAUC, video only versus audio only (95% CI)")]),
}
ALT_GRID = {"lr": "C ∈ {0.001, 0.01, 0.1, 1, 10}",
            "rf": "min_samples_leaf ∈ {20, 5, 1} × max_features ∈ {sqrt, 0.5}",
            "svm": "C ∈ {0.1, 1, 10, 100} × γ ∈ {0.1, 1, 10}/n_features",
            "mlp": "hidden layers ∈ {(32), (64, 32)} × L2 α ∈ {0.1, 0.01, 0.001, 0.0001}"}
ALT_IMPL = {"lr": "scikit-learn 1.7.2 LogisticRegression (L2, lbfgs, max_iter 5000, class_weight balanced); standardized",
            "rf": "scikit-learn 1.7.2 RandomForestClassifier (500 trees, class_weight balanced)",
            "svm": "scikit-learn 1.7.2 SVC (RBF, class_weight balanced; score = logistic of decision function); standardized",
            "mlp": "scikit-learn 1.7.2 MLPClassifier (ReLU, Adam lr 1e-3, batch 64, ≤500 epochs, early stopping on 10% "
                   "internal split, patience 20; balanced sample weights); standardized"}
ALT_KEYS = [("lr", "Logistic regression"), ("rf", "Random forest"), ("svm", "SVM (RBF)"), ("mlp", "MLP")]
TASK_TITLE = ["Abnormal classification", "OSA classification", "RBD classification", "OSA severity classification",
              "RBD severity classification"]


def f3(v):
    v = 0.0 if round(v, DIG) == 0 else v
    return f"{v:.{DIG}f}".replace("-", MINUS)


def fsign(v):
    return f"{0:.{DIG}f}" if round(v, DIG) == 0 else f"{v:+.{DIG}f}".replace("-", MINUS)


def fp(v):
    floor = 10 ** -DIG
    if v <= 0 or v < RES:
        return f"<{RES:.{DIG}f}" if RES >= floor else f"<{floor:.{DIG}f}"
    return f"<{floor:.{DIG}f}" if v < floor else f"{v:.{DIG}f}"


def style_header(c, bold=True):
    c.font = Font(name=FONT, bold=bold, size=10)
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def body(c, align="center"):
    c.font = Font(name=FONT, size=10)
    c.alignment = Alignment(horizontal=align, vertical="center")


def write_grid(ws, title, row_names, col_groups, cell_fn, notes, subhead=None):
    """Title, two header rows (group / column), optional italic sub-header, body rows, rules and notes."""
    ws["A1"] = title
    ws["A1"].font = Font(name=FONT, size=12)
    r0 = 3
    ws.cell(r0, 1, "Model"); style_header(ws.cell(r0, 1))
    ws.merge_cells(start_row=r0, start_column=1, end_row=r0 + 1, end_column=1)
    col = 2
    for glab, subs, _ in col_groups:
        ws.cell(r0, col, glab); style_header(ws.cell(r0, col))
        if len(subs) > 1:
            ws.merge_cells(start_row=r0, start_column=col, end_row=r0, end_column=col + len(subs) - 1)
        for j, s in enumerate(subs):
            style_header(ws.cell(r0 + 1, col + j, s))
        col += len(subs)
    ncol = col - 1
    r = r0 + 2
    if subhead:
        ws.cell(r, 1, subhead[0]); body(ws.cell(r, 1), "left")
        ws.cell(r, 1).font = Font(name=FONT, size=9, italic=True)
        col = 2
        for _, subs, gkey in col_groups:
            for s in subs:
                c = ws.cell(r, col, subhead[1](gkey, s)); body(c)
                c.font = Font(name=FONT, size=9, italic=True)
                col += 1
        r += 1
    first_body = r
    for rk, rlab in row_names:
        ws.cell(r, 1, rlab); body(ws.cell(r, 1), "left")
        col = 2
        for _, subs, gkey in col_groups:
            for s in subs:
                body(ws.cell(r, col, cell_fn(rk, gkey, s)))
                col += 1
        r += 1
    last = r - 1
    for cc in range(1, ncol + 1):
        ws.cell(r0, cc).border = Border(top=THIN)
        ws.cell(r0 + 1, cc).border = Border(bottom=THIN)
        ws.cell(last, cc).border = Border(bottom=THIN)
    rn = last + 2
    for n in notes:
        ws.cell(rn, 1, n).font = Font(name=FONT, size=9)
        ws.cell(rn, 1).alignment = Alignment(wrap_text=False, vertical="top")
        rn += 1
    ws.column_dimensions["A"].width = 30
    for cc in range(2, ncol + 1):
        ws.column_dimensions[get_column_letter(cc)].width = 21
    ws.row_dimensions[r0].height = 18
    ws.freeze_panes = ws.cell(first_body, 2)


def build(kind):
    spec = TABLES[kind]
    T = pd.read_csv(f"{paths.RESULTS}/{spec['csv']}", encoding="utf-8-sig")
    order = spec["models"]
    ref, ref_disp = order[0]
    get = lambda m, b, h: T[(T.model == m) & (T.bucket == b) & (T["head"] == h)].iloc[0]
    nn = {(b, h): (int(get(ref, b, h).n_pos), int(get(ref, b, h).n_neg)) for b in ("TEST", "EXT", "OOF") for h in HEADS}
    ntot = {b: int(get(ref, b, "Abnormal").n) for b in ("TEST", "EXT", "OOF")}
    notes = NOTES["common"] + NOTES[kind]
    src = [f"Source: results/{spec['csv']} (values, not formulas)."]
    pn = lambda b, h: f"{nn[(b, h)][0]}/{nn[(b, h)][1]}" if h in HEADS else "—"
    subs = HEADS + ["Average"]

    def auc_cell(m, b, s):
        if s == "Average":
            return f3(np.mean([get(m, b, h).AUC for h in HEADS]))
        r = get(m, b, s)
        return f"{f3(r.AUC)} ({f3(r.CI_lo)}–{f3(r.CI_hi)})"

    wb = Workbook()
    ws = wb.active; ws.title = "Table"
    write_grid(ws, spec["title"], order, [(f"{lab} (n = {ntot[b]})", subs, b) for b, lab in COH], auc_cell,
               notes + ["Positive/negative examinations per task are given under each column header."] + src,
               subhead=("Positive/negative, n", pn))

    for csv, models, sheet, title in spec["deltas"]:
        TT = pd.read_csv(f"{paths.RESULTS}/{csv}", encoding="utf-8-sig")
        ordr = models or [(m, d) for m, d in order if m in set(TT.model)]
        if title is None:
            title = f"ΔAUC versus {ref_disp} (95% CI)"
            diff = f"{'classifier' if kind == 'alt' else 'model'} − {ref_disp}"
        else:
            diff = f"{ordr[1][1]} − {ordr[0][1]}"
        g = lambda m, b, h: TT[(TT.model == m) & (TT.bucket == b) & (TT["head"] == h)].iloc[0]
        alts = [(m, d) for m, d in ordr if m != ordr[0][0]]
        F_ = TT[(TT.model != ordr[0][0]) & TT.bucket.isin(["TEST", "EXT"])]
        HB = {(m, b, h): (a_, p < RES) for m, b, h, p, a_ in zip(F_.model, F_.bucket, F_["head"], F_.p, F_.p_holm)}
        lab_h = lambda b, h: h + ("†" if nn[(b, h)][1] < 20 else "")
        grp = [(f"{lab} (n = {ntot[b]})", [lab_h(b, h) for h in HEADS], b) for b, lab in COH]
        ws = wb.create_sheet(sheet)

        def d_cell(m, b, x):
            r, hb = g(m, b, x.rstrip("†")), HB[(m, b, x.rstrip("†"))]
            return f"{fsign(r.dAUC)}{'*' if hb[0] < 0.05 else ''} ({fsign(r.d_lo)}, {fsign(r.d_hi)})"
        write_grid(ws, title, alts, grp, d_cell,
                   [f"ΔAUC = {diff}; paired bootstrap on the same resampled examinations (4,000 resamples). "
                    "* Holm-adjusted P < 0.05 (P values below). † fewer than 20 negative examinations.",
                    f"Bootstrap P values have a resolution of 2/4,000 = {RES:.4f}; P = 0 is shown as <{RES:.4f} "
                    "and enters the Holm adjustment as that bound (the adjusted value is then an upper bound, shown with <)."]
                   + (src if csv == spec["csv"] else
                      [f"Source: results/{csv} (values, not formulas); Holm adjustment within this sheet only."]))
        r = ws.max_row + 2
        ws.cell(r, 1, f"P value, two-sided bootstrap / Holm-adjusted across {len(F_)} comparisons").font = Font(name=FONT, size=10)
        r += 1
        col = 2
        for glab, hs, _ in grp:
            ws.cell(r, col, glab); style_header(ws.cell(r, col))
            ws.merge_cells(start_row=r, start_column=col, end_row=r, end_column=col + len(hs) - 1)
            for j, h in enumerate(hs):
                style_header(ws.cell(r + 1, col + j, h))
            col += len(hs)
        ws.cell(r, 1, "Model"); style_header(ws.cell(r, 1))
        ws.merge_cells(start_row=r, start_column=1, end_row=r + 1, end_column=1)
        rr = r + 2
        for m, d in alts:
            ws.cell(rr, 1, d); body(ws.cell(rr, 1), "left")
            col = 2
            for _, hs, b in grp:
                for h in hs:
                    pv, hb = g(m, b, h.rstrip("†")).p, HB[(m, b, h.rstrip("†"))]
                    body(ws.cell(rr, col, f"{fp(pv)} / {'<' if hb[1] else ''}{hb[0]:.{DIG}f}")); col += 1
            rr += 1
        for cc in range(1, col):
            ws.cell(r, cc).border = Border(top=THIN); ws.cell(r + 1, cc).border = Border(bottom=THIN)
            ws.cell(rr - 1, cc).border = Border(bottom=THIN)

    ws = wb.create_sheet("Operating point")
    write_grid(ws, "Sensitivity / specificity / macro-F1 at a probability threshold of 0.5", order,
               [(f"{lab} (n = {ntot[b]})", HEADS, b) for b, lab in COH],
               lambda m, b, s: (lambda r: f"{f3(r.Sens)} / {f3(r.Spec)} / {f3(r.macroF1)}")(get(m, b, s)),
               ["macro-F1 = mean of the F1 scores of the positive and negative classes."] + src)

    ws = wb.create_sheet("Dev OOF (reference)")
    write_grid(ws, "Development out-of-fold AUC (95% CI) — reference only, not a fair comparison", order,
               [(f"Internal development, 10-fold OOF (n = {ntot['OOF']})", subs, "OOF")], auc_cell,
               [OOF_NOTE[kind]] + src, subhead=("Positive/negative, n", pn))

    if kind == "alt":
        ws = wb.create_sheet("Hyperparameters")
        xh = json.load(open(f"{paths.CONFIGS}/stage2.json"))
        base = {k: v for k, v in dm.BASE.items() if k not in ("tree_method", "seed")}
        fx = lambda d: ", ".join(f"{k}={v}" for k, v in d.items() if k != "spw") + ", scale_pos_weight=n_neg/n_pos"
        rows = [("Classifier", "Implementation", "Search grid", "Abnormal", "OSA", "RBD"),
                ("XGBoost (main model)",
                 f"xgboost 3.2.0 (hist), ≤{dm.ROUNDS} rounds, early stopping {dm.EARLY_STOP} on the validation fold",
                 "abnormal stage: default (untuned); OSA/RBD heads: 6,912-configuration grid",
                 fx(base), fx(xh["OSA"]), fx(xh["RBD"]))]
        disp = dict(order)
        fmt = lambda d: ", ".join((f"gamma={v:g}/n_features" if k == "gamma_mult" else f"{k}={v:g}"
                                   if isinstance(v, float) else f"{k}={v}") for k, v in d.items() if k != "model") or "—"
        for key, name in ALT_KEYS:
            c = json.load(open(f"{paths.CONFIGS}/alt/{key}.json"))
            rows.append((disp.get(name, name), ALT_IMPL[key], ALT_GRID[key], fmt(c["Abnormal"]), fmt(c["OSA"]),
                         fmt(c["RBD"])))
        for i, row in enumerate(rows, start=1):
            for j, v in enumerate(row, start=1):
                c = ws.cell(i, j, v)
                if i == 1:
                    style_header(c)
                else:
                    c.font = Font(name=FONT, size=10)
                    c.alignment = Alignment(wrap_text=True, vertical="top")
        for j, w in enumerate([24, 60, 42, 30, 30, 30], start=1):
            ws.column_dimensions[get_column_letter(j)].width = w
        n = len(rows) + 2
        for t in ["Selection rule: per task, the configuration with the highest development out-of-fold AUC; "
                  "ties within 0.0001 resolved toward the simpler configuration (fixed before running).",
                  "The XGBoost abnormal stage uses scale_pos_weight = n_negative/n_positive with default parameters; "
                  "the heads use scale_pos_weight = n_negative/n_positive with the listed values."] + src:
            ws.cell(n, 1, t).font = Font(name=FONT, size=9); n += 1

    ws = wb.create_sheet("Data")
    for j, c in enumerate(T.columns, start=1):
        style_header(ws.cell(1, j, c))
    for i, row in enumerate(T.itertuples(index=False), start=2):
        for j, v in enumerate(row, start=1):
            if isinstance(v, (float, np.floating)) and not np.isfinite(v):
                v = None
            ws.cell(i, j, v).font = Font(name=FONT, size=10)
    ws.freeze_panes = "A2"
    out = f"{paths.RESULTS}/supp_table_{kind}.xlsx"
    wb.save(out)
    print("saved", out)


def operating():
    T = pd.read_csv(f"{paths.RESULTS}/clinical/table_optimal_threshold.csv", encoding="utf-8-sig").set_index(["Task", "Cohort"])
    num = lambda s: re.findall(r"[\d.]+", str(s))

    def cell(r, k, exact=None):
        v, lo, hi = num(r[k])
        if exact and lo == hi:                     # degenerate bootstrap interval -> Clopper-Pearson
            lo, hi = num(r[exact])
        return f"{v} ({lo}–{hi})"

    rows, ref = [], {}
    for t in TASK_TITLE:
        for c in ("Internal test", "External"):
            r = T.loc[(t, c)]
            ref[t] = r["Reference prev."]
            rows.append({"Task": t, "Cohort": c, "Threshold": f"{r['Threshold']:.4f}",
                         "Sensitivity": cell(r, "Sensitivity", "Sens exact CI"),
                         "Specificity": cell(r, "Specificity", "Spec exact CI"),
                         "PPV*": cell(r, "PPV (reference prev.)"), "NPV*": cell(r, "NPV (reference prev.)")})
    wb = Workbook()
    ws = wb.active; ws.title = "Table"
    ws["A1"] = "Clinical performance at the development out-of-fold Youden threshold, value (95% CI)"
    ws["A1"].font = Font(name=FONT, size=12)
    cols = list(rows[0])
    for j, c in enumerate(cols, start=1):
        style_header(ws.cell(3, j, c))
    for i, r in enumerate(rows, start=4):
        for j, c in enumerate(cols, start=1):
            body(ws.cell(i, j, r[c]), "left" if j <= 2 else "center")
    n = len(rows) + 5
    for t in ["*PPV and NPV at a reference prevalence applied to all cohorts: "
              + "; ".join(f"{t.replace(' classification', '')} {ref[t]:.4f}" for t in TASK_TITLE)
              + " (abnormal: general-population prevalence of OSA or RBD; other tasks: internal development set).",
              "CIs from 4,000 bootstrap resamples of examinations; sensitivities and specificities whose bootstrap interval "
              "is a single point are given with the exact Clopper–Pearson interval.",
              "Source: results/clinical/table_optimal_threshold.csv (values, not formulas)."]:
        ws.cell(n, 1, t).font = Font(name=FONT, size=9); n += 1
    ws.column_dimensions["A"].width = 30
    for j in range(2, len(cols) + 1):
        ws.column_dimensions[get_column_letter(j)].width = 22
    out = f"{paths.RESULTS}/supp_table_operating.xlsx"
    wb.save(out)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["operating", "fusion", "alt"])
    k = ap.parse_args().kind
    operating() if k == "operating" else build(k)


if __name__ == "__main__":
    main()
