"""Table 1 (demographic and clinical characteristics) and the per-exam demographics used by the figures.

Values from the matched PSG report (cohort/report_table.csv); recording duration = end of the last flow clip of
the clip-time table (alignment/clip_times_<centre>.npz). OSA severity bands of the table: 5 to < 10, 10 to < 15
and >= 15 events / h among exams labelled OSA-only or Both.

    python -m psgscreen.cohort.table1          -> cohort/demographics.csv, results/table1.xlsx
"""
import os

import numpy as np
import pandas as pd

from psgscreen import paths

DISPLAY = {"Normal": "Normal", "OSA-only": "OSA-only", "RBD-only": "RBD-only", "Both": "RBD with OSA"}


def durations():
    out = {}
    for c in paths.CENTERS:
        if os.path.exists(paths.clip_times(c)):
            z = np.load(paths.clip_times(c))
            for u in z.files:
                v = np.asarray(z[u], dtype=float)
                if v.ndim == 2 and len(v):
                    out[u.lower()] = float((v[:, 0] + v[:, 2] * v[:, 1]).max()) / 3600.0
    return out


def mean_sd(x, d=1):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return f"{x.mean():.{d}f} ± {x.std():.{d}f}" if len(x) else "—"


def n_pct(n, d):
    return f"{n:,} ({n / d * 100:.1f})" if d else "0 (—)"


def table(D, groups, diagnosis=True):
    rows = []

    def row(var, level, fn):
        rows.append({"Variable": var, "Level": level, **{g: fn(D[m]) for g, m in groups}})

    row("Age, years, mean ± SD", "", lambda g: mean_sd(g.age))
    for s, name in (("M", "Male"), ("F", "Female")):
        row("Sex, n (%)" if s == "M" else "", name,
            lambda g, s=s: n_pct(int((g.sex == s).sum()), int(g.sex.notna().sum())))
    if diagnosis:
        for i, c in enumerate(DISPLAY):
            row("Clinical diagnosis, n (%)" if i == 0 else "", DISPLAY[c],
                lambda g, c=c: n_pct(int((g.label == c).sum()), len(g)))
    for i, (name, lo, hi) in enumerate((("Mild (5–<10)", 5, 10), ("Moderate (10–<15)", 10, 15),
                                        ("Severe (≥15)", 15, np.inf))):
        def f(g, lo=lo, hi=hi):
            a = pd.to_numeric(g[g.label.isin(["OSA-only", "Both"])].ahi, errors="coerce").dropna()
            return n_pct(int(((a >= lo) & (a < hi)).sum()), len(a))
        row("OSA severity, OSA-only and RBD with OSA, n (%)" if i == 0 else "", name, f)
    row("Recording duration, hour, mean ± SD", "", lambda g: mean_sd(g.rec_h))
    row("BMI, kg/m², mean ± SD", "", lambda g: mean_sd(g.bmi))
    row("AHI, /hour, mean ± SD", "", lambda g: mean_sd(g.ahi))
    row("Time in bed, hour, mean ± SD", "", lambda g: mean_sd(g.tib_h))
    row("Total sleep time, hour, mean ± SD", "", lambda g: mean_sd(g.tst_h))
    return pd.DataFrame(rows)


def main():
    R = pd.read_csv(paths.ROSTER, encoding="utf-8-sig")
    T = pd.read_csv(paths.REPORT_TABLE, encoding="utf-8-sig")
    D = R.merge(T[["uuid", "age", "sex", "bmi", "ahi", "tib_min", "tst_min"]], on="uuid", how="left")
    D["tib_h"], D["tst_h"] = D.tib_min / 60.0, D.tst_min / 60.0
    D["rec_h"] = D.uuid.str.lower().map(durations())
    D["patient_id"] = D.exam_id.astype(str).str.extract(r"^\s*0*(\d+)")[0]
    os.makedirs(paths.COHORT, exist_ok=True)
    D[["uuid", "center", "patient_id", "age", "sex", "bmi", "ahi", "tib_h", "tst_h", "rec_h"]].to_csv(
        paths.DEMOGRAPHICS, index=False, encoding="utf-8-sig")

    internal = D.center.isin(paths.INTERNAL)
    allx = pd.Series(True, index=D.index)
    by_cohort = table(D, [(f"Internal Hospital A,B (n={int(internal.sum()):,})", internal),
                          (f"External Hospital C (n={int((~internal).sum()):,})", ~internal),
                          (f"Overall (n={len(D):,})", allx)])
    by_group = table(D, [(f"{DISPLAY[c]} (n={int((D.label == c).sum()):,})", D.label == c) for c in DISPLAY]
                     + [(f"Overall (n={len(D):,})", allx)], diagnosis=False)
    os.makedirs(paths.RESULTS, exist_ok=True)
    with pd.ExcelWriter(f"{paths.RESULTS}/table1.xlsx", engine="openpyxl") as w:
        by_group.to_excel(w, sheet_name="Table1_by_group", index=False)
        by_cohort.to_excel(w, sheet_name="Table1_by_cohort", index=False)
    print(by_cohort.to_string(index=False))


if __name__ == "__main__":
    main()
