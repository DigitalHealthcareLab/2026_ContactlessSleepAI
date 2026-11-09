"""One PSG report per exam, and the report values used by the study.

Matching (same centre and hospital number):
  recording date known   reports of that date; when none of them is a nocturnal study, nocturnal reports within
                         +-2 days (a daytime MSLT / MWT report can share the date)
  only the month known   reports of that month
  neither                all reports of the patient
Among the candidates: nocturnal before MSLT / MWT, diagnostic before titration / split-night, earlier date first,
then file-name order.
RBDSQ: the score written in the report impression; for Hospital C (YI) exams without one, the questionnaire
completed closest in time to the exam (YI registry export, matched by hospital number).

Inputs: data/cohort/exam_inventory.csv (uuid, exam_id, center, exam_date, exam_month), cohort/reports_parsed.parquet,
data/cohort/yi_questionnaires/{registry.xlsx, rbdsq_scores.xlsx}.

    python -m psgscreen.cohort.report_table                   -> cohort/report_table.csv
"""
import os

import numpy as np
import pandas as pd

from psgscreen import paths

FIELDS = ["date", "study_type", "age", "sex", "bmi", "tib_min", "tst_min", "ahi", "rbd_status", "rbd_evidence",
          "rbdsq"]
# column names of the YI registry exports
REGISTRY_ID, REGISTRY_HOSPITAL_NO = "연구등록번호", "(실명)등록번호"          # study id, hospital number
SCORE_SHEET, SCORE_DATE, SCORE_VALUE = "검사단위_score", "서식작성일자", "RBDSQ_prorated"   # per form: date, score


def hospital_number(exam_id):
    return pd.Series(exam_id).astype(str).str.extract(r"^\s*0*(\d+)")[0].values


def load_reports():
    P = pd.read_parquet(paths.REPORTS_PARSED)
    if "error" in P:
        P = P[P.error.isna()]
    P = P[P.hospital_id.notna()].copy()
    P["dt"] = pd.to_datetime(P.date, errors="coerce")
    P["ym"] = P.dt.dt.strftime("%Y-%m")
    P["night"] = ~P.study_type.astype(str).str.contains("MSLT|MWT", na=False)
    P["diag"] = P.study_type.astype(str) == "diagnostic"
    return P


def match(inv, P):
    groups = {k: g for k, g in P.groupby(["center", "hospital_id"], sort=False)}
    out = []
    for r, hid in zip(inv.itertuples(), hospital_number(inv.exam_id)):
        c = groups.get((r.center, hid))
        if c is not None and isinstance(r.exam_date, str) and r.exam_date:
            d0 = pd.Timestamp(r.exam_date)
            same = c[c.dt == d0]
            if not same.night.any():
                near = c[c.night & ((c.dt - d0).abs().dt.days <= 2)]
                same = near if len(near) else same
            c = same
        elif c is not None and isinstance(r.exam_month, str) and r.exam_month:
            c = c[c.ym == r.exam_month]
        if c is None or not len(c):
            out.append(dict(uuid=r.uuid, report_file=None))
            continue
        best = c.sort_values(["night", "diag", "dt"], ascending=[False, False, True], kind="stable").iloc[0]
        out.append(dict(uuid=r.uuid, report_file=best.file, **{f: best[f] for f in FIELDS}))
    return pd.DataFrame(out)


def questionnaire_rbdsq(T):
    """RBDSQ of the YI questionnaire nearest in time to each YI exam (NaN when the patient has none)."""
    d = paths.QUESTIONNAIRE
    if not os.path.exists(f"{d}/rbdsq_scores.xlsx"):
        return pd.Series(np.nan, index=T.index)
    reg = pd.read_excel(f"{d}/registry.xlsx").drop_duplicates(REGISTRY_ID).set_index(REGISTRY_ID)[REGISTRY_HOSPITAL_NO]
    S = pd.read_excel(f"{d}/rbdsq_scores.xlsx", sheet_name=SCORE_SHEET)
    S["hid"] = S[REGISTRY_ID].map(reg).astype("Int64").astype(str)
    S["sdt"] = pd.to_datetime(S[SCORE_DATE])
    out = pd.Series(np.nan, index=T.index)
    hid = hospital_number(T.exam_id)
    for i in np.flatnonzero((T.center == "YI").values):
        sub = S[S.hid == hid[i]]
        edt = pd.to_datetime(T.date.iloc[i], errors="coerce")
        if len(sub) and pd.notna(edt):
            out.iloc[i] = sub.loc[(sub.sdt - edt).abs().idxmin(), SCORE_VALUE]
    return out


def main():
    inv = pd.read_csv(paths.INVENTORY, encoding="utf-8-sig", dtype={"exam_date": str, "exam_month": str})
    T = inv[["uuid", "exam_id", "center"]].merge(match(inv, load_reports()), on="uuid", how="left")
    q = questionnaire_rbdsq(T)
    T["rbdsq_source"] = np.where(T.rbdsq.notna(), "report", np.where(q.notna(), "questionnaire", None))
    T["rbdsq"] = T.rbdsq.where(T.rbdsq.notna(), q)
    T = T.rename(columns={"date": "exam_date"})
    T.to_csv(paths.REPORT_TABLE, index=False, encoding="utf-8-sig")
    print(f"{len(T)} exams, {int(T.report_file.notna().sum())} with a report, "
          f"{int(T.ahi.notna().sum())} with AHI, RBDSQ {T.rbdsq_source.value_counts().to_dict()} -> {paths.REPORT_TABLE}")


if __name__ == "__main__":
    main()
