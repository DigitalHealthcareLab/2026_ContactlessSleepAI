"""Study cohort, labels and splits.

Inclusion, applied in this order to the exam inventory (exams with synchronized video and audio of the PSG night):
  1. not on the exclusion list   data/cohort/exclusions.csv (uuid, reason): recordings that failed the quality
                                 review (Hospital B) and PAP-treated or split-night studies identified at intake
  2. a matched report with an AHI
  3. not a daytime nap test (MSLT / MWT in the exam name)
  4. Hospital C (external): PSG night (date of the matched report) in 2025 or later
  5. age >= 19 years
Label (two axes): OSA = AHI >= 5; RBD = RBD status of the report impression "definite" or "probable"
  -> Normal, OSA-only, RBD-only, Both.
Splits: data/cohort/splits.csv (uuid, split, fold). They were assigned once by stratified random sampling within
centre and class (10 % locked internal test, the rest round-robin into 10 folds) and are fixed for all analyses;
the roster keeps the row order of this file (model training reads the exams in this order).
External exams are never used for training.

    python -m psgscreen.cohort.roster                         -> cohort/roster.csv (the default PSG_ROSTER)
"""
import numpy as np
import pandas as pd

from psgscreen import paths

MIN_AGE = 19
EXTERNAL_FROM_YEAR = 2025
NAP = r"MSLT|MLST|MWT"


def label(ahi, rbd_status):
    osa = ahi >= 5
    rbd = rbd_status.isin(["definite", "probable"])
    return np.select([osa & rbd, osa & ~rbd, ~osa & rbd], ["Both", "OSA-only", "RBD-only"], default="Normal")


def main():
    inv = pd.read_csv(paths.INVENTORY, encoding="utf-8-sig", dtype={"exam_date": str, "exam_month": str})
    rep = pd.read_csv(paths.REPORT_TABLE, encoding="utf-8-sig")
    excl = pd.read_csv(paths.EXCLUSIONS, encoding="utf-8-sig")
    D = inv.merge(rep[["uuid", "ahi", "rbd_status", "age", "exam_date"]].rename(columns={"exam_date": "night"}),
                  on="uuid", how="left")
    year = pd.to_numeric(D.night.astype(str).str[:4], errors="coerce")
    steps = [("exclusion list", ~D.uuid.isin(excl.uuid)),
             ("no report AHI", D.ahi.notna()),
             ("nap test", ~D.exam_id.str.contains(NAP, case=False, regex=True)),
             (f"external before {EXTERNAL_FROM_YEAR}", ~D.center.isin(paths.EXTERNAL) | (year >= EXTERNAL_FROM_YEAR)),
             (f"age < {MIN_AGE}", D.age >= MIN_AGE)]
    keep = pd.Series(True, index=D.index)
    print(f"inventory {len(D)}  {D.center.value_counts().to_dict()}")
    for name, ok in steps:
        drop = keep & ~ok
        keep &= ok
        print(f"  - {name:22s} {int(drop.sum()):4d}  {D[drop].center.value_counts().to_dict()}")
    D = D[keep].copy()
    D["label"] = label(D.ahi, D.rbd_status)

    S = pd.read_csv(paths.SPLITS, encoding="utf-8-sig")
    missing = ~D.uuid.isin(S.uuid)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} included exams have no split assignment")
    R = S[["uuid", "split", "fold"]].merge(D, on="uuid")[["uuid", "center", "exam_id", "label", "split", "fold"]]
    R.to_csv(f"{paths.COHORT}/roster.csv", index=False)
    print(f"cohort {len(R)}")
    print(pd.crosstab([R.center, R.split], R.label, margins=True).to_string())


if __name__ == "__main__":
    main()
