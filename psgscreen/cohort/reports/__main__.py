"""Parse every PSG report under <data>/reports/<centre>/ (.docx or .doc, any depth) into one table.

    python -m psgscreen.cohort.reports [--workers 6]          -> cohort/reports_parsed.parquet
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from psgscreen import paths
from psgscreen.cohort.reports.extract import read_blocks
from psgscreen.cohort.reports.parse import parse_report


def list_reports():
    files = []
    for c in paths.CENTERS:
        root = f"{paths.REPORTS}/{c}"
        for dirpath, _, names in os.walk(root):
            files += [(c, os.path.join(dirpath, n)) for n in names
                      if n.lower().endswith((".docx", ".doc")) and not n.startswith("~$")]
    return sorted(files, key=lambda t: (t[0], t[1]))


def parse_one(item):
    center, path = item
    row = dict(center=center, file=os.path.relpath(path, paths.REPORTS))
    try:
        row.update(parse_report(path, read_blocks(path)))
    except Exception as e:                       # unreadable or empty file: kept with the error message
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    files = list_reports()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(parse_one, files, chunksize=32))
    D = pd.DataFrame(rows)
    os.makedirs(paths.COHORT, exist_ok=True)
    D.to_parquet(paths.REPORTS_PARSED, index=False)
    print(f"{len(D)} reports ({D.center.value_counts().to_dict()}), "
          f"{int(D.get('error', pd.Series(dtype=str)).notna().sum())} unreadable -> {paths.REPORTS_PARSED}")


if __name__ == "__main__":
    main()
