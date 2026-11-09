"""Sleep-stage hypnogram of Hospital A exams (REM shading of the cumulative-time figure).

Stages come from the scoring text exports (tab-separated event list with Time, Epoch and Stage columns): per
30-s epoch the most frequent stage among its rows (S1-S4 / REM mapped to N1-N3 / R). Epochs without a row are
filled when the stages before and after the gap agree.

    python -m psgscreen.dataset.hypnogram            -> hypnogram/hypnogram_aligned.csv (uuid, epoch, stage, t_sec)
"""
import io
import os

import numpy as np
import pandas as pd

from psgscreen import paths
from psgscreen.dataset.store import uuid_of

EPOCH_SEC = 30.0
STAGE = {"W": "W", "N1": "N1", "N2": "N2", "N3": "N3", "R": "R", "REM": "R", "S1": "N1", "S2": "N2", "S3": "N3",
         "S4": "N3"}


def epoch_stages(path):
    raw = open(path, "rb").read()
    txt = None
    for enc in ("utf-16", "utf-16-le", "utf-8-sig", "cp949"):
        try:
            t = raw.decode(enc)
            if "Stage" in t and "Epoch" in t:
                txt = t
                break
        except Exception:
            continue
    if txt is None:
        return None
    lines = txt.splitlines()
    hi = next((i for i, l in enumerate(lines) if l.strip().startswith("Time\t")), None)
    if hi is None:
        return None
    D = pd.read_csv(io.StringIO("\n".join(lines[hi:])), sep="\t", dtype=str)
    D.columns = [c.strip() for c in D.columns]
    if "Epoch" not in D.columns or "Stage" not in D.columns:
        return None
    D["Epoch"] = pd.to_numeric(D.Epoch, errors="coerce")
    D["Stage"] = D.Stage.astype(str).str.strip().str.upper()
    d = D.dropna(subset=["Epoch"])
    d = d[d.Stage.isin(STAGE)]
    if not len(d):
        return None
    d = d.assign(Epoch=d.Epoch.astype(int), stage=d.Stage.map(STAGE))
    return d.groupby("Epoch").stage.agg(lambda x: x.value_counts().idxmax())


def fill_gaps(ep, st):
    full = np.arange(ep.min(), ep.max() + 1)
    out = np.full(len(full), "", dtype=object)
    out[ep - ep.min()] = st
    i = 0
    while i < len(out):
        if out[i] != "":
            i += 1
            continue
        j = i
        while j < len(out) and out[j] == "":
            j += 1
        if i > 0 and j < len(out) and out[i - 1] == out[j]:
            out[i:j] = out[i - 1]
        i = j
    m = out != ""
    return full[m], out[m]


def main():
    d = f"{paths.ANNOTATION_EXPORTS}/INT"
    inv = pd.read_csv(paths.INVENTORY, encoding="utf-8-sig")
    want = set(inv.loc[inv.center == "INT", "uuid"].str.lower())
    rows = []
    for f in sorted(os.listdir(d)):
        u = uuid_of(f)
        if not f.endswith(".txt") or u not in want:
            continue
        g = epoch_stages(os.path.join(d, f))
        if g is None:
            continue
        ep, st = g.index.values.astype(int), g.values.astype(object)
        o = np.argsort(ep)
        rows += [(u, int(e), s) for e, s in zip(*fill_gaps(ep[o], st[o]))]
    H = pd.DataFrame(rows, columns=["uuid", "epoch", "stage"])
    H["t_sec"] = (H.epoch - 1) * EPOCH_SEC
    os.makedirs(os.path.dirname(paths.HYPNOGRAM), exist_ok=True)
    H.to_csv(paths.HYPNOGRAM, index=False, encoding="utf-8-sig")
    print(f"{H.uuid.nunique()} exams, {len(H):,} epochs -> {paths.HYPNOGRAM}")


if __name__ == "__main__":
    main()
