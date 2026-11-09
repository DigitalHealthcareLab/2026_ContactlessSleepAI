"""Clip-time tables: for every 30-s optical-flow clip of an exam, its start time on the recording clock, sample
interval and number of valid flow samples.

  video   (Hospitals B, C) from the video files: each file is cut into clips of 900 frames, so clip c of a file
          starts at T_file + 900 c / fps and holds min(150, frames / 6 - 150 c) samples 6 / fps s apart;
          T_file = duration of the preceding files
  motion  (Hospital A) from the motion database: consecutive clips of the exam, each lasting
          (its number of samples) x 0.2 s

    python -m psgscreen.dataset.clip_times --center GN --source video   -> alignment/clip_times_<centre>.npz
"""
import argparse
import glob
import io
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from psgscreen import paths
from psgscreen.dataset.store import N_FLOW, Store, split_key, uuid_of

CLIP_FRAMES = 900


def video_table(folder):
    import cv2
    files = sorted(glob.glob(os.path.join(folder, "*.avi")) + glob.glob(os.path.join(folder, "*.mp4")),
                   key=lambda p: int(m.group(1)) if (m := re.search(r"_(\d+)\.(avi|mp4)$", p, re.I)) else 10 ** 9)
    rows, t = [], 0.0
    for p in files:
        cap = cv2.VideoCapture(p)
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if n <= 0 or not fps or fps <= 0:
            continue
        for c in range(math.ceil(n / CLIP_FRAMES)):
            rows.append((t + c * CLIP_FRAMES / fps, 6.0 / fps, max(0, min(N_FLOW, n // 6 - N_FLOW * c))))
        t += n / fps
    return np.asarray(rows, dtype=np.float32)


def motion_tables(center):
    st, n = Store(center), defaultdict(dict)
    suffix = b":scores_p95_smooth1.0"
    with st.env["motion"].begin() as txn:
        for k, v in txn.cursor():
            if k.endswith(suffix):
                base, idx = split_key(k[:-len(suffix)].decode("utf-8"))
                if idx is not None and uuid_of(base):
                    n[uuid_of(base)][idx] = int(np.load(io.BytesIO(st.dec.decompress(v)), allow_pickle=False).size)
    out = {}
    for u, d in n.items():
        rows, t = np.zeros((max(d) + 1, 3), np.float32), 0.0
        for j in range(len(rows)):
            rows[j] = (t, 0.2, d.get(j, 0))
            t += d.get(j, 0) * 0.2
        out[u] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", required=True, choices=list(paths.CENTERS))
    ap.add_argument("--source", required=True, choices=["video", "motion"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.source == "motion":
        tables = motion_tables(a.center)
    else:
        inv = pd.read_csv(paths.INVENTORY, encoding="utf-8-sig")
        want = set(inv.loc[inv.center == a.center, "uuid"].str.lower())
        folders = {}
        for d, _, _ in os.walk(f"{paths.VIDEO}/{a.center}"):
            u = uuid_of(os.path.basename(d))
            if u in want:
                folders.setdefault(u, d)
        with ProcessPoolExecutor(a.workers) as ex:
            tables = {u: t for u, t in zip(folders, ex.map(video_table, folders.values())) if len(t)}
    os.makedirs(os.path.dirname(paths.clip_times(a.center)), exist_ok=True)
    np.savez_compressed(paths.clip_times(a.center), **tables)
    print(f"{a.center}: {len(tables)} exams -> {paths.clip_times(a.center)}")


if __name__ == "__main__":
    main()
