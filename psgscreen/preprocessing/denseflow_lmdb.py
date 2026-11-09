"""Optical flow computed with the denseflow tool (part of Hospital A) -> the flow database.

denseflow (Docker image jinyong/denseflow:batch2048_umask_fps5; Farneback, options -s=1 -b=20 -cf -v -a=farn)
writes, per video file, the quantised flow as JPEG pairs flow_x_*.jpg / flow_y_*.jpg (5 fields per second,
bound 20 px). Each video's images are cut into clips of 150 fields, resized to 256 x 256 (INTER_LINEAR) and
padded with 128 (zero flow) to 150 fields; clip indices run on across the videos of an exam (sorted by name).
Exams with fewer than 100 video folders are skipped. Record: key "<exam folder>_<clip:04d>", zstd of the .npy
array (150 x 256 x 256 x 2, uint8).

    python -m psgscreen.preprocessing.denseflow_lmdb --root <denseflow output>      (writes flow_SC.lmdb)
"""
import argparse
import glob
import io
import math
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import lmdb
import numpy as np
import zstandard as zstd

from psgscreen import paths

MIN_VIDEOS = 100


def clips_of_video(video_dir, exam, start):
    fx_files = sorted(glob.glob(f"{video_dir}/flow_x_*.jpg"))
    fy_files = sorted(glob.glob(f"{video_dir}/flow_y_*.jpg"))
    if not fx_files or cv2.imread(fx_files[0], 0) is None:
        return []
    with ThreadPoolExecutor(max_workers=32) as pool:
        xs = list(pool.map(lambda p: cv2.imread(p, 0), fx_files))
        ys = list(pool.map(lambda p: cv2.imread(p, 0), fy_files))
    out = []
    for i in range(math.ceil(len(fx_files) / 150)):
        clip = np.full((150, 256, 256, 2), 128, dtype=np.uint8)
        frames = range(i * 150, min(i * 150 + 150, len(fx_files)))
        if any(xs[k] is None or ys[k] is None for k in frames):
            continue
        for j, k in enumerate(frames):
            clip[j, :, :, 0] = cv2.resize(xs[k], (256, 256), interpolation=cv2.INTER_LINEAR)
            clip[j, :, :, 1] = cv2.resize(ys[k], (256, 256), interpolation=cv2.INTER_LINEAR)
        buf = io.BytesIO()
        np.save(buf, clip, allow_pickle=False)
        out.append((f"{exam}_{start + i:04d}", zstd.ZstdCompressor(level=6).compress(buf.getvalue())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="denseflow output: <root>/<exam folder>/<video folder>/")
    a = ap.parse_args()
    env = lmdb.open(paths.lmdb("flow", "SC"), map_size=int(5e12))
    for exam_dir in sorted(glob.glob(f"{a.root}/*")):
        videos = sorted(glob.glob(f"{exam_dir}/*"))
        if len(videos) < MIN_VIDEOS:
            continue
        exam, clip = os.path.basename(exam_dir), 0
        for v in videos:
            n = len(os.listdir(v)) / 2
            if n == 0:
                continue
            with env.begin(write=True) as txn:
                for k, val in clips_of_video(v, exam, clip):
                    txn.put(k.encode("ascii"), val)
            clip += math.ceil(n / 150)
        print(exam, clip, flush=True)
    env.close()


if __name__ == "__main__":
    main()
