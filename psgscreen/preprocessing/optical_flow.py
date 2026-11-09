"""Dense optical flow of the PSG videos (GPU Farneback), 5 flow fields per second, stored per 30-s clip.

From every 6 consecutive frames (30 fps), the flow between frame 6k and frame 6k + 5 is computed on grey images
of 640 x 480 (resized with INTER_AREA when needed) by OpenCV CUDA Farneback (5 levels, pyramid scale 0.5,
window 15, 3 iterations, poly_n 5, poly_sigma 1.2). Displacements are clipped to +-20 px and quantised to uint8
(0 = -20 px, 255 = +20 px). Each video file is cut into clips of 150 flow fields (900 frames); the clip index runs
on across the files of an exam (every file starts a new clip). Record: key "<file stem without _NNNN>_<clip:06d>",
zstd(level 3) of the .npy array (n x 480 x 640 x 2, uint8).
Decoders: PyAV (Hospitals A, B) or OpenCV VideoCapture (Hospital C).

Part of Hospital A was processed with the denseflow tool instead (preprocessing/denseflow_lmdb.py).

    python -m psgscreen.preprocessing.optical_flow --center GN --decoder pyav --gpus 0 --workers_per_gpu 4
"""
import argparse
import io
import json
import math
import multiprocessing as mp
import os
import re
import subprocess

import numpy as np

from psgscreen import paths

STEP, BOUND, CLIP_FLOWS, CLIP_FRAMES = 6, 20.0, 150, 900
SIZE = (640, 480)


def file_number(path):
    """Position of a video file within its exam (the trailing _NNNN of the file name)."""
    m = re.search(r"_(\d+)\.avi$", path)
    return int(m.group(1)) if m else 0


def frame_count(path, decoder):
    """Number of video frames (ffprobe for PyAV videos, OpenCV otherwise); -1 if unknown."""
    try:
        if decoder == "opencv":
            import cv2
            cap = cv2.VideoCapture(path)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
            cap.release()
            return n if n > 0 else -1
        r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-select_streams", "v:0",
                            "-show_streams", path], capture_output=True, text=True, timeout=10)
        s = json.loads(r.stdout)["streams"][0]
        if s.get("nb_frames", "N/A") != "N/A":
            return int(s["nb_frames"])
        num, den = map(int, s.get("r_frame_rate", "30/1").split("/"))
        return int(float(s.get("duration", 0)) * num / den)
    except Exception:
        return -1


def frame_pairs(path, decoder):
    """(frame 6k, frame 6k + 5) as 640 x 480 grey images."""
    import cv2
    grey = lambda g: g if g.shape[1::-1] == SIZE else cv2.resize(g, SIZE, interpolation=cv2.INTER_AREA)
    prev = None
    if decoder == "pyav":
        import av
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for i, frame in enumerate(container.decode(stream)):
                if i % STEP == 0:
                    prev = grey(frame.to_ndarray(format="gray8"))
                elif i % STEP == STEP - 1 and prev is not None:
                    yield prev, grey(frame.to_ndarray(format="gray8"))
                    prev = None
        return
    cap = cv2.VideoCapture(path)
    i = 0
    while cap.grab():
        if i % STEP == 0 or (i % STEP == STEP - 1 and prev is not None):
            ok, frame = cap.retrieve()
            if not ok:
                break
            g = grey(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if i % STEP == 0:
                prev = g
            else:
                yield prev, g
                prev = None
        i += 1
    cap.release()


def _pack(clip):
    import zstandard as zstd
    buf = io.BytesIO()
    np.save(buf, clip, allow_pickle=False)
    return zstd.ZstdCompressor(level=3).compress(buf.getvalue())


def worker(gpu, decoder, tasks, out_q):
    import cv2
    cv2.cuda.setDevice(gpu)
    alg = cv2.cuda_FarnebackOpticalFlow.create(5, 0.5, False, 15, 3, 5, 1.2, 0)
    g0, g1 = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
    while True:
        task = tasks.get()
        if task is None:
            break
        path, exam_key, clip0 = task
        buf, clip, out = np.empty((CLIP_FLOWS, SIZE[1], SIZE[0], 2), np.uint8), clip0, []
        n = 0
        for a, b in frame_pairs(path, decoder):
            g0.upload(a); g1.upload(b)
            f = alg.calc(g0, g1, None).download()
            np.clip(f, -BOUND, BOUND, out=f)
            f += BOUND
            f *= 255.0 / (2.0 * BOUND)
            np.round(f, out=f)
            buf[n] = f
            n += 1
            if n == CLIP_FLOWS:
                out.append((f"{exam_key}_{clip:06d}", _pack(buf)))
                clip, n = clip + 1, 0
        if n:
            out.append((f"{exam_key}_{clip:06d}", _pack(buf[:n].copy())))
        out_q.put(out)
    out_q.put(None)


def main():
    import lmdb
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", required=True, choices=list(paths.CENTERS))
    ap.add_argument("--decoder", required=True, choices=["pyav", "opencv"])
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--workers_per_gpu", type=int, default=4)
    a = ap.parse_args()
    videos = sorted(os.path.join(d, f) for d, _, fs in os.walk(f"{paths.VIDEO}/{a.center}") for f in fs
                    if f.lower().endswith(".avi"))
    by_exam = {}
    for p in videos:
        by_exam.setdefault(os.path.basename(p).rsplit("_", 1)[0], []).append(p)
    env = lmdb.open(paths.lmdb("flow", a.center), map_size=7 * 1024 ** 4, subdir=True, lock=True, readahead=False)
    with env.begin() as txn:
        done = {k.decode("ascii") for k in txn.cursor().iternext(keys=True, values=False)}
    tasks = []
    for key, files in by_exam.items():
        clip = 0
        for p in sorted(files, key=file_number):
            n = frame_count(p, a.decoder)
            if n <= 0:
                continue
            k = math.ceil(n / CLIP_FRAMES)
            if not all(f"{key}_{clip + i:06d}" in done for i in range(k)):
                tasks.append((p, key, clip))
            clip += k
    print(f"{a.center}: {len(by_exam)} exams, {len(tasks)} video files to process")
    gpus = [int(g) for g in a.gpus.split(",")]
    ctx = mp.get_context("spawn")
    tq, oq = ctx.Queue(), ctx.Queue(maxsize=64)
    procs = [ctx.Process(target=worker, args=(g, a.decoder, tq, oq)) for g in gpus for _ in range(a.workers_per_gpu)]
    for p in procs:
        p.start()
    for t in tasks:
        tq.put(t)
    for _ in procs:
        tq.put(None)
    running = len(procs)
    while running:
        out = oq.get()
        if out is None:
            running -= 1
            continue
        with env.begin(write=True) as txn:
            for k, v in out:
                txn.put(k.encode("ascii"), v)
    for p in procs:
        p.join()
    env.close()


if __name__ == "__main__":
    main()
