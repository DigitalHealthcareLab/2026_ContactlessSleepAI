"""Motion features per flow clip (flow database -> three per-clip databases).

The uint8 flow is mapped back to pixels (u / 255 x 40 - 20) and, per flow field:
  motion    p95 of the pixel flow magnitude, then a 1-s moving average (5 fields)   key "<clip>:scores_p95_smooth1.0"
  magspec   histogram density of the pixel flow magnitude over 64 bins whose edges are square-root spaced
            on 0-20 px                                             key "<clip>:magnitude_spectrogram_bins64_max20_sqrt"
  flow_rep  frame means of the x and y displacement (clips with >= 5 fields; zero-padded to 150)    key "<clip>"
motion and magspec: zstd(level 3) of the .npy array; flow_rep: raw float32 (150 x-means, then 150 y-means).

    python -m psgscreen.preprocessing.motion --center YI [--workers 6]
"""
import argparse
import io
import multiprocessing as mp

import lmdb
import numpy as np
import zstandard as zstd

from psgscreen import paths

EDGES = np.linspace(0, np.sqrt(20.0), 65) ** 2
N_FLOW = 150
_G = {}


def to_pixels(u8):
    return (u8.astype(np.float32) / 255.0) * 40 - 20


def p95_motion(flow):
    fx, fy = to_pixels(flow[..., 0]), to_pixels(flow[..., 1])
    s = np.zeros(flow.shape[0], dtype=np.float32)
    for t in range(flow.shape[0]):
        s[t] = np.percentile(np.sqrt(fx[t] ** 2 + fy[t] ** 2), 95)
    return np.convolve(s, np.ones(5, dtype=np.float32) / 5, mode="same").astype(np.float32, copy=False)


def magnitude_histogram(flow):
    fx, fy = to_pixels(flow[..., 0]), to_pixels(flow[..., 1])
    mag = np.sqrt(fx ** 2 + fy ** 2)
    spec = np.zeros((mag.shape[0], 64), dtype=np.float32)
    for t in range(mag.shape[0]):
        spec[t] = np.histogram(mag[t], bins=EDGES, density=True)[0].astype(np.float32)
    return spec


def flow_means(flow):
    fx = (flow[..., 0].mean(axis=(1, 2)) / 255.0 * 40.0 - 20.0).astype(np.float32)
    fy = (flow[..., 1].mean(axis=(1, 2)) / 255.0 * 40.0 - 20.0).astype(np.float32)
    out = np.zeros(2 * N_FLOW, np.float32)
    n = min(len(fx), N_FLOW)
    out[:n], out[N_FLOW:N_FLOW + n] = fx[:n], fy[:n]
    return out


def _npy(a):
    b = io.BytesIO()
    np.save(b, a, allow_pickle=False)
    return b.getvalue()


def _init(center):
    _G["env"] = lmdb.open(paths.lmdb("flow", center), readonly=True, lock=False, readahead=False, meminit=False)
    _G["dec"], _G["comp"] = zstd.ZstdDecompressor(), zstd.ZstdCompressor(level=3)


def _one(key):
    with _G["env"].begin() as txn:
        buf = txn.get(key.encode("ascii"))
    flow = np.load(io.BytesIO(_G["dec"].decompress(buf)), allow_pickle=False)
    rep = flow_means(flow).tobytes() if flow.ndim == 4 and flow.shape[0] >= 5 else None
    return (key, _G["comp"].compress(_npy(p95_motion(flow))), _G["comp"].compress(_npy(magnitude_histogram(flow))), rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", required=True, choices=list(paths.CENTERS))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    with lmdb.open(paths.lmdb("flow", a.center), readonly=True, lock=False).begin() as txn:
        keys = [k.decode("ascii") for k in txn.cursor().iternext(keys=True, values=False)]
    out = {k: lmdb.open(paths.lmdb(k, a.center), map_size=1 << 40, subdir=True) for k in ("motion", "magspec", "flow_rep")}
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.center,)) as pool:
        for i in range(0, len(keys), 16000):
            res = pool.map(_one, keys[i:i + 16000], chunksize=64)
            with out["motion"].begin(write=True) as tm, out["magspec"].begin(write=True) as ts, \
                    out["flow_rep"].begin(write=True) as tr:
                for k, m, s, r in res:
                    tm.put(f"{k}:scores_p95_smooth1.0".encode("utf-8"), m)
                    ts.put(f"{k}:magnitude_spectrogram_bins64_max20_sqrt".encode("utf-8"), s)
                    if r is not None:
                        tr.put(k.encode("utf-8"), r)
            print(f"{min(i + 16000, len(keys))}/{len(keys)}", flush=True)
    for e in out.values():
        e.close()


if __name__ == "__main__":
    main()
