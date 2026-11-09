"""Readers of the per-clip feature databases (one LMDB per kind and centre, one record per 30-s clip).

Record keys are "<exam key>_<clip index>" (4- or 6-digit index); the exam key contains the exam uuid.
  audio     concatenated (name length, name, array length, .npy bytes) entries; "log_mel" (128 x frames, dB) and
            "cqt" (84 x frames), 31.25 frames / s
  motion    "<key>:scores_p95_smooth1.0", zstd-compressed .npy, 150 values (5 Hz): p95 optical-flow magnitude
  magspec   "<key>:magnitude_spectrogram_bins64_max20_sqrt", zstd-compressed .npy, 150 x 64: per-frame histogram
            of pixel flow magnitude (64 bins, square-root spaced over 0-20 px), as a density
  flow_rep  "<key>", raw float32, 300 values: mean flow x (150) then mean flow y (150)
"""
import io
import re

import lmdb
import numpy as np
import zstandard as zstd

from psgscreen import paths

N_FLOW = 150
AUDIO_HZ, FLOW_HZ = 16000 / 512, 5.0
CLIP_SEC = 30.0
MOTION_SUFFIX = "scores_p95_smooth1.0"
MAGSPEC_SUFFIX = "magnitude_spectrogram_bins64_max20_sqrt"
_UUID = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)
_IDX = re.compile(r"_(\d{4,6})$")


def uuid_of(s):
    m = _UUID.search(str(s))
    return m.group(1).lower() if m else ""


def split_key(key):
    """(exam key, clip index) of a record key."""
    m = _IDX.search(key)
    return (key[:m.start()], int(m.group(1))) if m else (key, None)


def key_variants(key):
    """The key with its clip index written with 4 and with 6 digits (the databases use both)."""
    base, idx = split_key(key)
    if idx is None:
        return [key]
    width = len(_IDX.search(key).group(1))
    return [f"{base}_{idx:0{width}d}", f"{base}_{idx:0{6 if width == 4 else 4}d}"]


class Store:
    def __init__(self, center):
        op = dict(readonly=True, lock=False, readahead=False, meminit=False, max_readers=256)
        self.env = {k: lmdb.open(paths.lmdb(k, center), **op) for k in ("audio", "motion", "magspec", "flow_rep")}
        self.dec = zstd.ZstdDecompressor()

    def audio(self, key):
        """(log-Mel, CQT) of a clip, or (None, None)."""
        with self.env["audio"].begin() as txn:
            raw = txn.get(key.encode("utf-8"))
        if raw is None:
            return None, None
        buf, data = io.BytesIO(raw), {}
        while True:
            n = buf.read(4)
            if not n:
                break
            name = buf.read(int.from_bytes(n, "little")).decode("utf-8")
            data[name] = np.load(io.BytesIO(buf.read(int.from_bytes(buf.read(8), "little"))), allow_pickle=False)
        mel, cqt = data.get("log_mel"), data.get("cqt")
        if mel is None or cqt is None:
            return None, None
        return mel[0].astype(np.float32), cqt[0].astype(np.float32)

    def _npz(self, kind, key, suffix):
        with self.env[kind].begin() as txn:
            buf = txn.get(f"{key}:{suffix}".encode("utf-8"))
        return None if buf is None else np.load(io.BytesIO(self.dec.decompress(buf)), allow_pickle=False)

    def motion_record(self, key, short_ok=False):
        """150 samples; with short_ok also the shorter last clip of a video file."""
        a = self._npz("motion", key, MOTION_SUFFIX)
        if a is None or not (0 < a.size <= N_FLOW if short_ok else a.size == N_FLOW):
            return None
        return np.asarray(a).reshape(-1).astype(np.float32, copy=False)

    def magspec_record(self, key, short_ok=False):
        a = self._npz("magspec", key, MAGSPEC_SUFFIX)
        if a is None or a.ndim != 2 or not (0 < a.shape[0] <= N_FLOW if short_ok else a.shape[0] == N_FLOW):
            return None
        return a.astype(np.float32, copy=False)

    def flow_rep_record(self, key):
        with self.env["flow_rep"].begin(buffers=True) as txn:
            buf = txn.get(key.encode("utf-8"))
        if buf is None:
            return None
        a = np.frombuffer(bytes(buf), dtype=np.float32)
        return np.stack([a[:N_FLOW], a[N_FLOW:]], axis=1) if a.size == 2 * N_FLOW else None

    def first(self, read, key):
        """Record of the first key variant that exists."""
        for k in key_variants(key):
            a = read(k)
            if a is not None:
                return a
        return None

    def audio_keys(self):
        with self.env["audio"].begin(buffers=True) as txn:
            return [bytes(k).decode("utf-8") for k in txn.cursor().iternext(keys=True, values=False)]

    def motion_clip_counts(self, index_span=False):
        """Number of motion records per exam uuid; with index_span the highest clip index + 1 instead."""
        n = {}
        with self.env["motion"].begin() as txn:
            for k in txn.cursor().iternext(keys=True, values=False):
                k = k.decode("utf-8")
                if k.endswith(":" + MOTION_SUFFIX):
                    u = uuid_of(k)
                    if not index_span:
                        n[u] = n.get(u, 0) + 1
                        continue
                    idx = split_key(k[:-len(MOTION_SUFFIX) - 1])[1]
                    if idx is not None:
                        n[u] = max(n.get(u, 0), idx + 1)
        return n
