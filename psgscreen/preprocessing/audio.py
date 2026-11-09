"""Audio features per 30-s clip.

The audio tracks of an exam's video files (sorted by name) are decoded with ffmpeg to 16 kHz mono and
concatenated; the waveform is cut into 30-s clips (the last one zero-padded). Per clip:
  log_mel  128-band Mel power spectrogram (n_fft 1024, hop 512, 50-8000 Hz) in dB, floored 80 dB below its
           maximum (torchaudio, GPU when available)
  cqt      constant-Q transform from 32.70 Hz, 84 bins, 12 per octave, hop 512, 20 log10 max(|C|, 1e-6) (librosa)
both float16, 1 x bins x 938 frames. Record: key "<exam folder>_<clip:04d>", entries (name, .npy) as read by
dataset/store.py.

    python -m psgscreen.preprocessing.audio --center GN
"""
import argparse
import io
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import lmdb
import numpy as np

from psgscreen import paths

SR, HOP, CLIP_SAMPLES = 16000, 512, 16000 * 30


def waveform(path):
    p = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vn", "-acodec", "pcm_s16le",
                        "-ar", str(SR), "-ac", "1", "-f", "s16le", "pipe:1"], capture_output=True, timeout=120)
    return (np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0) if p.stdout else None


def clips(files):
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = [w for w in pool.map(waveform, files) if w is not None]
    if not parts:
        return []
    audio = np.concatenate(parts).astype(np.float32)
    out = []
    for s in range(0, len(audio), CLIP_SAMPLES):
        c = audio[s:s + CLIP_SAMPLES]
        if len(c) < CLIP_SAMPLES:
            c = np.concatenate([c, np.zeros(CLIP_SAMPLES - len(c), np.float32)])
        out.append(c)
    return out


def cqt_db(x):
    import librosa
    c = librosa.cqt(x, sr=SR, hop_length=HOP, fmin=32.70, n_bins=84, bins_per_octave=12)
    return (20 * np.log10(np.maximum(np.abs(c), 1e-6))).astype(np.float16)[np.newaxis]


def record(entries):
    buf = io.BytesIO()
    for name, arr in entries.items():
        b = io.BytesIO()
        np.save(b, arr)
        k = name.encode("utf-8")
        buf.write(len(k).to_bytes(4, "little")); buf.write(k)
        buf.write(len(b.getvalue()).to_bytes(8, "little")); buf.write(b.getvalue())
    return buf.getvalue()


def main():
    import torch
    import torchaudio.transforms as T
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", required=True, choices=list(paths.CENTERS))
    ap.add_argument("--cqt_workers", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel = T.MelSpectrogram(sample_rate=SR, n_fft=1024, hop_length=HOP, n_mels=128, f_min=50.0, f_max=8000.0,
                           power=2.0).to(dev)
    to_db = T.AmplitudeToDB(stype="power", top_db=80.0).to(dev)
    exams = {}
    for d, _, fs in os.walk(f"{paths.VIDEO}/{a.center}"):
        avi = sorted((f for f in fs if f.lower().endswith(".avi")), key=lambda f: os.path.splitext(f)[0])
        if avi:
            exams[os.path.basename(d)] = [os.path.join(d, f) for f in avi]
    env = lmdb.open(paths.lmdb("audio", a.center), map_size=2 * 1024 ** 4, writemap=True)
    with env.begin() as txn:
        done = {k.decode("utf-8").rsplit("_", 1)[0] for k in txn.cursor().iternext(keys=True, values=False)}
    for exam, files in exams.items():
        if exam in done:
            continue
        segs = clips(files)
        if not segs:
            continue
        with ProcessPoolExecutor(a.cqt_workers) as pool:
            cqts = list(pool.map(cqt_db, segs))
        mels = []
        for i in range(0, len(segs), 64):
            x = torch.from_numpy(np.stack(segs[i:i + 64])[:, np.newaxis, :]).to(dev)
            with torch.no_grad():
                mels += list(to_db(mel(x)).cpu().numpy().astype(np.float16))
        with env.begin(write=True) as txn:
            for i, (m, c) in enumerate(zip(mels, cqts)):
                txn.put(f"{exam}_{i:04d}".encode("utf-8"), record({"log_mel": m, "cqt": c}))
        print(exam, len(segs), flush=True)
    env.close()


if __name__ == "__main__":
    main()
