"""Scoring exports -> event lists (time from the start of the recording, event title, duration in s).

  text  NeuroWorks text export (Hospitals A, B; UTF-16, tab-separated): time zero = the "Beginning of
        Recording" line; times before it are taken as after midnight; the duration is the last mm:ss token
  ent   Xltek .ent binary (Hospital C): annotation records with their sample stamps (and end stamps); time zero =
        the first "Video Recording ON" (else "Beginning of Recording" / "Start Recording", else the first stamp);
        the sample rate is read from the .erd header (offset 352, float64; default 512 Hz); deleted
        annotations are dropped
Event titles lose any text in parentheses. Output: annotations/<centre>/<export name>.ent_events.csv.

    python -m psgscreen.preprocessing.annotations --center YI
"""
import argparse
import csv
import glob
import os
import re
import struct

from psgscreen import paths

TIME = re.compile(r'^\d{2}:\d{2}:\d{2}$')
DURATION = re.compile(r'^\d+:\d+(?:\.\d+)?$')


def hms(sec):
    sec = int(round(sec))
    sign, sec = ("-" if sec < 0 else ""), abs(sec)
    return f"{sign}{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def clean_title(t):
    return re.sub(r'\s+', ' ', re.sub(r'\s*\(.*?\)\s*', '', t).strip())


def parse_text(path):
    lines = [ln for ln in open(path, "r", encoding="utf-16").read().splitlines() if ln.strip()]
    to_sec = lambda t: sum(int(x) * f for x, f in zip(t.split(':'), (3600, 60, 1)))
    t0 = next((to_sec(p[0].strip()) for p in (ln.split('\t') for ln in lines)
               if len(p) >= 2 and TIME.match(p[0].strip()) and p[-1].strip() == "Beginning of Recording"), None)
    if t0 is None:
        raise ValueError(f"{path}: no 'Beginning of Recording' line")
    rows = []
    for ln in lines:
        parts = [p.strip() for p in ln.split('\t')]
        if len(parts) < 2 or not TIME.match(parts[0]):
            continue
        t = to_sec(parts[0])
        t = t + 24 * 3600 if t < t0 else t
        event = clean_title(parts[-1])
        if event:
            d = [p for p in parts if DURATION.match(p)]
            dur = (float(d[-1].split(':')[0]) * 60 + float(d[-1].split(':')[1])) if d else None
            rows.append((hms(t - t0), event, dur))
    return rows


RE_ANN, RE_STAMP = re.compile(r'\(\."Type", "Annotation"\)'), re.compile(r'\(\."Stamp", (\d+)\)')
RE_END, RE_DELETED = re.compile(r'\(\."EndStamp", (\d+)\)'), re.compile(r'\(\."Deleted", 1\)')
RE_TEXT = re.compile(r'\(\."Text", "((?:[^"\\]|\\.)*)"\)')


def sample_rate(ent):
    for p in [ent[:-4] + ".erd"] + sorted(glob.glob(os.path.join(os.path.dirname(ent), "*.erd"))):
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    f.seek(352)
                    v = struct.unpack("<d", f.read(8))[0]
                if 50.0 <= v <= 20000.0:
                    return v
            except Exception:
                continue
    return 512.0


def parse_ent(path):
    raw = open(path, "rb").read().decode("latin-1")
    rate, recs, prev = sample_rate(path), [], 0
    for m in RE_ANN.finditer(raw):
        seg, prev = raw[prev:m.end()], m.end()
        st, tx = list(RE_STAMP.finditer(seg)), list(RE_TEXT.finditer(seg))
        if not st or not tx or RE_DELETED.search(seg[:st[-1].start()]):
            continue
        ends = [int(x.group(1)) for x in RE_END.finditer(seg[:st[-1].start()])]
        t = tx[-1].group(1).replace("\\xd", " ").replace("\\n", " ").replace("\\r", " ")
        t = t.replace("\r", " ").replace("\n", " ").replace('\\"', '"')
        recs.append((int(st[-1].group(1)), ends[-1] if ends else None,
                     re.sub(r"\s+", " ", re.sub(r"\s*\(.*?\)\s*", " ", t)).strip()))
    if not recs:
        return []
    first = lambda names: min((s for s, _, t in recs if t.casefold() in names), default=None)
    t0 = first({"video recording on"})
    t0 = first({"beginning of recording", "start recording"}) if t0 is None else t0
    t0 = min(s for s, _, _ in recs) if t0 is None else t0
    return [(hms((s - t0) / rate), t, round((e - s) / rate, 1) if (e is not None and e >= s) else None)
            for s, e, t in sorted(recs, key=lambda r: r[0]) if t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", required=True, choices=list(paths.CENTERS))
    a = ap.parse_args()
    src, out = f"{paths.ANNOTATION_EXPORTS}/{a.center}", f"{paths.ANNOTATIONS}/{a.center}"
    os.makedirs(out, exist_ok=True)
    ext = ".ent" if a.center == "YI" else ".txt"
    files = sorted(os.path.join(d, f) for d, _, fs in os.walk(src) for f in fs if f.endswith(ext))
    n_fail = 0
    for p in files:
        try:
            rows = parse_ent(p) if ext == ".ent" else parse_text(p)
        except Exception as e:
            print("failed:", os.path.basename(p), e)
            n_fail += 1
            continue
        name = os.path.basename(p)[:-4] if ext == ".ent" else os.path.basename(p).replace(".txt", "_events.csv")
        with open(os.path.join(out, f"{name}.ent_events.csv" if ext == ".ent" else name), "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["time", "event", "duration"])
            w.writerows([(t, ev, "" if d is None else d) for t, ev, d in rows])
    print(f"{len(files) - n_fail}/{len(files)} exports parsed -> {out}")


if __name__ == "__main__":
    main()
