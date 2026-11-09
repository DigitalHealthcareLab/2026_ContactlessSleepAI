"""Exam-level features: mean, maximum and SD over the night of each event's window probability (13 events x 3 = 39).

For fold k the window probabilities come from the fold-k detectors (model.vit) and fusion weights (model.fusion),
so the features of a development exam in fold k are out of sample. Only events with a fusion weight in every fold
are used. Windows of exams without event annotations are included (they are scored, not trained on).

  --combine fused   p = w * p_audio + (1 - w) * p_video with the fold's per-event weight w (main model)
            audio   audio detector only
            video   video detector only
  --dump_probs      also write each locked-test exam's fused window probabilities (input of model.trajectory)

    python -m psgscreen.model.exam_features [--combine fused] [--dump_probs]
        -> features[_audio|_video]/f<k>/<centre>_features.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from psgscreen import paths
from psgscreen.dataset import vit_inputs as vi
from psgscreen.dataset.store import CLIP_SEC
from psgscreen.model.fusion import window_probs

STATS = ("mean_prob", "max_prob", "std_prob")


def aggregate_exam(exam_id, probs, events, n_windows, stride_sec=CLIP_SEC):
    row = {"exam_id": exam_id, "n_windows": n_windows, "duration_sec": n_windows * stride_sec}
    for i, ev in enumerate(events):
        p = probs[:, i]
        row[f"{ev}__mean_prob"] = float(p.mean())
        row[f"{ev}__max_prob"] = float(p.max())
        row[f"{ev}__std_prob"] = float(p.std())
    return row


def common_events():
    """Events with a fusion weight in every fold."""
    sets = [set(json.load(open(f"{paths.FUSION}/f{k}/fusion_weights.json"))) for k in range(paths.N_FOLDS)]
    return sorted(set.intersection(*sets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combine", default="fused", choices=["fused", "audio", "video"])
    ap.add_argument("--dump_probs", action="store_true")
    a = ap.parse_args()
    out_root = paths.FEATURES if a.combine == "fused" else f"{paths.FEATURES}_{a.combine}"
    ros = pd.read_csv(paths.ROSTER, encoding="utf-8-sig")
    u2exam = {u.lower(): str(e) for u, e in zip(ros.uuid, ros.exam_id)}
    test = set(ros.loc[(ros.split == "test") & ros.center.isin(paths.INTERNAL), "uuid"].str.lower())
    rows = {c: vi.load_rows(c) for c in paths.CENTERS}
    names = [str(e) for e in rows[paths.CENTERS[0]]["event_names"]]
    events = common_events()
    idx = [names.index(e) for e in events]
    print(f"events used: {len(events)}")

    for k in range(paths.N_FOLDS):
        wts = json.load(open(f"{paths.FUSION}/f{k}/fusion_weights.json"))
        thr = json.load(open(f"{paths.FUSION}/f{k}/fused_thresholds.json"))
        w = np.array([wts[e][0] for e in events], np.float32)[None, :]
        out = f"{out_root}/f{k}"
        os.makedirs(out, exist_ok=True)
        for c in paths.CENTERS:
            if a.combine == "fused":
                P = w * window_probs("audio", k, c)[:, idx] + (1 - w) * window_probs("video", k, c)[:, idx]
            else:
                P = window_probs(a.combine, k, c)[:, idx]
            recs = []
            for u, s0, s1 in vi.exam_blocks(rows[c]["uuid"]):
                if u not in u2exam:
                    continue
                recs.append(aggregate_exam(u2exam[u], P[s0:s1], events, n_windows=s1 - s0))
                if a.dump_probs and a.combine == "fused" and u in test:
                    d = f"{paths.WINDOW_PROBS}/f{k}"
                    os.makedirs(d, exist_ok=True)
                    np.savez_compressed(f"{d}/{u2exam[u].replace('/', '_')}.npz", probs=P[s0:s1].astype(np.float32),
                                        events=np.array(events, dtype="<U64"),
                                        thr=np.array([float(thr[e]) for e in events], np.float32),
                                        stride_sec=np.float32(CLIP_SEC), exam_id=u2exam[u])
            pd.DataFrame(recs).to_csv(f"{out}/{c}_features.csv", index=False, encoding="utf-8-sig")
        json.dump(dict(fold=k, events=events, combine=a.combine), open(f"{out}/meta.json", "w"), indent=2)
        print(f"  fold {k} done", flush=True)


if __name__ == "__main__":
    main()
