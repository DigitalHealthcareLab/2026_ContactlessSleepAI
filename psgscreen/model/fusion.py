"""Late fusion of the audio and video detectors, per event and fold.

On the validation windows of fold k (development exams of Hospitals A and B in fold k+1, the exams that chose the
detectors' epoch) the audio weight w of p_fused = w * p_audio + (1 - w) * p_video is the first maximiser of the
validation AUC on a 0.005 grid, for each event separately (windows with an indeterminate label excluded). The
Youden threshold of the fused probability is stored with it. Events without positive or negative validation
windows in the fold get no weight.

    python -m psgscreen.model.fusion --fold 0
        -> models/fusion/f<k>/{fusion_weights.json, fused_thresholds.json, fusion_report.csv}
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from psgscreen import paths
from psgscreen.dataset import vit_inputs as vi
from psgscreen.model.folds import exam_roles

GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.005), 3)


def window_probs(modality, k, center):
    """Fold-k detector probabilities of every window of a centre (row order of dataset.vit_inputs)."""
    return np.load(f"{paths.VIT_MODELS}/{modality}_f{k}/preds/{center}.npy").astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    k = ap.parse_args().fold
    out = f"{paths.FUSION}/f{k}"
    os.makedirs(out, exist_ok=True)
    val = [u for u, r in exam_roles(k).items() if r == "val"]
    PA, PV, Y, events = [], [], [], None
    for c in paths.INTERNAL:
        r = vi.load_rows(c)
        m = np.isin(r["uuid"], val)
        PA.append(window_probs("audio", k, c)[m]); PV.append(window_probs("video", k, c)[m]); Y.append(r["y_multi"][m])
        events = [str(e) for e in r["event_names"]]
    PA, PV, Y = np.concatenate(PA), np.concatenate(PV), np.concatenate(Y)
    print(f"fold {k}: {len(Y):,} validation windows")

    weights, thresholds, rows = {}, {}, []
    for i, e in enumerate(events):
        keep = Y[:, i] >= 0
        y = Y[keep, i]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        pa, pv = PA[keep, i], PV[keep, i]
        a64, v64 = pa.astype(np.float64), pv.astype(np.float64)       # the weight search runs in float64
        aucs = np.array([roc_auc_score(y, w * a64 + (1 - w) * v64) for w in GRID])
        j = int(np.argmax(aucs)); w = float(GRID[j])
        fpr, tpr, thr = roc_curve(y, w * pa + (1 - w) * pv)           # threshold of the float32 fused probability
        weights[e] = [w, round(1 - w, 3)]
        thresholds[e] = float(thr[int(np.argmax(tpr - fpr))])
        rows.append(dict(event=e, n_pos=int(y.sum()), n=len(y), w_audio=w, auc_audio=roc_auc_score(y, pa),
                         auc_video=roc_auc_score(y, pv), auc_fused=float(aucs[j]), thr_fused=thresholds[e]))
        print(f"  {e:28s} w_audio={w:.3f}  AUC fused {aucs[j]:.4f}", flush=True)
    json.dump(weights, open(f"{out}/fusion_weights.json", "w"), indent=2)
    json.dump(thresholds, open(f"{out}/fused_thresholds.json", "w"), indent=2)
    pd.DataFrame(rows).to_csv(f"{out}/fusion_report.csv", index=False, encoding="utf-8-sig")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
