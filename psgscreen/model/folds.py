"""Exam roles in fold k of the nested 10-fold cross-validation of the event detectors and the fusion step.

Development exams of fold k are held out ("holdout"), fold k+1 selects the model ("val") and the other
development exams are trained on ("train"). Locked internal test exams and all external exams are "test".
"""
import numpy as np
import pandas as pd

from psgscreen import paths


def exam_roles(k):
    """{uuid: train | val | holdout | test} of fold k."""
    ros = pd.read_csv(paths.ROSTER, encoding="utf-8-sig")
    test = ((ros.split == "test") | ~ros.center.isin(paths.INTERNAL)).values
    fold = pd.to_numeric(ros.fold, errors="coerce").values
    kv = (k + 1) % paths.N_FOLDS
    role = np.where(test, "test", np.where(fold == k, "holdout", np.where(fold == kv, "val", "train")))
    return dict(zip(ros.uuid.str.lower(), role))
