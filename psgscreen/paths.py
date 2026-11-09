"""File locations used by every module.

Two roots, set by environment variables:
  PSG_DATA  inputs derived from the recordings and the PSG reports (feature databases, annotation exports,
            report files, exam inventory)
  PSG_WORK  everything this code writes (window dataset, ViT inputs, cohort tables, models, features,
            predictions, results and figures)
"""
import os

DATA = os.environ.get("PSG_DATA", "data")
WORK = os.environ.get("PSG_WORK", "work")

INTERNAL = ("INT", "GN")      # development + locked internal test
EXTERNAL = ("YI",)            # external validation
CENTERS = INTERNAL + EXTERNAL
N_FOLDS = 10
SEED = 42


def lmdb(kind, center):
    """Per-clip feature database: kind in {flow, motion, magspec, flow_rep, audio}."""
    return f"{DATA}/lmdb/{kind}_{center}.lmdb"


def clip_times(center):
    """Clip-time table: {uuid: (n_clips, 3) [start s, sample interval s, valid samples]}."""
    return f"{WORK}/alignment/clip_times_{center}.npz"


# inputs
VIDEO = f"{DATA}/video"                        # {center}/.../<exam folder named with the uuid>/*.avi|mp4
ANNOTATION_EXPORTS = f"{DATA}/annotation_exports"   # {center}/  scoring exports (.txt, .ent)
ANNOTATIONS = f"{DATA}/annotations"            # {center}/  parsed event lists (preprocessing.annotations)
REPORTS = f"{DATA}/reports"                    # PSG report files (.doc / .docx)
INVENTORY = f"{DATA}/cohort/exam_inventory.csv"
EXCLUSIONS = f"{DATA}/cohort/exclusions.csv"
SPLITS = f"{DATA}/cohort/splits.csv"
QUESTIONNAIRE = f"{DATA}/cohort/yi_questionnaires"

# cohort
COHORT = f"{WORK}/cohort"
ROSTER = os.environ.get("PSG_ROSTER", f"{COHORT}/roster.csv")
REPORTS_PARSED = f"{COHORT}/reports_parsed.parquet"
REPORT_TABLE = f"{COHORT}/report_table.csv"
DEMOGRAPHICS = f"{COHORT}/demographics.csv"   # per exam: uuid, center, patient_id, age, sex, bmi, ahi, tib_h, tst_h

# dataset
WINDOWS = f"{WORK}/windows"                    # 30-s windows: event targets and record keys
VIT_INPUTS = f"{WORK}/vit_inputs"              # {center}/  per-window video and audio input arrays
TRACES = f"{WORK}/traces"                      # event-locked raw signal (event-detection figure c, d)
HYPNOGRAM = f"{WORK}/hypnogram/hypnogram_aligned.csv"

# models and features
VIT_MODELS = f"{WORK}/models/vit"              # {modality}_f{fold}/
FUSION = f"{WORK}/models/fusion"
FEATURES = f"{WORK}/features"                  # + features_audio, features_video (single-modality inputs)
WINDOW_PROBS = f"{WORK}/window_probs"
HPARAM = f"{WORK}/hparam"

# predictions, results, figures
PREDICTIONS = f"{WORK}/predictions"
RESULTS = f"{WORK}/results"
FIGURES = f"{WORK}/figures"

CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def configs(variant=None):
    """Hyper-parameter folder of the main model (variant None) or of the audio_only / video_only inputs."""
    return CONFIGS if variant is None else f"{CONFIGS}/{variant}"


SEVERITY_PARAMS = {"osa_severity": f"{CONFIGS}/severity_osa.json",
                   "rbd_severity": f"{CONFIGS}/severity_rbd.json"}


def disorder_predictions(variant=None):
    """Disorder predictions of the main model, or of the audio_only / video_only exam features."""
    return f"{PREDICTIONS}/{variant or 'disorder'}.csv"


def severity_predictions(target):
    return f"{PREDICTIONS}/severity_{target}.csv"


def alt_predictions(model):
    return f"{PREDICTIONS}/alt/{model}.csv"
