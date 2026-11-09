"""Event vocabulary shared by the detectors, tables and figures."""

EXCLUDED = "anychinemg activity"       # not used by any downstream model (too few positive windows in some folds)
RBD_RELATED = {"limb movement", "rbd", "candidate signal"}     # all other events are OSA-related
NAMES = {"rbd": "RBD symptom", "candidate signal": "indeterminate event",
         "spo2 threshold event": "SpO2 threshold event"}
_RAW = {v: k for k, v in NAMES.items()}


def name(e):
    """Display name of an annotation event."""
    return NAMES.get(e, e)


def raw(display_name):
    """Annotation event of a display name."""
    return _RAW.get(display_name, display_name)
