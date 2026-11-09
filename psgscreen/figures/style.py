"""Colours, line styles and fonts shared by all figures (Okabe-Ito based, colour-vision-deficiency safe)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

HEAD = {"Normal": "#56B4E9", "Abnormal": "#CC79A7", "OSA": "#E69F00", "RBD": "#0072B2", "Severe OSA": "#D55E00"}
COHORT = {"Internal": "#2f2f2f", "External": "#009E73"}
COHORT_LS = {"Internal": "-", "External": (0, (5, 2))}
MODALITY = {"Audio": "#D55E00", "Video": "#166082"}
FUSED = "#54278f"
NEUTRAL = "#8f8d87"
DISPLAY_CENTER = {"INT": "Hospital A", "GN": "Hospital B", "YI": "Hospital C"}


def use_fonts():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Nimbus Sans", "Liberation Sans", "FreeSans", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.titleweight"] = "normal"
    plt.rcParams["figure.titleweight"] = "normal"


def tag_dpi(path, dpi=600):
    """Re-save a PNG with a resolution tag only (pixels unchanged): panels are drawn at twice their final size
    and 300 dpi, so a 600-dpi tag makes them the size of their box in the hand-made layout."""
    with Image.open(path) as im:
        im.load()
        im.save(path, dpi=(dpi, dpi))
