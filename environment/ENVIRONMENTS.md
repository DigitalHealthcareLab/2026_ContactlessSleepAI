# Environments

Versions of the environments that produced the published results (Linux, CentOS 7.9; 2× Xeon Platinum 8253;
2× Quadro RTX 8000). `scripts/run_all.sh` names each interpreter by its variable.

| Variable | Used for | Python and packages |
|---|---|---|
| `PY` | window table, fusion, exam features, disorder and severity classifiers, evaluation, Figures 3, 5, 6, supplementary figures, Table 1, scoring-export parsing, denseflow conversion | Python 3.10.8 — `requirements-analysis.txt` |
| `PYB` | report parsing and matching, clinical metrics, Figure 4 | Python 3.12.4 — `requirements-reports.txt` |
| `PYP` | audio Mel spectrogram, motion features | Python 3.10.8 — `requirements-preprocess.txt`; FFmpeg command-line tool |
| `PYF` | optical flow, Hospitals A and B (PyAV decoder) | Python 3.10.12, numpy 2.2.6, PyAV, lmdb 1.7.5, zstandard 0.24.0, OpenCV 4.13.0 built with CUDA 12.4; FFmpeg (`ffprobe`) |
| `PYFC` | optical flow, Hospital C (OpenCV decoder) | Python 3.7.12, numpy 1.21.6, lmdb 1.3.0, zstandard 0.21.0, OpenCV 4.3.0 built with CUDA 10 |
| `PYV` | ViT input arrays, ViT event detectors | Python 3.10.8 — `requirements-vit.txt` (CUDA 12.4); pretrained weights `vit_small_patch16_224.augreg_in21k` from the timm / Hugging Face hub |

Part of Hospital A was processed with denseflow (C++; Docker image `jinyong/denseflow:batch2048_umask_fps5`,
OpenCV 4.5.2, CUDA 11.3.1; Farneback, options `-s=1 -b=20 -cf -v -a=farn`).

Notes
* The ViT detectors were trained on one Quadro RTX 8000 per fold and modality (1.5–3 h per epoch on a shared GPU,
  3 epochs); GPU training is not bit-reproducible.
* XGBoost models are trained on the CPU (`tree_method=hist`) with `nthread` set explicitly; SHAP values are
  XGBoost `pred_contribs` (the `shap` package is not used).
* Figure 4 is drawn with matplotlib 3.8.4 (`PYB`) and the other figures with matplotlib 3.10.8 (`PY`); other
  matplotlib versions give the same content with slightly different rendering.
* `huggingface_hub` is pinned below 1.0: timm 1.0.19 does not import with later versions.
* `torch`, `torchvision` and `torchaudio` are the CUDA 12.4 builds (`+cu124`) from the PyTorch index, e.g.
  `pip install -r environment/requirements-vit.txt --extra-index-url https://download.pytorch.org/whl/cu124`.
* On CentOS 7 the analysis environment needs a newer `libgcc_s` than the system one for the pyarrow wheel that
  scikit-learn loads (e.g. `LD_LIBRARY_PATH` pointing to a conda `lib` directory).
