<div align="center">

# Contactless audiovisual screening for obstructive sleep apnea and REM sleep behavior disorder from overnight recordings

Jin Yong Kim<sup>1,†</sup>, Bo Kyu Choi<sup>2,†</sup>, Jungyon Yum<sup>3</sup>, Doyeon Kook<sup>2</sup>, Joonho Kim<sup>4</sup>,
Soomi Cho<sup>4</sup>, Woo-Seok Ha<sup>4</sup>, Wonwoo Lee<sup>3</sup>, Min Kyung Chu<sup>4</sup>, Won-Joo Kim<sup>2</sup>,
Kyung Min Kim<sup>4,\*</sup>, Yu Rang Park<sup>1,\*</sup>

<sup>1</sup>Department of Biomedical Systems Informatics, Yonsei University College of Medicine, Seoul, Republic of Korea<br>
<sup>2</sup>Department of Neurology, Gangnam Severance Hospital, Yonsei University College of Medicine, Seoul, Republic of Korea<br>
<sup>3</sup>Department of Neurology, Yongin Severance Hospital, Yonsei University College of Medicine, Yongin, Republic of Korea<br>
<sup>4</sup>Department of Neurology, Severance Hospital, Yonsei University College of Medicine, Seoul, Republic of Korea<br>
<sup>†</sup>Equal contribution &nbsp;·&nbsp; <sup>\*</sup>Co-corresponding authors

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg)
![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4-EE4C2C.svg)
![XGBoost 3.2](https://img.shields.io/badge/XGBoost-3.2-189FDD.svg)

</div>

## Overview

Polysomnography (PSG) is the diagnostic standard for sleep disorders, but it requires an overnight stay in a sleep
laboratory with many sensors attached to the body. This repository contains the complete analysis code of a
two-stage framework that screens for **obstructive sleep apnea (OSA)** and **REM sleep behavior disorder (RBD)**
from the **video and audio** recorded during the night. No body-attached sensor signal, demographic variable or
questionnaire score is a model input.

- **Stage 1 — sleep-event detection.** Every 30-second window of the night is scored for 13 PSG-annotated sleep
  events (apneas and hypopneas, oxygen desaturation, arousal, snoring, limb movement, RBD symptom and others) by
  two Vision Transformers: one reads the optical flow of the infrared video, the other the log-Mel spectrogram of
  the audio. Their outputs are combined per event with a learned late-fusion weight.
- **Stage 2 — disorder classification.** The whole-night event probabilities are summarised per event (mean,
  maximum and standard deviation), and XGBoost classifiers detect abnormal sleep (OSA and/or RBD), then OSA and RBD,
  and grade OSA and RBD severity.
- **Event-based explanations.** The stage-2 inputs are sleep events, so SHAP values explain each prediction in
  terms of the events a clinician scores.
- **Whole night, no pre-selection.** The framework runs on the unselected recording; it needs no sleep staging and
  no selection of event-containing segments.

Recordings of 1,558 patients from two hospitals were used for development (1,402) and internal held-out testing
(156); 402 patients from a third hospital, never used for training, model selection or threshold choice, form the
external validation set.

## Results at a glance

Area under the ROC curve (AUROC) with exam-level bootstrap 95 % CI (4,000 resamples). Each held-out patient is
scored by the mean of the 10 cross-validation fold models.

| Task | Internal held-out test | External validation |
|---|---|---|
| Abnormal sleep (OSA and/or RBD) vs normal | **0.9695** (0.9423–0.9883) | **0.9184** (0.8855–0.9469) |
| OSA, among patients with abnormal sleep | **0.9519** (0.9009–0.9863) | **0.9610** (0.9258–0.9878) |
| RBD, among patients with abnormal sleep | **0.8991** (0.8407–0.9475) | **0.8050** (0.7429–0.8637) |
| Severe OSA (AHI ≥ 30), among patients with OSA | 0.8748 (0.7995–0.9400) | 0.8927 (0.8546–0.9266) |
| RBD severity (RBDSQ ≥ development median), among patients with RBD | 0.6296 (0.3778–0.8667) | 0.4971 (0.3059–0.6853) |

Patients evaluated (positives): abnormal sleep 156 (114) internal and 402 (346) external; the OSA and RBD
classifiers act on patients with abnormal sleep, 114 (OSA 104, RBD 54) internal and 346 (OSA 327, RBD 56)
external; severe OSA 104 (42) and 327 (138); RBD severity 21 (12) and 37 (18), the patients with an RBD Screening
Questionnaire (RBDSQ) score.

- **Event detection.** Fusing the two modalities raised the mean window-level AUROC over the 13 events from
  0.8015 (audio only) and 0.8375 (video only) to 0.8694 (internal held-out test, mean of the 10 fold detectors).
- **Recording time.** Scored on the first *t* minutes of the night, the AUROC became non-inferior to the
  full-night AUROC (margin 0.05) after 1 h 45 min for OSA, 3 h 00 min for abnormal sleep and 3 h 25 min for RBD
  (Hospital A internal held-out test).

## Method

### Cohort

Adults (18 years or older) who underwent in-laboratory diagnostic PSG with synchronized video and audio.

| Hospital | Code | Role | Patients |
|---|---|---|---|
| Sinchon Severance Hospital (Hospital A) | `SC` | development and internal held-out test | 1,478 |
| Gangnam Severance Hospital (Hospital B) | `GN` | development and internal held-out test | 80 |
| Yongin Severance Hospital (Hospital C) | `YI` | external validation only | 402 |

Patients per set and diagnostic class:

| Set | Patients | Normal | OSA only | RBD only | OSA and RBD |
|---|---:|---:|---:|---:|---:|
| Development (10 cross-validation folds) | 1,402 | 396 | 557 | 95 | 354 |
| Internal held-out test | 156 | 42 | 60 | 10 | 44 |
| External validation | 402 | 56 | 290 | 19 | 37 |

Labels come from the PSG report: OSA = apnea–hypopnea index (AHI) ≥ 5 events/h; RBD = definite or probable RBD in
the report impression; abnormal sleep = OSA and/or RBD. The internal held-out test set is a stratified random 10 %
of each hospital and class, fixed once and used for no model decision.

### Stage 1 — sleep-event detection

- **Windows and targets.** The night is cut into non-overlapping 30-second windows. A window is positive for an
  event when the event covers at least 50 % of the window or at least 50 % of the event lies in the window;
  windows with a smaller overlap, or within 1 s of an event, are indeterminate for that event and ignored;
  13 event types: obstructive apnea, central apnea, mixed apnea, hypopnea, obstructive hypopnea, arousal, oxygen
  desaturation, SpO2 threshold event, oximeter event, limb movement, snoring, RBD symptom and indeterminate event.
  1,219,337 windows (713,808 with at least one event) train the detectors and the fusion weights.
- **Video input.** Dense optical flow of the infrared video (5 flow fields per second) as three channels (flow-x,
  flow-y, flow magnitude) in 30 one-second bins of 64 × 64 pixels, upsampled to 128 × 128; tubelets of
  6 s × 16 × 16 pixels give 5 × 8 × 8 = 320 tokens.
- **Audio input.** Log-Mel spectrogram, 128 Mel bands × 30 s; patches of 16 Mel bands × about 1 s give
  8 × 30 = 240 tokens.
- **Detectors.** A separate ViT-S/16 per modality, pre-trained on ImageNet-21k (timm
  `vit_small_patch16_224.augreg_in21k`); the CLS token feeds one logit per event.
- **Late fusion.** For each event, p<sub>fused</sub> = w · p<sub>audio</sub> + (1 − w) · p<sub>video</sub>, with
  the audio weight w chosen on a 0.005 grid to maximise the validation AUROC of the fold.

<details>
<summary>Training details</summary>

- Loss: binary cross-entropy per event on the windows with a definite label for that event (indeterminate labels
  are ignored), positive-class weight (n<sub>negative</sub> / n<sub>positive</sub>)<sup>0.25</sup>.
- Optimiser: AdamW, learning rate 1 × 10<sup>−4</sup>, weight decay 0.05 (not on biases, norms, position
  embeddings and the CLS token); warm-up over 0.1 epoch, then cosine decay over a 5-epoch schedule of which 3
  epochs are run; batch 64; gradient clipping 1.0; drop path 0.1; mixed precision (fp16).
- Augmentation: video left–right flip (with the sign of flow-x) and shifts of up to 4 pixels; audio two frequency
  masks (≤ 16 bands) and two time masks (≤ 32 frames).
- Inputs are standardised with the mean and SD of 20,000 training windows of the fold. The epoch with the highest
  mean validation AUROC over the 13 events is kept.
- The detectors have a 14th output (`anychinemg activity`) that no downstream model uses (too few positive windows
  in some folds).

</details>

### Stage 2 — disorder classification

- **Exam features.** Mean, maximum and SD over the night of each event's fused probability: 13 × 3 = 39 features.
- **Hierarchical classifier (XGBoost).** An abnormal-sleep classifier separates abnormal from normal sleep in all
  patients; two independent binary heads, OSA and RBD, are trained on patients with abnormal sleep, so a patient
  can be positive for both. All three use the same 39 features.
- **Severity (XGBoost, same features).** Severe OSA (AHI ≥ 30 vs 5 ≤ AHI < 30) among patients with OSA; RBD
  severity (RBDSQ at or above the development median vs below) among patients with RBD and an RBDSQ score.
- **Class weighting and hyper-parameters.** `scale_pos_weight` = n<sub>negative</sub> / n<sub>positive</sub> of
  the training fold (its square root for the severity classifiers). The abnormal-sleep classifier uses fixed
  parameters (max_depth 4, eta 0.05, subsample 0.8, colsample_bytree 0.8, lambda 1); the OSA and RBD heads use the
  best of 6,912 configurations by development out-of-fold AUROC. Up to 1,000 boosting rounds (severity: 600),
  stopped early after 40 rounds without improvement on the validation fold. Seed 42 throughout.

### Validation design

- **Nested 10-fold cross-validation on the development set.** In fold *k* the detectors and the classifiers are
  trained on eight folds, select the epoch or stop early on fold *k* + 1, and leave fold *k* out. Every development
  patient therefore receives out-of-fold features and predictions.
- **Held-out and external patients** are scored by the mean of the 10 fold models. Operating thresholds are the
  development out-of-fold Youden thresholds, applied unchanged to every cohort.
- **The external hospital** contributes no training data and takes no part in model selection or threshold choice.

## Results

### Sleep-event detection

<p align="center"><img src="docs/figures/fig3_event_detection.png" width="100%" alt="Event-detection AUROC by modality, late-fusion weights and event-locked signals"></p>

(a) Window-level AUROC of the audio-only, video-only and fused detectors on the internal held-out test (bar = mean,
error bar = SD over the 10 fold detectors). (b) Late-fusion weight of the audio and video detectors per event.
(c, d) Audio fluctuation and 95th-percentile flow magnitude around the event midpoint for the three most
audio-weighted and the three most video-weighted events, against windows without the event.

Audio carried most of the weight for snoring (84 %), oximeter events (77 %) and hypopnea (69 %); video carried most
of it for limb movement (88 %), central apnea (84 %) and RBD symptoms (77 %).

### Sleep disorder detection

<p align="center"><img src="docs/figures/fig4_disorder_detection.png" width="100%" alt="ROC curves, calibration and decision curves for abnormal sleep, OSA and RBD"></p>

(a) ROC curves with 95 % confidence bands on the internal held-out test and the external validation set; the
markers are the development-derived thresholds. (b) Calibration curves with calibration slope and
calibration-in-the-large (CITL). (c) Decision curves at a reference prevalence.

### Interpretability

<p align="center"><img src="docs/figures/fig5_shap.png" width="100%" alt="Event-level SHAP values of the three classifiers and three patient-level waterfall plots"></p>

(a) Event-level SHAP values (log-odds) of the abnormal-sleep, OSA and RBD classifiers on the internal held-out
test; colour = the patient's level of the event, orange labels = OSA-related events, blue labels = RBD-related
events. (b) SHAP waterfall plots of three patients of the internal held-out test (normal, OSA, RBD).

Oxygen desaturation, obstructive apnea and RBD symptoms contributed most to the abnormal-sleep predictions, oxygen
desaturation and obstructive hypopnea and apnea to the OSA predictions, and RBD symptoms and indeterminate events to
the RBD predictions.

### How much of the night is needed

<p align="center"><img src="docs/figures/fig6_recording_time.png" width="100%" alt="AUROC and individual predicted probabilities as a function of cumulative recording time"></p>

AUROC (coloured, left axis) when each patient of the Hospital A internal held-out test is scored on the first
*t* minutes of the recording, in 5-minute steps; the dotted line marks the earliest time from which the AUROC is
non-inferior to the full night. Black lines (right axis) are the predicted probabilities of representative
patients; grey bands are the 30-minute periods in the top quartile of the share of patients in REM sleep.

## Intended use and limitations

- The framework is meant for **screening**, to identify people who may benefit from a full sleep evaluation. It
  does not replace diagnostic PSG; a definitive RBD diagnosis requires the PSG electromyogram, which video and
  audio cannot observe.
- All recordings come from hospital sleep laboratories in South Korea. Performance at home and in other
  populations is untested.
- RBD discrimination drops at the external hospital (AUROC 0.8991 → 0.8050), and at the development-derived
  threshold the external RBD sensitivity is 0.0893 (5 of 56). Calibration also differs from ideal in both cohorts.
  Probabilities and thresholds should be recalibrated before use at a new site.
- The severity grading of RBD rests on few patients with an RBDSQ score and on a screening questionnaire, not on
  an objective measure; its external AUROC does not differ from chance.

## Repository structure

```
.
├── psgscreen/
│   ├── preprocessing/   recordings → per-clip databases (optical flow, motion records, log-Mel spectrogram);
│   │                    scoring-export parsers
│   ├── cohort/          PSG report parsing and matching, cohort, labels and splits, Table 1
│   ├── dataset/         30-s window table, clip timing, ViT input arrays, hypnogram, event-locked traces
│   ├── model/           ViT event detectors, late fusion, exam features, disorder and severity classifiers,
│   │                    hyper-parameter search, cumulative-time scores, SHAP, alternative classifiers
│   ├── evaluation/      detection and disorder performance, clinical metrics (operating points, calibration,
│   │                    decision curves), model comparisons, supplementary tables
│   ├── figures/         main and supplementary figures
│   ├── paths.py         file locations
│   └── events.py        event names
├── configs/             selected hyper-parameters: main model, audio_only/ and video_only/ inputs, alt/ classifiers
├── scripts/run_all.sh   every step in execution order
├── environment/         environments and package versions
└── docs/figures/        figures shown in this README
```

Every module starts with a description of what it computes and how; `python -m psgscreen.<module> -h` lists its
options.

## Data

Patient data (videos, audio, PSG reports, scoring exports and cohort tables) and the trained model weights are not
part of this repository. The code reads its inputs from `PSG_DATA` and writes everything else to `PSG_WORK`.

| `PSG_DATA/…` | Content |
|---|---|
| `video/<centre>/…/<exam folder>/*.avi` | PSG video files of an exam (the folder name contains the exam uuid; files numbered `<stem>_<NNNN>.avi`) |
| `annotation_exports/<centre>/` | Scoring exports: NeuroWorks text exports (`.txt`, Hospitals A and B), Xltek `.ent` / `.erd` (Hospital C) |
| `reports/<centre>/` | PSG report files (`.doc` / `.docx`) |
| `cohort/exam_inventory.csv` | One row per exam with synchronized video and audio: `uuid, exam_id, center, exam_date, exam_month` |
| `cohort/exclusions.csv` | `uuid, reason`: recordings that failed the quality review and PAP-treated or split-night studies |
| `cohort/splits.csv` | `uuid, split, fold`: internal held-out test (`test`) or development (`dev`, folds 0–9), fixed for all analyses |
| `cohort/yi_questionnaires/` | Hospital C questionnaire exports (`registry.xlsx`, `rbdsq_scores.xlsx`) for RBDSQ scores missing from the report |

Centre codes: `SC` = Hospital A, `GN` = Hospital B, `YI` = Hospital C. Preprocessing writes the per-clip databases
to `PSG_DATA/lmdb/<kind>_<centre>.lmdb` (kind: flow, motion, magspec, flow_rep, audio) and the parsed event lists to
`PSG_DATA/annotations/<centre>/`.

## Installation

The published results were produced with six Python environments, one per group of steps. Their versions are in
[`environment/ENVIRONMENTS.md`](environment/ENVIRONMENTS.md) and the `environment/requirements-*.txt` files.

| Variable | Used for | Requirements |
|---|---|---|
| `PY` | windows, fusion, exam features, classifiers, evaluation, most figures | `requirements-analysis.txt` (Python 3.10) |
| `PYB` | report parsing, clinical metrics, Figure 4 | `requirements-reports.txt` (Python 3.12) |
| `PYP` | audio and motion features | `requirements-preprocess.txt` (Python 3.10) |
| `PYF` | optical flow, Hospitals A and B | OpenCV built with CUDA, PyAV (see `ENVIRONMENTS.md`) |
| `PYFC` | optical flow, Hospital C | OpenCV built with CUDA, Python 3.7 (see `ENVIRONMENTS.md`) |
| `PYV` | ViT input arrays and event detectors | `requirements-vit.txt` (Python 3.10, CUDA 12.4) |

For example, the analysis and ViT environments (the CUDA 12.4 builds of PyTorch come from the PyTorch index):

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r environment/requirements-analysis.txt

python3.10 -m venv .venv_vit
.venv_vit/bin/pip install -r environment/requirements-vit.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

## How to run

```bash
git clone https://github.com/DigitalHealthcareLab/2026_ContactlessSleepAI.git
cd 2026_ContactlessSleepAI
export PSG_DATA=/path/to/data      # inputs (see "Data")
export PSG_WORK=/path/to/work      # everything the code writes
PY=… PYB=… PYP=… PYF=… PYFC=… PYV=… bash scripts/run_all.sh
```

`scripts/run_all.sh` lists every step in execution order with the interpreter it runs under. Each step reads only
files written by earlier steps, so steps can also be run one at a time with their command from that script, e.g.

```bash
python -m psgscreen.model.vit --modality video --fold 0     # one detector (GPU)
python -m psgscreen.model.disorder                          # two-stage disorder classifier
python -m psgscreen.figures.event_detection                 # Figure 3
```

| Step | Modules |
|---|---|
| 1. Per-clip data: optical flow, motion records, log-Mel spectrogram, parsed scoring exports | `preprocessing.optical_flow`, `.motion`, `.audio`, `.annotations` |
| 2. Cohort: report values, labels, splits | `cohort.reports`, `.report_table`, `.roster` |
| 3. Windows, ViT inputs, hypnogram, event-locked traces, Table 1 | `dataset.clip_times`, `.windows`, `.vit_inputs`, `.hypnogram`, `.event_traces`; `cohort.table1` |
| 4. Detectors (per fold and modality), late fusion, exam features, classifiers | `model.vit`, `.fusion`, `.exam_features`, `.hparam_search`, `.disorder`, `.severity`, `.trajectory`, `.shap_values`, `.alt_classifiers` |
| 5. Evaluation | `evaluation.detection`, `.metrics`, `.clinical`, `.model_comparison`, `.supp_tables` |
| 6. Figures | `figures.*` |

`PSG_ROSTER` overrides the cohort file (default `PSG_WORK/cohort/roster.csv`). Optical flow, audio features and the
ViT detectors use a GPU (one detector took 1.5–3 h per epoch on a Quadro RTX 8000); all XGBoost models run on the
CPU. The video input arrays take about 370 KB per window (about 0.6 TB for the cohort). Part of Hospital A was
processed with the denseflow tool (`preprocessing/denseflow_lmdb.py`; settings in `scripts/run_all.sh`).

## Figures and tables

Figures are written to `PSG_WORK/figures/<module>/` and tables to `PSG_WORK/results/`. Panel letters and
multi-panel layouts were assembled by hand.

| Item | Module | Upstream |
|---|---|---|
| Figure 1 (study design), Figure 2 (flowchart) | — (drawn by hand) | |
| Figure 3, event detection (a–d) | `figures.event_detection` | `evaluation.detection`, `model.fusion`, `dataset.event_traces` |
| Figure 4, ROC, calibration and decision curves | `figures.disorder_detection` | `model.disorder`, `evaluation.clinical` |
| Figure 5, SHAP (a, b) | `figures.shap` | `model.shap_values` |
| Figure 6, cumulative recording time | `figures.time_course` | `model.trajectory`, `dataset.hypnogram` |
| Table 1 | `cohort.table1` | `cohort.report_table`, `dataset.clip_times` |
| Event detection per event (text) | `evaluation.detection` → `detection_table.csv` | |
| Disorder and severity classification (text) | `evaluation.metrics` → `table_disorder.csv`, `table_severity.csv` | `model.disorder`, `model.severity` |
| Supplementary Figure S1, severity ROC | `figures.disorder_detection --severity` | `model.severity` |
| Supplementary Figure S2, severity SHAP | `figures.severity_shap` | `model.severity` |
| Supplementary Figure S3, group-mean cumulative probability | `figures.group_probability` | `model.trajectory`, `evaluation.clinical` |
| Supplementary Tables S1 (recording set-up), S2 (event definitions) | — (compiled by hand) | |
| Supplementary Table S3, operating points | `evaluation.supp_tables operating` | `evaluation.clinical` |
| Supplementary Table S4, input modality and fusion strategy | `evaluation.supp_tables fusion` | `model.disorder --variant`, `evaluation.model_comparison fusion` |
| Supplementary Tables S5, S6, alternative classifiers | `evaluation.supp_tables alt` | `model.alt_classifiers`, `evaluation.model_comparison alt` |

## Citation

The paper is not yet published; citation details will be added here on publication.

## License

Released under the MIT License; see [`LICENSE`](LICENSE).
