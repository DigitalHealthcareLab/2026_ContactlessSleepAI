# psgscreen — video and audio screening for OSA and RBD from PSG recordings

Analysis code of the manuscript. From the synchronized video and audio of a polysomnography (PSG) night, two
Vision Transformer event detectors (one for the optical flow of the video, one for the log-Mel spectrogram of the
audio) score every 30-s window; their outputs are fused per event, summarised over the night and fed to a
two-stage XGBoost classifier (Normal vs abnormal, then OSA and RBD). Development and locked internal test data
come from Hospitals A and B, external validation data from Hospital C.

Patient data (videos, audio, PSG reports, scoring exports, cohort tables) and trained weights are not part of this
repository.

Centre codes in the code: `INT` = Hospital A, `GN` = Hospital B (internal), `YI` = Hospital C (external).

## Layout

| Path | Content |
|---|---|
| `psgscreen/preprocessing/` | Recordings → per-clip databases: optical flow, motion records, audio Mel spectrogram, scoring-export parsers |
| `psgscreen/cohort/` | PSG report parsing, report matching, cohort and labels, Table 1 |
| `psgscreen/dataset/` | 30-s window table (event targets), clip timing, ViT input arrays, hypnogram, event-locked traces |
| `psgscreen/model/` | ViT event detectors, late fusion, exam features, disorder and severity classifiers, hyper-parameter search, cumulative-time scores, SHAP, alternative classifiers |
| `psgscreen/evaluation/` | Detection and disorder performance, clinical metrics (operating points, calibration, decision curves), model comparisons, supplementary tables |
| `psgscreen/figures/` | Main and supplementary figures |
| `psgscreen/paths.py`, `events.py` | File locations; event names |
| `configs/` | Selected hyper-parameters (main model; `audio_only/`, `video_only/` inputs; `alt/` classifiers) |
| `scripts/run_all.sh` | Every step in execution order |
| `environment/` | Environments and package versions |

Every module starts with a description of what it computes and how; `python -m psgscreen.<module> -h` lists its
options.

## Inputs

Two roots are set by environment variables: `PSG_DATA` (inputs) and `PSG_WORK` (everything the code writes).

| `PSG_DATA/…` | Content |
|---|---|
| `video/<centre>/…/<exam folder>/*.avi` | PSG video files of an exam (folder name contains the exam uuid; files numbered `<stem>_<NNNN>.avi`) |
| `annotation_exports/<centre>/` | Scoring exports: NeuroWorks text exports (`.txt`, Hospitals A and B), Xltek `.ent` / `.erd` (Hospital C) |
| `reports/<centre>/` | PSG report files (`.doc` / `.docx`) |
| `cohort/exam_inventory.csv` | One row per exam with synchronized video and audio: `uuid, exam_id, center, exam_date, exam_month` |
| `cohort/exclusions.csv` | `uuid, reason`: recordings that failed the quality review and PAP-treated or split-night studies |
| `cohort/splits.csv` | `uuid, split, fold`: locked test (`test`) or development (`dev`, folds 0–9), fixed for all analyses |
| `cohort/yi_questionnaires/` | Hospital C questionnaire exports (`registry.xlsx`, `rbdsq_scores.xlsx`) for RBDSQ scores missing in the report |

`preprocessing` writes the per-clip databases to `PSG_DATA/lmdb/<kind>_<centre>.lmdb` (kind: flow, motion,
magspec, flow_rep, audio) and the parsed event lists to `PSG_DATA/annotations/<centre>/`.
Part of Hospital A was processed with the denseflow tool (`preprocessing/denseflow_lmdb.py`).

## Running

```bash
export PSG_DATA=/path/to/data PSG_WORK=/path/to/work
PY=… PYB=… PYP=… PYF=… PYFC=… PYV=… bash scripts/run_all.sh
```

The interpreters are listed at the top of `scripts/run_all.sh` and described in `environment/ENVIRONMENTS.md`.
Steps can be run one at a time with the commands in that script; each step reads only files written by earlier
steps. `PSG_ROSTER` overrides the cohort file (default `PSG_WORK/cohort/roster.csv`). Optical flow, audio features
and the ViT detectors use a GPU; all XGBoost models run on the CPU (`tree_method=hist`). The video input arrays
take about 370 KB per window (about 0.6 TB for the cohort).

Main steps:

1. **Per-clip data** (30-s clips): dense optical flow at 5 fields / s; motion records (p95 flow magnitude,
   flow-magnitude histogram, mean flow); 128-band log-Mel spectrogram of the audio.
2. **Cohort**: one PSG report per exam → AHI, RBD status, demographics → inclusion criteria, labels
   (OSA = AHI ≥ 5; RBD = report impression of definite or probable RBD) and the fixed splits.
3. **Windows**: 30-s windows with 14 event targets; per window a video input (fx, fy and |f| in 30 one-second
   bins at 64 × 64 pixels) and an audio input (log-Mel, 128 bands × 30 s).
4. **Event detectors**: one ViT-S/16 per modality and fold of a nested 10-fold cross-validation; late fusion with
   a per-event audio weight chosen on the validation fold.
5. **Exam features**: mean, maximum and SD of each event's fused probability over the night (13 events × 3).
6. **Disorder classifier**: stage 1 Abnormal vs Normal; stage 2 OSA and RBD heads on abnormal exams. Locked test and
   external exams are scored by the mean of the 10 fold models. Seed 42 throughout.

## Figures and tables

Figures are written to `PSG_WORK/figures/<module>/`, tables to `PSG_WORK/results/`. Panel letters and multi-panel
layouts were assembled by hand.

| Item | Module | Upstream |
|---|---|---|
| Figure 1 (study design), Figure 2 (flowchart) | — (drawn by hand) | |
| Figure 3, event detection (a–d) | `figures.event_detection` | `evaluation.detection`, `model.fusion`, `dataset.event_traces` |
| Figure 4, ROC, calibration, decision curves | `figures.disorder_detection` | `model.disorder`, `evaluation.clinical` |
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

## License

MIT; see `LICENSE`.
