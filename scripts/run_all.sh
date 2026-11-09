#!/bin/bash
# All steps in execution order. Inputs: README.md; environments: environment/ENVIRONMENTS.md.
# Python interpreters (default: python on PATH):
#   PY    analysis (Python 3.10; xgboost, scikit-learn, matplotlib)
#   PYB   report parsing, clinical metrics, Figure 4 (Python 3.12; statsmodels, olefile)
#   PYP   audio and motion features (torch, torchaudio, librosa)
#   PYF   optical flow, Hospitals A and B (OpenCV built with CUDA, PyAV)
#   PYFC  optical flow, Hospital C (OpenCV built with CUDA, Python 3.7)
#   PYV   ViT input arrays and event detectors (torch, timm, numba; GPU)
# CPU-bound steps take --nthread / --workers; set them to the share of the machine you may use.
set -eu
: "${PSG_DATA:?set PSG_DATA}" "${PSG_WORK:?set PSG_WORK}"
PY=${PY:-python}; PYB=${PYB:-python}; PYP=${PYP:-python}; PYF=${PYF:-python}; PYFC=${PYFC:-python}; PYV=${PYV:-python}
M="-m psgscreen"

# ---- 1. preprocessing: recordings -> per-clip feature databases (PSG_DATA/lmdb) ----
# Part of Hospital A was processed with the denseflow tool instead of optical_flow: denseflow (Docker image
# jinyong/denseflow:batch2048_umask_fps5, options -s=1 -b=20 -cf -v -a=farn) on each video, then
#   $PY $M.preprocessing.denseflow_lmdb --root <denseflow output>
$PYF  $M.preprocessing.optical_flow --center INT --decoder pyav
$PYF  $M.preprocessing.optical_flow --center GN --decoder pyav
$PYFC $M.preprocessing.optical_flow --center YI --decoder opencv
for C in INT GN YI; do
  $PYP $M.preprocessing.motion --center $C
  $PYP $M.preprocessing.audio --center $C
  $PY  $M.preprocessing.annotations --center $C
done

# ---- 2. cohort: reports -> per-exam values -> roster ----
$PYB $M.cohort.reports
$PYB $M.cohort.report_table
$PY  $M.cohort.roster

# ---- 3. windows, ViT inputs, event traces ----
$PY $M.dataset.clip_times --center INT --source motion
$PY $M.dataset.clip_times --center GN --source video
$PY $M.dataset.clip_times --center YI --source video
for C in INT GN YI; do $PY $M.dataset.windows --center $C; done
$PY $M.dataset.windows --center INT --no_annotation
for C in INT GN YI; do
  for S in rows video audio; do $PYV $M.dataset.vit_inputs $S --center $C; done
done
$PY $M.dataset.hypnogram
for C in INT GN; do
  $PY $M.dataset.event_traces --center $C
  $PY $M.dataset.event_traces --center $C --events hypopnea --name hypopnea
done
$PY $M.cohort.table1

# ---- 4. models ----
for K in 0 1 2 3 4 5 6 7 8 9; do
  $PYV $M.model.vit --modality audio --fold $K
  $PYV $M.model.vit --modality video --fold $K
  $PY  $M.model.fusion --fold $K
done
$PY $M.model.exam_features --dump_probs
$PY $M.model.exam_features --combine audio
$PY $M.model.exam_features --combine video
# stage-2 hyper-parameter search for each input (configs/stage2.json and configs/<variant>/stage2.json hold the
# selected values)
for V in main audio_only video_only; do
  O=""; [ $V = main ] || O="--variant $V"
  for S in $(seq 0 15); do $PY $M.model.hparam_search grid --shard $S/16 $O; done
  $PY $M.model.hparam_search pick $O
done
$PY $M.model.disorder
$PY $M.model.disorder --variant audio_only
$PY $M.model.disorder --variant video_only
for T in osa_severity rbd_severity; do $PY $M.model.severity --target $T; done
$PY $M.model.trajectory
$PY $M.model.shap_values
# alternative classifiers (configs/alt/<model>.json hold the selected values)
for A in lr rf svm mlp; do
  $PY $M.model.alt_classifiers grid --model $A
  $PY $M.model.alt_classifiers select --model $A
  $PY $M.model.alt_classifiers run --model $A
done

# ---- 5. evaluation ----
$PY  $M.evaluation.detection
$PY  $M.evaluation.metrics disorder
$PY  $M.evaluation.metrics severity
$PYB $M.evaluation.clinical
$PY  $M.evaluation.model_comparison alt
$PY  $M.evaluation.model_comparison fusion
for K in operating fusion alt; do $PY $M.evaluation.supp_tables $K; done

# ---- 6. figures ----
$PY  $M.figures.event_detection                 # Figure 3
$PYB $M.figures.disorder_detection              # Figure 4
$PY  $M.figures.shap                            # Figure 5
$PY  $M.figures.time_course                     # Figure 6
$PY  $M.figures.disorder_detection --severity   # Supplementary Figure S1
$PY  $M.figures.severity_shap                   # Supplementary Figure S2
$PY  $M.figures.group_probability               # Supplementary Figure S3
