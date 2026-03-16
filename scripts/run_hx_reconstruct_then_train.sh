#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

SONGFORMDB_DIR="${SONGFORMDB_DIR:-$ROOT_DIR/data/SongFormDB}"
HX_JSONL="${HX_JSONL:-$SONGFORMDB_DIR/HX/SongFormDB-HX.jsonl}"
HX_ROOT="${HX_ROOT:-$SONGFORMDB_DIR}"
HX_MEL_DIR="${HX_MEL_DIR:-$SONGFORMDB_DIR/HX}"
HX_AUDIO_DIR="${HX_AUDIO_DIR:-$ROOT_DIR/data/songform-hx-audio}"
ALIGNED_DIR="${ALIGNED_DIR:-$ROOT_DIR/data/songform-hx-aligned}"

RUN_NAME="${RUN_NAME:-hx_$(date +%Y%m%d_%H%M%S)}"
CKPT_DIR="${CKPT_DIR:-$ROOT_DIR/checkpoints/$RUN_NAME}"

RECON_LIMIT="${RECON_LIMIT:-0}"
RECON_MATCH="${RECON_MATCH:-}"
RECON_SKIP_EXISTING="${RECON_SKIP_EXISTING:-1}"
USE_CUDA_KERNEL="${USE_CUDA_KERNEL:-1}"

EPOCHS="${EPOCHS:-30}"
ARCH="${ARCH:-ms_transformer}"
BOUNDARY_BATCH_SIZE="${BOUNDARY_BATCH_SIZE:-8}"
BOUNDARY_LR="${BOUNDARY_LR:-1e-3}"
BOUNDARY_EVAL_INTERVAL="${BOUNDARY_EVAL_INTERVAL:-1}"
BOUNDARY_EVAL_TOLERANCE="${BOUNDARY_EVAL_TOLERANCE:-0.5}"
BOUNDARY_EVAL_THRESHOLDS="${BOUNDARY_EVAL_THRESHOLDS:-}"

CLASSIFIER_BATCH_SIZE="${CLASSIFIER_BATCH_SIZE:-32}"
CLASSIFIER_LR="${CLASSIFIER_LR:-1e-3}"
CLASSIFIER_HIDDEN="${CLASSIFIER_HIDDEN:-512}"
CLASSIFIER_WEIGHT_POWER="${CLASSIFIER_WEIGHT_POWER:-0.75}"
CLASSIFIER_PATIENCE="${CLASSIFIER_PATIENCE:-3}"
CLASSIFIER_MIN_DELTA="${CLASSIFIER_MIN_DELTA:-0.0}"
CLASSIFIER_GRAD_CLIP="${CLASSIFIER_GRAD_CLIP:-1.0}"
CLASSIFIER_INPUT_TYPE="${CLASSIFIER_INPUT_TYPE:-mel}"
CLASSIFIER_SEGMENT_FRAMES="${CLASSIFIER_SEGMENT_FRAMES:-256}"

VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.0}"
SEED="${SEED:-42}"
MIN_LABEL_COUNT="${MIN_LABEL_COUNT:-20}"
MIN_SEG_SECONDS="${MIN_SEG_SECONDS:-0.5}"
BALANCE_SAMPLES="${BALANCE_SAMPLES:-0}"
MAX_TRAIN_ITEMS="${MAX_TRAIN_ITEMS:-0}"
MAX_LABELS="${MAX_LABELS:-12}"
WORKERS="${WORKERS:-}"

AUDIO_SAMPLE_RATE="${AUDIO_SAMPLE_RATE:-22050}"
AUDIO_N_MELS="${AUDIO_N_MELS:-80}"
AUDIO_HOP_LENGTH="${AUDIO_HOP_LENGTH:-1024}"
AUDIO_N_FFT="${AUDIO_N_FFT:-2048}"

LIMIT="$RECON_LIMIT" MATCH="$RECON_MATCH" SKIP_EXISTING="$RECON_SKIP_EXISTING" USE_CUDA_KERNEL="$USE_CUDA_KERNEL" \
  bash "$ROOT_DIR/scripts/reconstruct_hx_wavs.sh"

mkdir -p "$ALIGNED_DIR"
python3 "$ROOT_DIR/scripts/align_salami_audio.py" \
  --hx_jsonl "$HX_JSONL" \
  --hx_root "$HX_ROOT" \
  --hx_mel_dir "$HX_MEL_DIR" \
  --hx_audio_dir "$HX_AUDIO_DIR" \
  --out_dir "$ALIGNED_DIR" \
  --require_wav

mkdir -p "$CKPT_DIR"

balance_flag=()
if [[ "$BALANCE_SAMPLES" == "1" ]]; then
  balance_flag+=(--balance_samples)
fi

workers_flag=()
if [[ -n "$WORKERS" ]]; then
  workers_flag+=(--workers "$WORKERS")
fi

set +u
python3 -m model.train_pipeline \
  --data_dir "$ALIGNED_DIR" \
  --ckpt_dir "$CKPT_DIR" \
  --epochs "$EPOCHS" \
  --arch "$ARCH" \
  --val_ratio "$VAL_RATIO" \
  --test_ratio "$TEST_RATIO" \
  --seed "$SEED" \
  --eval_tolerance "$BOUNDARY_EVAL_TOLERANCE" \
  --boundary_batch_size "$BOUNDARY_BATCH_SIZE" \
  --boundary_lr "$BOUNDARY_LR" \
  --boundary_eval_interval "$BOUNDARY_EVAL_INTERVAL" \
  --boundary_eval_thresholds "$BOUNDARY_EVAL_THRESHOLDS" \
  --audio_sample_rate "$AUDIO_SAMPLE_RATE" \
  --audio_n_mels "$AUDIO_N_MELS" \
  --audio_hop_length "$AUDIO_HOP_LENGTH" \
  --audio_n_fft "$AUDIO_N_FFT" \
  --classifier_batch_size "$CLASSIFIER_BATCH_SIZE" \
  --classifier_lr "$CLASSIFIER_LR" \
  --classifier_hidden "$CLASSIFIER_HIDDEN" \
  --classifier_weight_power "$CLASSIFIER_WEIGHT_POWER" \
  --classifier_patience "$CLASSIFIER_PATIENCE" \
  --classifier_min_delta "$CLASSIFIER_MIN_DELTA" \
  --classifier_grad_clip "$CLASSIFIER_GRAD_CLIP" \
  --classifier_input_type "$CLASSIFIER_INPUT_TYPE" \
  --classifier_segment_frames "$CLASSIFIER_SEGMENT_FRAMES" \
  --min_label_count "$MIN_LABEL_COUNT" \
  --min_seg_seconds "$MIN_SEG_SECONDS" \
  --max_train_items "$MAX_TRAIN_ITEMS" \
  --max_labels "$MAX_LABELS" \
  "${balance_flag[@]}" \
  "${workers_flag[@]}"
set -u
