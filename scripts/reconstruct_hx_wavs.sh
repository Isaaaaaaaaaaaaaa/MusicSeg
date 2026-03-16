#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SONGFORMDB_DIR="${SONGFORMDB_DIR:-$ROOT_DIR/data/SongFormDB}"
BIGVGAN_REPO_DIR="${BIGVGAN_REPO_DIR:-$ROOT_DIR/BigVGAN}"
HX_MEL_DIR="${HX_MEL_DIR:-$SONGFORMDB_DIR/HX}"
HX_AUDIO_DIR="${HX_AUDIO_DIR:-$ROOT_DIR/data/songform-hx-audio}"
CKPT_FILE="${CKPT_FILE:-$SONGFORMDB_DIR/utils/HX/ckpt/g_00276000.ckpt}"

LIMIT="${LIMIT:-0}"
MATCH="${MATCH:-}"
USE_CUDA_KERNEL="${USE_CUDA_KERNEL:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

export PYTHONPATH="$BIGVGAN_REPO_DIR:${PYTHONPATH:-}"

CUDA_AVAILABLE="$(python3 -c "import torch; print('1' if torch.cuda.is_available() else '0')")"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if [[ -z "${USE_CUDA_KERNEL}" ]]; then
  USE_CUDA_KERNEL="$CUDA_AVAILABLE"
fi

if [[ ! -d "$HX_MEL_DIR" ]]; then
  echo "missing HX_MEL_DIR: $HX_MEL_DIR"
  exit 1
fi
if [[ ! -f "$CKPT_FILE" ]]; then
  echo "missing CKPT_FILE: $CKPT_FILE"
  exit 1
fi
if [[ ! -d "$BIGVGAN_REPO_DIR" ]]; then
  echo "missing BIGVGAN_REPO_DIR: $BIGVGAN_REPO_DIR"
  exit 1
fi

mkdir -p "$HX_AUDIO_DIR"

CUDA_KERNEL_ARG=""
if [[ "$USE_CUDA_KERNEL" == "1" ]]; then
  CUDA_KERNEL_ARG="--use_cuda_kernel"
fi

SKIP_EXISTING_ARG=""
if [[ "$SKIP_EXISTING" == "1" ]]; then
  SKIP_EXISTING_ARG="--skip_existing"
fi

python3 "$SONGFORMDB_DIR/utils/HX/inference_e2e.py" \
  --input_mels_dir "$HX_MEL_DIR" \
  --output_dir "$HX_AUDIO_DIR" \
  --checkpoint_file "$CKPT_FILE" \
  $CUDA_KERNEL_ARG \
  $SKIP_EXISTING_ARG \
  --limit "$LIMIT" \
  --match "$MATCH"
