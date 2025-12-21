#!/bin/bash

#SBATCH --job-name=pipeline_01_04b
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
#SBATCH --partition=ou_bcs_low
#SBATCH --time=02:45:00
#SBATCH --array=6
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --mail-type=FAIL,END

# Full pipeline script (01-04b) for batch processing with SLURM
# IMPORTANT: Submit SLURM jobs from the project root directory using:
#   sbatch scripts/run_pipeline_01_to_04b_batch.sh
# This script runs all 5 steps sequentially for each video in the array
#
# With optimizations enabled (default):
#   - Sequential frame reading: 5-10x faster for step 03
#   - Batch embeddings: 2-4x faster for step 04
#   - Expected time for steps 03-04: ~10-15 minutes
#
# To disable optimizations, set environment variables:
#   NO_SEQUENTIAL=1 sbatch run_pipeline_01_to_04b_batch.sh
#   NO_BATCH=1 sbatch run_pipeline_01_to_04b_batch.sh
#   BATCH_SIZE=64 sbatch run_pipeline_01_to_04b_batch.sh
#   MODEL_NAME=buffalo_l sbatch run_pipeline_01_to_04b_batch.sh
#   SIMILARITY_THRESHOLD=0.5 sbatch run_pipeline_01_to_04b_batch.sh

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate friends_char_track

# Set up paths
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    # Running via SLURM - assume submitted from project root
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    # Running locally - use script location
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SCRIPTS_DIR="$REPO_ROOT/scripts"
TASK_FILE="$REPO_ROOT/data/episode_id.txt"
LOG_DIR="$REPO_ROOT/logs"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Check if task file exists
if [ ! -f "$TASK_FILE" ]; then
    echo "ERROR: Task file not found: $TASK_FILE"
    exit 1
fi

# Mode for step 04b: copy, move, or symlink (default: symlink for efficiency)
MODE="${MODE:-symlink}"

# Optimization flags (can be set via environment variables)
# Set NO_SEQUENTIAL=1 to disable sequential frame reading
# Set NO_BATCH=1 to disable batch embedding processing
# Set BATCH_SIZE=64 to use custom batch size
# Set MODEL_NAME=buffalo_l to use InsightFace buffalo_l model (default: vggface2)
USE_SEQUENTIAL="${NO_SEQUENTIAL:-}"
USE_BATCH="${NO_BATCH:-}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MODEL_NAME="${MODEL_NAME:-buffalo_l}"
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-0.5}"

# Get the video name for this array task
TASK_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_FILE")

if [ -z "$TASK_ID" ]; then
    echo "ERROR: Could not read TASK_ID from line ${SLURM_ARRAY_TASK_ID} of $TASK_FILE"
    exit 1
fi

echo "=========================================="
echo "SLURM Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing video: $TASK_ID"
echo "Mode for 04b: $MODE"
echo "Embedding model: $MODEL_NAME"
echo "Similarity threshold: $SIMILARITY_THRESHOLD"
echo "Node: $(hostname)"
if [ -z "$USE_SEQUENTIAL" ]; then
    echo "Optimization: Sequential frame reading ENABLED (5-10x faster)"
else
    echo "Optimization: Sequential frame reading DISABLED"
fi
if [ -z "$USE_BATCH" ]; then
    echo "Optimization: Batch embeddings ENABLED (batch_size=$BATCH_SIZE, 2-4x faster)"
else
    echo "Optimization: Batch embeddings DISABLED"
fi
echo "=========================================="
echo ""

# Change to scripts directory
cd "$SCRIPTS_DIR" || exit 1

# Build command with optimization flags using bash array (safer than eval)
CMD=("./run_pipeline_01_to_04b.sh" "$TASK_ID" --mode "$MODE")
if [ -n "$USE_SEQUENTIAL" ]; then
    CMD+=(--no-sequential)
fi
if [ -n "$USE_BATCH" ]; then
    CMD+=(--no-batch)
fi
CMD+=(--batch-size "$BATCH_SIZE")
CMD+=(--model-name "$MODEL_NAME")
CMD+=(--similarity-threshold "$SIMILARITY_THRESHOLD")

# Run the full pipeline script
echo "Starting pipeline execution..."
echo "Command: ${CMD[*]}"
"${CMD[@]}"

# Capture exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "SUCCESS: Pipeline completed for $TASK_ID"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "FAILED: Pipeline failed for $TASK_ID with exit code $EXIT_CODE"
    echo "=========================================="
    exit $EXIT_CODE
fi
