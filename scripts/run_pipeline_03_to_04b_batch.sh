#!/bin/bash

#SBATCH --job-name=pipeline_03_04b
#SBATCH --output=/orcd/home/002/yibei/face-track/logs/%x_%j.out
#SBATCH --error=/orcd/home/002/yibei/face-track/logs/%x_%j.err
#SBATCH --partition=ou_bcs_low
#SBATCH --time=00:20:00
#SBATCH --array=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

# Partial pipeline script (03-04b) for batch processing with SLURM
# This script runs steps 03-04b sequentially for each video in the array
# Assumes steps 01-02 have already completed
#
# With optimizations enabled (default):
#   - Sequential frame reading: 5-10x faster for step 03
#   - Batch embeddings: 2-4x faster for step 04
#   - Expected time: ~10-15 minutes per episode
#
# To disable optimizations, set environment variables:
#   NO_SEQUENTIAL=1 sbatch run_pipeline_03_to_04b_batch.sh
#   NO_BATCH=1 sbatch run_pipeline_03_to_04b_batch.sh
#   BATCH_SIZE=64 sbatch run_pipeline_03_to_04b_batch.sh

# Source micromamba (adjust if using conda instead)
# For micromamba:
eval "$(micromamba shell hook --shell bash)"
micromamba activate friends_char_track

# For conda, use instead:
# source $HOME/miniconda3/etc/profile.d/conda.sh
# conda activate friends_char_track

# Set up paths - use absolute paths for reliability in SLURM
REPO_ROOT="/orcd/home/002/yibei/face-track"
SCRIPTS_DIR="$REPO_ROOT/scripts"
TASK_FILE="$REPO_ROOT/data/episode_id.txt"
LOG_DIR="$REPO_ROOT/logs"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Mode for step 04b: copy, move, or symlink (default: symlink for efficiency)
MODE="${MODE:-symlink}"

# Optimization flags (can be set via environment variables)
# Set NO_SEQUENTIAL=1 to disable sequential frame reading
# Set NO_BATCH=1 to disable batch embedding processing
# Set BATCH_SIZE=64 to use custom batch size
USE_SEQUENTIAL="${NO_SEQUENTIAL:-}"
USE_BATCH="${NO_BATCH:-}"
BATCH_SIZE="${BATCH_SIZE:-32}"

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

# Build command with optimization flags
CMD="./run_pipeline_03_to_04b.sh \"$TASK_ID\" --mode \"$MODE\""
if [ -n "$USE_SEQUENTIAL" ]; then
    CMD="$CMD --no-sequential"
fi
if [ -n "$USE_BATCH" ]; then
    CMD="$CMD --no-batch"
fi
CMD="$CMD --batch-size \"$BATCH_SIZE\""

# Run the partial pipeline script (03-04b)
echo "Starting partial pipeline execution (steps 03-04b)..."
echo "Command: $CMD"
eval $CMD

# Capture exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "SUCCESS: Partial pipeline (03-04b) completed for $TASK_ID"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "FAILED: Partial pipeline (03-04b) failed for $TASK_ID with exit code $EXIT_CODE"
    echo "=========================================="
    exit $EXIT_CODE
fi
