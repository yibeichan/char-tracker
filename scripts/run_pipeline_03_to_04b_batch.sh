#!/bin/bash

#SBATCH --job-name=pipeline_03_04b
#SBATCH --output=/orcd/home/002/yibei/face-track/logs/%x_%j.out
#SBATCH --error=/orcd/home/002/yibei/face-track/logs/%x_%j.err
#SBATCH --partition=ou_bcs_low
#SBATCH --time=01:00:00
#SBATCH --array=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

# Partial pipeline script (03-04b) for batch processing with SLURM
# This script runs steps 03-04b sequentially for each video in the array
# Assumes steps 01-02 have already completed

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
echo "=========================================="
echo ""

# Change to scripts directory
cd "$SCRIPTS_DIR" || exit 1

# Run the partial pipeline script (03-04b)
echo "Starting partial pipeline execution (steps 03-04b)..."
./run_pipeline_03_to_04b.sh "$TASK_ID" --mode "$MODE"

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
