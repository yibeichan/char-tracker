#!/bin/bash

#SBATCH --job-name=annotation_eps
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
#SBATCH --partition=ou_bcs_low
#SBATCH --time=01:00:00
#SBATCH --array=23,43,73
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=4G
#SBATCH --mail-type=FAIL,END

# Batch processing script for ANNOTATION EPISODES ONLY
# Processes the 18 representative episodes selected for annotation (Phase 1-2)
#
# Episode List (18 half-episodes, 3 per season):
#   Season 1: s01e03b (6), s01e12a (23), s01e22a (43)
#   Season 2: s02e02b (52), s02e13a (73), s02e23b (94)
#   Season 3: s03e04a (103), s03e11b (118), s03e21a (137)
#   Season 4: s04e03b (152), s04e14a (173), s04e22b (190)
#   Season 5: s05e02a (197), s05e12b (218), s05e22a (237)
#   Season 6: s06e05b (252), s06e13a (267), s06e23b (288)
#
# Usage:
#   # Process ALL 18 annotation episodes:
#   sbatch scripts/run_pipeline_annotation_episodes.sh
#
#   # Process only Phase 1 pilot (5 episodes):
#   # Line numbers: 6 (s01e03b), 118 (s03e11b), 237 (s05e22a), 73 (s02e13a), 252 (s06e05b)
#   sbatch --array=6,118,237,73,252 scripts/run_pipeline_annotation_episodes.sh
#
#   # Process a single episode:
#   sbatch --array=6 scripts/run_pipeline_annotation_episodes.sh
#
# IMPORTANT: Submit from the project root directory:
#   cd /home/yibei/char-tracker
#   sbatch scripts/run_pipeline_annotation_episodes.sh

# ============================================================================
# LINE NUMBER REFERENCE (for --array override)
# ============================================================================
# Phase 1 Pilot (5 episodes) - Recommended first batch:
#   6   = friends_s01e03b  (Season 1 beginning)
#   118 = friends_s03e11b  (Season 3 middle)
#   237 = friends_s05e22a  (Season 5 end)
#   73  = friends_s02e13a  (Season 2 middle)
#   252 = friends_s06e05b  (Season 6 beginning)
#
# Phase 1 Extension (5 more episodes):
#   43  = friends_s01e22a
#   173 = friends_s04e14a
#   52  = friends_s02e02b
#   137 = friends_s03e21a
#   197 = friends_s05e02a
#
# Phase 2 Completion (8 remaining):
#   23  = friends_s01e12a
#   94  = friends_s02e23b
#   103 = friends_s03e04a
#   152 = friends_s04e03b
#   190 = friends_s04e22b
#   218 = friends_s05e12b
#   267 = friends_s06e13a
#   288 = friends_s06e23b
# ============================================================================

# Load cuDNN module for onnxruntime-gpu (required for libcudnn.so.9)
module load cudnn/9.8.0.87-cuda12

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate friends_char_track

# Set up paths
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SCRIPTS_DIR="$REPO_ROOT/scripts"
TASK_FILE="$REPO_ROOT/data/episode_id.txt"
LOG_DIR="$REPO_ROOT/logs"

mkdir -p "$LOG_DIR"

if [ ! -f "$TASK_FILE" ]; then
    echo "ERROR: Task file not found: $TASK_FILE"
    exit 1
fi

# Configuration - can be overridden via environment variables
MODE="${MODE:-symlink}"
USE_SEQUENTIAL="${NO_SEQUENTIAL:-}"
USE_BATCH="${NO_BATCH:-}"
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-0.5}"

# Output directory name constants (must match src/utils.py)
OUTPUT_DIR_FACE_TRACKING_BY_CLUSTER="04b_face_tracking_by_cluster"

# Get the episode for this array task
# Handle both SLURM and local execution
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # Running via SLURM - use array task ID as line number
    TASK_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_FILE")
    if [ -z "$TASK_ID" ]; then
        echo "ERROR: Could not read TASK_ID from line ${SLURM_ARRAY_TASK_ID} of $TASK_FILE"
        exit 1
    fi
else
    # Running locally - require episode ID as argument
    if [ -z "$1" ]; then
        echo "=========================================="
        echo "ERROR: Not running via SLURM and no episode specified"
        echo ""
        echo "Usage (SLURM - recommended):"
        echo "  sbatch scripts/run_pipeline_annotation_episodes.sh"
        echo "  sbatch --array=6,73,118,237,252 scripts/run_pipeline_annotation_episodes.sh"
        echo ""
        echo "Usage (Local - for testing single episode):"
        echo "  ./scripts/run_pipeline_annotation_episodes.sh friends_s01e03b"
        echo ""
        echo "Available annotation episodes:"
        grep -v "^#" "$REPO_ROOT/data/annotation_episodes.txt" | grep -v "^$"
        echo "=========================================="
        exit 1
    fi
    TASK_ID="$1"
    # Verify episode exists in episode_id.txt
    if ! grep -q "^${TASK_ID}$" "$TASK_FILE"; then
        echo "ERROR: Episode '$TASK_ID' not found in $TASK_FILE"
        exit 1
    fi
fi

echo "=========================================="
echo "ANNOTATION EPISODE PROCESSING"
echo "=========================================="
echo "SLURM Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Episode: $TASK_ID"
echo "Similarity threshold: $SIMILARITY_THRESHOLD"
echo "Node: $(hostname)"
echo ""
echo "Purpose: Prepare episode for annotation in ClusterMark"
echo "=========================================="
echo ""

cd "$SCRIPTS_DIR" || exit 1

# Build command
CMD=("./run_pipeline_01_to_04b.sh" "$TASK_ID" --mode "$MODE")
if [ -n "$USE_SEQUENTIAL" ]; then
    CMD+=(--no-sequential)
fi
if [ -n "$USE_BATCH" ]; then
    CMD+=(--no-batch)
fi
CMD+=(--similarity-threshold "$SIMILARITY_THRESHOLD")

echo "Command: ${CMD[*]}"
echo ""
"${CMD[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "SUCCESS: $TASK_ID ready for annotation"
    echo ""
    echo "Next steps:"
    echo "1. Find cluster images in: \$SCRATCH_DIR/output/${OUTPUT_DIR_FACE_TRACKING_BY_CLUSTER}/$TASK_ID/"
    echo "2. Create ZIP for ClusterMark upload"
    echo "3. Annotate clusters and export JSON"
    echo "4. Run: python scripts/04c_refine_with_annotations.py $TASK_ID <annotation.json>"
    echo "5. Run: python scripts/08_generate_character_timestamps.py $TASK_ID <annotation.json>"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "FAILED: $TASK_ID with exit code $EXIT_CODE"
    echo "=========================================="
    exit $EXIT_CODE
fi
