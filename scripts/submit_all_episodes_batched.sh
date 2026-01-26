#!/bin/bash

# Batch submission script for all 292 episodes
# Submits 10 jobs at a time, waits 2 hours between batches
# Designed to run in tmux for long-running submission process
#
# Usage:
#   tmux new -s batch-submit
#   ./scripts/submit_all_episodes_batched.sh
#
# The script is resumable - if interrupted, it will skip already submitted jobs

set -e

# Configuration
BATCH_SIZE=10
BATCH_DELAY_SECONDS=7200  # 2 hours = 7200 seconds
TOTAL_EPISODES=292

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK_FILE="$REPO_ROOT/data/episode_id.txt"
SUBMISSION_LOG="$REPO_ROOT/logs/submission_log.txt"
PID_FILE="$REPO_ROOT/logs/batch_submit.pid"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Create logs directory
mkdir -p "$(dirname "$SUBMISSION_LOG")"
mkdir -p "$(dirname "$PID_FILE")"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        log_error "Another instance is already running (PID: $OLD_PID)"
        log_info "If this is incorrect, remove $PID_FILE"
        exit 1
    else
        log_warning "Found stale PID file, removing..."
        rm -f "$PID_FILE"
    fi
fi

# Write our PID
echo $$ > "$PID_FILE"
trap "rm -f '$PID_FILE'" EXIT

# Check task file exists
if [ ! -f "$TASK_FILE" ]; then
    log_error "Task file not found: $TASK_FILE"
    exit 1
fi

# Count total lines in task file
TOTAL_LINES=$(wc -l < "$TASK_FILE")
if [ "$TOTAL_LINES" -ne "$TOTAL_EPISODES" ]; then
    log_warning "Expected $TOTAL_EPISODES episodes, found $TOTAL_LINES in $TASK_FILE"
fi

# Function to get submitted job IDs from log
get_submitted_jobs() {
    if [ -f "$SUBMISSION_LOG" ]; then
        grep "^SUBMITTED:" "$SUBMISSION_LOG" 2>/dev/null | awk '{print $3}' | sort -u
    fi
}

# Function to check if a line number was already submitted
is_already_submitted() {
    local line_num=$1
    if [ -f "$SUBMISSION_LOG" ]; then
        grep -q "^SUBMITTED:.*line:${line_num}" "$SUBMISSION_LOG" 2>/dev/null
    else
        return 1
    fi
}

# Function to submit a single job
submit_job() {
    local line_num=$1
    local episode_id=$2

    if is_already_submitted "$line_num"; then
        echo -e "${CYAN}SKIP${NC}    Line $line_num: $episode_id (already submitted)"
        return 0
    fi

    # Submit the job
    local job_id
    job_id=$(sbatch --array="$line_num" "$SCRIPT_DIR/run_pipeline_annotation_episodes.sh" 2>&1 | grep -oP 'Submitted batch job \K\d+' || echo "")

    if [ -n "$job_id" ]; then
        echo -e "${GREEN}SUBMIT${NC}  Line $line_num: $episode_id -> Job $job_id"
        echo "SUBMITTED: $(date '+%Y-%m-%d %H:%M:%S') line:$line_num episode:$episode_id job:$job_id" >> "$SUBMISSION_LOG"
        return 0
    else
        echo -e "${RED}FAILED${NC}  Line $line_num: $episode_id"
        echo "FAILED: $(date '+%Y-%m-%d %H:%M:%S') line:$line_num episode:$episode_id" >> "$SUBMISSION_LOG"
        return 1
    fi
}

# Main submission loop
log_info "=========================================="
log_info "BATCH SUBMISSION SCRIPT STARTED"
log_info "=========================================="
log_info "Total episodes: $TOTAL_EPISODES"
log_info "Batch size: $BATCH_SIZE"
log_info "Delay between batches: ${BATCH_DELAY_SECONDS}s ($((BATCH_DELAY_SECONDS / 3600))h)"
log_info "Submission log: $SUBMISSION_LOG"
log_info "=========================================="
echo ""

# Count already submitted jobs
ALREADY_SUBMITTED=0
if [ -f "$SUBMISSION_LOG" ]; then
    ALREADY_SUBMITTED=$(grep -c "^SUBMITTED:" "$SUBMISSION_LOG" 2>/dev/null || echo "0")
fi
REMAINING=$((TOTAL_EPISODES - ALREADY_SUBMITTED))
ESTIMATED_BATCHES=$(( (REMAINING + BATCH_SIZE - 1) / BATCH_SIZE ))
ESTIMATED_HOURS=$((ESTIMATED_BATCHES * BATCH_DELAY_SECONDS / 3600))

log_info "Already submitted: $ALREADY_SUBMITTED"
log_info "Remaining: $REMAINING"
log_info "Estimated batches: $ESTIMATED_BATCHES"
log_info "Estimated time to complete: ${ESTIMATED_HOURS}h"
echo ""

submitted_this_run=0
batch_num=0

for line_num in $(seq 1 "$TOTAL_EPISODES"); do
    # Get episode ID from task file
    episode_id=$(sed -n "${line_num}p" "$TASK_FILE")

    if [ -z "$episode_id" ]; then
        log_warning "Empty line $line_num in $TASK_FILE"
        continue
    fi

    # Check if already submitted
    if is_already_submitted "$line_num"; then
        # Skip silently, just show progress
        submitted_this_run=$((submitted_this_run + 1))
        if [ $((submitted_this_run % 20)) -eq 0 ]; then
            log_info "Progress: $submitted_this_run/$TOTAL_EPISODES processed..."
        fi
        continue
    fi

    # Submit the job
    batch_num=$((batch_num + 1))
    in_batch=$((in_batch + 1))

    submit_job "$line_num" "$episode_id"
    submitted_this_run=$((submitted_this_run + 1))

    # If we've submitted a full batch, wait before the next batch
    if [ "$in_batch" -ge "$BATCH_SIZE" ]; then
        # Check if there are more to submit
        remaining=$((TOTAL_EPISODES - line_num))

        if [ "$remaining" -gt 0 ]; then
            echo ""
            log_info "Batch complete. Submitted $in_batch jobs this batch."
            log_info "Remaining episodes: $remaining"
            log_info "Next batch in ${BATCH_DELAY_SECONDS}s ($((BATCH_DELAY_SECONDS / 60)) minutes)..."
            log_info "Current jobs in queue:"
            squeue -u "$USER" --format="%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R" 2>/dev/null | head -11 || log_warning "Could not query job queue"
            log_info "=========================================="

            sleep "$BATCH_DELAY_SECONDS"
            echo ""
            in_batch=0
        fi
    fi
done

echo ""
log_success "=========================================="
log_success "ALL $TOTAL_EPISODES EPISODES PROCESSED!"
log_success "=========================================="
log_info "Final submission count in this run: $submitted_this_run"
log_info "Total submissions in log: $(grep -c "^SUBMITTED:" "$SUBMISSION_LOG" 2>/dev/null || echo "0")"
log_info "Submission log: $SUBMISSION_LOG"
log_success "=========================================="
