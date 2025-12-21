#!/bin/bash

# Partial pipeline script to process a single video through steps 03-04b
# Assumes steps 01-02 have already completed
# Usage: ./run_pipeline_03_to_04b.sh <video_name> [--mode copy|move|symlink]

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output (only when output is to a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# Function to print colored messages
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check if file exists
check_file_exists() {
    local file=$1
    local description=$2

    if [ ! -f "$file" ]; then
        log_error "$description not found: $file"
        return 1
    fi
    log_success "$description found: $file"
    return 0
}

# Function to check if directory exists
check_dir_exists() {
    local dir=$1
    local description=$2

    if [ ! -d "$dir" ]; then
        log_error "$description not found: $dir"
        return 1
    fi
    log_success "$description found: $dir"
    return 0
}

# Parse arguments
if [ $# -lt 1 ]; then
    log_error "Usage: $0 <video_name> [--mode copy|move|symlink] [--no-sequential] [--no-batch] [--batch-size N] [--model-name vggface2|buffalo_l]"
    exit 1
fi

VIDEO_NAME=$1
MODE="copy"  # Default mode for 04b
NO_SEQUENTIAL=""  # Empty = use sequential (optimized)
NO_BATCH=""  # Empty = use batch processing (optimized)
BATCH_SIZE="32"  # Default batch size
MODEL_NAME="vggface2"  # Default embedding model

# Parse optional arguments
shift
while [ $# -gt 0 ]; do
    case $1 in
        --mode)
            if [ $# -lt 2 ]; then
                log_error "Missing argument for --mode. Must be one of 'copy', 'move', or 'symlink'."
                exit 1
            fi
            if [[ "$2" =~ ^(copy|move|symlink)$ ]]; then
                MODE="$2"
                shift 2
            else
                log_error "Invalid mode: '$2'. Must be one of 'copy', 'move', or 'symlink'."
                exit 1
            fi
            ;;
        --no-sequential)
            NO_SEQUENTIAL="--no-sequential"
            shift
            ;;
        --no-batch)
            NO_BATCH="--no-batch"
            shift
            ;;
        --batch-size)
            if [ $# -lt 2 ]; then
                log_error "Missing argument for --batch-size."
                exit 1
            fi
            BATCH_SIZE="$2"
            shift 2
            ;;
        --model-name)
            if [ $# -lt 2 ]; then
                log_error "Missing argument for --model-name."
                exit 1
            fi
            if [[ "$2" =~ ^(vggface2|buffalo_l)$ ]]; then
                MODEL_NAME="$2"
                shift 2
            else
                log_error "Invalid model-name: '$2'. Must be 'vggface2' or 'buffalo_l'."
                exit 1
            fi
            ;;
        *)
            log_error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Load environment variables safely
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [ -f "$ENV_FILE" ]; then
    log_info "Loading environment from: $ENV_FILE"
    set -a  # Automatically export all variables
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +a  # Stop automatically exporting
else
    log_warning ".env file not found at: $ENV_FILE"
fi

# Get SCRATCH_DIR from environment
if [ -z "${SCRATCH_DIR:-}" ]; then
    log_error "SCRATCH_DIR environment variable is not set"
    exit 1
fi

log_info "SCRATCH_DIR: $SCRATCH_DIR"
log_info "Video name: $VIDEO_NAME"
log_info "Mode for 04b: $MODE"
log_info "Embedding model: $MODEL_NAME"
if [ -z "$NO_SEQUENTIAL" ]; then
    log_info "Step 03: Sequential frame reading ENABLED (5-10x faster)"
else
    log_warning "Step 03: Sequential frame reading DISABLED (using legacy mode)"
fi
if [ -z "$NO_BATCH" ]; then
    log_info "Step 04: Batch processing ENABLED (batch_size=$BATCH_SIZE, 2-4x faster)"
else
    log_warning "Step 04: Batch processing DISABLED (using sequential mode)"
fi

# Define paths
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_FILE="${SCRATCH_DIR}/data/mkv2mp4/${VIDEO_NAME}.mp4"

# Input paths required from previous steps
SCENE_OUTPUT="${SCRATCH_DIR}/output/scene_detection/${VIDEO_NAME}.txt"
FACE_DETECTION_OUTPUT="${SCRATCH_DIR}/output/face_detection/${VIDEO_NAME}.json"

# Output paths for steps 03-04b
TRACKING_DIR="${SCRATCH_DIR}/output/face_tracking/${VIDEO_NAME}"
TRACKED_FACES="${TRACKING_DIR}/${VIDEO_NAME}_tracked_faces.json"
SELECTED_FRAMES="${TRACKING_DIR}/${VIDEO_NAME}_selected_frames_per_face.json"
CLUSTERING_OUTPUT="${SCRATCH_DIR}/output/face_clustering/${VIDEO_NAME}_matched_faces_with_clusters.json"
REORGANIZE_DIR="${SCRATCH_DIR}/output/face_tracking_by_cluster/${VIDEO_NAME}"

# Check prerequisites from steps 01-02
log_info "Checking prerequisites from previous steps..."
check_file_exists "$VIDEO_FILE" "Input video" || exit 1
check_file_exists "$SCENE_OUTPUT" "Scene detection output (step 01)" || exit 1
check_file_exists "$FACE_DETECTION_OUTPUT" "Face detection output (step 02)" || exit 1

echo ""
log_info "=========================================="
log_info "Starting partial pipeline (03-04b) for: $VIDEO_NAME"
log_info "=========================================="
echo ""

# Change to scripts directory for all pipeline steps
cd "$SCRIPTS_DIR"

# ============================================================================
# Step 03: Within-Scene Tracking
# ============================================================================
log_info "=========================================="
log_info "STEP 03: Within-Scene Tracking"
log_info "=========================================="

cmd_step03=("python" "03_within_scene_tracking.py" "$VIDEO_NAME")
if [ -n "$NO_SEQUENTIAL" ]; then
    cmd_step03+=("--no-sequential")
fi
"${cmd_step03[@]}"

# Validate outputs
check_dir_exists "$TRACKING_DIR" "Tracking directory" || exit 1
check_file_exists "$TRACKED_FACES" "Tracked faces output" || exit 1
check_file_exists "$SELECTED_FRAMES" "Selected frames output" || exit 1

# Check if any face images were saved
IMAGE_COUNT=$(find "$TRACKING_DIR" -name "*.jpg" 2>/dev/null | wc -l)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    log_warning "No face images found in $TRACKING_DIR"
    log_warning "This might indicate no faces were tracked in the video"
else
    log_success "Found $IMAGE_COUNT face images in tracking directory"
fi
echo ""

# ============================================================================
# Step 04: Face Clustering
# ============================================================================
log_info "=========================================="
log_info "STEP 04: Face Clustering"
log_info "=========================================="

cmd_step04=("python" "04_face_clustering.py" "$VIDEO_NAME")
if [ -n "$NO_BATCH" ]; then
    cmd_step04+=("--no-batch")
fi
cmd_step04+=("--batch-size" "$BATCH_SIZE")
cmd_step04+=("--model-name" "$MODEL_NAME")
"${cmd_step04[@]}"

# Validate output
check_file_exists "$CLUSTERING_OUTPUT" "Face clustering output" || exit 1
echo ""

# ============================================================================
# Step 04b: Reorganize by Cluster
# ============================================================================
log_info "=========================================="
log_info "STEP 04b: Reorganize by Cluster"
log_info "=========================================="

python 04b_reorganize_by_cluster.py "$VIDEO_NAME" --mode "$MODE" --create-zip

# Validate output
check_dir_exists "$REORGANIZE_DIR" "Reorganized cluster directory" || exit 1

# Count cluster directories
CLUSTER_COUNT=$(find "$REORGANIZE_DIR" -maxdepth 1 -type d -name "*_cluster-*" 2>/dev/null | wc -l)
if [ "$CLUSTER_COUNT" -eq 0 ]; then
    log_warning "No cluster directories found in $REORGANIZE_DIR"
else
    log_success "Found $CLUSTER_COUNT cluster directories"
fi
echo ""

# ============================================================================
# Pipeline Complete
# ============================================================================
log_info "=========================================="
log_success "PARTIAL PIPELINE (03-04b) COMPLETE!"
log_info "=========================================="
echo ""
log_info "Output locations:"
log_info "  - Face tracking:      $TRACKING_DIR"
log_info "  - Face clustering:    $CLUSTERING_OUTPUT"
log_info "  - Reorganized faces:  $REORGANIZE_DIR"
log_info "  - Cluster zip file:   ${SCRATCH_DIR}/output/face_tracking_by_cluster/${VIDEO_NAME}.zip"
echo ""
log_success "Steps 03-04b completed successfully for video: $VIDEO_NAME"
