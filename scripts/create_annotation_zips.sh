#!/bin/bash

# Script to create zip files for successfully completed annotation episodes
# These episodes have completed all steps (01-04) and are ready for annotation

# Load environment
source /orcd/home/002/yibei/char-tracker/.env

# Successfully completed episodes
EPISODES=(
    "friends_s02e02b"
    "friends_s02e23b"
    "friends_s03e04a"
    "friends_s03e21a"
    "friends_s04e03b"
    "friends_s04e14a"
    "friends_s04e22b"
    "friends_s05e02a"
    "friends_s05e12b"
    "friends_s05e22a"
    "friends_s06e05b"
    "friends_s06e13a"
    "friends_s06e23b"
)

# Output directory for zip files
ZIP_OUTPUT_DIR="${SCRATCH_DIR}/annotation_zips"
mkdir -p "$ZIP_OUTPUT_DIR"

echo "=========================================="
echo "Creating annotation zip files"
echo "=========================================="
echo "Source: ${SCRATCH_DIR}/output/face_tracking"
echo "Output: ${ZIP_OUTPUT_DIR}"
echo "Total episodes: ${#EPISODES[@]}"
echo "=========================================="
echo ""

# Counter for progress
SUCCESS_COUNT=0
FAIL_COUNT=0

micromamba activate friends_char_track

# Process each episode
for episode in "${EPISODES[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing: $episode"
    
    # Run 04b script to create symlink structure and zip
    python scripts/04b_reorganize_by_cluster.py \
        "$episode" \
        --mode symlink \
        --create-zip
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Created zip for: $episode"
        ((SUCCESS_COUNT++))
        
        # Check if zip file was created
        ZIP_FILE="${SCRATCH_DIR}/output/face_tracking/${episode}/${episode}_annotation.zip"
        if [ -f "$ZIP_FILE" ]; then
            ZIP_SIZE=$(du -h "$ZIP_FILE" | cut -f1)
            echo "  → Zip file: $ZIP_FILE (${ZIP_SIZE})"
        fi
    else
        echo "[FAILED] Error processing: $episode"
        ((FAIL_COUNT++))
    fi
    
    echo ""
done

echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Total episodes: ${#EPISODES[@]}"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "=========================================="

# List all created zip files
echo ""
echo "Created zip files:"
find "${SCRATCH_DIR}/output/face_tracking" -name "*_annotation.zip" -type f -exec ls -lh {} \; | tail -n 20
