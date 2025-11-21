# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A video processing pipeline for detecting, tracking, clustering, and identifying faces in TV episodes (Friends dataset). The system processes video through seven sequential stages, from raw video to character appearance events per scene.

## Pipeline Architecture

The pipeline follows a strict sequential data flow where each stage depends on outputs from previous stages:

1. **Scene Detection** (`01_scene_detection.py`) - Segments video into scenes using PySceneDetect (adaptive/content/hash detectors)
2. **Face Detection** (`02_face_detection.py`) - Uses MTCNN via facenet-pytorch to detect faces in all frames
3. **Within-Scene Tracking** (`03_within_scene_tracking.py`) - Tracks faces across frames within each scene using IoU matching with best-match assignment and optional median box smoothing, selects top 3 frames per unique face
4. **Face Clustering** (`04_face_clustering.py`) - Clusters unique faces across scenes using FaceNet embeddings and similarity thresholding
4b. **Cluster Reorganization** (`04b_reorganize_by_cluster.py`) - Optional: reorganizes face images into cluster-based directory structure
5. **Character Embedding** (`05_char_embedding.py`) - Extracts reference embeddings for the 6 main characters (Chandler, Joey, Monica, Phoebe, Rachel, Ross) over episode windows
6. **Cluster-Face Matching** (`06_cluster_face_match.py`) - Matches clusters to known characters using cosine similarity against reference embeddings
7. **Event Generation** (`07_get_char_event.py`) - Produces final CSV with character presence per scene

## Core Modules

- `src/scene_detector.py` - SceneDetector class wrapping PySceneDetect detectors
- `src/face_detector.py` - FaceDetector using MTCNN for detection and annotation
- `src/face_tracker.py` - FaceTracker (IoU-based tracking), FrameSelector (quality-based frame selection)
- `src/face_clusterer.py` - FaceEmbedder (configurable: InceptionResnetV1 or insightface buffalo_l), FaceClusterer (similarity-based graph clustering)
- `src/utils.py` - Currently minimal; intended for shared utilities

## Environment Setup

Required environment variables in `.env`:
- `SCRATCH_DIR` - Storage for all data and outputs (data, intermediate outputs, final results)

Setup environment:
```bash
# Create environment from env.yaml
micromamba env create -f env.yaml

# Activate environment
micromamba activate face-track
```

Dependencies (see `env.yaml` for complete list): PyTorch with CUDA 12.4, facenet-pytorch, insightface, opencv, python-dotenv, scenedetect, tqdm, scikit-learn, scipy, networkx, pandas, natsort, pytest, onnxruntime

## Development Commands

Run pipeline stages sequentially:
```bash
python scripts/01_scene_detection.py <episode_id> --detector adaptive
python scripts/02_face_detection.py <episode_id>
python scripts/03_within_scene_tracking.py <episode_id> --iou-threshold 0.5 --max-gap 2
python scripts/04_face_clustering.py <episode_id> [--model-name vggface2|senet50_256]
python scripts/04b_reorganize_by_cluster.py <episode_id> --mode copy  # Optional
python scripts/05_char_embedding.py <season_id> [--model-name vggface2|senet50_256]
python scripts/06_cluster_face_match.py <episode_id>
python scripts/07_get_char_event.py  # Processes all episodes from data/episode_id.txt
```

**Embedding Models:**
- `vggface2` (default): InceptionResnetV1 pretrained on VGGFace2 (512-dim, 160×160 input)
- `senet50_256`: insightface buffalo_l model (512-dim, 112×112 input, ArcFace loss)

**IMPORTANT:** Embedding models are incompatible. Switching models requires re-running:
- Stage 4 (clustering) for all episodes
- Stage 5 (character embeddings) for all seasons
- Stage 6 (cluster matching) for all episodes

For batch processing via SLURM:
```bash
sbatch scripts/06_cluster_face_match.sh  # Adjust --array range in .sh file
```

Run tests:
```bash
pytest                    # All tests
pytest -k test_name       # Specific test
```

## Data Flow Patterns

All scripts follow this pattern:
1. Load `.env` for directory paths
2. Read input from `$NESE_DIR/output/<prior_stage>/` or `$SCRATCH_DIR/output/<prior_stage>/`
3. Write output to appropriate output directory
4. Use episode ID format: `friends_s<season>e<episode>` (e.g., `friends_s01e01b`)

Key output paths (all under `$SCRATCH_DIR`):
- Scene detection: `$SCRATCH_DIR/output/scene_detection/<episode_id>.txt`
- Face detection: `$SCRATCH_DIR/output/face_detection/<episode_id>.json`
- Face tracking: `$SCRATCH_DIR/output/face_tracking/<episode_id>/<episode_id>_selected_frames_per_face.json`
- Face tracking (by cluster): `$SCRATCH_DIR/output/face_tracking_by_cluster/<episode_id>/<episode_id>_cluster-XX/`
- Face clustering: `$SCRATCH_DIR/output/face_clustering/<episode_id>_matched_faces_with_clusters.json`
- Character embeddings: `$SCRATCH_DIR/output/char_ref_embs/<season>_e<start>-e<end>_char_<id>_embeddings.npy`
- Cluster matching: `$SCRATCH_DIR/output/cluster_face_matching/<episode_id>_cluster-face_matching.json`
- Final events: `$SCRATCH_DIR/output/face_event/<episode_id>_event.csv`

## Important Implementation Details

**Hardcoded paths**: All scripts add `/om2/user/yibei/face-track/src` to `sys.path`. When developing, export `PYTHONPATH=$(pwd)/src` to use relative imports.

**GPU usage**: CUDA is enabled when available. Check `nvidia-smi` before SLURM submissions.

**IoU tracking**: `FaceTracker` uses best-match assignment (highest IoU above threshold) with configurable parameters:
- `iou_threshold=0.5` - Minimum IoU to link detections (default: 0.5)
- `max_gap=2` - Max missing frames before track dies (default: 2)
- `box_expansion=0.1` - Expands boxes 10% before IoU calculation to tolerate head movement (default: 0.1)
- `use_median_box=True` - Uses median of last 5 detections for stability (default: True)

**Clustering parameters**:
- Face clustering: `similarity_threshold=0.6`, `max_iterations=100` (see `scripts/04_face_clustering.py:84`)
- Character matching: Uses dynamic thresholds based on internal similarity distribution with `scale_factor=0.5`

**Episode windows**: Character embedding (stage 5) uses 2-episode sliding windows to ensure sufficient training data across episodes.

## Code Style

Follow existing patterns: PEP 8, snake_case, 4-space indentation. Commit messages are concise lowercase imperatives (e.g., "refine clustering", "adjust min_n_faces"). Keep docstrings focused on parameter meanings and data flow between stages.

## Testing

Place tests in `tests/<module>_test.py`. Mock heavy video I/O operations. Focus on JSON schema validation to ensure compatibility between pipeline stages.
