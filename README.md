# char-tracker

A comprehensive video face tracking and clustering pipeline for analyzing character appearances in video content.

## Pipeline Overview

The pipeline consists of 5 sequential steps:

1. **Scene Detection** (`01_scene_detection.py`) - Detects scene boundaries in videos
2. **Face Detection** (`02_face_detection.py`) - Detects faces in each frame
3. **Within-Scene Tracking** (`03_within_scene_tracking.py`) - Tracks faces across frames within scenes
4. **Face Clustering** (`04_face_clustering.py`) - Clusters similar faces across the entire video
5. **Reorganize by Cluster** (`04b_reorganize_by_cluster.py`) - Organizes face images by character cluster

## Running the Pipeline

### Single Video Processing

To process a single video through the entire pipeline (steps 01-04b):

```bash
cd scripts
./run_pipeline_01_to_04b.sh <video_name> [--mode copy|move|symlink]
```

**Arguments:**
- `<video_name>`: Name of the video file (without .mp4 extension)
- `--mode`: (Optional) How to organize files in step 04b
  - `copy` (default): Copy files to cluster directories
  - `move`: Move files to cluster directories
  - `symlink`: Create symbolic links to original files

**Example:**
```bash
./run_pipeline_01_to_04b.sh friends_s01e01
./run_pipeline_01_to_04b.sh friends_s01e02 --mode symlink
```

### Batch Processing with SLURM

#### Full Pipeline (Recommended)

Process multiple videos through the entire pipeline (01-04b) using SLURM array jobs:

```bash
cd scripts

# Run with default settings (symlink mode)
sbatch run_pipeline_01_to_04b_batch.sh

# Or specify a different mode using environment variable
MODE=copy sbatch run_pipeline_01_to_04b_batch.sh
```

**Resource Allocation:**
- **Time**: 8 hours per video
- **Memory**: 4 GB (required for face detection step)
- **GPU**: 1 GPU per job
- **Array**: Processes videos listed in `data/episode_id.txt`


**Note:** Log files are automatically created in `logs/` directory at the repository root.

#### Individual Steps (Advanced)

For running steps separately with more control:

```bash
sbatch 01_scene_detection.sh
sbatch 02_face_detection.sh
sbatch 03_within_scene_tracking.sh
sbatch 04_face_clustering.sh
```

## Prerequisites

1. **Environment Setup:**
   - micromamba environment named `char-tracker`
   - Required packages specified in `env.yaml`

2. **Environment Variables:**
   - Set `SCRATCH_DIR` in `.env` file pointing to your data directory

3. **Data Structure:**
   ```
   ${SCRATCH_DIR}/
   ├── data/
   │   └── mkv2mp4/          # Input videos (.mp4)
   └── output/
       ├── scene_detection/   # Scene boundary files (.txt)
       ├── face_detection/    # Face detection results (.json)
       ├── face_tracking/     # Tracked faces per video
       ├── face_clustering/   # Clustering results
       └── face_tracking_by_cluster/  # Reorganized by character
   ```

## Pipeline Features

### Error Handling
- Validates input video exists before starting
- Checks output files after each step
- Exits on error to prevent cascading failures
- Colored logging for easy debugging

### Transition Validation
The pipeline script automatically validates:
- Input video file exists
- Scene detection output created
- Face detection output created
- Tracking directory and files created
- Face images saved during tracking
- Clustering output created
- Cluster directories created

### Logging
- **INFO** (Blue): General information
- **SUCCESS** (Green): Successful operations
- **WARNING** (Yellow): Non-critical issues
- **ERROR** (Red): Critical failures

## Individual Step Details

### Step 01: Scene Detection
```bash
python 01_scene_detection.py <video_name> [--detector content|adaptive|hash]
```
Detects scene boundaries using PySceneDetect.

### Step 02: Face Detection
```bash
python 02_face_detection.py <video_name>
```
Detects faces in video frames using a face detection model.

### Step 03: Within-Scene Tracking
```bash
python 03_within_scene_tracking.py <video_name> [options]
```
Options:
- `--iou-threshold`: Minimum IoU for tracking (default: 0.5)
- `--max-gap`: Max missing frames before track ends (default: 2)
- `--box-expansion`: Box expansion ratio (default: 0.1)
- `--use-median-box/--no-median-box`: Use median box for stability

### Step 04: Face Clustering
```bash
python 04_face_clustering.py <video_name>
```
Clusters faces using embeddings and similarity thresholds.

### Step 04b: Reorganize by Cluster
```bash
python 04b_reorganize_by_cluster.py <video_name> [--mode copy|move|symlink]
```
Organizes face images into directories by character cluster.

## Output Files

Each video generates:
- `{video_name}.txt` - Scene boundaries
- `{video_name}.json` - Face detection results
- `{video_name}_tracked_faces.json` - Tracking results
- `{video_name}_selected_frames_per_face.json` - Selected representative frames
- `{video_name}_matched_faces_with_clusters.json` - Clustering results
- `{video_name}/` - Directory with cluster subdirectories

## Monitoring SLURM Jobs

After submitting a batch job with `sbatch`, you can monitor progress using:

```bash
# Check job status
squeue -u $USER

# View specific job details
squeue -j <job_id>

# Check logs in real-time
tail -f logs/pipeline_01_04b_<job_id>.out
tail -f logs/pipeline_01_04b_<job_id>.err

# Cancel a job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# View completed job info
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

**Log Files:**
- Output logs: `logs/pipeline_01_04b_<job_id>.out`
- Error logs: `logs/pipeline_01_04b_<job_id>.err`
- One log pair per array task

## Troubleshooting

### SLURM Issues

**Job pending (PD) for long time:**
- Check cluster load with `squeue`
- Verify resource requests are reasonable
- Check node exclusions in script

**Job fails immediately:**
- Check log files in `logs/` directory
- Verify conda environment exists: `conda env list`
- Ensure TASK_FILE exists and has correct format
- Verify SCRATCH_DIR is set in `.env`

**Out of memory errors:**
- Step 02 (face detection) needs 16GB
- Increase `--mem` if processing high-resolution videos
- Check `sacct` output for MaxRSS to see actual memory usage

**GPU not available:**
- Verify partition allows GPU access
- Check GPU availability: `sinfo -p <partition>`

### Pipeline Issues

**Pipeline fails at step 03:**
- Check if scenes were detected in step 01
- Verify face detection found faces in step 02

**No cluster directories created:**
- May indicate no faces were successfully tracked
- Check intermediate outputs from steps 02 and 03

**SCRATCH_DIR not set:**
- Create `.env` file with `SCRATCH_DIR=/path/to/your/data`
