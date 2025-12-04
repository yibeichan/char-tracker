#!/bin/bash

#SBATCH --job-name=within_scene_track
#SBATCH --output=/home/yibei/char-tracker/logs/%x_%j.out 
#SBATCH --error=/home/yibei/char-tracker/logs/%x_%j.err 
#SBATCH --partition=normal
#SBATCH --exclude=node[030-070]
#SBATCH --time=04:30:00 
#SBATCH --array=61
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate char-tracker

TASK_FILE="/home/yibei/char-tracker/data/episode_id.txt"

TASK_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $TASK_FILE)

echo "Processing: $TASK_ID"

cd /home/yibei/char-tracker/scripts
python 03_within_scene_tracking.py "${TASK_ID}"
