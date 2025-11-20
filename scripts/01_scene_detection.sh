#!/bin/bash

#SBATCH --job-name=scene_detect 
#SBATCH --output=/om2/user/yibei/face-track/logs/%x_%j.out 
#SBATCH --error=/om2/user/yibei/face-track/logs/%x_%j.err 
#SBATCH --partition=normal
#SBATCH --exclude=node[030-070]
#SBATCH --time=00:10:00 
#SBATCH --array=1-292
#SBATCH --ntasks=1 
#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

source $HOME/miniconda3/etc/profile.d/conda.sh
# Activate your Conda environment
conda activate friends_char_track

TASK_FILE="/om2/user/yibei/face-track/data/episode_id.txt"

TASK_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $TASK_FILE)

echo "Processing: $TASK_ID"

cd /om2/user/yibei/face-track/scripts
python 01_scene_detection.py "${TASK_ID}"