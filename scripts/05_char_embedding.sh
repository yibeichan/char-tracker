#!/bin/bash

#SBATCH --job-name=face_embedding
#SBATCH --output=/om2/user/yibei/face-track/logs/%x_%j.out 
#SBATCH --error=/om2/user/yibei/face-track/logs/%x_%j.err 
#SBATCH --partition=normal
#SBATCH --exclude=node[030-070]
#SBATCH --time=00:15:00 
#SBATCH --array=0-5
#SBATCH --ntasks=1 
#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

source $HOME/miniconda3/etc/profile.d/conda.sh
# Activate your Conda environment
conda activate friends_char_track

seasons=("s01" "s02" "s03" "s04" "s05" "s06")

TASK_ID=${seasons[$SLURM_ARRAY_TASK_ID]}

echo "Processing: $TASK_ID"

cd /om2/user/yibei/face-track/scripts
python 05_char_embedding.py "${TASK_ID}"