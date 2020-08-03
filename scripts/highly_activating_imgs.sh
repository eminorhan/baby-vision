#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --array=0
#SBATCH --job-name=activating_imgs
#SBATCH --output=activating_imgs_%A_%a.out

module purge
module load cuda-10.1

python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/highly_activating_imgs.py '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/S_clean_labeled_data_1fps_4/' --n_out 26 --model-path 'mobilenetV2_S_5fps_2000cls_coloraug_labeled.tar' 

echo "Done"
