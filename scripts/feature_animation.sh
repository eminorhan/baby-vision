#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1080ti:2
#SBATCH --mem=150GB
#SBATCH --time=1:00:00
#SBATCH --array=0
#SBATCH --job-name=feature_animation
#SBATCH --output=feature_animation_%A_%a.out

module purge
module load cuda-10.1

python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/feature_animation.py '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/feature_animation_imgs_intphys/' --model-path '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/self_supervised_models/TC-SAY.tar' --batch-size 900 --n_out 6269 --feature-idx 600

echo "Done"
