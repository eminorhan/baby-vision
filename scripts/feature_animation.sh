#!/bin/bash

#SBATCH --nodes=1
#SBATCH --exclude=hpc1,hpc2,hpc3,hpc4,hpc5,hpc6,hpc7,hpc8,hpc9,vine3,vine4,vine6,vine11,vine12,lion17,rose7,rose8,rose9
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=100GB
#SBATCH --time=1:00:00
#SBATCH --array=0
#SBATCH --job-name=feature_animation
#SBATCH --output=feature_animation_%A_%a.out

module purge
module load cuda-10.1

python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/feature_animation.py '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/feature_animation_imgs/' --model-path '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/self_supervised_models/mobilenetV2_S_5fps_2000cls_coloraug.tar' --batch-size 900 --n_out 2765 --feature-idx 300

echo "Done"
