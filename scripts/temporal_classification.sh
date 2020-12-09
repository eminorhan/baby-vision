#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=150GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=tempclas
#SBATCH --output=tempclas_%A_%a.out

module purge
module load cuda-10.1

#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/temporal_classification.py --model 'resnext50_32x4d' --n_out 6269 --resume 'resnext50_32x4d_augmentstrong_batch256_True_SAY_5_288_epoch_15.tar' --start-epoch 16 '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/SAY_data_5fps_2000cls_pytorch/'

#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/temporal_classification.py --model 'resnext50_32x4d' --n_out 2765 '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/S_data_5fps_2000cls_pytorch/'

#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/temporal_classification.py --model 'resnext50_32x4d' --n_out 1786 '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/A_data_5fps_2000cls_pytorch/'

python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/temporal_classification.py --model 'resnext50_32x4d' --n_out 1718 '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/Y_data_5fps_2000cls_pytorch/'

echo "Done"
