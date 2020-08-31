#!/bin/bash

#SBATCH --nodes=1
#SBATCH --exclude=loopy8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2080ti:4
#SBATCH --mem=100GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=temp_class
#SBATCH --output=temp_class_%A_%a.out

module purge
module load cuda-10.1

#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/temporal_classification.py --n_out 6269 '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/SAY_data_5fps_2000cls_pytorch/'
python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/temporal_classification.py --n_out 2765 '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/S_data_5fps_2000cls_pytorch/'
#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/temporal_classification.py --n_out 1786 --resume 'mobilenetV2_A_5fps_2000cls_coloraug.tar'
#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/temporal_classification.py --n_out 1718 --resume 'mobilenetV2_Y_5fps_2000cls_coloraug.tar'
#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/temporal_classification.py --n_out 6269 --resume 'mobilenetV2_SAY_5fps_2000cls_coloraug_5.tar'
#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/temporal_classification.py --n_out 6269 --resume ''

echo "Done"
