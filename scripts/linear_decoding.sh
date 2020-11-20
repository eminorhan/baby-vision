#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:titanrtx:2
#SBATCH --mem=150GB
#SBATCH --time=12:00:00
#SBATCH --array=0
#SBATCH --job-name=linear_decoding
#SBATCH --output=linear_decoding_%A_%a.out

module purge
module load cuda-10.1

#python -u /misc/vlgscratch4/LakeGroup/emin/baby_vision/linear_decoding.py '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/S_clean_labeled_data_1fps_4/' --model-name 'random' --num-classes 26 --subsample

#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/linear_decoding.py '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/S_clean_labeled_data_1fps_4/' --model-name 'moco_img_0005' --num-classes 26 --subsample

#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/linear_decoding.py '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/S_clean_labeled_data_1fps_4/' --model-name 'moco_temp_0005' --num-classes 26 --subsample 

#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/linear_decoding.py '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/S_clean_labeled_data_1fps_5/' --model-name 'TC-S' --num-outs 2765

#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/linear_decoding.py '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/S_clean_labeled_data_1fps_4/' --model-name 'mobilenetV2_A_5fps_2000cls_coloraug' --num-outs 1786

#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/linear_decoding.py '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/S_clean_labeled_data_1fps_4/' --model-name 'mobilenetV2_Y_5fps_2000cls_coloraug' --num-outs 1718

python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/linear_decoding.py '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/S_clean_labeled_data_1fps_5/' --model-name 'TC-SAY' --num-outs 6269

#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/linear_decoding.py '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/S_clean_labeled_data_1fps_5/' --model-name 'TC-S' --num-outs 2765

echo "Done"
