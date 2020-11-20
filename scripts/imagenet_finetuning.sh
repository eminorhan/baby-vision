#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=150GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=finetune_imgnet
#SBATCH --output=finetune_imgnet_%A_%a.out

module purge
module load cuda-10.1

#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/imagenet_finetuning.py --freeze-trunk --n_out 6269 --resume 'resnext50_32x4d_augmentstrong_batch256_True_SAY_5_288_epoch_10.tar'
#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/imagenet_finetuning.py --freeze-trunk --n_out 2765 --resume 'resnext50_32x4d_augmentstrong_batch256_True_S_5_288_epoch_10.tar'
#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/imagenet_finetuning.py --freeze-trunk --n_out 1786 --resume 'resnext50_32x4d_augmentstrong_batch256_True_A_5_288_epoch_10.tar'
#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/imagenet_finetuning.py --freeze-trunk --n_out 1718 --resume 'resnext50_32x4d_augmentstrong_batch256_True_Y_5_288_epoch_10.tar'

#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/imagenet_finetuning.py --frac-retained 0.01 --n_out 6269 --resume 'resnext50_32x4d_augmentstrong_batch256_True_SAY_5_288_epoch_10.tar'
python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/imagenet_finetuning.py --freeze-trunk --n_out 1000 --resume 'ft_IN_resnext50_32x4d_augmentstrong_batch256_True_SAY_5_288_epoch_10.tar'

echo "Done"
