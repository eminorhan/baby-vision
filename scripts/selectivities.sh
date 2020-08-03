#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64GB
#SBATCH --time=1:00:00
#SBATCH --array=0
#SBATCH --job-name=selectivity
#SBATCH --output=selectivity_%A_%a.out

module purge
module load cuda-10.1

python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/selectivities.py \
--n_out 1000 \
--model-path '' \
--layer 18 \
'/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/S_clean_labeled_data_1fps_4/'

echo "Done"
