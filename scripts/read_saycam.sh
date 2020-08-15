#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=read_saycam
#SBATCH --output=read_saycam_%A_%a.out

module purge
module load cuda-10.1

python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/read_saycam.py \
  --save-dir '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/S_data_5fps_1000cls_pytorch/' \
  --fps 5 \
  --seg-len 576 \
  '/misc/vlgscratch4/LakeGroup/emin/headcam/data_2/S'

echo "Done"
