#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:titanrtx:4
#SBATCH --mem=150GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=moco_img
#SBATCH --output=moco_img_%A_%a.out

module purge
module load cuda-10.1

python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/moco_img.py \
  -a resnext50_32x4d \
  --lr 0.015 \
  --batch-size 256 \
  --mlp \
  --moco-t 0.2 \
  --aug-plus --cos \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed \
  --world-size 1 --rank 0 \
  --start-epoch 0 \
  --resume '' \
  '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/S_data_5fps_2000cls_pytorch/'

echo "Done"
