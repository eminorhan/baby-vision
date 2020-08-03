#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --time=6:00:00
#SBATCH --array=0
#SBATCH --job-name=hog
#SBATCH --output=hog_%A_%a.out

module purge

python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/hog_baseline.py '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/toybox_1fps/'

echo "Done"
