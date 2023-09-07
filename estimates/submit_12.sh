#!/bin/bash

#SBATCH --partition=iaifi_gpu_mig
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:00:00
#SBATCH --output=/n/home07/yitians/all_sky_gegenschein/axion-mirror/outputs/slurm/%x_%A_%a.out
#SBATCH --error=/n/home07/yitians/all_sky_gegenschein/axion-mirror/outputs/slurm/%x_%A_%a.err
#SBATCH --account=iaifi_lab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yitians@mit.com

source /n/home07/yitians/setup_torch.sh
cd /n/home07/yitians/all_sky_gegenschein/axion-mirror/estimates

echo '1_egrs:'
python 1_egrs.py --config $TELESCOPE

echo '2_gsr:'
python 2_gsr.py	--config $TELESCOPE

echo 'complete!'