#!/bin/bash

#SBATCH --job-name=boo_srlow
#SBATCH --partition=gpu_mig
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:00:00
#SBATCH --output=/n/home07/yitians/all_sky_gegenschein/axion-mirror/outputs/slurm/%x_%j.out
#SBATCH --error=/n/home07/yitians/all_sky_gegenschein/axion-mirror/outputs/slurm/%x_%j.err
#SBATCH --account=iaifi_lab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yitians@mit.com

source /n/home07/yitians/setup_torch.sh

cd /n/home07/yitians/all_sky_gegenschein/axion-mirror/scripts

srun python populate_graveyard.py --var srlow
#srun python <<< "print('hello world')"