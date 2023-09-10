#!/bin/bash

#SBATCH --job-name=pg_srhigh
#SBATCH --array=13
#SBATCH --partition=iaifi_gpu_mig
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:00:00
#SBATCH --output=/n/home07/yitians/all_sky_gegenschein/axion-mirror/outputs/slurm/%x_%a.out
#SBATCH --error=/n/home07/yitians/all_sky_gegenschein/axion-mirror/outputs/slurm/%x_%a.err
#SBATCH --account=iaifi_lab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yitians@mit.com

source /n/home07/yitians/setup_torch.sh

cd /n/home07/yitians/all_sky_gegenschein/axion-mirror/scripts

ISTART=$((SLURM_ARRAY_TASK_ID*10))
IEND=$((SLURM_ARRAY_TASK_ID*10+10)) # end excluded

srun python populate_graveyard.py --var srhigh --istart $ISTART --iend $IEND

echo "Complete!"