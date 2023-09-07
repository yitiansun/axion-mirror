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

# only for local runs
TELESCOPE=$1

source /n/home07/yitians/setup_torch.sh
cd /n/home07/yitians/all_sky_gegenschein/axion-mirror/estimates

VAR_FLAGS=("base" "ti1" "tf30" "tf300")

for VAR_FLAG in "${VAR_FLAGS[@]}"; do
    echo $TELESCOPE "obs" $VAR_FLAG
    python 4_snr_obs.py --config $TELESCOPE --var $VAR_FLAG
done

echo 'complete!'