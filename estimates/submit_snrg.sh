#!/bin/bash

#SBATCH --array=5
#SBATCH --partition=shared
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=0-23:00:00
#SBATCH --output=/n/home07/yitians/all_sky_gegenschein/axion-mirror/outputs/slurm/%x_%a.out
#SBATCH --error=/n/home07/yitians/all_sky_gegenschein/axion-mirror/outputs/slurm/%x_%a.err
#SBATCH --account=iaifi_lab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yitians@mit.com

source /n/home07/yitians/setup_torch.sh
cd /n/home07/yitians/all_sky_gegenschein/axion-mirror/estimates

VAR_FLAGS=("base" "ti1" "tf30" "tf300" "srlow" "srhigh")
VAR_FLAG=${VAR_FLAGS[${SLURM_ARRAY_TASK_ID}]}

echo $TELESCOPE "graveyard" $VAR_FLAG
python 3_snr.py --config $TELESCOPE --pop graveyard --var $VAR_FLAG

echo 'complete!'

#SBATCH --gres=gpu:1