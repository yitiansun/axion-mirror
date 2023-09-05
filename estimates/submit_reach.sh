#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:00:00
#SBATCH --output=/n/home07/yitians/all_sky_gegenschein/axion-mirror/outputs/slurm/%x_%A_%a.out
#SBATCH --error=/n/home07/yitians/all_sky_gegenschein/axion-mirror/outputs/slurm/%x_%A_%a.err
#SBATCH --account=iaifi_lab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yitians@mit.com

TELESCOPE='CHIME'

source /n/home07/yitians/setup_torch.sh
cd /n/home07/yitians/all_sky_gegenschein/axion-mirror/estimates

echo 'reach egrs:'
python reach.py --config $TELESCOPE --src egrs --save egrs
echo 'reach gsr:'
python reach.py	--config $TELESCOPE --src gsr --save gsr
echo 'reach snrf-base:'
python reach.py	--config $TELESCOPE --src snr-fullinfo-base --save snrf-base
echo 'reach snrp-base'
python reach.py	--config $TELESCOPE --src snr-partialinfo-base --save snrp-base
echo 'reach snrg-base:'
python reach.py	--config $TELESCOPE --src snr-graveyard-base --save snrg-base
echo 'reach total-base:'
python reach.py	--config $TELESCOPE --src egrs gsr snr-fullinfo-base snr-partialinfo-base snr-graveyard-base --save total-base

echo 'reach complete!'