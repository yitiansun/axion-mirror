#!/bin/bash

CONFIG=$1

source /n/home07/yitians/setup_torch.sh
cd /n/home07/yitians/all_sky_gegenschein/axion-mirror/estimates

VAR_FLAG="nofree"

echo "reach snrf-$VAR_FLAG"
python reach.py	--config $CONFIG --source snr-fullinfo-$VAR_FLAG --save_name snrf-$VAR_FLAG
echo "reach snrp-$VAR_FLAG"
python reach.py	--config $CONFIG --source snr-partialinfo-$VAR_FLAG --save_name snrp-$VAR_FLAG
echo "reach snro-$VAR_FLAG"
python reach.py	--config $CONFIG --source snr-obs-$VAR_FLAG --save_name snro-$VAR_FLAG
echo "reach total-$VAR_FLAG"
python reach.py	--config $CONFIG --source egrs gsr snr-obs-$VAR_FLAG snr-graveyard-base --save_name total-$VAR_FLAG

echo 'reach nofree complete!'