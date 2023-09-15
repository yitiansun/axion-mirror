#!/bin/bash

CONFIG=$1

source /n/home07/yitians/setup_torch.sh
cd /n/home07/yitians/all_sky_gegenschein/axion-mirror/estimates

VAR_FLAG="nofree"

echo "reach snrf-$VAR_FLAG"
python reach.py	--config $CONFIG --src snr-fullinfo-$VAR_FLAG --save snrf-$VAR_FLAG
echo "reach snrp-$VAR_FLAG"
python reach.py	--config $CONFIG --src snr-partialinfo-$VAR_FLAG --save snrp-$VAR_FLAG
echo "reach snro-$VAR_FLAG"
python reach.py	--config $CONFIG --src snr-obs-$VAR_FLAG --save snro-$VAR_FLAG
echo "reach total-$VAR_FLAG"
python reach.py	--config $CONFIG --src egrs gsr snr-obs-$VAR_FLAG snr-graveyard-base --save total-$VAR_FLAG

echo 'reach nofree complete!'