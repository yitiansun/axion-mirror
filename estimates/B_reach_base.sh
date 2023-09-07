#!/bin/bash

CONFIG=$1

source /n/home07/yitians/setup_torch.sh
cd /n/home07/yitians/all_sky_gegenschein/axion-mirror/estimates

echo 'reach egrs:'
python reach.py --config $CONFIG --src egrs --save egrs
echo 'reach gsr:'
python reach.py	--config $CONFIG --src gsr --save gsr
echo 'reach snrf-base:'
python reach.py	--config $CONFIG --src snr-fullinfo-base --save snrf-base
echo 'reach snrp-base'
python reach.py	--config $CONFIG --src snr-partialinfo-base --save snrp-base
echo 'reach snro-base:'
python reach.py	--config $CONFIG --src snr-obs-base --save snro-base
echo 'reach snrg-base:'
python reach.py	--config $CONFIG --src snr-graveyard-base --save snrg-base
echo 'reach total-base:'
python reach.py	--config $CONFIG --src egrs gsr snr-fullinfo-base snr-partialinfo-base snr-graveyard-base --save total-base

echo 'reach complete!'