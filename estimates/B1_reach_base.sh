#!/bin/bash

CONFIG=$1

source /n/home07/yitians/setup_torch.sh
cd /n/home07/yitians/all_sky_gegenschein/axion-mirror/estimates

echo 'reach egrs:'
python reach.py --config $CONFIG --source egrs --save_name egrs
echo 'reach gsr:'
python reach.py	--config $CONFIG --source gsr --save_name gsr
echo 'reach snrf-base:'
python reach.py	--config $CONFIG --source snr-fullinfo-base --save_name snrf-base
echo 'reach snrp-base'
python reach.py	--config $CONFIG --source snr-partialinfo-base --save_name snrp-base
echo 'reach snro-base:'
python reach.py	--config $CONFIG --source snr-obs-base --save_name snro-base
echo 'reach snrg-base:'
python reach.py	--config $CONFIG --source snr-graveyard-base --save_name snrg-base
echo 'reach total-base:'
python reach.py	--config $CONFIG --source egrs gsr snr-fullinfo-base snr-partialinfo-base snr-graveyard-base --save_name total-base

echo 'reach base complete!'