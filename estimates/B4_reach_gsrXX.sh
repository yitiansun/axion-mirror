#!/bin/bash

CONFIG=$1

source /n/home07/yitians/setup_torch.sh
cd /n/home07/yitians/all_sky_gegenschein/axion-mirror/estimates

echo "reach gsrAH"
python reach.py	--config $CONFIG --source gsrAH --save_name gsrAH
echo "reach total-gsrAH"
python reach.py	--config $CONFIG --source egrs gsrAH snr-obs-base snr-graveyard-base --save_name total-gsrAH

echo "reach gsrBH"
python reach.py	--config $CONFIG --source gsrBH --save_name gsrBH
echo "reach total-gsrBH"
python reach.py	--config $CONFIG --source egrs gsrBH snr-obs-base snr-graveyard-base --save_name total-gsrBH

echo 'reach gsrXX complete!'