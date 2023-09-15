#!/bin/bash

TELESCOPE=$1

source /n/home07/yitians/setup_torch.sh
cd /n/home07/yitians/all_sky_gegenschein/axion-mirror/estimates

VAR_FLAG='nofree'

echo $TELESCOPE "fullinfo" $VAR_FLAG
python 3_snr.py --config $TELESCOPE --pop fullinfo --var $VAR_FLAG

echo $TELESCOPE "partialinfo" $VAR_FLAG
python 3_snr.py --config $TELESCOPE --pop partialinfo --var $VAR_FLAG

echo $TELESCOPE "obs" $VAR_FLAG
python 4_snr_obs.py --config $TELESCOPE --var $VAR_FLAG

echo 'complete!'