#!/bin/bash

TELESCOPE=$1

source /n/home07/yitians/setup_torch.sh
cd /n/home07/yitians/all_sky_gegenschein/axion-mirror/estimates

VAR_FLAGS=("base" "ti1" "tf30" "tf300")

for VAR_FLAG in "${VAR_FLAGS[@]}"; do
    echo $TELESCOPE "obs" $VAR_FLAG
    python 4_snr_obs.py --config $TELESCOPE --var $VAR_FLAG
done

echo 'complete!'