#!/bin/bash

TELESCOPE=$1

sbatch --export=TELESCOPE=$TELESCOPE --job-name="$TELESCOPE-12" submit_12.sh
sbatch --export=TELESCOPE=$TELESCOPE --job-name="$TELESCOPE-snrf" submit_snrf.sh
sbatch --export=TELESCOPE=$TELESCOPE --job-name="$TELESCOPE-snrp" submit_snrp.sh
sbatch --export=TELESCOPE=$TELESCOPE --job-name="$TELESCOPE-snrg" submit_snrg.sh