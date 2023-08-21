TELESCOPE='HIRAX1024'

echo '1_egrs:'
python 1_egrs.py --config $TELESCOPE
echo '2_gsr:'
python 2_gsr.py	--config $TELESCOPE
echo '3_snr fullinfo:'
python 3_snr.py --config $TELESCOPE --pop fullinfo
echo '3_snr partialinfo:'
python 3_snr.py --config $TELESCOPE --pop partialinfo
echo '3_snr graveyard:'
python 3_snr.py --config $TELESCOPE --pop graveyard
echo 'reach egrs:'
python reach.py --config $TELESCOPE --src egrs --save egrs
echo 'reach gsr:'
python reach.py	--config $TELESCOPE --src gsr --save gsr
echo 'reach snrf:'
python reach.py	--config $TELESCOPE --src snr-fullinfo --save snrf
echo 'reach snrp'
python reach.py	--config $TELESCOPE --src snr-partialinfo --save snrp
echo 'reach snrg:'
python reach.py	--config $TELESCOPE --src snr-graveyard --save snrg
echo 'reach total:'
python reach.py	--config $TELESCOPE --src egrs gsr snr-fullinfo snr-partialinfo snr-graveyard --save total
