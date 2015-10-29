#!/bin/sh
echo -e "Converting the .csv's ...\n"
python ./dataConverter.py '.csv' './data/dat*/csv/*.csv'

PNGPATH=()
PNGPATH+=("'./data/dat1/png/trn/*'")
PNGPATH+=("'./data/dat2/png/trn/*'")
PNGPATH+=("'./data/dat3/png/trn/*'")
PNGPATH+=("'./data/dat1/png/tst/*'")
PNGPATH+=("'./data/dat2/png/tst/*'")
PNGPATH+=("'./data/dat3/png/tst/*'")
echo -e "converting the .pngs ...\n"
for pth in $PNGPATH
do
    :
    echo $pth    
    #python ./dataConverter.py '.png' $pth
done
echo -e "done ...\n"