#!/bin/sh
echo -e "Converting the .csv's ...\n"
python ./dataConverter.py '.csv' './data/dat*/csv/*.csv'

pngpath[0]="'./data/dat1/png/trn/*.png'"
pngpath[1]="'./data/dat2/png/trn/*.png'"
pngpath[2]="'./data/dat3/png/trn/*.png'"
pngpath[3]="'./data/dat1/png/tst/*.png'"
pngpath[4]="'./data/dat2/png/tst/*.png'"
pngpath[5]="'./data/dat3/png/tst/*.png'"
echo -e "converting the .pngs ...\n"

count=0
for i in {0..5}
do
    :
    python ./dataConverter.py '.png' ${pngpath[$i]}
done
echo -e "done ...\n"
