#!/bin/sh
echo -e "Converting the .csv's ...\n"
python ./dataConverter.py '.csv' './data/dat*/csv/*.csv'

echo -e "Making splits ...\n"
python ./create_validation_split.py './data/dat1/csv' './data/dat1' '0.1'
python ./create_validation_split.py './data/dat2/csv' './data/dat2' '0.1'
python ./create_validation_split.py './data/dat3/csv' './data/dat3' '0.1'
python ./create_validation_split.py './data/dat4/csv' './data/dat4' '0.1'
python ./create_validation_split.py './data/dat5/csv' './data/dat5' '0.1'
python ./create_validation_split.py './data/dat6/csv' './data/dat6' '0.1'
python ./create_validation_split.py './data/dat7/csv' './data/dat7' '0.1'
python ./create_validation_split.py './data/dat8/csv' './data/dat8' '0.1'


#pngpath[0]="'./data/dat1/png/trn/*.png'"
#pngpath[1]="'./data/dat2/png/trn/*.png'"
#pngpath[2]="'./data/dat3/png/trn/*.png'"
#pngpath[3]="'./data/dat1/png/tst/*.png'"
#pngpath[4]="'./data/dat2/png/tst/*.png'"
#pngpath[5]="'./data/dat3/png/tst/*.png'"
#echo -e "converting the .pngs ...\n"

#count=0
#for i in {0..5}
#do
#    :
#    python ./dataConverter.py '.png' ${pngpath[$i]}
#done
echo -e "done ...\n"
