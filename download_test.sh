#!/bin/bash

file="data/csv/L12PO_VAL_NO_DUPLICATES.ZIP"
if [ -f "$file" ]
then
    echo "Datasets already downloaded"
else
    echo "Downloading datasets"
    wget -O data/csv/L12PO_VAL_NO_DUPLICATES.ZIP "https://www.dropbox.com/s/gspi26ef7zvfu62/L12PO_VAL_NO_DUPLICATES.ZIP?dl=1"
fi
rm -rf data/csv/test
mkdir data/csv/test
unzip data/csv/L12PO_VAL_NO_DUPLICATES.ZIP -d data/csv/test
rm -rf data/csv/test/*trn*
rm -rf data/csv/test/*val*
