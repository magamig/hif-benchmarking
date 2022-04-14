#!/bin/bash

wget https://www.cs.columbia.edu/CAVE/databases/multispectral/zip/complete_ms_data.zip
unzip complete_ms_data -d data/GT/aux/

mkdir -p data/GT/CAVE/
cp -r data/GT/aux/*/* data/GT/CAVE/

rm -r data/GT/aux/
rm complete_ms_data.zip