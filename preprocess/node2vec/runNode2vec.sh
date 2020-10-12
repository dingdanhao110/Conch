#!/usr/bin/env bash
#python3 ./main.py  --input ..\..\data\cora\homograph.txt --output ..\..\data\cora\AP_128.emb --dimensions 128 --workers 56 --num-walks 1000 --window-size 10 --walk-length 100
python3 ./main.py  --input ../../data/dblp2/homograph.txt --output ../../data/dblp2/APC_16.emb --dimensions 16 --workers 56