#!/bin/bash

# run.sh

# --
#
mkdir -p "experiment/freebase/"

lr="0.001 0.0005 0.0003 0.0001 0.00005"
prep="node_embedding"
aggr="attention"
head="1 2 4 8"
dropout="0 0.1 0.3 0.5 0.7"
attn_dropout="0 0.1 0.3 0.5 0.7"
count="1"
for l in $lr; do
for p in $prep; do
for a in $aggr; do
for h in $head; do
for d in $dropout; do
for ad in $attn_dropout; do
python3 ./train.py \
    --problem-path ../../data/freebase/ \
    --problem yago \
    --epochs 10000 \
    --batch-size 999999 \
    --lr-init $l \
    --lr-schedule constant\
    --prep-class $p \
    --dropout $d \
    --attn-dropout $ad \
    --aggregator-class $a \
    --n-head $h\
    > "experiment/freebase/fb_"$count".txt" 
let count++
done
done
done
done
done
done