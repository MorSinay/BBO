#!/usr/bin/env bash

#identifier=$1
#dim=$2
index=$1
#device=$4
aux="${@:2}"

echo device=device dim=$dim index=$index


args="--algorithm=first_order --game=CMP --trust-alg=relu --robust-scaler-lr=0.1"

####CUDA_VISIBLE_DEVICES=$device, python main.py --replay-memory-factor=256 --identifier=$identifier --action-space=$dim --problem-index=$index $args $aux &

CUDA_VISIBLE_DEVICES=0, python main.py --replay-memory-factor=256 --identifier=r_256_r1_r_no_m --action-space=40 --problem-index=$index $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --replay-memory-factor=512 --identifier=r_512_r1_r_no_m --action-space=40 --problem-index=$index $args $aux &
