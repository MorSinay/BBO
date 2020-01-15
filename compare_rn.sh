#!/usr/bin/env bash

identifier=$1
dim=$2
index=$3
device=$4
aux="${@:5}"

echo device=device dim=$dim index=$index

args="--algorithm=first_order --game=CMP"

CUDA_VISIBLE_DEVICES=$device, python main.py --n-explore=64  --identifier=$identifier --action-space=$dim --problem-index=$index $args $aux &
#CUDA_VISIBLE_DEVICES=0, python main.py --spline --robust-scaler-lr=0.15 --learn-iteration=40 --stop-con=100 --best-explore-update --warmup-factor=1 --epsilon=0.2 --epsilon-factor=1 --identifier=spline --action-space=$dim --problem-index=$index $args $aux &

