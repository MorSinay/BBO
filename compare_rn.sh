#!/usr/bin/env bash

dim=$1
index=$2
device=$3
aux="${@:4}"

echo device=device dim=$dim index=$index

args="--algorithm=first_order --game=CMP"

CUDA_VISIBLE_DEVICES=$device, python main.py  --best-explore-update --identifier=1_relu --action-space=$dim --problem-index=$index $args $aux &
#CUDA_VISIBLE_DEVICES=0, python main.py --spline --robust-scaler-lr=0.15 --learn-iteration=40 --stop-con=100 --best-explore-update --warmup-factor=1 --epsilon=0.2 --epsilon-factor=1 --identifier=spline --action-space=$dim --problem-index=$index $args $aux &

