#!/usr/bin/env bash

identifier=$1
dim=$2
index=$3
device=$4
aux="${@:5}"

echo identifier=$identifier device=$device dim=$dim index=$index


args="--algorithm=first_order --game=CMP --epsilon=0.1 --epsilon-factor=1 --explore=rand"

CUDA_VISIBLE_DEVICES=$device, python main.py --identifier=$identifier --action-space=$dim --problem-index=$index $args $aux &

