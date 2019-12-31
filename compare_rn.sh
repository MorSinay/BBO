#!/usr/bin/env bash

algorithm=$1
dim=$2
index=$3
aux="${@:4}"

echo algorithm=$algorithm dim=$dim index=$index

args="--best-explore-update --algorithm=$algorithm --game=CMP --budget=800 --replay-memory-factor=80 --warmup-minibatch=1 --n-explore=1024 --batch=1024 --learn-iteration=20"

CUDA_VISIBLE_DEVICES=0, python main.py --agent=trust --identifier=trust --action-space=$dim --problem-index=$index $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --agent=robust --identifier=robust --action-space=$dim --problem-index=$index $args $aux &
CUDA_VISIBLE_DEVICES=2, python main.py --agent=single --identifier=single --action-space=$dim --problem-index=$index $args $aux &


