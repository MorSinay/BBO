#!/usr/bin/env bash

algorithm=$1
dim=$2
index=$3
aux="${@:4}"

echo algorithm=$algorithm dim=$dim index=$index

args="--algorithm=$algorithm --game=CMP --budget=2000 --replay-memory-factor=32 --warmup-minibatch=10 --n-explore=128 --batch=1024 --learn-iteration=20"

CUDA_VISIBLE_DEVICES=0, python main.py --pi-lr=1e-3 --identifier=lr3 --action-space=$dim --problem-index=$index $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --pi-lr=1e-2 --identifier=lr2 --action-space=$dim --problem-index=$index $args $aux &
CUDA_VISIBLE_DEVICES=2, python main.py --epsilon=0.1 --identifier=epsilon1 --action-space=$dim --problem-index=$index $args $aux &
CUDA_VISIBLE_DEVICES=3, python main.py --epsilon=0.2 --identifier=epsilon2 --action-space=$dim --problem-index=$index $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --epsilon=0.3 --identifier=epsilon3 --action-space=$dim --problem-index=$index $args $aux &
CUDA_VISIBLE_DEVICES=2, python main.py --spline --identifier=spline --action-space=$dim --problem-index=$index $args $aux &


