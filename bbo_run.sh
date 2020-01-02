#!/usr/bin/env bash

resume=$1
algorithm=$2
identifier=$3
aux="${@:4}"

echo resume $1 algorithm $2

loc=`dirname "%0"`

args="--best-explore-update --algorithm=$algorithm --identifier=$identifier --game=RUN --budget=2000 --replay-memory-factor=32 --warmup-minibatch=10 --n-explore=128 --batch=1024 --learn-iteration=20"

CUDA_VISIBLE_DEVICES=1, python main.py --action-space=1 --resume=$resume --load-last-model $args $aux &

CUDA_VISIBLE_DEVICES=1, python main.py --action-space=2 --resume=$resume --load-last-model $args $aux &
CUDA_VISIBLE_DEVICES=0, python main.py --action-space=3 --resume=$resume --load-last-model $args $aux &

CUDA_VISIBLE_DEVICES=1, python main.py --action-space=5 --resume=$resume --load-last-model $args $aux &
CUDA_VISIBLE_DEVICES=2, python main.py --action-space=10 --resume=$resume --load-last-model $args $aux &

CUDA_VISIBLE_DEVICES=2, python main.py --action-space=20 --resume=$resume --load-last-model $args $aux &
CUDA_VISIBLE_DEVICES=0, python main.py --action-space=40 --resume=$resume --load-last-model $args $aux &

#CUDA_VISIBLE_DEVICES=1, python main.py --action-space=784 --resume=$resume --load-last-model $args $aux &








