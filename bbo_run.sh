#!/usr/bin/env bash

resume=$1
index=$2
aux="${@:3}"

echo resume $1 index $2

args="--algorithm=bbo --game=reinforce --tensorboard --budget=10000"

CUDA_VISIBLE_DEVICES=1, python main.py --identifier=debug2 --tensor --action-space=2 --resume=$resume --load-last-model $args $aux &
CUDA_VISIBLE_DEVICES=2, python main.py --identifier=debug3 --tensor --action-space=3 --resume=$resume --load-last-model $args $aux &

CUDA_VISIBLE_DEVICES=3, python main.py --identifier=debug5 --tensor --action-space=5 --resume=$resume --load-last-model $args $aux &
CUDA_VISIBLE_DEVICES=3, python main.py --identifier=debug10 --tensor --action-space=10 --resume=$resume --load-last-model $args $aux &

CUDA_VISIBLE_DEVICES=2, python main.py --identifier=debug20 --tensor --action-space=20 --resume=$resume --load-last-model $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --identifier=debug40 --tensor --action-space=40 --resume=$resume --load-last-model $args $aux &