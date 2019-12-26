#!/usr/bin/env bash

algorithm=$1
dim=$2
index=$3
aux="${@:4}"

echo algorithm=$algorithm dim=$dim index=$index

args="--game=CMP --algorithm=$algorithm --debug --grad-clip=0 --budget=1500  --replay-memory-factor=80 --batch=512 --epsilon=0.1 --no-best-explore-update"

CUDA_VISIBLE_DEVICES=3, python main.py  --pi-lr=1e-2 --learn-iteration=6 --explore=cone --identifier=i20lr2 --action-space=$dim --problem-index=$index $args $aux &


