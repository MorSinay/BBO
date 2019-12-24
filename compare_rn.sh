#!/usr/bin/env bash

algorithm=$1
dim=$2
index=$3
aux="${@:4}"

echo algorithm=$algorithm dim=$dim index=$index

args="--game=CMP --algorithm=$algorithm --grad-clip=0 --budget=800  --replay-memory-factor=80 --batch=1024 --epsilon=0.1 --no-best-explore-update --no-bandage"

CUDA_VISIBLE_DEVICES=1, python main.py  --learn-iteration=4 --explore=cone --identifier=debug --action-space=$dim --problem-index=$index $args $aux &


