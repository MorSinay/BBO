#!/usr/bin/env bash

algorithm=$1
dim=$2
index=$3
aux="${@:4}"

echo algorithm=$algorithm dim=$dim index=$index

args="--best-explore-update --agent=trust --learn-iteration=10 --pi-lr=1e-2 --algorithm=$algorithm --grad-clip=0 --game=CMP --budget=1500 --explore=cone --alpha=1 --replay-memory-factor=100 --warmup-factor=1 --n-explore=1024 --batch=1024 --epsilon=0.1"

CUDA_VISIBLE_DEVICES=3, python main.py  --identifier=debug --action-space=$dim --problem-index=$index $args $aux &


