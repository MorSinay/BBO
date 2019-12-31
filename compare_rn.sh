#!/usr/bin/env bash

algorithm=$1
dim=$2
index=$3
aux="${@:4}"

echo algorithm=$algorithm dim=$dim index=$index

args="--best-explore-update --algorithm=$algorithm --game=CMP --budget=1500 --replay-memory-factor=80 --warmup-minibatch=1 --n-explore=1024 --batch=1024 --learn-iteration=20"

CUDA_VISIBLE_DEVICES=1, python main.py  --identifier=debug --action-space=$dim --problem-index=$index $args $aux &


