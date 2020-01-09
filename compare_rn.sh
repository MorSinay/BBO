#!/usr/bin/env bash

algorithm=$1
dim=$2
index=$3
aux="${@:4}"

echo algorithm=$algorithm dim=$dim index=$index

args="--algorithm=$algorithm --game=CMP --budget=150000 --warmup-minibatch=5 --printing-interval=50"

CUDA_VISIBLE_DEVICES=1, python main.py --best-explore-update --warmup-factor=1 --epsilon=0.2 --epsilon-factor=0.8 --identifier=bst_e2_f8 --action-space=$dim --problem-index=$index $args $aux &

