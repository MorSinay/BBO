#!/usr/bin/env bash

algorithm=$1
dim=$2
index=$3
aux="${@:4}"

echo algorithm=$algorithm dim=$dim index=$index

args="--algorithm=$algorithm --game=CMP --budget=100000 --replay-memory-factor=512 --warmup-minibatch=20 --n-explore=64 --batch=1024 --learn-iteration=40 --printing-interval=50"

CUDA_VISIBLE_DEVICES=0, python main.py --best-explore-update --loss=mse --warmup-factor=1 --cone-angle=2 --epsilon=0.2 --epsilon-factor=0.75 --grad-clip=1e-1 --identifier=bst_ep2 --action-space=$dim --problem-index=$index $args $aux &
CUDA_VISIBLE_DEVICES=0, python main.py --best-explore-update --loss=mse --warmup-factor=1 --cone-angle=2 --epsilon=0.2 --epsilon-factor=0.75 --grad-clip=0 --identifier=bst_clip_ep2 --action-space=$dim --problem-index=$index $args $aux &

