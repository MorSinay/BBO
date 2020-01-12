#!/usr/bin/env bash

algorithm=$1
dim=$2
index=$3
aux="${@:4}"

echo algorithm=$algorithm dim=$dim index=$index

args="--algorithm=$algorithm --game=CMP --budget=150000 --warmup-minibatch=5 --printing-interval=50"

#CUDA_VISIBLE_DEVICES=1, python main.py --robust-scaler-lr=0.15 --learn-iteration=60 --stop-con=100 --best-explore-update --warmup-factor=1 --epsilon=0.2 --epsilon-factor=1 --identifier=debug --action-space=$dim --problem-index=$index $args $aux &
CUDA_VISIBLE_DEVICES=0, python main.py --spline --robust-scaler-lr=0.15 --learn-iteration=40 --stop-con=100 --best-explore-update --warmup-factor=1 --epsilon=0.2 --epsilon-factor=1 --identifier=spline --action-space=$dim --problem-index=$index $args $aux &

