#!/usr/bin/env bash

resume=$1
algorithm=$2
dim=$3
aux="${@:4}"

echo resume $1 algorithm $2 dim $3

loc=`dirname "%0"`

args="--best-explore-update --algorithm=$algorithm --action-space=$dim --game=RUN --budget=150000 --warmup-minibatch=5 --printing-interval=100"

CUDA_VISIBLE_DEVICES=0, python main.py --trust-factor=0.5 --epsilon=0.1 --epsilon-factor=1 --identifier=e1_f1_tr5 --resume=$resume --load-last-model $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --trust-factor=0.5 --epsilon=0.2 --epsilon-factor=1 --identifier=e2_f1_tr5 --resume=$resume --load-last-model $args $aux &
CUDA_VISIBLE_DEVICES=2, python main.py --trust-factor=0.5 --epsilon=0.3 --epsilon-factor=0.75 --identifier=e3_f75_tr5 --resume=$resume --load-last-model $args $aux &

#CUDA_VISIBLE_DEVICES=0, python main.py --spline --learn-iteration=20 --epsilon=0.1 --epsilon-factor=1 --identifier=e1_f1_spline --resume=$resume --load-last-model $args $aux &






