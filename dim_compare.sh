#!/usr/bin/env bash

dim=$1
aux="${@:2}"

echo dim $1

loc=`dirname "%0"`

args="--action-space=$dim --game=DIM  --epsilon=0.1 --epsilon-factor=0.97"

CUDA_VISIBLE_DEVICES=0, python main.py --algorithm=first_order --trust-alg=relu --explore=ball --identifier=ball_relu_f $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --algorithm=first_order --trust-alg=relu --explore=cone --identifier=ball_cone_relu_f $args $aux &
CUDA_VISIBLE_DEVICES=2, python main.py --algorithm=first_order --trust-alg=log --explore=ball --identifier=ball_log_f $args $aux &
CUDA_VISIBLE_DEVICES=0, python main.py --algorithm=first_order --trust-alg=log --explore=cone --identifier=ball_log_cone_f $args $aux &






