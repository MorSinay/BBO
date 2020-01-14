#!/usr/bin/env bash

dim=$1
aux="${@:2}"

echo dim $1

loc=`dirname "%0"`

args="--best-explore-update --algorithm=first_order --action-space=$dim --game=RUN --budget=150000 --warmup-minibatch=5 --printing-interval=100 --learn-iteration=60"

CUDA_VISIBLE_DEVICES=0, python main.py --start=0 --stop-con=100 --trust-factor=0.75 --epsilon=0.2 --epsilon-factor=1 --identifier=n_th_es --resume=0 --load-last-model $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --start=1 --stop-con=100 --trust-factor=0.75 --epsilon=0.2 --epsilon-factor=1 --identifier=n_th_es --resume=0 --load-last-model $args $aux &
CUDA_VISIBLE_DEVICES=2, python main.py --start=2 --stop-con=100 --trust-factor=0.75 --epsilon=0.2 --epsilon-factor=1 --identifier=n_th_es --resume=0 --load-last-model $args $aux &
CUDA_VISIBLE_DEVICES=3, python main.py --start=3 --stop-con=100 --trust-factor=0.75 --epsilon=0.2 --epsilon-factor=1 --identifier=n_th_es --resume=0 --load-last-model $args $aux &
#CUDA_VISIBLE_DEVICES=1, python main.py --trust-factor=0.5 --epsilon=0.2 --epsilon-factor=1 --identifier=e2_f1_tr5 --resume=$resume --load-last-model $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --trust-factor=0.5 --epsilon=0.3 --epsilon-factor=0.75 --identifier=e3_f75_tr5 --resume=$resume --load-last-model $args $aux &

#CUDA_VISIBLE_DEVICES=0, python main.py --spline --learn-iteration=20 --epsilon=0.1 --epsilon-factor=1 --identifier=e1_f1_spline --resume=$resume --load-last-model $args $aux &






