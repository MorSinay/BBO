#!/usr/bin/env bash

resume=$1
algorithm=$2
aux="${@:3}"

echo resume $1 algorithm $2

loc=`dirname "%0"`

#case "$algorithm" in
#    ("value") args="--algorithm=$algorithm  --beta-lr=1e-3 --game=RUN --budget=400 --explore=grad_rand --delta=0.1 --epsilon=0.1 --update-step=n_step --no-best-explore-update --grad-clip --no-bandage";;
#    ("first_order") args="--algorithm=$algorithm  --beta-lr=1e-3 --game=RUN --budget=400 --explore=grad_rand --delta=0.01 --epsilon=0.01 --update-step=n_step --no-best-explore-update --grad-clip --no-bandage";;
#    ("second_order") args="--algorithm=$algorithm  --beta-lr=1e-3 --game=RUN --budget=400 --explore=grad_rand --delta=0.01 --epsilon=0.01 --update-step=n_step --no-best-explore-update --grad-clip --no-bandage";;
#    ("anchor") args="--algorithm=$algorithm  --beta-lr=1e-3 --game=RUN --budget=400 --explore=grad_rand --delta=0.01 --epsilon=0.01 --update-step=n_step --no-best-explore-update --grad-clip --no-bandage";;
#    (*) echo "$algorithm: Not Implemented" ;;
#esac

args="--algorithm=$algorithm  --beta-lr=1e-3 --game=RUN --budget=800 --explore=grad_direct --alpha=1 --replay-memory-factor=30 --batch=100 --delta=0.1 --epsilon=0.1 --update-step=n_step --no-best-explore-update --grad-clip=1e-3 --no-bandage"

CUDA_VISIBLE_DEVICES=0, python main.py --identifier=debug1 --action-space=1 --resume=$resume --load-last-model $args $aux &

CUDA_VISIBLE_DEVICES=1, python main.py --identifier=debug2 --action-space=2 --resume=$resume --load-last-model $args $aux &
CUDA_VISIBLE_DEVICES=2, python main.py --identifier=debug3 --action-space=3 --resume=$resume --load-last-model $args $aux &

CUDA_VISIBLE_DEVICES=0, python main.py --identifier=debug5 --action-space=5 --resume=$resume --load-last-model $args $aux &
CUDA_VISIBLE_DEVICES=0, python main.py --identifier=debug10 --action-space=10 --resume=$resume --load-last-model $args $aux &

#CUDA_VISIBLE_DEVICES=2, python main.py --identifier=debug20 --action-space=20 --resume=$resume --load-last-model $args $aux &
#CUDA_VISIBLE_DEVICES=1, python main.py --identifier=debug40 --action-space=40 --resume=$resume --load-last-model $args $aux &

#CUDA_VISIBLE_DEVICES=1, python main.py --identifier=debug784 --action-space=784 --resume=$resume --load-last-model $args $aux &








