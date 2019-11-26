#!/usr/bin/env bash

dim=$1
index=$2
aux="${@:3}"

echo dim $1 index $2

#args="--game=VALUE --budget=100 --action-space=$dim --resume=0 --load-last-model --problem-index=$index --no-bandage --delta=0.1 --epsilon=0.1"
#args="--game=GRAD --grad --budget=100 --action-space=$dim --resume=0 --load-last-model --problem-index=$index --no-bandage --delta=0.01 --epsilon=0.01"

args="--game=CMP --budget=800 --delta=0.1 --epsilon=0.1 --best-explore-update --bandage --action-space=$dim --problem-index=$index --beta-lr=1e-1"

CUDA_VISIBLE_DEVICES=0, python main.py --grad --algorithm=grad_d --explore=grad_direct --update-step=n_step $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --grad --algorithm=grad_r --explore=grad_rand --update-step=n_step $args $aux &
CUDA_VISIBLE_DEVICES=0, python main.py --algorithm=value_d --explore=grad_direct --update-step=n_step $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --algorithm=value_r --explore=grad_rand --update-step=n_step $args $aux &


#CUDA_VISIBLE_DEVICES=0, python main.py --algorithm=bbo --explore=grad_direct --update-step=n_step $args $aux &
#CUDA_VISIBLE_DEVICES=1, python main.py --grad --algorithm=grad --explore=grad_direct --update-step=n_step $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --algorithm=rand --explore=rand --update-step=n_step $args $aux &

#CUDA_VISIBLE_DEVICES=0, python main.py --algorithm=directnoupdate --explore=grad_direct --update-step=no_update $args $aux &
#CUDA_VISIBLE_DEVICES=1, python main.py --algorithm=randnoupdate --explore=rand --update-step=no_update $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --algorithm=guidedrandnoupdate --explore=grad_rand --update-step=no_update $args $aux &

