#!/usr/bin/env bash

dim=$1
index=$2
aux="${@:3}"

echo dim $1 index $2

args="--game=CMP --budget=100 --normalize --no-best-explore-update --no-bandage --update-step=n_step --explore=grad_direct --action-space=$dim --problem-index=$index"

CUDA_VISIBLE_DEVICES=0, python main.py --algorithm=first_order --no-grad-clip --identifier=lr_1 --beta-lr=1e-1 $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --algorithm=second_order --no-grad-clip --identifier=lr_1 --beta-lr=1e-1 $args $aux &
CUDA_VISIBLE_DEVICES=2, python main.py --algorithm=value --no-grad-clip --identifier=lr_1 --beta-lr=1e-1 $args $aux &
CUDA_VISIBLE_DEVICES=0, python main.py --algorithm=anchor --no-grad-clip --identifier=lr_1 --beta-lr=1e-1 $args $aux &

CUDA_VISIBLE_DEVICES=0, python main.py --algorithm=first_order --no-grad-clip --identifier=lr_2 --beta-lr=1e-2 $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --algorithm=second_order --no-grad-clip --identifier=lr_2 --beta-lr=1e-2 $args $aux &
CUDA_VISIBLE_DEVICES=2, python main.py --algorithm=value --no-grad-clip --identifier=lr_2 --beta-lr=1e-2 $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --algorithm=anchor --no-grad-clip --identifier=lr_2 --beta-lr=1e-2 $args $aux &

CUDA_VISIBLE_DEVICES=0, python main.py --algorithm=first_order --grad-clip --identifier=lr_1_c --beta-lr=1e-1 $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --algorithm=second_order --grad-clip --identifier=lr_1_c --beta-lr=1e-1 $args $aux &
CUDA_VISIBLE_DEVICES=2, python main.py --algorithm=value --grad-clip --identifier=lr_1_c --beta-lr=1e-1 $args $aux &
CUDA_VISIBLE_DEVICES=2, python main.py --algorithm=anchor --grad-clip --identifier=lr_1_c --beta-lr=1e-1 $args $aux &

CUDA_VISIBLE_DEVICES=0, python main.py --algorithm=first_order --grad-clip --identifier=lr_2_c --beta-lr=1e-2 $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --algorithm=second_order --grad-clip --identifier=lr_2_c --beta-lr=1e-2 $args $aux &
CUDA_VISIBLE_DEVICES=2, python main.py --algorithm=value --grad-clip --identifier=lr_2_c --beta-lr=1e-2 $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --algorithm=anchor --grad-clip --identifier=lr_2_c --beta-lr=1e-2 $args $aux &

#args="--game=LR --algorithm=first_order --explore=grad_direct --update-step=n_step --budget=150 --no-best-explore-update --no-bandage --action-space=$dim --problem-index=$index"

#CUDA_VISIBLE_DEVICES=0, python main.py --identifier=lr_1 --no-normalize --beta-lr=1e-1 $args $aux &
#CUDA_VISIBLE_DEVICES=1, python main.py --identifier=lr_2 --no-normalize --beta-lr=1e-2 $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --identifier=lr_3 --no-normalize --beta-lr=1e-3 $args $aux &
#CUDA_VISIBLE_DEVICES=0, python main.py --identifier=lr_4 --no-normalize --beta-lr=1e-4 $args $aux &
#CUDA_VISIBLE_DEVICES=1, python main.py --identifier=lr_5 --no-normalize --beta-lr=1e-5 $args $aux &

#CUDA_VISIBLE_DEVICES=2, python main.py --identifier=lr_1_n --normalize --beta-lr=1e-1 $args $aux &
#CUDA_VISIBLE_DEVICES=0, python main.py --identifier=lr_2_n --normalize --beta-lr=1e-2 $args $aux &
#CUDA_VISIBLE_DEVICES=1, python main.py --identifier=lr_3_n --normalize --beta-lr=1e-3 $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --identifier=lr_4_n --normalize --beta-lr=1e-4 $args $aux &
#CUDA_VISIBLE_DEVICES=1, python main.py --identifier=lr_5_n --normalize --beta-lr=1e-5 $args $aux &





#CUDA_VISIBLE_DEVICES=0, python main.py --algorithm=bbo --explore=grad_direct --update-step=n_step $args $aux &
#CUDA_VISIBLE_DEVICES=1, python main.py --grad --algorithm=grad --explore=grad_direct --update-step=n_step $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --algorithm=rand --explore=rand --update-step=n_step $args $aux &

#CUDA_VISIBLE_DEVICES=0, python main.py --algorithm=directnoupdate --explore=grad_direct --update-step=no_update $args $aux &
#CUDA_VISIBLE_DEVICES=1, python main.py --algorithm=randnoupdate --explore=rand --update-step=no_update $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --algorithm=guidedrandnoupdate --explore=grad_rand --update-step=no_update $args $aux &

