#!/usr/bin/env bash

algorithm=$1
dim=$2
index=$3
aux="${@:4}"

echo algorithm=$1 dim=$2 index=$3

args="--game=CMP --explore=grad_direct --epsilon=0.1 --norm=robust_scaler --budget=400 --grad-clip=1e-3 --best-explore-update --no-bandage --update-step=n_step --action-space=$dim --problem-index=$index"

CUDA_VISIBLE_DEVICES=3, python main.py --algorithm=$algorithm --alpha=1 --replay-memory-factor=30 --batch=100 --identifier=direct --beta-lr=1e-3 $args $aux &

#CUDA_VISIBLE_DEVICES=1, python main.py --algorithm=first_order --explore=grad_direct --identifier=direct --beta-lr=1e-3 $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --algorithm=second_order --explore=grad_direct --identifier=direct --beta-lr=1e-3 $args $aux &
#CUDA_VISIBLE_DEVICES=0, python main.py --algorithm=anchor --explore=grad_direct --identifier=direct --beta-lr=1e-3 $args $aux &


#args="--game=LR --algorithm=first_order --explore=grad_direct --update-step=n_step --budget=150 --no-best-explore-update --no-bandage --action-space=$dim --problem-index=$index"

#CUDA_VISIBLE_DEVICES=0, python main.py --identifier=lr_1 --beta-lr=1e-1 $args $aux &
#CUDA_VISIBLE_DEVICES=1, python main.py --identifier=lr_2 --beta-lr=1e-2 $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --identifier=lr_3 --beta-lr=1e-3 $args $aux &
#CUDA_VISIBLE_DEVICES=0, python main.py --identifier=lr_4 --beta-lr=1e-4 $args $aux &
#CUDA_VISIBLE_DEVICES=1, python main.py --identifier=lr_5 --beta-lr=1e-5 $args $aux &

#CUDA_VISIBLE_DEVICES=2, python main.py --identifier=lr_1_n --beta-lr=1e-1 $args $aux &
#CUDA_VISIBLE_DEVICES=0, python main.py --identifier=lr_2_n --beta-lr=1e-2 $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --identifier=lr_4_n --beta-lr=1e-4 $args $aux &
#CUDA_VISIBLE_DEVICES=1, python main.py --identifier=lr_5_n --beta-lr=1e-5 $args $aux &





#CUDA_VISIBLE_DEVICES=0, python main.py --algorithm=bbo --explore=grad_direct --update-step=n_step $args $aux &
#CUDA_VISIBLE_DEVICES=1, python main.py --grad --algorithm=grad --explore=grad_direct --update-step=n_step $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --algorithm=rand --explore=rand --update-step=n_step $args $aux &

#CUDA_VISIBLE_DEVICES=0, python main.py --algorithm=directnoupdate --explore=grad_direct --update-step=no_update $args $aux &
#CUDA_VISIBLE_DEVICES=1, python main.py --algorithm=randnoupdate --explore=rand --update-step=no_update $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --algorithm=guidedrandnoupdate --explore=grad_rand --update-step=no_update $args $aux &

