#!/usr/bin/env bash

dim=$1
index=$2
aux="${@:3}"

echo dim $1 index $2
#args="--game=VALUE --budget=5 --action-space=$dim --resume=0 --load-last-model --problem-index=$index --no-bandage --explore=grad_direct --delta=0.1 --epsilon=0.1 --beta-optim=sgd"
#args="--game=GRAD --grad --budget=5 --action-space=$dim --resume=0 --load-last-model --problem-index=$index --no-bandage --explore=grad_direct --delta=0.01 --epsilon=0.01 --beta-optim=sgd"
args="--game=SEC_GRAD --grad --debug --budget=5 --action-space=$dim --resume=0 --load-last-model --problem-index=$index --no-bandage --explore=grad_direct --delta=0.01 --epsilon=0.01 --beta-optim=sgd"

#CUDA_VISIBLE_DEVICES=1, python main.py --identifier=sgd --grad --algorithm=grad_mid  --mid-val $args $aux &

CUDA_VISIBLE_DEVICES=0, python main.py --identifier=best_explore_update --algorithm=n_step --update-step=n_step --best-explore-update $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --identifier=best_explore_update --algorithm=best_step --update-step=best_step --best-explore-update $args $aux &
CUDA_VISIBLE_DEVICES=2, python main.py --identifier=best_explore_update --algorithm=first_vs_last --update-step=first_vs_last --best-explore-update $args $aux &
CUDA_VISIBLE_DEVICES=0, python main.py --identifier=best_explore_update --algorithm=no_update --update-step=no_update --best-explore-update $args $aux &

CUDA_VISIBLE_DEVICES=1, python main.py --identifier=no_best_explore_update --algorithm=n_step --update-step=n_step --no-best-explore-update $args $aux &
CUDA_VISIBLE_DEVICES=2, python main.py --identifier=no_best_explore_update --algorithm=best_step --update-step=best_step --no-best-explore-update $args $aux &
CUDA_VISIBLE_DEVICES=0, python main.py --identifier=no_best_explore_update --algorithm=first_vs_last --update-step=first_vs_last --no-best-explore-update $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --identifier=no_best_explore_update --algorithm=no_update --update-step=no_update --no-best-explore-update $args $aux &