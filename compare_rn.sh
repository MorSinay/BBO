#!/usr/bin/env bash

dim=$1
index=$2

echo dim $1 index $2
args="--game=MOR --budget=20 --action-space=$dim --resume=0 --load-last-model --problem-index=$index"

CUDA_VISIBLE_DEVICES=1, python main.py --identifier=sgd --algorithm=grad_direct_1 --grad --explore=grad_direct --delta=0.1 --epsilon=0.1 --beta-optim=sgd $args &
CUDA_VISIBLE_DEVICES=1, python main.py --identifier=sgd --algorithm=grad_rand_1 --grad --explore=grad_rand --delta=0.1 --epsilon=0.1 --beta-optim=sgd $args &
CUDA_VISIBLE_DEVICES=1, python main.py --identifier=sgd --algorithm=grad_uniform_1 --grad --explore=rand --delta=0.1 --epsilon=0.1 --beta-optim=sgd $args &

CUDA_VISIBLE_DEVICES=2, python main.py --identifier=sgd --algorithm=grad_direct_01 --grad --explore=grad_direct --delta=0.01 --epsilon=0.01 --beta-optim=sgd $args &
CUDA_VISIBLE_DEVICES=2, python main.py --identifier=sgd --algorithm=grad_rand_01 --grad --explore=grad_rand --delta=0.01 --epsilon=0.01 --beta-optim=sgd $args &
CUDA_VISIBLE_DEVICES=2, python main.py --identifier=sgd --algorithm=grad_uniform_01 --grad --explore=rand --delta=0.01 --epsilon=0.01 --beta-optim=sgd $args &

CUDA_VISIBLE_DEVICES=3, python main.py --identifier=sgd --algorithm=value_direct_1 --explore=grad_direct --delta=0.1 --epsilon=0.1 --beta-optim=sgd $args &
CUDA_VISIBLE_DEVICES=3, python main.py --identifier=sgd --algorithm=value_rand_1 --explore=grad_rand --delta=0.1 --epsilon=0.1 --beta-optim=sgd $args &
CUDA_VISIBLE_DEVICES=3, python main.py --identifier=sgd --algorithm=value_uniform_1 --explore=rand --delta=0.1 --epsilon=0.1 --beta-optim=sgd $args &

CUDA_VISIBLE_DEVICES=1, python main.py --identifier=sgd --algorithm=value_direct_01 --explore=grad_direct --delta=0.01 --epsilon=0.01 --beta-optim=sgd $args &
CUDA_VISIBLE_DEVICES=2, python main.py --identifier=sgd --algorithm=value_rand_01 --explore=grad_rand --delta=0.01 --epsilon=0.01 --beta-optim=sgd $args &
CUDA_VISIBLE_DEVICES=3, python main.py --identifier=sgd --algorithm=value_uniform_01 --explore=rand --delta=0.01 --epsilon=0.01 --beta-optim=sgd $args &
