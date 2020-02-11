#!/usr/bin/env bash

dim=$1
aux="${@:2}"

echo dim $1

loc=`dirname "%0"`

args="--action-space=$dim --game=DBG --budget=9000"

CUDA_VISIBLE_DEVICES=0, python main.py --algorithm=EGL --replay-memory-factor=4 --identifier=dbg --problem-index=15 $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --algorithm=IGL --replay-memory-factor=4 --identifier=dbg --problem-index=15 $args $aux &
