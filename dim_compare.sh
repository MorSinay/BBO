#!/usr/bin/env bash

dim=$1
aux="${@:2}"

echo dim $1

loc=`dirname "%0"`

args="--action-space=$dim --game=RUN"

CUDA_VISIBLE_DEVICES=0, python main.py --algorithm=first_order --spline --identifier=spline_15_1 $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --algorithm=value --spline --identifier=spline_15_1 $args $aux &
CUDA_VISIBLE_DEVICES=2, python main.py --algorithm=first_order --identifier=15_1 $args $aux &






