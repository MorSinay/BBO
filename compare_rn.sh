#!/usr/bin/env bash

index=$1
aux="${@:2}"

args="--game=CMP --budget=9000 --action-space=2 --problem-index=$index"

CUDA_VISIBLE_DEVICES=1, python main.py --identifier=x --spline --algorithm=EGL $args $aux &
CUDA_VISIBLE_DEVICES=2, python main.py --identifier=xy --algorithm=EGL $args $aux &

CUDA_VISIBLE_DEVICES=3, python main.py --identifier=x --spline --algorithm=IGL $args $aux &
CUDA_VISIBLE_DEVICES=3, python main.py --identifier=xy --algorithm=IGL $args $aux &
