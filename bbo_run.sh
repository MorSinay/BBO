#!/usr/bin/env bash

algorithm=$1
identifier=$2
aux="${@:3}"

echo resume $1 algorithm $2

loc=`dirname "%0"`

args="--algorithm=$algorithm --start=3 --identifier=$identifier --game=RUN --resume=0 --load-last-model"

#CUDA_VISIBLE_DEVICES=0, python main.py --spline --action-space=1 $args $aux &


#CUDA_VISIBLE_DEVICES=0, python main.py --spline --action-space=2 $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --spline --action-space=3 $args $aux &
#CUDA_VISIBLE_DEVICES=3, python main.py --spline --action-space=5 $args $aux &
#CUDA_VISIBLE_DEVICES=0, python main.py --spline --action-space=10 $args $aux &
#CUDA_VISIBLE_DEVICES=1, python main.py --spline --action-space=20 $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --spline --action-space=40 $args $aux &


#CUDA_VISIBLE_DEVICES=0, python main.py --action-space=784 $args $aux --algorithm=EGL --identifier=24_1 &
#CUDA_VISIBLE_DEVICES=1, python main.py --action-space=784 $args $aux --explore=cone --algorithm=EGL --identifier=24_1_cone &
#CUDA_VISIBLE_DEVICES=2, python main.py --action-space=784 $args $aux --algorithm=value --identifier=24_1 &
#CUDA_VISIBLE_DEVICES=1, python main.py --action-space=784 $args $aux --explore=cone --algorithm=value --identifier=24_1_cone &








