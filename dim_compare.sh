#!/usr/bin/env bash

dim=$1
aux="${@:2}"

echo dim $1

loc=`dirname "%0"`

args="--action-space=$dim --game=ABL --algorithm=IGL --seed=150 --start=1"

#CUDA_VISIBLE_DEVICES=1, python main.py --pertub=0 --replay-memory-factor=1  --identifier=r1_pertub_0 $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --pertub=0 --replay-memory-factor=16 --identifier=r16_pertub_0 $args $aux &
CUDA_VISIBLE_DEVICES=2, python main.py --pertub=0 --replay-memory-factor=32 --identifier=pertub_0 $args $aux &
#CUDA_VISIBLE_DEVICES=0, python main.py --pertub=1e-1 --replay-memory-factor=32 --identifier=pertub_1 $args $aux &


#CUDA_VISIBLE_DEVICES=0, python main.py --start=0 --replay-memory-factor=32 --identifier=fc_tr_map_re32 $args $aux &
#CUDA_VISIBLE_DEVICES=1, python main.py --start=1 --replay-memory-factor=32 --identifier=fc_tr_map_re32 $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --start=2 --replay-memory-factor=32 --identifier=fc_tr_map_re32 $args $aux &
#CUDA_VISIBLE_DEVICES=3, python main.py --start=3 --replay-memory-factor=32 --identifier=fc_tr_map_re32 $args $aux &

#CUDA_VISIBLE_DEVICES=0, python main.py --start=0 --replay-memory-factor=16 --identifier=fc_tr_map_re16 $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --start=1 --replay-memory-factor=16 --identifier=fc_tr_map_re16 $args $aux &
#CUDA_VISIBLE_DEVICES=0, python main.py --start=2 --replay-memory-factor=16 --identifier=fc_tr_map_re16 $args $aux &
#CUDA_VISIBLE_DEVICES=1, python main.py --start=3 --replay-memory-factor=16 --identifier=fc_tr_map_re16 $args $aux &




#CUDA_VISIBLE_DEVICES=0, python main.py --pertub=1e-1 --seed=150 --start=0 --identifier=pertub_1 $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --pertub=1e-2 --seed=150 --start=0 --identifier=pertub_2 $args $aux &
#CUDA_VISIBLE_DEVICES=1, python main.py --pertub=0 --seed=150 --start=0 --identifier=pertub_0 $args $aux &

#CUDA_VISIBLE_DEVICES=3, python main.py --pertub=1e-2 --seed=150 --start=1 --identifier=pertub_2 $args $aux &
#CUDA_VISIBLE_DEVICES=3, python main.py --pertub=1e-1 --seed=150 --start=1 --identifier=pertub_1 $args $aux &
#CUDA_VISIBLE_DEVICES=0, python main.py --pertub=0 --seed=150 --start=1 --identifier=pertub_0 $args $aux &



#CUDA_VISIBLE_DEVICES=1, python main.py --no-trust-region --r-norm-alg=none --identifier=fc_no_tr_no_map $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --trust-region --r-norm-alg=none --identifier=fc_tr_no_map $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --spline --trust-region --r-norm-alg=log --identifier=spline_mor_tr_map $args $aux &
#CUDA_VISIBLE_DEVICES=3, python main.py --trust-region --r-norm-alg=log --identifier=fc_tr_map $args $aux &


#CUDA_VISIBLE_DEVICES=1, python main.py --n-explore=128 --start=0 --identifier=128 $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --n-explore=128 --start=1 --identifier=128 $args $aux &
#CUDA_VISIBLE_DEVICES=3, python main.py --n-explore=128 --start=3 --identifier=128 $args $aux &

#CUDA_VISIBLE_DEVICES=1, python main.py --n-explore=256 --start=0 --identifier=256 $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --n-explore=256 --start=1 --identifier=256 $args $aux &
#CUDA_VISIBLE_DEVICES=3, python main.py --n-explore=256 --start=3 --identifier=256 $args $aux &


#CUDA_VISIBLE_DEVICES=2, python main.py --trust-region --r-norm-alg=log --explore=cone --n-explore=128 --start=2 --identifier=fc_tr_map_cone_126 $args $aux &
#CUDA_VISIBLE_DEVICES=3, python main.py --trust-region --r-norm-alg=log --explore=cone --n-explore=256 --start=2 --identifier=fc_tr_map_cone_256 $args $aux &
#CUDA_VISIBLE_DEVICES=0, python main.py --trust-region --r-norm-alg=log --n-explore=256 --identifier=fc_tr_map_256 $args $aux &
#CUDA_VISIBLE_DEVICES=0, python main.py --trust-region --r-norm-alg=log --n-explore=64 --identifier=fc_tr_map_64 $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --trust-region --r-norm-alg=log --n-explore=128 --identifier=fc_tr_map_128 $args $aux &
#CUDA_VISIBLE_DEVICES=0, python main.py --trust-region --r-norm-alg=log --explore=cone --identifier=fc_tr_map_cone $args $aux &




