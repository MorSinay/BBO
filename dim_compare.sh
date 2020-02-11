#!/usr/bin/env bash

dim=$1
aux="${@:2}"

echo dim $1

loc=`dirname "%0"`

args="--action-space=$dim --game=ABL --algorithm=EGL"

#CUDA_VISIBLE_DEVICES=3, python main.py --pertub=0 --start=0 --seed=150 --replay-memory-factor=4 --identifier=r4_pertub_0 $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --pertub=1e-2 --start=0 --seed=150 --replay-memory-factor=1 --identifier=r1_pertub_2 $args $aux &

#CUDA_VISIBLE_DEVICES=3, python main.py --pertub=1e-2 --start=1 --seed=150 --replay-memory-factor=4 --identifier=r4_pertub_2 $args $aux &

#CUDA_VISIBLE_DEVICES=1, python main.py --pertub=1e-1 --start=0 --seed=150 --replay-memory-factor=1 --identifier=r1_pertub_1 $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --pertub=1e-2 --start=1 --seed=150 --replay-memory-factor=1 --identifier=r1_pertub_2 $args $aux &
#CUDA_VISIBLE_DEVICES=0, python main.py --pertub=1e-1 --start=1 --seed=150 --replay-memory-factor=1 --identifier=r1_pertub_1 $args $aux &



#CUDA_VISIBLE_DEVICES=2, python main.py --pertub=0 --seed=150 --replay-memory-factor=4 --identifier=pertub_0 $args $aux &
#CUDA_VISIBLE_DEVICES=0, python main.py --pertub=1e-1 --seed=150 --replay-memory-factor=4 --identifier=pertub_1 $args $aux &


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





args="--action-space=$dim --game=ABL --algorithm=EGL"
#CUDA_VISIBLE_DEVICES=1, python main.py --start=0 --stop=46 --trust-region --r-norm-alg=log --spline --identifier=spline_tr_map $args $aux &

#CUDA_VISIBLE_DEVICES=0, python main.py --start=60 --stop=106 --trust-region --r-norm-alg=log --spline --identifier=spline_tr_map $args $aux &
CUDA_VISIBLE_DEVICES=1, python main.py --start=345 --stop=346 --trust-region --r-norm-alg=log --spline --identifier=spline_tr_map $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --start=180 --stop=226 --trust-region --r-norm-alg=log --spline --identifier=spline_tr_map $args $aux &

#CUDA_VISIBLE_DEVICES=1, python main.py --start=240 --stop=286 --trust-region --r-norm-alg=log --spline --identifier=spline_tr_map $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --start=300 --stop=346 --trust-region --r-norm-alg=log --spline --identifier=spline_tr_map $args $aux &

#CUDA_VISIBLE_DEVICES=1, python main.py --start=0 --stop=166 --trust-region --r-norm-alg=none --identifier=fc_tr_no_map $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --start=120 --stop=121 --no-trust-region --r-norm-alg=none --identifier=fc_no_tr_no_map $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --start=0 --stop=166 --no-trust-region --r-norm-alg=log --identifier=fc_no_tr_map $args $aux &


#CUDA_VISIBLE_DEVICES=3, python main.py --start=0 --stop=46 --trust-region --r-norm-alg=log --identifier=fc_tr_map $args $aux &
#CUDA_VISIBLE_DEVICES=1, python main.py --start=60 --stop=106 --trust-region --r-norm-alg=log --identifier=fc_tr_map $args $aux &
#CUDA_VISIBLE_DEVICES=2, python main.py --start=120 --stop=166 --trust-region --r-norm-alg=log --identifier=fc_tr_map $args $aux &




