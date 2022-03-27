#!/bin/bash

DATA_DIR="/home/dengpanfu/data"
EXP_DIR="/home/dengpanfu/project/lupnl/simclr"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python lup_simclr.py \
    --data_path "${DATA_DIR}/lupnl_lmdb/lmdb" \
    --info_path "${DATA_DIR}/lupnl_lmdb/keys.pkl" \
    --eval_path "${DATA_DIR}/reid" \
    --eval_name "market" \
    --snap_dir "${EXP_DIR}/snapshots" \
    --log_dir "${EXP_DIR}/logs" \
    -a "resnet50" \
    --lr 0.3 \
    -j 32 \
    --batch-size 2048 \
    --optimizer "LARS" \
    --dist-url 'tcp://localhost:13701' \
    --multiprocessing-distributed 1 \
    --world-size 1 \
    --rank 0 \
    --T 0.1 \
    --aug_type 'ori-cj+sre' \
    --cos 1 \
    --mix 1 \
    --auto_resume 1 \
    --save_freq 20 \
    --print-freq 10 \
    --epochs 200 \
    --mean_type "lmdb_200_20" \
    --eval_freq -1 \
    --warmup_epochs 10