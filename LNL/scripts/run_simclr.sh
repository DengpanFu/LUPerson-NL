#!/bin/bash

DATA_DIR="/home/dengpanfu/data"
EXP_DIR="/home/dengpanfu/project/lupnl/simclr"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python lnl.py \
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
    --dist-url 'tcp://localhost:13701' \
    --multiprocessing-distributed 1 \
    --world-size 1 \
    --rank 0 \
    --T 0.1 \
    --aug_type 'ori-cj+sre' \
    --cos 1 \
    --snap_dir 'snapshots/debug' \
    --log_dir 'logs/debug' \
    --mix 1 \
    --auto_resume 1 \
    --load_mode "epoch" \
    --save_freq 20 \
    --print-freq 10 \
    --epochs 100 \
    --mean_type "lmdb_300_30" \
    --eval_freq 1 \
    --schedule "30,60,90" \
    --warmup_epochs 10