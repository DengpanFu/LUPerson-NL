#!/bin/bash

DATA_DIR="/home/dengpanfu/data"
EXP_DIR="/home/dengpanfu/project/lupnl/lnl"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python lnl.py \
    --data_path "${DATA_DIR}/lupnl_lmdb/lmdb" \
    --info_path "${DATA_DIR}/lupnl_lmdb/keys.pkl" \
    --eval_path "${DATA_DIR}/reid" \
    --eval_name "market" \
    --snap_dir "${EXP_DIR}/snapshots" \
    --log_dir "${EXP_DIR}/logs" \
    -a "resnet50" \
    --lr 0.4 \
    -j 32 \
    --batch-size 1536 \
    --multiprocessing-distributed 1 \
    --dist-url "tcp://localhost:23791" \
    --world-size 1 \
    --rank 0 \
    --mix 1 \
    --auto_resume 1 \
    --save_freq 10 \
    --print-freq 10 \
    --eval_freq 3 \
    --schedule "40,80" \
    --mean_type "lmdb_200_20" \
    --aug_type "ori-cj+sre" \
    --optimizer "SGD" \
    --epochs 90 \
    --cos 0 \
    --moco_dim 128 \
    --cls_dim 256 \
    --T 0.07 \
    --moco_k 65536 \
    --alpha 0.5 \
    --pseudo_th 0.8 \
    --start_clean_epoch 11 \
    --start_supcont_epoch 15 