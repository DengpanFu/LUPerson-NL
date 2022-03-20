#!/bin/bash

DIR="/home/store1/dengpanfu/LUP/lmdbs"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python lup_supcont.py \
    --data_path "${DIR}/lmdb_3000/lmdb" \
    --info_path "${DIR}/lmdb_3000/keys.pkl" \
    --eval_path "/home/dengpanfu/projects/fast-reid/datasets/data" \
    --eval_name "market" \
    -a resnet50 \
    --embed_dim 2048 \
    --lr 0.1 \
    --optimizer 'LARS' \
    -j 32 \
    --batch-size 2048 \
    --dist-url 'tcp://localhost:13701' \
    --multiprocessing-distributed 1 \
    --world-size 1 \
    --rank 0 \
    --snap_dir 'snapshots/debug' \
    --log_dir 'logs/debug' \
    --mix 1 \
    --auto_resume 1 \
    --load_mode 'epoch' \
    --save_freq 20 \
    --print-freq 10 \
    --epochs 100 \
    --mean_type "lmdb_300_30" \
    --eval_freq 1 \
    --schedule '30,60,90' \
    --cos 1