#!/bin/bash

RAF_PATH=/nas_homes/jihyun/RAF_DB
# RESNET_PATH=/home/jihyun/code/pseudo_cutmix_fer/model/resnet50_ft_weight.pkl
# BATCH_SIZE=64
rank_margin=0.3
# RANK_ALPHA=0.4

for RANK_ALPHA in 0.2 0.3 0.4 0.5 0.6 1
do

    EXP_NAME=RAF_${focal_gamma}_${rank_margin}_${RANK_ALPHA}

    python train_RAF.py \
        --wandb=${EXP_NAME} \
        --rank_margin=$rank_margin \
        --rank_alpha=$RANK_ALPHA \
        --focal_gamma=$focal_gamma \
        --raf_path=${RAF_PATH} \
        --batch_size=64 \
        --gpu=3
done
