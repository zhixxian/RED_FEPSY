#!/bin/bash

rank_margin=0.3
focal_gamma=5

for RANK_ALPHA in 0.2 0.3 0.4 0.5 0.6 1
do
    EXP_NAME=affectnet_${focal_gamma}_${rank_margin}_${RANK_ALPHA}

    python train_Affect.py \
        --wandb=${EXP_NAME} \
        --rank_margin=$rank_margin \
        --rank_alpha=$RANK_ALPHA \
        --focal_gamma=$focal_gamma \
        --raf_path=${RAF_PATH} \
        --batch_size=64 \
        --gpu=3
done

