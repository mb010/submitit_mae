#!/bin/bash

JOB_DIR=/share/nas2/mbowles/submitit_mae/experiments/mae_vit_large_patch16_75_mask
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO

# nodes>1 breaks with nccl error. See logs for examples.
python submitit_pretrain.py \
    --job_dir ${JOB_DIR} \
    --nodes 1 \
    --ngpus 1 \
    --use_a100 \
    --batch_size 128 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --data_path /share/nas2_5/mbowles/data/imagenet21k_resized
