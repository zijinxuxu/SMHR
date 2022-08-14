#!/usr/bin/env bash
set -ex

# Training
GPU_ID=0
NAME='MHR_demo'
task='simplified' # simplified, artificial
mode='test'
dataset='HO3D' # FreiHAND, HO3D, Joint
# Network configuration

BATCH_SIZE=1

# Reconstruction resolution
# NOTE: one can change here to reconstruct mesh in a different resolution.
Input_RES=224 # 224,384

CHECKPOINTS_PATH='data/HO3D-2d-simplified.pth'

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python demo.py \
    ${task} \
    --gpus 0 \
    --mode ${mode} \
    --dataset ${dataset} \
    --batch_size ${BATCH_SIZE} \
    --default_resolution ${Input_RES} \
    --photometric_loss \
    --load_model ${CHECKPOINTS_PATH} \
    # --pick_hand \