#!/usr/bin/env bash
set -ex

# Training
GPU_ID=0
NAME='SMHR_eval'
task='artificial' 
mode='test' # train, test, eval
dataset='FreiHAND' # modify datasets in data/joint_dataset.txt
# Network configuration

BATCH_SIZE=1

# Reconstruction resolution
Input_RES=384 # 224,512

CHECKPOINTS_PATH='data/Multi_pho.pth'

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch --master_port 12508 --nproc_per_node 1 demo.py \
    --task ${task} \
    --gpus 0 \
    --mode ${mode} \
    --dataset ${dataset} \
    --batch_size ${BATCH_SIZE} \
    --default_resolution ${Input_RES} \
    --photometric_loss \
    --load_model ${CHECKPOINTS_PATH} \
    --pick_hand \
