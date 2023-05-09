#!/usr/bin/env bash
set -ex

# Training
GPU_ID=0
NAME='SMHR_eval'
task='simplified' 
mode='test' # train, test, eval
dataset='FreiHAND' # modify datasets in data/joint_dataset.txt
# Network configuration

BATCH_SIZE=1

# Reconstruction resolution
Input_RES=224 # 224,512

CHECKPOINTS_PATH='data/Single_pho.pth'

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch --master_port 12508 --nproc_per_node 1 main.py \
    --task ${task} \
    --gpus 0 \
    --mode ${mode} \
    --dataset ${dataset} \
    --batch_size ${BATCH_SIZE} \
    --default_resolution ${Input_RES} \
    --reproj_loss \
    --brightness \
    --bone_loss \
    --arch csp_50 \
    --avg_center \
    --config_info Single-2d-right-pers-pho \
    --photometric_loss \
    --load_model ${CHECKPOINTS_PATH} \
    # --lr 5e-5 \
    # --load_model ${CHECKPOINTS_PATH} \
    # set to true when using pca rather than euler angles
    # --using_pca \ 
    # set to true when using heatmaps of keypoints
    #--heatmaps \ 
    # for FreiHAND and HO3D dataset without detection
    #--no_det \ 
    # used when both left/right hand exists.
    # --pick_hand \
    # set to true when using bone_dir_loss. 
    #--bone_loss \
    # set to true when using pho_loss.
    # --photometric_loss \ 
