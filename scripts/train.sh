#!/usr/bin/env bash
set -ex

# Training
GPU_ID=0,1,2,3
NAME='SMHR_train'
task='simplified' 
mode='train' # train, test, eval
dataset='Joint' # modify datasets in data/joint_dataset.txt
# Network configuration

BATCH_SIZE=200

# Reconstruction resolution
Input_RES=224 # 224,512

CHECKPOINTS_PATH='/home/zijinxuxu/codes/SMHR/outputs/logs/simplified/default/logs_2022-07-06-19-43-/model_150.pth'

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    ${task} \
    --gpus 0,1,2,3 \
    --mode ${mode} \
    --dataset ${dataset} \
    --batch_size ${BATCH_SIZE} \
    --default_resolution ${Input_RES} \
    --reproj_loss \
    --brightness \
    --bone_loss \
    --arch csp_50 \
    --avg_center \
    --config_info the-same-as-latest-csp_18 \
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