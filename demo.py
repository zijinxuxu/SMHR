from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from turtle import pos

import _init_paths

import os
from os.path import join
import pickle
import json
import sys

import torch
import torch.utils.data
from opts import opts
from models.model import create_model
from utils.utils import load_model, save_model
from models.utils import _sigmoid, _tranpose_and_gather_feat
from torch.utils.data.sampler import *
import time
import glob
import cv2
import numpy as np
from models.hand3d.Mano_render import ManoRender
from datasets.artificial import ArtificialDataset
from datasets.simplified import SimplifiedDataset
from datasets.joint_dataset import JointDataset
from trains.simplified import SimplifiedTrainer
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import Textures as MeshTextures
import matplotlib.pyplot as plt
import torch.nn.functional as F
# from psbody.mesh import Mesh
import matplotlib.patches as patches
from scipy.io import loadmat
import math
from math import cos, sin
from utils.utils import drawCirclev2
import random

def get_dataset(task):
  if task == 'simplified':
    class Dataset(JointDataset, SimplifiedDataset):
      pass
  else:
    class Dataset(JointDataset, ArtificialDataset):
      pass        
  return Dataset

# import torch.distributed as dist
def seed_torch(seed=0):
  random.seed(seed)
  os.environ['PYTHONHASHSEED']=str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True


def main(opt):
  # setup 
  seed_torch(opt.seed)
  # torch.manual_seed(opt.seed)
  # torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark
  Dataset = get_dataset(opt.task)
  opt = opts.update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

  render = ManoRender(opt).cuda().eval()

  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
  if opt.load_model != '':
      model = load_model(model, opt.load_model)
  model.cuda().eval()
  # base_dir = '/media/zijinxuxu/Seagate Backup Plus Drive/Hands_data/OneHand10K/Evaluation_data'
  base_dir = 'assets/Single/HO3D'
  img_list = []
  fileid_list = os.listdir(base_dir)
  for fileid in fileid_list:
      post_fix = fileid.split('.')[1]
      if post_fix != 'jpg' and post_fix != 'png':
        continue
      # fileid = fileid.split('.')[0]          
      img_rgb_path = os.path.join(base_dir, fileid) # v3 is .jpg  
      img_list.append(img_rgb_path)
  # img_list = glob.glob('/home/zijinxuxu/Downloads/egohands/egohands_data/_LABELLED_SAMPLES/CARDS_COURTYARD_S_H/*.jpg')
  # img_list = sorted(glob.glob('/mnt/SSD/AFLW/AFLW2000/*.jpg'))
  # img_list = sorted(glob.glob('/mnt/SSD/LS3D/LS3D-W/300W-Testset-3D/*.png'))
  mean = np.array([0.485, 0.456, 0.406],
                  dtype=np.float32).reshape(1, 1, 3)
  std = np.array([0.229, 0.224, 0.225],
                 dtype=np.float32).reshape(1, 1, 3)

  out = 'outputs'
  if not os.path.exists(out):
    os.makedirs(out)

  with torch.no_grad():
    for i, img_file in enumerate(img_list):
      # print(i)
      image = cv2.imread(img_file)

      h, w, _ = image.shape
      padh, padw = max(h, w) - h, max(h, w) - w
      image = np.pad(image, ((padh, 0), (0, padw), (0, 0)), mode='constant')
      image = cv2.resize(image, (opt.input_res, opt.input_res))
      save_img_0 = image.copy()

      pre_img = preprocess(image, mean, std)
      pre_img = torch.from_numpy(pre_img).permute(2, 0, 1).unsqueeze(0).cuda()

      folder, fname = img_file.split('/')[-2:]
      folder = os.path.join(out, folder)
      if not os.path.exists(folder):
        os.makedirs(folder)

      outputs, ind = model(pre_img)

      params = _tranpose_and_gather_feat(outputs[0]['params'][-1], ind)
      # decode part
      B, C = params.size(0), params.size(1)
      global_orient_coeff_l_up, pose_coeff_l, betas_coeff_l, global_transl_coeff_l_up, \
          global_orient_coeff_r_up, pose_coeff_r, betas_coeff_r, global_transl_coeff_r_up  = \
          render.Split_coeff(params.view(-1, params.size(2)), ind.view(-1))


      hand_verts_r_up, hand_joints_r_up, full_pose_r = render.Shape_formation(global_orient_coeff_r_up, pose_coeff_r, betas_coeff_r, global_transl_coeff_r_up,'right')              
      hand_verts_l_up, hand_joints_l_up, full_pose_l = render.Shape_formation(global_orient_coeff_l_up, pose_coeff_l, betas_coeff_l, global_transl_coeff_l_up,'left')
      lms21_orig_l_up = render.get_Landmarks(hand_joints_l_up)
      lms21_orig_r_up = render.get_Landmarks(hand_joints_r_up)       

      if opt.pick_hand:
          # assign left/right hand
          handmap = _sigmoid(outputs[0]['handmap'])
          hand_label_pred = _tranpose_and_gather_feat(handmap, ind)

          handness = torch.zeros((hand_label_pred.shape[1], )).to('cuda') + 5
          for h_id in range(hand_label_pred.shape[1]):
              if hand_label_pred[0][h_id][0] > hand_label_pred[0][h_id][1]:
                  handness[h_id] = 0
              else:
                  handness[h_id] = 1 
      else: # single right hand
        handness = torch.ones((C, )).to('cuda')
      idx_lms = handness.unsqueeze(1).unsqueeze(2).expand_as(lms21_orig_r_up)
      lms21_all = torch.where(idx_lms==0,lms21_orig_l_up,lms21_orig_r_up)
      idx_verts = handness.unsqueeze(1).unsqueeze(2).expand_as(hand_verts_r_up)
      verts_all = torch.where(idx_verts==0,hand_verts_l_up,hand_verts_r_up)
      idx_joints = handness.unsqueeze(1).unsqueeze(2).expand_as(hand_joints_r_up)
      joints_all = torch.where(idx_joints==0,hand_joints_l_up,hand_joints_r_up)              

      if opt.photometric_loss:
          pre_textures = _tranpose_and_gather_feat(outputs[0]['texture'], ind)
          device = pre_textures.device
          render.gamma = _tranpose_and_gather_feat(outputs[0]['light'], ind)
          Albedo = pre_textures.view(-1,778,3) #[b, 778, 3]
          texture_mean = torch.tensor([0.45,0.35,0.3]).float().to(device)
          texture_mean = texture_mean.unsqueeze(0).unsqueeze(0).repeat(1,Albedo.shape[1],1)#[1, 778, 3]
          Albedo = Albedo + texture_mean
          Texture, lighting = render.Illumination(Albedo, verts_all.view(-1,778,3))
          # for render
          # rotShape = rotShape.view(B, C, -1, 3)
          rotShape = verts_all.view(B, C, -1, 3)
          Texture = Texture.view(B, C, -1, 3)
          nV = rotShape.size(2)
          Verts, Faces, Textures = [], [], []
          valid = []
          tag_drop = False
          for i in range(len(rotShape)):
            # detach vertex to avoid influence of color
            V_ = rotShape[i].detach()
            if V_.size(0) == 0:
              # to simplify the code, drop the whole batch with any zero_hand image.
              Verts = []
              break
            valid.append(i)
            T_ = Texture[i]
            F_ = torch.from_numpy(render.MANO_R.faces.astype(np.int64)).to(device).view(1,-1)
            range_ = torch.arange(V_.size(0)).view(-1, 1) * nV
            F_ = F_.expand(V_.size(0), -1) + range_.cuda()
            Verts.append(V_.view(-1, 3))
            Textures.append(T_.view(-1, 3))
            Faces.append(F_.view(-1, 3).float())

          if len(Verts) > 0:
            # meshes = Meshes(verts=Verts, faces=Faces,
            #                 textures=TexturesVertex(verts_features=Textures))
            meshes = Meshes(verts=Verts, faces=Faces,
                            textures=MeshTextures(verts_rgb=Textures))
            rendered, gpu_masks, depth = render(meshes)
            rendered = torch.flip(rendered,[1])
            gpu_masks = torch.flip(gpu_masks,[1])
            depth = torch.flip(depth,[1]) 
            gpu_masks = gpu_masks.detach().float()


      # vis
      drawCirclev2(image,lms21_all.reshape(-1,2),(0,0,255),1)  
      save_path = os.path.join(folder,fname)
      for k in range(len(ind[0])):
          image = showHandJoints(image,lms21_all[k].detach().cpu().numpy(),save_path)


      if opt.photometric_loss and rendered is not None:
        render_img = rendered[0].detach().cpu().numpy()[:, :, ::-1].astype(np.float32)
        render_msk = gpu_masks[0].detach().cpu().numpy()
        fname_fitted = 'new-fitted-' + fname
        save_fitted_path = os.path.join(folder,fname_fitted)
        fname_render = 'new-render-'+fname
        save_render_path = os.path.join(folder,fname_render)

        cv2.imwrite(save_fitted_path, render_img*255+save_img_0*((1-render_msk).reshape(save_img_0.shape[0],save_img_0.shape[1],1)))
        cv2.imwrite(save_render_path, render_img*255)


def preprocess(image, mean, std):
  return (image.astype(np.float32) / 255. - mean) / std

def _nms(heat, kernel=3):
  pad = (kernel - 1) // 2
  if kernel == 2:
    hm_pad = F.pad(heat, [0, 1, 0, 1])
    hmax = F.max_pool2d(hm_pad, (kernel, kernel), stride=1, padding=pad)
  else:
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
  keep = (hmax == heat).float()
  return heat * keep

def _topk(scores, K):
    b, c, h, w = scores.size()
    assert c == 1
    topk_scores, topk_inds = torch.topk(scores.view(b, -1), K)

    topk_inds = topk_inds % (h * w)
    topk_ys = (topk_inds // w).int().float()
    topk_xs = (topk_inds % w).int().float()
    return topk_scores, topk_inds, topk_ys, topk_xs


def showHandJoints(imgInOrg, gtIn, filename=None):
    '''
    Utility function for displaying hand annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param filename: dump image name
    :return:
    '''
    import cv2

    imgIn = np.copy(imgInOrg)

    # Set color for each finger
    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]

    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [0, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [0, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [0, 17],
             [17, 18],
             [18, 19],
             [19, 20]
             ]

    PYTHON_VERSION = sys.version_info[0]

    gtIn = np.round(gtIn).astype(np.int)

    if gtIn.shape[0]==1:
        imgIn = cv2.circle(imgIn, center=(gtIn[0][0], gtIn[0][1]), radius=3, color=joint_color_code[0],
                             thickness=-1)
    else:

        for joint_num in range(gtIn.shape[0]):

            color_code_num = (joint_num // 4)
            if joint_num in [0, 4, 8, 12, 16]:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=1, color=joint_color, thickness=-1)
            else:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=1, color=joint_color, thickness=-1)

        for limb_num in range(len(limbs)):

            x1 = gtIn[limbs[limb_num][0], 1]
            y1 = gtIn[limbs[limb_num][0], 0]
            x2 = gtIn[limbs[limb_num][1], 1]
            y2 = gtIn[limbs[limb_num][1], 0]
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if length < 150 and length > 5:
                deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
                polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                           (int(length / 2), 3),
                                           int(deg),
                                           0, 360, 1)
                color_code_num = limb_num // 4
                if PYTHON_VERSION == 3:
                    limb_color = list(map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]))
                else:
                    limb_color = map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num])

                cv2.fillConvexPoly(imgIn, polygon, color=limb_color)


    if filename is not None:
        cv2.imwrite(filename, imgIn)

    return imgIn

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
