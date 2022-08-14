from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from sys import maxsize

import numpy as np
import torch
import torch.nn.functional as F
# from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import Textures as MeshTextures
import torch.nn as nn
from torch.utils import data
import math
import sys

from models.hand3d.Mano_render import ManoRender
from models.losses import FocalLoss, bone_direction_loss, calculate_psnr
from models.losses import RegL1Loss, RegWeightedL1Loss, NormLoss
from models.utils import _sigmoid, _tranpose_and_gather_feat
from .base_trainer import BaseTrainer
from utils.utils import drawCirclev2
import copy
import cv2
from scipy.optimize import minimize

class CtdetLoss(torch.nn.Module):
  def __init__(self, opt, render=None, facenet=None):
    super(CtdetLoss, self).__init__()
    self.opt = opt
    self.crit = FocalLoss()
    self.crit_reg = RegL1Loss()
    if opt.reproj_loss:
      self.crit_reproj = RegL1Loss()
    if opt.photometric_loss or opt.reproj_loss:
      self.crit_norm = NormLoss()
    if opt.off:
      self.crit_lms = RegWeightedL1Loss()
    self.render = render
    self.facenet = facenet

  def bce_loss(self, pred, gt):
    return F.binary_cross_entropy(pred, gt, reduction='none')

  def l1_loss(self, pred, gt, is_valid=None):
    loss = F.l1_loss(pred, gt, reduction='none')
    if is_valid is not None:
        loss *= is_valid
    return loss

  def soft_argmax_1d(self, heatmap1d):     
      heatmap_size = heatmap1d.shape[2]
      heatmap1d = F.softmax(heatmap1d * heatmap_size, 2)
      coord = heatmap1d * torch.arange(heatmap_size).float().cuda()
      coord = coord.sum(dim=2, keepdim=True)
      return coord

  def forward(self, outputs, batch):
    pass

  def multi_pred(self, outputs, ind, batch):
    pass

  def pred(self, outputs, ind, batch):
    pass

  def test(self, outputs, ind, batch):
    pass
  
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

def _topk(scores, K):
    b, c, h, w = scores.size()
    assert c == 1
    topk_scores, topk_inds = torch.topk(scores.view(b, -1), K)

    topk_inds = topk_inds % (h * w)
    topk_ys = (topk_inds // w).int().float()
    topk_xs = (topk_inds % w).int().float()
    return topk_scores, topk_inds, topk_ys, topk_xs

def _nms(heat, kernel=5):
    pad = (kernel - 1) // 2
    if kernel == 2:
        hm_pad = F.pad(heat, [0, 1, 0, 1])
        hmax = F.max_pool2d(hm_pad, (kernel, kernel), stride=1, padding=pad)
    else:
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep    

def align_uv(t, uv, vertex2xyz, K):
  xyz = vertex2xyz + t
  proj = np.matmul(K, xyz.T).T
  # projection_ = proj[..., :2] / ( proj[..., 2:])
  # proj = np.matmul(K, xyz.T).T
  uvz = np.concatenate((uv, np.ones([uv.shape[0], 1])), axis=1) * xyz[:, 2:]
  loss = (proj - uvz)**2
  return loss.mean()


class SimplifiedTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    self.render = ManoRender(opt) if opt.reproj_loss or opt.photometric_loss else None
    self.facenet = None
    super(SimplifiedTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    loss_stats = ['loss', 'hm_loss']
    if opt.pick_hand:
      loss_stats.append('pick_loss')
    if not opt.center_only:
      if opt.heatmaps:
        loss_stats.append('heatmaps_loss')
      if opt.reproj_loss:
        loss_stats.append('reproj_loss_all')
        loss_stats.append('norm_loss')
      if opt.bone_loss:
        loss_stats.append('bone_direc_loss')
      if opt.mode == 'train_3d':
        if self.opt.dataset == 'RHD':
          loss_stats.append('pose_loss')
        else:
          loss_stats.append('pose_loss')
          loss_stats.append('verts_loss')
          loss_stats.append('mano_loss')
          loss_stats.append('shape_loss')
      if opt.photometric_loss:
        loss_stats.append('photometric_loss')
        loss_stats.append('seg_loss')
    loss = CtdetLoss(opt, self.render, self.facenet)
    return loss_stats, loss
