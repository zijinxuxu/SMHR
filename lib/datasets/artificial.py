from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sys import flags

import torch.utils.data as data
import numpy as np
import cv2
import os
from utils.image import get_affine_transform, affine_transform, affine_transform_array
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils import data_augment, data_generators
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

#calculating least sqaures problem
def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg

def lms2bbox(self, uv):
    idx = np.where((uv[:,:,0] >= 0)&(uv[:,:,1] >= 0)&(uv[:,:,0] < self.opt.size_train[0])&(uv[:,:,1] < self.opt.size_train[0])) 
    if len(idx[0])==0:
      return None     
    x_min = uv[idx][:,0].min()
    x_max = uv[idx][:,0].max()
    y_min = uv[idx][:,1].min()
    y_max = uv[idx][:,1].max()
    bbox = np.array([[x_min, y_min, x_max, y_max]])
    return bbox
    
def draw_lms(img, lms):
  for id_lms in lms:
    for id in range(len(id_lms)):
      cv2.circle(img, (int(id_lms[id,0]), int(id_lms[id,1])), 2, (0,0,255), 2)
  return img

class ArtificialDataset(data.Dataset):
  def normal(self, img):
    res = img.astype(np.float32) / 255.
    return (res - self.mean) / self.std

  def pad(self, img, stride):
    img = self.normal(img)
    height,width = img.shape[:2]
    padh = math.ceil(height / stride) * stride - height
    padw = math.ceil(width / stride) * stride - width
    result = np.pad(img, ((0,padh), (0,padw), (0,0)), mode='constant')
    assert result.shape[0] % stride == 0 and result.shape[1] % stride == 0
    return result


  def __getitem__(self, index):
    pass