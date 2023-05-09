from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tkinter.messagebox import NO
from torch.functional import split

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from lib.utils.image import flip, color_aug
from lib.utils.image import get_affine_transform, affine_transform, affine_transform_array
from lib.utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from lib.utils.image import draw_dense_reg
from lib.utils import data_augment, data_generators
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
from PIL import Image
from scipy import stats, ndimage
import pickle
from lib.models.networks.manolayer import ManoLayer, rodrigues_batch
from models.hand3d.Mano_render import ManoRender

def draw_lms(img, lms):
  for id_lms in lms:
    for id in range(len(id_lms)):
      cv2.circle(img, (int(id_lms[id,0]), int(id_lms[id,1])), 2, (255,0,0), 2)
  return img

def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]

def lms2bbox(uv):
    # idx = np.where((uv[:,:,0] >= 0)&(uv[:,:,1] >= 0)&(uv[:,:,0] < self.opt.size_train[0])&(uv[:,:,1] < self.opt.size_train[0])) 
    # if len(idx[0])==0:
    #   return None     
    x_min = uv[:,0].min()
    x_max = uv[:,0].max()
    y_min = uv[:,1].min()
    y_max = uv[:,1].max()

    box_w = x_max - x_min
    box_h = y_max - y_min
    # vis
    # cv2.rectangle(mask, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,255,0), 1)
    # save_dir = os.path.join('Data_process/cache/', mask_path.split('/')[-1])
    # cv2.imwrite('Data_process/cache/mask_{}'.format(mask_path.split('/')[-1]), mask)
    bbox = np.array([[x_min, y_min, x_max, y_max]])
    return bbox

def uv2map(uv, size=(224, 224)):
    kernel_size = (size[0] * 13 // size[0] - 1) // 2
    gaussian_map = np.zeros((uv.shape[0], size[0], size[1]))
    size_transpose = np.array(size)
    gaussian_kernel = cv2.getGaussianKernel(2 * kernel_size + 1, (2 * kernel_size + 2)/4.)
    gaussian_kernel = np.dot(gaussian_kernel, gaussian_kernel.T)
    gaussian_kernel = gaussian_kernel/gaussian_kernel.max()

    for i in range(gaussian_map.shape[0]):
        if (uv[i] >= 0).prod() == 1 and (uv[i][1] <= size_transpose[0]) and (uv[i][0] <= size_transpose[1]):
            s_pt = np.array((uv[i][1], uv[i][0]))
            p_start = s_pt - kernel_size
            p_end = s_pt + kernel_size
            p_start_fix = (p_start >= 0) * p_start + (p_start < 0) * 0
            k_start_fix = (p_start >= 0) * 0 + (p_start < 0) * (-p_start)
            p_end_fix = (p_end <= (size_transpose - 1)) * p_end + (p_end > (size_transpose - 1)) * (size_transpose - 1)
            k_end_fix = (p_end <= (size_transpose - 1)) * kernel_size * 2 + (p_end > (size_transpose - 1)) * (2*kernel_size - (p_end - (size_transpose - 1)))
            gaussian_map[i, p_start_fix[0]: p_end_fix[0] + 1, p_start_fix[1]: p_end_fix[1] + 1] = \
                gaussian_kernel[k_start_fix[0]: k_end_fix[0] + 1, k_start_fix[1]: k_end_fix[1] + 1]

    return gaussian_map

def process_bbox(in_box, original_img_shape):
    
    # aspect ratio preserving bbox
    bbox = in_box.copy()
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = original_img_shape[1]/original_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = int(w*2.)
    bbox[3] = int(h*2.)
    bbox[0] = int(c_x - bbox[2]/2.)
    bbox[1] = int(c_y - bbox[3]/2.)

    return bbox

def read_depth_img(depth_filename):
    """Read the depth image in dataset and decode it"""
    # depth_filename = os.path.join(base_dir, split, seq_name, 'depth', file_id + '.png')

    # _assert_exist(depth_filename)

    depth_scale = 0.00012498664727900177
    depth_img = cv2.imread(depth_filename)

    dpt = depth_img[:, :, 2] + depth_img[:, :, 1] * 256
    dpt = dpt * depth_scale

    return dpt

def depthToPCL(dpt, T, background_val=0.):
    # get valid points and transform
    pts = np.asarray(np.where(~np.isclose(dpt, background_val))).transpose()
    pts = np.concatenate([pts[:, [1, 0]] + 0.5, np.ones((pts.shape[0], 1), dtype='float32')], axis=1)
    pts = np.dot(np.linalg.inv(np.asarray(T)), pts.T).T
    pts = (pts[:, 0:2] / pts[:, 2][:, None]).reshape((pts.shape[0], 2))

    # replace the invalid data
    depth = dpt[(~np.isclose(dpt, background_val))]

    # get x and y data in a vectorized way
    row = (pts[:, 0] - 160.) / 241.42 * depth
    col = (pts[:, 1] - 120.) / 241.42 * depth

    # combine x,y,depth
    return np.column_stack((row, col, depth))

def loadDepthMap(filename):
    """
    Read a depth-map
    :param filename: file name to load
    :return: image data of depth image
    """

    img = Image.open(filename)  # open image

    assert len(img.getbands()) == 1  # ensure depth image
    imgdata = np.asarray(img, np.float32)

    return imgdata

def fix_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        # print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1
        
class InterHandDataset(data.Dataset):

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

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
  
  def myaxis2Rmat(self, axis):
      # input is bs x 3 or bs x 45
      bs = axis.shape[0]
      rotation_mat = rodrigues_batch(axis.view(-1, 3))
      rotation_mat = rotation_mat.view(bs, -1, 3, 3)
      return rotation_mat
  
  def Rmat2axis(self, R):
      # R: bs x 3 x 3
      R = R.view(-1, 3, 3)
      temp = (R - R.permute(0, 2, 1)) / 2
      L = temp[:, [2, 0, 1], [1, 2, 0]]  # bs x 3
      sin = torch.norm(L, dim=1, keepdim=False)  # bs
      L = L / (sin.unsqueeze(-1) + 1e-8)

      temp = (R + R.permute(0, 2, 1)) / 2
      temp = temp - torch.eye((3), dtype=R.dtype, device=R.device)
      temp2 = torch.matmul(L.unsqueeze(-1), L.unsqueeze(1))
      temp2 = temp2 - torch.eye((3), dtype=R.dtype, device=R.device)
      temp = temp[:, 0, 0] + temp[:, 1, 1] + temp[:, 2, 2]
      temp2 = temp2[:, 0, 0] + temp2[:, 1, 1] + temp2[:, 2, 2]
      cos = 1 - temp / (temp2 + 1e-8)  # bs

      sin = torch.clamp(sin, min=-1 + 1e-7, max=1 - 1e-7)
      theta = torch.asin(sin)

      # prevent in-place operation
      theta2 = torch.zeros_like(theta)
      theta2[:] = theta
      idx1 = (cos < 0) & (sin > 0)
      idx2 = (cos < 0) & (sin < 0)
      theta2[idx1] = 3.14159 - theta[idx1]
      theta2[idx2] = -3.14159 - theta[idx2]
      axis = theta2.unsqueeze(-1) * L

      return axis.view(-1, 3)

  def pca2axis_left(self, pca):
      rotation_axis = pca.mm(self.mano_layer['left'].hands_components[:pca.shape[1]])  # bs * 45
      rotation_axis = rotation_axis #+ self.mano_layer['left'].hands_mean
      return rotation_axis  # bs * 45
  def pca2axis_right(self, pca):
      rotation_axis = pca.mm(self.mano_layer['right'].hands_components[:pca.shape[1]])  # bs * 45
      rotation_axis = rotation_axis #+ self.mano_layer['right'].hands_mean
      return rotation_axis  # bs * 45

  def __getitem__(self, index):
    if self.opt.dataset == 'InterHandNew':
      if self.split == 'train_3d':
         self.split = 'train'
      mano_path = {'left': os.path.join('/home/zijinxuxu/codes/SMHR-InterHand/lib/models/hand3d/mano_core', 'MANO_LEFT.pkl'),
                  'right': os.path.join('/home/zijinxuxu/codes/SMHR-InterHand/lib/models/hand3d/mano_core', 'MANO_RIGHT.pkl')}            
      self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None, use_pca=True),
                          'left': ManoLayer(mano_path['left'], center_idx=None, use_pca=True)}
      # self.render = ManoRender(self.opt)      
      fix_shape(self.mano_layer)          
      img = cv2.imread(os.path.join(self.data_path, self.split, 'img', '{}.jpg'.format(index)))
      # mask = cv2.imread(os.path.join(self.data_path, self.split, 'mask', '{}.jpg'.format(index)))
      # dense = cv2.imread(os.path.join(self.data_path, self.split, 'dense', '{}.jpg'.format(index)))

      with open(os.path.join(self.data_path, self.split, 'anno', '{}.pkl'.format(index)), 'rb') as file:
          data = pickle.load(file)

      R = data['camera']['R']
      T = data['camera']['t']
      camera = data['camera']['camera']

      hand_dict = {}
      for hand_type in ['left', 'right']:

          params = data['mano_params'][hand_type]
          handV, handJ = self.mano_layer[hand_type](torch.from_numpy(params['R']).float(),
                                                    torch.from_numpy(params['pose']).float(),
                                                    torch.from_numpy(params['shape']).float(),
                                                    trans=torch.from_numpy(params['trans']).float())
          handV_root, handJ_root = self.mano_layer[hand_type](torch.eye(3).unsqueeze(0),
                                                    torch.from_numpy(params['pose']).float(),
                                                    torch.from_numpy(params['shape']).float(),
                                                    trans=torch.from_numpy(params['trans']).float())
          # rotation_axis = self.Rmat2axis(torch.from_numpy(params['R']).float())
          if hand_type == 'left':
            pose_axis = self.pca2axis_left(torch.from_numpy(params['pose']).float())      
          else:
            pose_axis = self.pca2axis_right(torch.from_numpy(params['pose']).float())
          # handV_, handJ_, full_pose_r = self.render.Shape_formation(rotation_axis,pose_axis, torch.from_numpy(params['shape']).float(), torch.from_numpy(params['trans']).float(),hand_type,True)              
          
          handV = handV[0].numpy()
          handJ = handJ[0].numpy()
          handV = handV @ R.T + T
          handJ = handJ @ R.T + T

          handV2d = handV @ camera.T
          handV2d = handV2d[:, :2] / handV2d[:, 2:]
          handJ2d = handJ @ camera.T
          handJ2d = handJ2d[:, :2] / handJ2d[:, 2:]

          hand_dict[hand_type] = {#'hms': hms,
                                  'verts3d': handV_root[0].numpy(), 'joints3d': handJ_root[0].numpy(),
                                  'verts2d': handV2d, 'joints2d': handJ2d,
                                  'R': R @ params['R'][0],
                                  'pose': params['pose'][0],
                                  'shape': params['shape'][0],
                                  'camera': camera,
                                  'pose_axis': pose_axis
                                  }

      if True: # do augmentation operation here to get 10mm accuracy, now get 15mm accuracy.
        c = np.array([img.shape[0] / 2., img.shape[1] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.
        min_scale, max_scale = 0.8, 1.5
        s = s * np.random.choice(np.arange(min_scale, max_scale, 0.01))
        center = np.array([img.shape[0] / 2., img.shape[1] / 2.], dtype=np.float32)
        center_noise = 5
        c[0] = np.random.randint(low=int(center[0] - center_noise), high=int(center[0] + center_noise))
        c[1] = np.random.randint(low=int(center[1] - center_noise), high=int(center[1] + center_noise))
        # max_size = max(twohand_bbox[0, 2:] - twohand_bbox[0, :2] + 1) 
        # min_scale, max_scale = max_size / 0.7 / s, max_size / 0.6 / s
        # s = s * np.random.choice(np.arange(min_scale, max_scale, 0.01))   
        rot = np.random.randint(low=-120, high=120) # not defined yet.

        trans_input,inv_trans = get_affine_transform(c, s, rot, [self.opt.size_train[0], self.opt.size_train[1]])
        img = cv2.warpAffine(img, trans_input,
                            (self.opt.size_train[0], self.opt.size_train[1]),
                            flags=cv2.INTER_LINEAR)
        # if self.opt.depth:
        #   depth = cv2.warpAffine(depth, trans_input,
        #                       (self.opt.size_train[0], self.opt.size_train[1]),
        #                       flags=cv2.INTER_LINEAR)
          # depth = np.clip(depth,0,2) / 2.0        # clip to [0,2] in meters, and normalize to [0,1]         

        # if mask is not None:
        #   mask = cv2.warpAffine(mask, trans_input, (self.opt.size_train[0], self.opt.size_train[1]),
        #                         flags=cv2.INTER_NEAREST)
        #   # save mask to [64,64]
        #   mask_gt =((mask[:,:,2]>10).astype(np.uint8) | (mask[:,:,1]>10).astype(np.uint8))
        #   mask_gt = cv2.resize(mask_gt,(64,64))

        # if dense is not None:
        #   dense = cv2.warpAffine(dense, trans_input,
        #                       (self.opt.size_train[0], self.opt.size_train[1]),
        #                       flags=cv2.INTER_LINEAR)
        #   dense = cv2.resize(dense,(64,64))
        # rotate lms and joints
        tx, ty = trans_input[0,2], trans_input[1,2]
        cx, cy, fx, fy= camera[0,2],camera[1,2],camera[0,0],camera[1,1]
        t0 = (trans_input[0,0] * cx + trans_input[0,1] * cy +tx -cx) / fx
        t1 = (trans_input[1,0] * cx + trans_input[1,1] * cy +ty -cy) / fy
        rot_point = np.array([[np.cos(rot / 180. * np.pi), np.sin(rot / 180. * np.pi), t0],
                [-np.sin(rot / 180. * np.pi), np.cos(rot / 180. * np.pi), t1],
                [0, 0, 1]])
        rot_point[:2,:2] = trans_input[:2,:2].copy()

        hms = np.zeros((42, 64, 64), dtype=np.float32)
        hms_idx = 0          
        for hand_type in ['left', 'right']:
          # for hIdx in range(7):
          #   hm = cv2.imread(os.path.join(self.data_path, self.split, 'hms', '{}_{}_{}.jpg'.format(index, hIdx, hand_type)))
          #   hm = cv2.resize(hm,(256,256))
          #   hm = cv2.warpAffine(hm, trans_input,
          #                       (self.opt.size_train[0], self.opt.size_train[1]),
          #                       flags=cv2.INTER_LINEAR)            
          #   hm = cv2.resize(hm,(64,64))
          #   hm = hm.transpose(2, 0, 1) / 255.
          #   for kk in range(hm.shape[0]):
          #     hms[hms_idx] = hm[kk]
          #     hms_idx = hms_idx + 1

          hand_dict[hand_type]['joints2d'] = affine_transform_array(hand_dict[hand_type]['joints2d'],trans_input)

          hand_dict[hand_type]['camera'] = np.matmul(hand_dict[hand_type]['camera'],rot_point)
          # hand_dict[hand_type]['verts3d'] = np.matmul(hand_dict[hand_type]['verts3d'],rot_point.T)
          ## proj to image for vis
          if False:
            handV2d = hand_dict[hand_type]['joints3d'] @ hand_dict[hand_type]['camera'].T
            handV2d = handV2d[:, :2] / handV2d[:, 2:]
            lms_img = draw_lms(img, handV2d.reshape(1,-1,2))
            cv2.imwrite('local_lms_mixed_img.jpg', lms_img)
            
      else:
        pass
        # save mask to [64,64] 
        # mask_gt =((mask[:,:,2]>10).astype(np.uint8) | (mask[:,:,1]>10).astype(np.uint8))
        # mask_gt = cv2.resize(mask_gt,(64,64))
        # dense = cv2.resize(dense,(64,64))

      bbox_left = lms2bbox(hand_dict['left']['joints2d'])
      ct_left = (bbox_left[0, 2:] + bbox_left[0, :2]) / 2
      left_w, left_h = (bbox_left[:, 2] - bbox_left[:, 0])/0.8, (bbox_left[:, 3] - bbox_left[:, 1])/0.8

      bbox_right = lms2bbox(hand_dict['right']['joints2d'])
      ct_right = (bbox_right[0, 2:] + bbox_right[0, :2]) / 2
      right_w, right_h = (bbox_right[:, 2] - bbox_right[:, 0])/0.8, (bbox_right[:, 3] - bbox_right[:, 1])/0.8

      # calculate 1/8 gts
      down = self.opt.down_ratio
      #down = 4 # test for IntagHand format
      self.max_objs = 2 
      self.num_classes = 2 # left/right hand
      heatmap_h, heatmap_w = self.opt.size_train[1] // down, self.opt.size_train[0] // down
      hm = np.zeros((self.num_classes, heatmap_h, heatmap_w), dtype=np.float32)
      hm_lms = np.zeros((42, heatmap_h, heatmap_w), dtype=np.float32)
      wh = np.zeros((self.max_objs, 2), dtype=np.float32)
      off_hm = np.zeros((self.max_objs, 2), dtype=np.float32)
      off_lms = np.zeros((self.max_objs, 21 *2), dtype=np.float32)
      ind = np.zeros((self.max_objs), dtype=np.int64)
      reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

      if True:
        w, h = left_w / down, left_h / down               
        lms21_down =  hand_dict['left']['joints2d'] / down    
        hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        hp_radius = max(0, int(hp_radius))
        ct_int = (ct_left / down).astype(np.int32)
        for kk in range(21):
          draw_umich_gaussian(hm_lms[kk], (lms21_down[kk]).astype(np.int32), hp_radius)  
          off_lms[0, kk * 2: kk * 2 + 2] = lms21_down[kk, :2] - ct_int    
        draw_umich_gaussian(hm[0], ct_int, hp_radius)    
        wh[0] = 1. * w, 1. * h
        ind[0] = ct_int[1] * heatmap_w + ct_int[0]
        off_hm[0] = (ct_left / down) - ct_int
        reg_mask[0] = 1
      
      if True:
        w, h = right_w / down, right_h / down   
        lms21_down =  hand_dict['right']['joints2d'] / down        
        hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        hp_radius = max(0, int(hp_radius))
        ct_int = (ct_right / down).astype(np.int32)
        for kk in range(21):
          draw_umich_gaussian(hm_lms[21+kk], (lms21_down[kk]).astype(np.int32), hp_radius)
          off_lms[1, kk * 2: kk * 2 + 2] = lms21_down[kk, :2] - ct_int      
        draw_umich_gaussian(hm[1], ct_int, hp_radius)   
        wh[1] = 1. * w, 1. * h
        ind[1] = ct_int[1] * heatmap_w + ct_int[0]
        off_hm[1] = (ct_right / down) - ct_int
        reg_mask[1] = 1

      # address the outscreen aug case.
      if ind[0] >= heatmap_h*heatmap_w or ind[0] <0:
        ind[0] = 0
      if ind[1] >= heatmap_h*heatmap_w or ind[1] <0:
        ind[1] = 0

        #vis
      if False:
        joints_left = hand_dict['left']['joints3d']
        K = hand_dict['left']['camera']
        lms = hand_dict['left']['joints2d']
        for xy in lms[:]:
            cv2.circle(img, (int(xy[0]), int(xy[1])), 1, (0,0,255), 1)
        cv2.imwrite('img_new_proj.jpg',img) 
        cv2.rectangle(img, (int(ct_left[0]-left_w/2), int(ct_left[1]-left_h/2)), (int(ct_left[0]+left_w/2), int(ct_left[1]+left_h/2)), (0,255,0), 1)          
        cv2.imwrite('new_left_bbox.png', img)
        cv2.circle(img, (int(ct_left[0]),int(ct_left[1])), 1, (255, 0, 0), 2)
        cv2.imwrite('new_left_bbox.png', img)
        handV2d = joints_left @ K.T
        handV2d = handV2d[:, :2] / handV2d[:, 2:]  
        for xy in handV2d[:]:
            cv2.circle(img, (int(xy[0]), int(xy[1])), 1, (0,255,0), 1)
        cv2.imwrite('img_proj.jpg',img)
                        
      ret = {'hm': hm}
      # if self.opt.heatmaps:
      ret.update({'hms': hm_lms})

      ret.update({'valid': reg_mask.astype(np.int64), 'ind': ind})
      ret.update({'wh': wh}) # used for perceptual loss
      if self.opt.off:
        ret.update({'off_lms': off_lms})
        ret.update({'off_hm': off_hm})
      # add the followings for IntagHand format.  
      # ret.update({'rot_point': rot_point.astype(np.float32)})
      
      ret.update({'input': self.normal(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)})
      ret.update({'image': img.copy()})
      # ret.update({'mask_gt': mask_gt.astype(np.float32)})
      # ret.update({'dense': dense.astype(np.float32)})
      # ret.update({'hms': hms.astype(np.float32)})
      ret.update({'file_id':index})
      ret.update({'lms_left_gt': hand_dict['left']['joints2d'].astype(np.float32)})
      ret.update({'lms_right_gt': hand_dict['right']['joints2d'].astype(np.float32)})
      ret.update({'joints_left_gt': hand_dict['left']['joints3d'].astype(np.float32)})
      ret.update({'verts_left_gt': hand_dict['left']['verts3d'].astype(np.float32)})
      ret.update({'joints_right_gt': hand_dict['right']['joints3d'].astype(np.float32)})
      ret.update({'verts_right_gt': hand_dict['right']['verts3d'].astype(np.float32)})
      ret.update({'camera_left': hand_dict['left']['camera'].astype(np.float32)})
      ret.update({'camera_right': hand_dict['right']['camera'].astype(np.float32)})
      ret.update({'pose_left': np.array(hand_dict['left']['pose_axis'][0],dtype=np.float32)})
      ret.update({'pose_right': np.array(hand_dict['right']['pose_axis'][0],dtype=np.float32)})
      ret.update({'shape_left': hand_dict['left']['shape'].astype(np.float32)})
      ret.update({'shape_right': hand_dict['right']['shape'].astype(np.float32)})
      # vis
      if False:
        lms_img = draw_lms(img, ret['lms_left_gt'].reshape(1,-1,2))
        cv2.imwrite('local_lms_mixed_img.jpg', lms_img)      
        if self.opt.photometric_loss:
          cv2.imwrite('new_mask.png',ret['mask'])              
      return ret
      
    if self.split == 'train' or self.split == 'train_3d' or self.split == 'val' or self.split == 'test':
      # np.random.seed(317)
      ret, x_img, depth = self.augment_centernet(self.data[index],self.split)
      # dpt, M, com = self.get_pcl(depth*1000)
      img = x_img.copy()
      if self.opt.brightness and np.random.randint(0, 2) == 0:
        img = data_augment._brightness(x_img, min=0.8, max=1.2)

      ret.update({'file_id': index})
      ret.update({'dataset': self.data[index]['dataset']})
      if 'id' in self.data[index] and self.split == 'test':
        ret.update({'id': (self.data[index]['id'])}) 
        ret.update({'frame_num': int(self.data[index]['imgpath'][-10:-4])})       
      ret.update({'input': self.normal(img).transpose(2, 0, 1)})
      if depth is not None:
        ret.update({'depth': depth.astype(np.float32).reshape(1,self.opt.size_train[0],self.opt.size_train[1])}) 

      # img_new = (img / 255.)[:, :, ::-1].astype(np.float32)
      ret.update({'image': img.copy()})

      if 'mano_coeff' in self.data[index]:
        ret.update({'joints': (self.data[index]['joints']).astype(np.float32).reshape(1,-1)})
        ret.update({'mano_coeff': self.data[index]['mano_coeff'].astype(np.float32).reshape(1,-1)})
        ret.update({'K': np.array(self.data[index]['K'].astype(np.float32).reshape(1,-1))})

      # lms_img = draw_lms(img, ret['lms_left_gt'].reshape(ret['lms_left_gt'].shape[0],-1,2))
      # cv2.imwrite('local_lms_mixed_img.jpg', lms_img)      
      # cv2.imwrite('new_image.jpg',x_img)
      # if self.opt.photometric_loss:
      #   cv2.imwrite('new_mask.png',img_data['mask'])

      return ret


  def augment_centernet(self, img_data, split):
    assert 'imgpath' in img_data
    img_data_aug = copy.deepcopy(img_data)
    img = cv2.imread(os.path.join(self.opt.pre_fix, img_data_aug['imgpath']))
    depth = cv2.imread(os.path.join(self.opt.pre_fix, img_data_aug['depthpath']), cv2.IMREAD_ANYDEPTH) / 1000. if 'depthpath' in img_data and self.opt.depth else None
    # depth = depth / 1000. # transpose to meters
    # depth = read_depth_img(os.path.join(self.opt.pre_fix, img_data_aug['depthpath'])) if 'depthpath' in img_data and self.opt.depth else None
    # mask[:,:,2] because HO3D has red hand and blue object.
    mask_path = img_data_aug['imgpath'].replace('rgb','mask') # for H2O mask file
    mask = cv2.imread(os.path.join(self.opt.pre_fix, mask_path))
    # mask = cv2.imread(os.path.join(self.opt.pre_fix, mask_path))[:, :, 2] \
    #   if 'maskpath' in img_data_aug and self.opt.photometric_loss else None # 
    if img is None:
      print('what',os.path.join(self.opt.pre_fix, img_data_aug['imgpath']))    
    img_height, img_width = img.shape[:2]
    if mask is not None:
      mask_height, mask_width = mask.shape[:2]
      if (mask_height != img_height) or (mask_width != img_width):
        mask = cv2.resize(mask,(img_width,img_height))
        
    c = np.array([img_width / 2., img_height / 2.], dtype=np.float32)
    s = max(img_height, img_width) * 1.
    rot = 0

    # gts = np.repeat(gts,5,axis=0)
    if 'lms' in img_data_aug:
      lms = np.copy(img_data_aug['lms']).astype(np.float32).reshape(-1,2)
      lms_left = np.copy(img_data_aug['lms'][:21,:2]).astype(np.float32).reshape(-1,2)
      lms_right = np.copy(img_data_aug['lms'][21:,:2]).astype(np.float32).reshape(-1,2)

    if 'joints' in img_data_aug:
      joints_left = np.copy(img_data_aug['joints'][:21,:3]).astype(np.float32).reshape(-1,3)
      joints_right = np.copy(img_data_aug['joints'][21:,:3]).astype(np.float32).reshape(-1,3)

    if 'K' in img_data_aug:
      K = np.copy(img_data_aug['K']).astype(np.float32)
    

    # vis
    if False:
        for xy in lms_left[:]:
            cv2.circle(img, (int(xy[0]), int(xy[1])), 2, (0,255,255), 2)
        cv2.imwrite('img.jpg',img)    
        rot_vec = np.array([[0, 0, 0]]).astype(np.float32)
        tran_vec = np.array([[0, 0, 0]]).astype(np.float32)

        img_points, _ = cv2.projectPoints(
            joints_left, rot_vec, tran_vec, K, None)
        for xy in img_points[:]:
            cv2.circle(img, (int(xy[0][0]), int(xy[0][1])), 1, (0,0,255), 1)
        cv2.imwrite('img_proj.jpg',img)

    # choose center and crop as (512,512)
    if True:
      twohand_bbox = lms2bbox(lms)
      center = (twohand_bbox[0, 2:] + twohand_bbox[0, :2]) / 2
      center_noise = 10
      c[0] = np.random.randint(low=int(center[0] - center_noise), high=int(center[0] + center_noise))
      c[1] = np.random.randint(low=int(center[1] - center_noise), high=int(center[1] + center_noise))    
      max_size = max(twohand_bbox[0, 2:] - twohand_bbox[0, :2] + 1) 
      min_scale, max_scale = max_size / 0.85 / s, max_size / 0.6 / s
      s = s * np.random.choice(np.arange(min_scale, max_scale, 0.01))   
    rot = np.random.randint(low=-60, high=60) # not defined yet.
                   
    trans_input,inv_trans = get_affine_transform(c, s, rot, [self.opt.size_train[0], self.opt.size_train[1]])
    img = cv2.warpAffine(img, trans_input,
                         (self.opt.size_train[0], self.opt.size_train[1]),
                         flags=cv2.INTER_LINEAR)
    if depth is not None:                    
      depth = cv2.warpAffine(depth, trans_input,
                          (self.opt.size_train[0], self.opt.size_train[1]),
                          flags=cv2.INTER_LINEAR)        
      # depth = np.clip(depth,0,2) / 2.0        # clip to [0,2] in meters, and normalize to [0,1]         

    if mask is not None:
      mask = cv2.warpAffine(mask, trans_input, (self.opt.size_train[0], self.opt.size_train[1]),
                            flags=cv2.INTER_NEAREST)

    # rotate lms and joints
    tx, ty = trans_input[0,2], trans_input[1,2]
    cx, cy, fx, fy= K[0,2],K[1,2],K[0,0],K[1,1]
    t0 = (trans_input[0,0] * cx + trans_input[0,1] * cy +tx -cx) / fx
    t1 = (trans_input[1,0] * cx + trans_input[1,1] * cy +ty -cy) / fy
    rot_point = np.array([[np.cos(rot / 180. * np.pi), np.sin(rot / 180. * np.pi), t0],
            [-np.sin(rot / 180. * np.pi), np.cos(rot / 180. * np.pi), t1],
            [0, 0, 1]])
    rot_point[:2,:2] = trans_input[:2,:2].copy()
    
    if lms is not None:
      lms = affine_transform_array(lms,trans_input)
    joints_left = np.matmul(joints_left,rot_point.T)
    joints_right = np.matmul(joints_right,rot_point.T)

    # calculate gts 
    # check whether both hand appear
    # K[0][0] = K[0][0]*trans_input[0][0]
    # K[1][1] = K[1][1]*trans_input[1][1]
    # K[0][2] = K[0][2]*trans_input[0][0] + trans_input[0][2]
    # K[1][2] = K[1][2]*trans_input[1][1] + trans_input[1][2]
    # self.fx = K[0][0]
    # self.fy = K[1][1] 
    # self.ux = K[0][2]
    # self.uy = K[1][2] 

    valid_left = 1 if img_data_aug['mano_coeff'][0] == 1 else 0    
    bbox_left = lms2bbox(lms[:21])
    ct_left = (bbox_left[0, 2:] + bbox_left[0, :2]) / 2
    left_w, left_h = (bbox_left[:, 2] - bbox_left[:, 0])/0.7, (bbox_left[:, 3] - bbox_left[:, 1])/0.7

    valid_right = 1 if img_data_aug['mano_coeff'][62] == 1 else 0
    bbox_right = lms2bbox(lms[21:])
    ct_right = (bbox_right[0, 2:] + bbox_right[0, :2]) / 2
    right_w, right_h = (bbox_right[:, 2] - bbox_right[:, 0])/0.7, (bbox_right[:, 3] - bbox_right[:, 1])/0.7

    # calculate 1/8 gts
    down = self.opt.down_ratio
    down = 4 # test for IntagHand format
    heatmap_h, heatmap_w = self.opt.size_train[1] // down, self.opt.size_train[0] // down
    hm = np.zeros((self.num_classes, heatmap_h, heatmap_w), dtype=np.float32)
    hm_lms = np.zeros((42, heatmap_h, heatmap_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    off_hm = np.zeros((self.max_objs, 2), dtype=np.float32)
    off_lms = np.zeros((self.max_objs, 21 *2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

    if valid_left:
      w, h = left_w / down, left_h / down               
      lms21_down =  lms[:21] / down    
      hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
      hp_radius = max(0, int(hp_radius))
      ct_int = (ct_left / down).astype(np.int32)
      for kk in range(21):
        draw_umich_gaussian(hm_lms[kk], (lms21_down[kk]).astype(np.int32), hp_radius)  
        off_lms[0, kk * 2: kk * 2 + 2] = lms21_down[kk, :2] - ct_int    
      draw_umich_gaussian(hm[0], ct_int, hp_radius)    
      wh[0] = 1. * w, 1. * h
      ind[0] = ct_int[1] * heatmap_w + ct_int[0]
      off_hm[0] = (ct_left / down) - ct_int
      reg_mask[0] = 1
     
    if valid_right:
      w, h = right_w / down, right_h / down   
      lms21_down =  lms[21:] / down        
      hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
      hp_radius = max(0, int(hp_radius))
      ct_int = (ct_right / down).astype(np.int32)
      for kk in range(21):
        draw_umich_gaussian(hm_lms[21+kk], (lms21_down[kk]).astype(np.int32), hp_radius)
        off_lms[1, kk * 2: kk * 2 + 2] = lms21_down[kk, :2] - ct_int      
      draw_umich_gaussian(hm[1], ct_int, hp_radius)   
      wh[1] = 1. * w, 1. * h
      ind[1] = ct_int[1] * heatmap_w + ct_int[0]
      off_hm[1] = (ct_right / down) - ct_int
      reg_mask[1] = 1

    # address the outscreen aug case.
    if ind[0] >= heatmap_h*heatmap_w or ind[0] <0:
      ind[0] = 0
    if ind[1] >= heatmap_h*heatmap_w or ind[1] <0:
      ind[1] = 0

    if mask is not None:
      img_data_aug['mask'] = mask
      #vis
    if False:
      for xy in lms[:]:
          cv2.circle(img, (int(xy[0]), int(xy[1])), 1, (0,0,255), 1)
      cv2.imwrite('img_new_proj.jpg',img) 
      cv2.rectangle(img, (int(ct_left[0]-left_w/2), int(ct_left[1]-left_h/2)), (int(ct_left[0]+left_w/2), int(ct_left[1]+left_h/2)), (0,255,0), 1)          
      cv2.imwrite('new_left_bbox.png', img)
      cv2.circle(img, (int(ct_left[0]),int(ct_left[1])), 1, (255, 0, 0), 2)
      cv2.imwrite('new_left_bbox.png', img)
      rot_vec = np.array([[0, 0, 0]]).astype(np.float32)
      tran_vec = np.array([[0, 0, 0]]).astype(np.float32)      
      img_points, _ = cv2.projectPoints(
          joints_left, rot_vec, tran_vec, K, None)
      for xy in img_points[:]:
          cv2.circle(img, (int(xy[0][0]), int(xy[0][1])), 1, (0,255,0), 1)
      cv2.imwrite('img_proj.jpg',img)
                         
    ret = {'hm': hm}
    # if self.opt.heatmaps:
    ret.update({'hms': hm_lms})

    ret.update({'valid': reg_mask.astype(np.int64), 'ind': ind})
    ret.update({'lms': lms.astype(np.float32)})
    ret.update({'wh': wh}) # used for perceptual loss
    ret.update({'K_new': K.reshape(1,-1)})
    mask_gt =((mask[:,:,2]>10).astype(np.uint8) | (mask[:,:,1]>10).astype(np.uint8))
    mask_gt = cv2.resize(mask_gt,(64,64))
    ret.update({'mask_gt': mask_gt.astype(np.float32)})
    if self.opt.off:
      ret.update({'off_lms': off_lms})
      ret.update({'off_hm': off_hm})
    # add the followings for IntagHand format.  
    ret.update({'lms_left_gt': lms[:21].astype(np.float32)})
    ret.update({'lms_right_gt': lms[21:].astype(np.float32)})      
    ret.update({'joints_left_gt': joints_left})
    ret.update({'joints_right_gt': joints_right})
    ret.update({'camera_left': K.astype(np.float32)})
    ret.update({'camera_right': K.astype(np.float32)})
    ret.update({'rot_point': rot_point.astype(np.float32)})
    
    return ret, img, depth

