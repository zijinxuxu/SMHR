from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.functional import split

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform, affine_transform_array
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
from utils import data_augment, data_generators
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

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
    idx = np.where((uv[:,0] > 0)&(uv[:,1] > 0)) 
    if len(idx[0])==0:
      return np.zeros((1, 4), dtype=np.float32)     
    x_min = uv[idx,0].min()
    x_max = uv[idx,0].max()
    y_min = uv[idx,1].min()
    y_max = uv[idx,1].max()

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

class SimplifiedDataset(data.Dataset):
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
  def add_handmean(self,mano):
    mano = mano.squeeze()
    hand_mean_right = np.array([ 0.1117, -0.0429,  0.4164,  0.1088,  0.0660,  0.7562, -0.0964,  0.0909,
         0.1885, -0.1181, -0.0509,  0.5296, -0.1437, -0.0552,  0.7049, -0.0192,
         0.0923,  0.3379, -0.4570,  0.1963,  0.6255, -0.2147,  0.0660,  0.5069,
        -0.3697,  0.0603,  0.0795, -0.1419,  0.0859,  0.6355, -0.3033,  0.0579,
         0.6314, -0.1761,  0.1321,  0.3734,  0.8510, -0.2769,  0.0915, -0.4998,
        -0.0266, -0.0529,  0.5356, -0.0460,  0.2774])
    mano[3:48] = mano[3:48] + hand_mean_right
    return mano


  def __getitem__(self, index):
    if self.split == 'train' or self.split == 'train_3d' or self.split == 'val':
      ret, x_img = self.augment_centernet(self.data[index])
      img = x_img.copy()
      # add img noise
      if self.opt.brightness and np.random.randint(0, 2) == 0:
        img = data_augment.add_noise(img.astype(np.float32),
                                      noise=0.0,
                                      scale=255.0,
                                      alpha=0.3, beta=0.05).astype(np.uint8)

      ret.update({'file_id': index})
      ret.update({'dataset': self.data[index]['dataset']})
      if 'id' in self.data[index] and self.split == 'test':
        ret.update({'id': (self.data[index]['id'])}) 
        ret.update({'frame_num': int(self.data[index]['imgpath'][-10:-4])})       
      ret.update({'input': self.normal(img).transpose(2, 0, 1)})
      ret.update({'image': img.copy()})
      img_new = (img / 255.)[:, :, ::-1].astype(np.float32)
      ret.update({'render': img_new.copy()})
      if 'mano_coeff' in self.data[index]:
        if self.data[index]['dataset']==1: # HO3D use flaten mano value is larger than FreiHAND.
          ret.update({'mano_coeff': (self.add_handmean(self.data[index]['mano_coeff'])).astype(np.float32).reshape(1,-1)})
        # ret.update({'mano_coeff': self.data[index]['mano_coeff'].astype(np.float32).reshape(1,-1)})
        ret.update({'K': np.array(self.data[index]['K'].astype(np.float32).reshape(1,-1))})

      # lms_img = draw_lms(img, ret['lms'].reshape(ret['lms'].shape[0],-1,2))
      # cv2.imwrite('local_lms_mixed_img.jpg', lms_img)      
      # cv2.imwrite('new_image.jpg',x_img)
      # if self.opt.photometric_loss:
      #   cv2.imwrite('new_mask.png',img_data['mask'])

      return ret

   
    elif self.split == 'test':
      imgIn = cv2.imread(os.path.join(self.opt.pre_fix, self.data[index]['filepath']))
      h, w, _ = imgIn.shape
      ret = {'file_id': index}
      if 'bboxes' in self.data[index]:
        if 'handness' in self.data[index]:
          x1,y1,x2,y2 = self.data[index]['bboxes'][0][0] if self.data[index]['handness'] == 0 else self.data[index]['bboxes'][1][0]
          ret.update({'handness': np.array([self.data[index]['handness']])}) 
        else:
          x1,y1,x2,y2 = self.data[index]['bboxes']
        # imgIn = cv2.rectangle(imgIn, (int(x1), int(y1)),
        #               (int(x2), int(y2)), (0, 0, 255), thickness=3)
        # cv2.imwrite('imgIn.jpg',imgIn)
        c = np.array([x1+x2,y1+y2])/2
        s = max(y2-y1, x2-x1) / 0.5 #* 2.
        rot = 0
        trans_input,inv_trans = get_affine_transform(c, s, rot, [self.opt.size_train[0], self.opt.size_train[1]])
        imgIn = cv2.warpAffine(imgIn, trans_input,
                            (self.opt.size_train[0], self.opt.size_train[1]),
                            flags=cv2.INTER_LINEAR)
        # cv2.circle(imgOut, (int(112),int(112)), 2, (255,0,255), 2)
        # cv2.imwrite('imgOut.jpg',imgOut)
        ret.update({'inv_trans': inv_trans.astype(np.float32)}) 

      if self.opt.pick_hand:
        if np.random.randint(0, 2) == 1:
          imgIn = cv2.flip(imgIn, 1) 
          ret.update({'handness':np.array([0],dtype=np.int64)}) # left hand   
        else:
          ret.update({'handness':np.array([1],dtype=np.int64)}) # stay right hand  
      if 'xyz' in self.data[index] and 'handness' in self.data[index]: 
        xyz = np.copy(self.data[index]['xyz'][:21,:3]).astype(np.float32).reshape(1,-1,3) if self.data[index]['handness']==0 \
          else np.copy(self.data[index]['xyz'][21:,:3]).astype(np.float32).reshape(1,-1,3)        
        ret.update({'xyz': xyz.reshape(1,-1)})        
      if 'lms42' in self.data[index]:
        lms21 = np.copy(self.data[index]['lms42'][:21,:2]).astype(np.float32).reshape(1,-1,2) if self.data[index]['handness']==0 \
          else np.copy(self.data[index]['lms42'][21:,:2]).astype(np.float32).reshape(1,-1,2)        
        ret.update({'lms21_orig': lms21.copy().reshape(1,-1)}) 

        for i in range(lms21.shape[1]):
          # cv2.circle(imgIn, (int(lms21[:, i, 0]),int(lms21[:, i, 1])), 3, (0, 0, 255), 2)
          lms21[:,i, :2] = affine_transform_array(lms21[:,i, :2].reshape(1,-1), trans_input)
          # cv2.circle(imgOut, (int(lms21[:, i, 0]),int(lms21[:, i, 1])), 3, (0, 255, 0), 2)
        ret.update({'lms21': lms21.reshape(1,-1)})   
        # cv2.imwrite('lms_in.jpg',imgIn)
        # cv2.imwrite('lms_out.jpg',imgOut)
        # for i in range(lms21.shape[1]):
        #   lms21[:,i, :2] = affine_transform_array(lms21[:,i, :2].reshape(1,-1), inv_trans)
        #   cv2.circle(imgIn, (int(lms21[:, i, 0]),int(lms21[:, i, 1])), 3, (255, 0, 0), 1)
        # cv2.imwrite('lms_out_in.jpg',imgIn)    

      # cv2.circle(image, (int(center_orig[0] * ratio ),int(center_orig[1] * ratio )), 3, 100, 3)
      # cv2.imwrite('final_bbox.jpg',image)
        # ret.update({'ind': np.array([int(ind)])})      
      ret.update({'dataset': self.data[index]['dataset']})
      ret.update({'valid': np.ones((1,), dtype=np.bool)})
      ret.update({'input': self.normal(imgIn).transpose(2, 0, 1)})
      # img_new = (imgIn / 255.)[:, :, ::-1].astype(np.float32)
      ret.update({'image': imgIn.copy()})
      if 'K' in self.data[index]:
        ret.update({'K_aug': np.array(self.data[index]['K'],dtype=np.float32)})
      if 'scale' in self.data[index]:
        ret.update({'scale': np.array(self.data[index]['scale'])})
      if 'root_3d' in self.data[index]:
          ret.update({'root_3d': np.array(self.data[index]['root_3d'])})        
      return ret


  def augment_centernet(self, img_data):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    img_data_aug = copy.deepcopy(img_data)
    image = cv2.imread(os.path.join(self.opt.pre_fix, img_data_aug['filepath']))
    mask = cv2.imread(os.path.join(self.opt.pre_fix, img_data_aug['maskpath'])) if 'maskpath' in img_data_aug else None # and self.opt.photometric_loss 
    if mask is None:
      print('what',os.path.join(self.opt.pre_fix, img_data_aug['maskpath']))
    else:
      mask = mask[:, :, 2]  
    img_height, img_width = image.shape[:2]
    img = image.copy()

    c = np.array([img_width / 2., img_height / 2.], dtype=np.float32)
    s = max(img_height, img_width) * 1.
    rot = 0

    if 'lms21' in img_data_aug:
      lms = np.copy(img_data_aug['lms21']).astype(np.float32).reshape(-1,2)
    if mask is not None:
      mask_height, mask_width = mask.shape[:2]
      if (mask_height != img_height) or (mask_width != img_width):
        mask = cv2.resize(mask,(img_width,img_height))

    if True: # random crop 
      hand_bbox = lms2bbox(lms)
      center = (hand_bbox[0, 2:] + hand_bbox[0, :2]) / 2
      center_noise = 5
      c[0] = np.random.randint(low=int(center[0] - center_noise), high=int(center[0] + center_noise))
      c[1] = np.random.randint(low=int(center[1] - center_noise), high=int(center[1] + center_noise))    
      max_size = max(hand_bbox[0, 2:] - hand_bbox[0, :2] + 1) 
      min_scale, max_scale = max_size / 0.6 / s, max_size / 0.3 / s
      s = s * np.random.choice(np.arange(min_scale, max_scale, 0.01))   
      rot = np.random.randint(low=-40, high=40) # not defined yet.
    
    trans_input,inv_trans = get_affine_transform(c, s, rot, [self.opt.size_train[0], self.opt.size_train[1]])
    img = cv2.warpAffine(img, trans_input,
                         (self.opt.size_train[0], self.opt.size_train[1]),
                         flags=cv2.INTER_LINEAR)

    if mask is not None:
      mask = cv2.warpAffine(mask, trans_input, (self.opt.size_train[0], self.opt.size_train[1]),
                            flags=cv2.INTER_NEAREST)

    if lms is not None:
      lms = affine_transform_array(lms,trans_input)

    img_data_aug['lms21'] = lms

    if mask is not None:
      img_data_aug['mask'] = mask

    bbox = lms2bbox(lms)
    ct_hand = (bbox[0, 2:] + bbox[0, :2]) / 2
    hand_w, hand_h = (bbox[0, 2] - bbox[0, 0]), (bbox[0, 3] - bbox[0, 1])

    # vis
    if False:
      cv2.rectangle(mask, (bbox[:, 0], bbox[:, 1]), (bbox[:, 2], bbox[:, 3]), 255, 1)
      lms_img = draw_lms(img, lms.reshape(-1,21,2))
      cv2.imwrite('local_lms_mixed_img.jpg', lms_img)      
      cv2.imwrite('new_bboxes.png', mask)
      cv2.imwrite('old_lms.jpg', image)
      cv2.imwrite('new_lms.jpg', img)

    # calculate 1/4 gts
    down = self.opt.down_ratio
    # down = 4 # test for IntagHand format
    heatmap_h, heatmap_w = self.opt.size_train[1] // down, self.opt.size_train[0] // down
    hm = np.zeros((self.num_classes, heatmap_h, heatmap_w), dtype=np.float32)
    hm_lms = np.zeros((21, heatmap_h, heatmap_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    off_hm = np.zeros((self.max_objs, 2), dtype=np.float32)
    off_lms = np.zeros((self.max_objs, 21 *2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

    if True:
      w, h = hand_w / down, hand_h / down               
      lms21_down =  lms[:21] / down    
      hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
      hp_radius = max(0, int(hp_radius))
      ct_int = (ct_hand / down).astype(np.int32)
      for kk in range(21):
        draw_umich_gaussian(hm_lms[kk], (lms21_down[kk]).astype(np.int32), hp_radius)  
        off_lms[0, kk * 2: kk * 2 + 2] = lms21_down[kk, :2] - ct_int    
      draw_umich_gaussian(hm[0], ct_int, hp_radius)    
      wh[0] = 1. * w, 1. * h
      ind[0] = ct_int[1] * heatmap_w + ct_int[0]
      off_hm[0] = (ct_hand / down) - ct_int
      reg_mask[0] = 1

    # address the outscreen aug case.
    if ind[0] >= heatmap_h*heatmap_w or ind[0] <0:
      ind[0] = 0
    ret = {'hm': hm}
    if self.opt.heatmaps:
      ret.update({'hms': hm_lms})

    ret.update({'valid': reg_mask.astype(np.bool), 'ind': ind})
    ret.update({'lms': lms.astype(np.float32)})
    ret.update({'wh': wh}) # used for perceptual loss
    ret.update({'mask': mask.astype(np.float32)})
    if self.opt.off:
      ret.update({'off_lms': off_lms})
      ret.update({'off_hm': off_hm})

    return ret, img
