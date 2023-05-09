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
  
  def add_handmean(self,mano):
    mano = mano.squeeze()
    hand_mean_right = np.array([ 0.1117, -0.0429,  0.4164,  0.1088,  0.0660,  0.7562, -0.0964,  0.0909,
         0.1885, -0.1181, -0.0509,  0.5296, -0.1437, -0.0552,  0.7049, -0.0192,
         0.0923,  0.3379, -0.4570,  0.1963,  0.6255, -0.2147,  0.0660,  0.5069,
        -0.3697,  0.0603,  0.0795, -0.1419,  0.0859,  0.6355, -0.3033,  0.0579,
         0.6314, -0.1761,  0.1321,  0.3734,  0.8510, -0.2769,  0.0915, -0.4998,
        -0.0266, -0.0529,  0.5356, -0.0460,  0.2774])
    mano[3:48] = mano[3:48] + hand_mean_right
    return mano.reshape(1,-1)

  def __getitem__(self, index):
    if self.split == 'train' or self.split == 'train_3d' or self.split == 'test' or self.split == 'val':
      idx = self.random_idx[index]
      # img_data, x_img = self.augment_centernet(self.data[idx[0]])
      img_data, x_img = self.add_background(self.data[idx[0]])

      flag_first = True
      for idx_ in idx[:]:
        tmp_data = copy.deepcopy(self.data[idx_])
        orig_width = 224 # x_img.shape[1], FreiHAND
        if np.random.randint(0, 2) == 0 and self.opt.pick_hand:
          tmp_data['lms21'] = tmp_data['lms21'].astype(np.float32).reshape(-1,2)
          tmp_data['lms21'][:, 0] = orig_width - tmp_data['lms21'][:, 0]
          top_low_x = orig_width - tmp_data['bboxes'][0,2]
          bottom_high_x = orig_width - tmp_data['bboxes'][0,0]
          tmp_data['bboxes'][0,0] = top_low_x
          tmp_data['bboxes'][0,2] = bottom_high_x
          if 'handness' not in tmp_data:
            tmp_data.update({'handness':[0]})
          else:
            handness = 1 - tmp_data['handness'][0]
            tmp_data.update({'handness':[handness]})
        else:
          if 'handness' not in tmp_data:
            tmp_data.update({'handness':[1]})
          else:
            tmp_data.update({'handness':tmp_data['handness']})

        img_data, x_img = self.augment_append(tmp_data, img_data, x_img)

      img = x_img.copy()
      if self.opt.brightness and np.random.randint(0, 2) == 0:
        img = data_augment._brightness(x_img, min=0.4, max=1.6)
      ret = self.calc_gt(img_data)
      ret.update({'file_id': index})
      ret.update({'dataset': self.data[index]['dataset']})      
      ret.update({'input': self.normal(img).transpose(2, 0, 1)})
      if self.opt.photometric_loss:
        mask = img_data['mask'] if 'mask' in img_data else np.zeros([self.opt.size_train[1], self.opt.size_train[0]], dtype=np.uint8)
        if 'mask' in img_data:
          mask = (mask > 50).astype(np.uint8)
        ret.update({'mask': mask})

      img_new = (img / 255.)[:, :, ::-1].astype(np.float32)
      ret.update({'render': img_new.copy()})
      ret.update({'image': img.copy()})
      # if 'mano_coeff' in self.data[index]:
      #   ret.update({'xyz': (self.data[index]['xyz']).astype(np.float32).reshape(1,-1)})
      #   ret.update({'mano_coeff': self.data[index]['mano_coeff'].astype(np.float32).reshape(1,-1)})
      #   ret.update({'K': np.array(self.data[index]['K'].astype(np.float32).reshape(1,-1))})      
      # vis
      if False:
        sv_img = draw_lms(x_img, ret['lms21'].reshape(-1,21,2))
        cv2.imwrite('append_lms_mixed_img.jpg', sv_img)
        cv2.imwrite('append_mask_mix.png',img_data['mask'])
        cv2.imwrite('append_hm.png',ret['hm'][0]*255)

      return ret
      
    elif self.split == 'eval':
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
        s = max(y2-y1, x2-x1) / 0.4 #* 2.
        rot = 0
        trans_input,inv_trans = get_affine_transform(c, s, rot, [self.opt.size_train[0], self.opt.size_train[1]])
        imgIn = cv2.warpAffine(imgIn, trans_input,
                            (self.opt.size_train[0], self.opt.size_train[1]),
                            flags=cv2.INTER_LINEAR)
        # cv2.circle(imgOut, (int(112),int(112)), 2, (255,0,255), 2)
        # cv2.imwrite('imgOut.jpg',imgOut)
        ret.update({'inv_trans': inv_trans.astype(np.float32)}) 
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
      ret.update({'mask': np.ones((1,), dtype=np.bool)})
      ret.update({'input': self.normal(imgIn).transpose(2, 0, 1)})
      img_new = (imgIn / 255.)[:, :, ::-1].astype(np.float32)
      ret.update({'image': img_new.copy()})
      if 'K' in self.data[index]:
        ret.update({'K': np.array(self.data[index]['K'])})
      if 'scale' in self.data[index]:
        ret.update({'scale': np.array(self.data[index]['scale'])})
      if 'root_3d' in self.data[index]:
          ret.update({'root_3d': np.array(self.data[index]['root_3d'])})        
      return ret

    elif self.split == 'aug_train':
      img = cv2.imread(os.path.join(self.opt.pre_fix, self.data[index]['filepath'])) 
      mask = cv2.imread(os.path.join(self.opt.pre_fix, self.data[index]['maskpath']))[:, :, 2] \
        if 'maskpath' in self.data[index] and self.opt.photometric_loss else None 
      if mask is not None:
          mask = (mask > 50).astype(np.uint8)

      ret = {'file_id': self.data[index]['file_id']}
      ret.update({'input': self.normal(img).transpose(2, 0, 1)})
      img_new = (img / 255.)[:, :, ::-1].astype(np.float32)
      ret.update({'image': img_new.copy()})
      ret.update({'ind': self.data[index]['ind']})
      ret.update({'box': self.data[index]['box']})
      ret.update({'lms21': self.data[index]['lms21']})
      ret.update({'mask': self.data[index]['mask']})
      ret.update({'skin_mask': mask})
      # get hms
      # box[idx_] = gts[idx, 1] / heatmap_h, gts[idx, 0] / heatmap_w, gts[idx, 3] / heatmap_h, gts[idx, 2] / heatmap_w
      down = self.opt.down_ratio
      heatmap_h, heatmap_w = self.opt.size_train[1] // down, self.opt.size_train[0] // down
      hm = np.zeros((self.num_classes, heatmap_h, heatmap_w), dtype=np.float32)
      gts = np.copy(self.data[index]['box']*heatmap_h) # y1,x1,y2,x2
      # gts = gts[:, :4] / down
      for idx, flag in enumerate(self.data[index]['mask']):
        if flag:
          ct = (np.array([gts[idx, 1],gts[idx, 0]]) + np.array([gts[idx, 3],gts[idx, 2]]) )/ 2
          ct_int = ct.astype(np.int32)
          assert 0 <= ct_int[0] < heatmap_w and 0 <= ct_int[1] < heatmap_h

          h, w = gts[idx, 2] - gts[idx, 0], gts[idx, 3] - gts[idx, 1]
          hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
          hp_radius = max(0, int(hp_radius))
          draw_umich_gaussian(hm[0], ct_int, hp_radius)
      ret.update({'hm': hm})
      # lms_img = draw_lms(img, ret['lms21'].reshape(ret['lms21'].shape[0],-1,2))
      # cv2.imwrite('local_lms_mixed_img{}.jpg'.format(index), lms_img)
      # save_hm = np.squeeze(hm[0])*255
      # cv2.imwrite('local_lms_mixed_mask{}.jpg'.format(index), mask*255)
      # gt_hm_h = self.data[index]['ind'][0]//(self.opt.input_res/8)
      # gt_hm_w = self.data[index]['ind'][0]%(self.opt.input_res/8)
      # cv2.circle(save_hm, (int(gt_hm_w), int(gt_hm_h)), 1, 200, 1)
      # cv2.imwrite('heatmap{}.png'.format(index), save_hm)

      return ret


  def augment_append(self, new_data, img_data, x_img):
    image = cv2.imread(os.path.join(self.opt.pre_fix, new_data['filepath']))
    mask = cv2.imread(os.path.join(self.opt.pre_fix, new_data['maskpath']))[:, :, 2] \
      if 'maskpath' in new_data else None # and self.opt.photometric_loss 
    img_height, img_width = image.shape[:2]
    img = cv2.flip(image, 1) if new_data['handness'][0] == 0 else image.copy()
    mask = cv2.flip(mask, 1) if new_data['handness'][0] == 0 else mask.copy()
    if mask is not None:
      mask_height, mask_width = mask.shape[:2]
      if (mask_height != img_height) or (mask_width != img_width):
        mask = cv2.resize(mask,(img_width,img_height))
    # vis
    if False:
      for id in range(len(new_data['lms21'])):
        cv2.circle(img, (int(new_data['lms21'][id,0]), int(new_data['lms21'][id,1])), 2, (255,0,0), 2)
      cv2.imwrite('local_lms_mixed_img.jpg', img)
      cv2.imwrite('local_mask_mix.png',mask)
    pad = 40
    img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
    mask = np.pad(mask, ((pad, pad), (pad, pad)), mode='constant') if mask is not None else mask
    # cv2.imwrite('append_lms_1.jpg', img)
    gts = np.copy(new_data['bboxes']).astype(np.float32)

    if 'lms21' in new_data:
      lms = np.copy(new_data['lms21']).astype(np.float32).reshape(1,-1,2)
    elif 'lms5' in new_data:
      lms = np.copy(new_data['lms5']).astype(np.float32)
    else:
      lms = None
    lms68 = np.copy(new_data['lms68']).astype(np.float32) if 'lms68' in new_data else None

    max_size = max(gts[0, 2:] - gts[0, :2] + 1)
    try:
      low_x = max(0,int(gts[0, 0]))
      low_y = max(0,int(gts[0, 1]))
      high_x = min(img.shape[1], int(gts[0, 2]+2*pad))
      high_y = min(img.shape[0], int(gts[0, 3]+2*pad))
      patch_img = img[low_y:high_y, low_x:high_x]
      patch_mask = mask[low_y:high_y, low_x:high_x] if mask is not None else mask
    except:
      print("where is wrong?!")
    # cv2.imwrite('append_lms_patch.jpg', patch_img)
    if not img_data['full']:
      if img_data['boarder_l'] > (self.min_size*1):
        # get random size between[80,120]
        size = self.min_size * np.exp(np.random.choice(np.arange(0, np.log(min(img_data['boarder_l'], self.max_size)/self.min_size), 0.01)))
        scale = size / (max_size + 2*pad)
        try:
          patch_img = cv2.resize(patch_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
          patch_mask = cv2.resize(patch_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) if patch_mask is not None else patch_mask
        except:
          print("where is wrong here?!")
        h, w, _ = patch_img.shape
        assert w*h > 0 and w<self.opt.size_train[0] and h<self.opt.size_train[0]
        new_r = min(img_data['boarder_l'] + 20, self.opt.size_train[0])
        # seperate two hands adj. set [40]
        if max(new_r - 40, w) < new_r + 1:
          new_r = int(np.random.choice(np.arange(max(new_r - 40, w), new_r+1, 1)))
          # put hand bottom at [192+h/2]
          new_b = int(np.random.choice(np.arange(min(self.opt.size_train[0] - 1, self.opt.size_train[0]//2 + h*1.5), self.opt.size_train[0], 1)))
          # print(new_r, new_b, w, h, img_data['boarder_l'], img_data['boarder_t'])
          try:
            # try remove background using mask
            bg = (patch_mask > 10).astype(np.uint8)
            patch_img = patch_img * bg.reshape(bg.shape[0],bg.shape[1],1) 
            orig_bg = x_img[new_b-h:new_b, new_r-w:new_r].copy() * bg.reshape(bg.shape[0],bg.shape[1],1)     
            x_img[new_b-h:new_b, new_r-w:new_r] = x_img[new_b-h:new_b, new_r-w:new_r] + patch_img.copy() - orig_bg.copy()           
          except:
            print("something wrong")
          if 'mask' in img_data:
            img_data['mask'][new_b-h:new_b, new_r-w:new_r] = patch_mask.copy() if patch_mask is not None else patch_mask
          # plt.imshow(x_img[..., ::-1])
          # plt.show()
          # cv2.imwrite('append_lms_x_img.jpg', x_img)

          corner1 = gts[0, :2].astype(np.int) - pad
          corner2 = np.array([new_r - w, new_b - h], dtype=np.int)
          gts = self.process(gts.reshape((gts.shape[0], -1, 2)), scale, corner1, corner2)
          gts = gts.reshape(-1, 4)
          # vis_box(x_img, gts)
          lms = self.process(lms, scale, corner1, corner2)
          if False:
            lms_img = draw_lms(x_img, lms)
            cv2.imwrite('local_lms_mixed_img.jpg', lms_img)
            cv2.imwrite('local_mask_mix.png',img_data['mask'])
          # maybe no need to camera space?
          # lms[..., 1] = self.opt.size_train[0] - lms[..., 1]
          # vis(x_img, lms68)
          img_data['boarder_l'] = min(gts[0][0] - 20, img_data['boarder_l'])
          img_data['boarder_t'] = min(gts[0][1] - 20, img_data['boarder_t'])
          img_data['bboxes'] = np.concatenate((img_data['bboxes'], gts), 0)
          # img_data['lms68'] = np.concatenate((img_data['lms68'], lms68), 0)
          img_data['lms21'] = np.concatenate((img_data['lms21'], lms), 0)
          img_data['handness'] = np.concatenate((img_data['handness'], new_data['handness']), 0)

          img_data['keep'] = np.concatenate((img_data['keep'], np.array([img_data['bboxes'].shape[0] - 1])), 0)
          # img_data['lms68_keep'] = np.concatenate((img_data['lms68_keep'], np.array([img_data['bboxes'].shape[0] - 1])), 0)
          # add mano_coeff
          if 'mano_coeff' in new_data:
            # if new_data['dataset']==1: # HO3D use flaten mano value is larger than FreiHAND.
            img_data['mano_coeff'] = np.concatenate((img_data['mano_coeff'], self.add_handmean(new_data['mano_coeff'])), 0)
          if 'K' in new_data:
            img_data['K'] = np.concatenate((img_data['K'], new_data['K']), 0)       
        # plt.imshow(x_img)
        # plt.show()
          # print('ok',new_data['filepath'])
      else:
        img_data['full'] = True
        img_data['boarder_l'] = self.opt.size_train[0]-20 # i want to the top hand not close to right.
        # img_data['boarder_t'] = min(self.opt.size_train[0]//2-50, img_data['boarder_t']) # i want to the top hand not close to middle.
        # print('skip',new_data['filepath'])
    else:
      if img_data['boarder_l'] > self.min_size and img_data['boarder_t'] > self.min_size:
        size = self.min_size * np.exp(np.random.choice(np.arange(0, np.log(min(img_data['boarder_l'], img_data['boarder_t'], self.max_size) / self.min_size), 0.01)))
        scale = size / (max_size + 2*pad)
        try:
          patch_img = cv2.resize(patch_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
          patch_mask = cv2.resize(patch_mask, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_NEAREST) if patch_mask is not None else patch_mask
        except:
          print("where is wrong here?!")
        h, w, _ = patch_img.shape
        assert w*h > 0 and w<self.opt.size_train[0] and h<self.opt.size_train[0]
        new_r = min(img_data['boarder_l'] + 20, self.opt.size_train[0])
        if max(new_r - 40, w) < new_r + 1:
          new_r = int(np.random.choice(np.arange(max(new_r - 40, w), new_r+1, 1)))
          new_b = int(np.random.choice(np.arange(h, max(h, img_data['boarder_t']) + 1, 1)))
          # print(new_r, new_b, w, h, img_data['boarder_l'], img_data['boarder_t'])
          try:
            # try remove background using mask
            bg = (patch_mask > 10).astype(np.uint8)
            patch_img = patch_img * bg.reshape(bg.shape[0],bg.shape[1],1) 
            orig_bg = x_img[new_b-h:new_b, new_r-w:new_r].copy() * bg.reshape(bg.shape[0],bg.shape[1],1)     
            x_img[new_b-h:new_b, new_r-w:new_r] = x_img[new_b-h:new_b, new_r-w:new_r] + patch_img.copy() - orig_bg.copy()           
          except:
            print("something wrong")
          if 'mask' in img_data:
            img_data['mask'][new_b - h:new_b, new_r - w:new_r] = patch_mask.copy() if patch_mask is not None else patch_mask
          # plt.imshow(x_img[..., ::-1])
          # plt.show()

          corner1 = gts[0, :2].astype(np.int) - pad
          corner2 = np.array([new_r - w, new_b - h], dtype=np.int)
          gts = self.process(gts.reshape((gts.shape[0], -1, 2)), scale, corner1, corner2)
          gts = gts.reshape(-1, 4)
          # vis_box(x_img, gts)
          lms = self.process(lms, scale, corner1, corner2)
          if False:
            lms_img = draw_lms(x_img, lms)
            cv2.imwrite('local_lms_mixed_img.jpg', lms_img)
            cv2.imwrite('local_mask_mix.png',img_data['mask'])
          # lms[..., 1] = self.opt.size_train[0] - lms[..., 1]
          # vis(x_img, lms68)
          img_data['boarder_l'] = min(gts[0][0] - 20, img_data['boarder_l'])
          img_data['bboxes'] = np.concatenate((img_data['bboxes'], gts), 0)
          # img_data['lms68'] = np.concatenate((img_data['lms68'], lms68), 0)
          img_data['lms21'] = np.concatenate((img_data['lms21'], lms), 0)
          img_data['keep'] = np.concatenate((img_data['keep'], np.array([img_data['bboxes'].shape[0] - 1])), 0)
          img_data['handness'] = np.concatenate((img_data['handness'], new_data['handness']), 0)
              # add mano_coeff
          if 'mano_coeff' in new_data:
            img_data['mano_coeff'] = np.concatenate((img_data['mano_coeff'], self.add_handmean(new_data['mano_coeff'])), 0)
          if 'K' in new_data:
            img_data['K'] = np.concatenate((img_data['K'], new_data['K']), 0)
        # print('ok',new_data['filepath'])
          # img_data['lms68_keep'] = np.concatenate((img_data['lms68_keep'], np.array([img_data['bboxes'].shape[0] - 1])), 0)
      else:
        pass
        # print('skip',new_data['filepath'])
    # img_data['lms68'][..., 1] = self.opt.size_train[0] - img_data['lms68'][..., 1]
    # # add mano_coeff
    # if 'mano_coeff' in new_data:
    #   img_data['mano_coeff'] = np.concatenate((img_data['mano_coeff'], new_data['mano_coeff']), 0)
    return img_data, x_img

  def process(self, array, scale, corner1, corner2):
    return scale * (array - corner1[np.newaxis, np.newaxis, :]) + corner2[np.newaxis, np.newaxis, :]

  def add_background(self, img_data):
    img_data_aug = copy.deepcopy(img_data)
    idx = np.random.randint(0,7) 
    bg_path = os.path.join(self.opt.root_dir, 'assets/bg/bg_{}.jpeg'.format(idx))
    image = cv2.imread(bg_path)
    h,w = image.shape[:2]
    s = min(h,w)
    c = np.array([w / 2., h / 2.], dtype=np.float32)    
    rot = np.random.randint(low=-12, high=12)   
    trans_input,inv_trans = get_affine_transform(c, s, rot, [self.opt.size_train[0], self.opt.size_train[1]])
    img = cv2.warpAffine(image, trans_input,
                         (self.opt.size_train[0], self.opt.size_train[1]),
                         flags=cv2.INTER_LINEAR)    
    mask = img[:,:,0]*0
    # img_data_aug['bboxes'] = new_gts
    img_data_aug['boarder_l'] = self.opt.size_train[0]-50 # not too close to right side
    img_data_aug['boarder_t'] = self.opt.size_train[1] * 3 / 4
    img_data_aug['full'] = False
    if 'lms21' in img_data_aug:
      img_data_aug['lms21'] = img_data_aug['lms21'].reshape(1,-1,2)
    # if mask is not None:
    #   img_data_aug['mask'] = mask
    img_data_aug['keep'] = np.array(range(1))
    img_data_aug.update({'handness':[0]})
    img_data_aug.update({'mask':mask.reshape(self.opt.size_train[0], self.opt.size_train[1])})

    return img_data_aug, img

  def augment_centernet(self, img_data):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    img_data_aug = copy.deepcopy(img_data)
    image = cv2.imread(os.path.join(self.opt.pre_fix, img_data_aug['filepath']))
    mask = cv2.imread(os.path.join(self.opt.pre_fix, img_data_aug['maskpath']))[:, :, 2] \
      if 'maskpath' in img_data_aug else None # and self.opt.photometric_loss
    img_height, img_width = image.shape[:2]
    img = image.copy()

    if 'handness' in img_data_aug:
      img = cv2.flip(image, 1) if img_data_aug['handness'][0] == 0 else image.copy()
      mask = cv2.flip(mask, 1) if img_data_aug['handness'][0] == 0 else mask.copy()
    if mask is not None:
      mask_height, mask_width = mask.shape[:2]
      if (mask_height != img_height) or (mask_width != img_width):
        mask = cv2.resize(mask,(img_width,img_height))

    c = np.array([img_width / 2., img_height / 2.], dtype=np.float32)
    s = max(img_height, img_width) * 1.
    gts = np.copy(img_data_aug['bboxes']).astype(np.float32)
    if 'lms21' in img_data_aug:
      lms = np.copy(img_data_aug['lms21']).astype(np.float32).reshape(1,-1,2)
    else:
      lms = None        

    # restrict the face in [self.min_size 96, self.max_size 320]
    max_size = max(gts[0, 2:] - gts[0, :2] + 1)
    center = (gts[0, 2:] + gts[0, :2]) / 2
    if self.opt.no_det:
      c[0], c[1] = center[0], center[1]
      min_scale, max_scale = max_size / 0.6 / s, max_size / 0.3 / s
      s = s * np.random.choice(np.arange(min_scale, max_scale, 0.02))
      if True: # add nosie
        center_noise = 5
        c[0] = np.random.randint(low=int(center[0] - center_noise), high=int(center[0] + center_noise))
        c[1] = np.random.randint(low=int(center[1] - center_noise), high=int(center[1] + center_noise))    

    else:
      # restrict to [min,max], max is of 224*0.6~512*0.6
      min_scale, max_scale = self.opt.size_train[0] * max_size / self.max_size / s, self.opt.size_train[0] * max_size / self.min_size / s
      s = s * np.random.choice(np.arange(min_scale, max_scale, 0.05))
      s = self.opt.size_train[0]
      x1 = int(center[0] - (s - max_size) // 2)
      x2 = int(max(center[0] - max_size // 4, x1))
      y1 = int(center[1] - (s - max_size) // 2)
      y2 = int(max(center[1] - max_size // 8, y1))
      c[0] = np.random.randint(low=x1, high=x2+1)
      c[1] = np.random.randint(low=y1, high=y2+1)

    rot = np.random.randint(low=-60, high=60) if self.opt.input_res == 224 else 0
    if False: # use original img
      s = 224
      c = np.array([img_width / 2., img_height / 2.], dtype=np.float32)    
      rot = 0 
    trans_input,inv_trans = get_affine_transform(c, s, rot, [self.opt.size_train[0], self.opt.size_train[1]])
    img = cv2.warpAffine(img, trans_input,
                         (self.opt.size_train[0], self.opt.size_train[1]),
                         flags=cv2.INTER_LINEAR)
    if mask is not None:
      mask = cv2.warpAffine(mask, trans_input, (self.opt.size_train[0], self.opt.size_train[1]),
                            flags=cv2.INTER_NEAREST)

    # try remove background using mask
    # cv2.imwrite('append_lms_0.jpg', img)
    #!!!try to see if this matters?!!!
    blank = (img < 5).astype(np.uint8) # filling the blank
    orig_img = cv2.resize(image,(self.opt.size_train[0],self.opt.size_train[1]))
    orig_bg = orig_img * blank *0.25     
    img = (orig_bg + img.copy()).astype(np.uint8) 
    # cv2.imwrite('append_lms_1.jpg', img)  

    keep = np.array(range(len(gts)))
    if len(gts) > 0:
      # transform lms first
      if lms is not None:
        for i in range(lms.shape[1]):
          if lms[:, i, :2].min()==-1 or lms[:, i, :2].min() == 0:
            # print("this is a null lm")
            continue
          # cv2.circle(img_old, (lms[:, i, 0],lms[:, i, 1]), 3, (0, 0, 255), 2)
          lms[:, i, :2] = affine_transform_array(lms[:, i, :2], trans_input)
          # cv2.circle(img, (lms[:, i, 0],lms[:, i, 1]), 3, (0, 255, 0), 2)

      # calculate gts from new_lms, this may be wrong when multi hands occur...
      new_gts = lms2bbox(self,lms)
      if len(new_gts) > 0:
        w, h = new_gts[:, 2] - new_gts[:, 0], new_gts[:, 3] - new_gts[:, 1]
        tags = np.logical_or(w >= 16, h >= 16)
        keep = keep[tags]
      else:
        print('no gts found!')
    img_data_aug['bboxes'] = new_gts
    img_data_aug['boarder_l'] = new_gts[0, 0]
    img_data_aug['boarder_t'] = new_gts[0, 1]
    img_data_aug['full'] = False
    if 'lms21' in img_data_aug:
      img_data_aug['lms21'] = lms
    if mask is not None:
      img_data_aug['mask'] = mask
    img_data_aug['keep'] = keep

    return img_data_aug, img 

  def calc_gt(self, img_data):
    self.max_objs = 4 # we set 5 to address 3rd hand skip.
    gts = np.copy(img_data['bboxes'][1:])
    mano_coeff_gt = np.copy(img_data['mano_coeff'][1:])
    K_gt = np.copy(img_data['K'].reshape(-1,3,3)[1:])
    gt_lms21 = np.copy(img_data['lms21'][1:]) if 'lms21' in img_data else None
    down = self.opt.down_ratio
    heatmap_h, heatmap_w = self.opt.size_train[1] // down, self.opt.size_train[0] // down
    hm = np.zeros((self.num_classes, heatmap_h, heatmap_w), dtype=np.float32)
    heatmaps = np.zeros((21, self.opt.size_train[1]//2, self.opt.size_train[0]//2), dtype=np.float32)
    if gt_lms21 is not None:
      ind = np.zeros((self.max_objs,), dtype=np.int64)
      reg_mask = np.zeros((self.max_objs,), dtype=np.bool)
      n =  21
      lms21 = np.zeros((self.max_objs, n * 2), dtype=np.float32)
      box = np.zeros((self.max_objs, 4), dtype=np.float32) # for perceptual loss
      handness = np.zeros((self.max_objs, ), dtype=np.int64) + 5 # 0 for left/ 1 for right
      handmap = np.zeros((2, heatmap_h, heatmap_w), dtype=np.float32) # two classes for left/right hands type.
      mano_coeff = np.zeros((self.max_objs, 61), dtype=np.float32)
      K = np.zeros((self.max_objs, 3,3), dtype=np.float32)
         
    if len(gts) > 0:
      gts = gts[:, :4] / down
      if gt_lms21 is not None:
        lms21_down = gt_lms21 / down        
      idx_ = 0
      for k, idx in enumerate(img_data['keep'][:-1]):
        # use mean position as default.
        if self.opt.avg_center:
          idxx = np.where((lms21_down[idx,:,0] >0)&(lms21_down[idx,:,1] >0))
          ct = np.array([lms21_down[idx,idxx,0].mean(),lms21_down[idx,idxx,1].mean()])
        else:
          ct = np.array(gts[idx, 0:2] + gts[idx, 2:4]) / 2
        ct_int = ct.astype(np.int32)
        assert 0 <= ct_int[0] < heatmap_w and 0 <= ct_int[1] < heatmap_h
        w, h = gts[idx, 2] - gts[idx, 0], gts[idx, 3] - gts[idx, 1]
        hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        hp_radius = max(0, int(hp_radius))
        draw_umich_gaussian(hm[0], ct_int, hp_radius)
        
        if gt_lms21 is not None and idx_ < self.max_objs:
          ind[idx_] = heatmap_w * ct_int[1] + ct_int[0]
          handness[idx_] = img_data['handness'][1:][idx_]
          draw_umich_gaussian(handmap[img_data['handness'][1:][idx_]], ct_int, hp_radius)
          for i in range(gt_lms21.shape[1]):
            draw_umich_gaussian(heatmaps[i], (gt_lms21[0][i]/2).astype(np.int32), hp_radius)
            
          reg_mask[idx_] = 1
          lms21[idx_] = gt_lms21[idx].reshape(-1)
          mano_coeff[idx_] = mano_coeff_gt[idx]
          K[idx_] = K_gt[idx]
          box[idx_] = gts[idx, 1] / heatmap_h, gts[idx, 0] / heatmap_w, gts[idx, 3] / heatmap_h, gts[idx, 2] / heatmap_w
          idx_ += 1     
        else:
          print('this is not right')

    ret = {'hm': hm}
    if self.opt.heatmaps:
      ret.update({'heatmaps': heatmaps})
    if gt_lms21 is not None:
      ret.update({'valid': reg_mask, 'ind': ind})
      ret.update({'lms21': lms21})
      ret.update({'lms21_full': copy.deepcopy(lms21)})
      ret.update({'box': box}) # used for perceptual loss   
      ret.update({'handness': handness}) 
      ret.update({'handmap': handmap})
      ret.update({'mano_coeff': mano_coeff})
      ret.update({'K': K})

    return ret