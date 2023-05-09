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
from models.losses import RegL1Loss, RegWeightedL1Loss, NormLoss, NormLossLeft
from models.utils import _sigmoid, _tranpose_and_gather_feat
from .base_trainer import BaseTrainer
from utils.utils import drawCirclev2
import copy
import cv2
from scipy.optimize import minimize
from lib.models.networks.manolayer import ManoLayer

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
      self.crit_normleft = NormLossLeft()
    if opt.off:
      self.crit_lms = RegWeightedL1Loss()
    self.render = render
    self.facenet = facenet
    self.mano_path = {'left': self.render.lhm_path,
                'right': self.render.rhm_path}           
    self.mano_layer = {'right': ManoLayer(self.mano_path['right'], center_idx=None, use_pca=True),
                        'left': ManoLayer(self.mano_path['left'], center_idx=None, use_pca=True)}
    
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
  
  def projection_batch(self, scale, trans2d, label3d, img_size=256):
    """orthodox projection
    Input:
        scale: (B)
        trans2d: (B, 2)
        label3d: (B x N x 3)
    Returns:
        (B, N, 2)
    """
    scale = (scale + 3)* img_size  # init to 684, 0.1m to 65pixel
    if scale.dim() == 1:
        scale = scale.unsqueeze(-1).unsqueeze(-1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(-1)
    trans2d = trans2d * img_size / 2 + img_size / 2  # bs x 2
    trans2d = trans2d.unsqueeze(1)

    label2d = scale * label3d[..., :2] + trans2d
    return label2d    
  
  def batch_orth_proj(self, X, camera, mode='2d',keep_dim=False):
    camera = camera.view(-1, 1, 3)
    X_camed = X[:,:,:2] * camera[:, :, 0].unsqueeze(-1)
    X_camed += camera[:, :, 1:]
    if keep_dim:
        X_camed = torch.cat([X_camed, X[:,:,2].unsqueeze(-1)],-1)
    return X_camed

  def forward(self, outputs, mode, batch):
    opt = self.opt
    hm_loss, heatmaps_loss = 0, 0
    pick_loss, pose_loss = 0,0
    if opt.reproj_loss:
      reproj_loss, norm_loss = 0, 0
      reproj_loss_all = 0
    if opt.bone_loss:
      bone_loss, bone_direc_loss = 0, 0
    if opt.photometric_loss:
      norm_loss, var_loss = 0, 0
      photometric_loss, seg_loss = 0, 0
    if opt.perceptual_loss:
      perceptual_loss = 0
    if opt.gcn_decoder:
      S_loss, gcn_reproj_loss = 0, 0
    if opt.off:
      off_hm_loss, off_lms_loss, wh_loss = 0, 0, 0
    if opt.discrepancy:
      discrepancy_loss = 0
    loss = 0

    for s in range(opt.num_stacks):
      output = outputs[s]
      output['hm'] = _sigmoid(output['hm'])
      handness = None

      if mode == 'val' or mode == 'test':
        hms = output['hm'].clone().detach()
        score = 0.5
        hms = _nms(hms, 5)
        # K = int((hms[0] > score).float().sum())
        K = 4 if self.opt.default_resolution == 384 else 1
        topk_scores, pred_ind, topk_ys, topk_xs = _topk(hms[:,:1,:,:], K)  
      ind_pred = pred_ind if (mode == 'val' or mode == 'test') and self.opt.default_resolution == 224  else batch['ind']
      if opt.pick_hand:
        output['handmap'] = _sigmoid(output['handmap'])
        hand_label_pred = _tranpose_and_gather_feat(output['handmap'], ind_pred)
        pred_handness = torch.max(hand_label_pred,2)[1]
        handness = pred_handness if mode == 'val' or mode == 'test' else batch['handness']
        pick_loss = pick_loss + self.crit(output['handmap'], batch['handmap']) / opt.num_stacks if 'handmap' in batch else 0


      hm_loss = hm_loss + self.crit(output['hm'], batch['hm']) / opt.num_stacks if 'hm' in batch else 0
      if opt.heatmaps:
        device = batch['lms21'].device
        heatmaps_loss = heatmaps_loss + self.crit(output['uv_prior'], batch['heatmaps']) / opt.num_stacks / output['uv_prior'].shape[1] / 16
        heatmaps = output['uv_prior']
        heatmaps = _nms(heatmaps, 5)
        K = batch['ind'].shape[1]
        # first estimate wrist_x_y and cat others
        topk_scores, topk_inds, topk_ys, topk_xs = _topk(heatmaps[:,0:1,:,:], K)    
        joint_coord = torch.stack((topk_xs,topk_ys),dim=2)        
        for k in range(20):
          topk_scores, topk_inds, topk_ys, topk_xs = _topk(heatmaps[:,k+1:k+2,:,:], K)    
          tmp_coord = torch.stack((topk_xs,topk_ys),dim=2)
          joint_coord = torch.cat((joint_coord,tmp_coord),2)
        joint_coord = joint_coord*2 # transform to 224x224   
        # heatmaps_loss = heatmaps_loss + torch.abs(joint_coord-batch['lms21']).mean(dim=(1,2))

      ret_rendered, ret_gpu_masks,verts_all_gt,joints_all_gt = None, None,None, None
      file_id = batch['file_id'].detach().cpu().numpy().astype(np.int)[0]
      if not opt.center_only:
        if opt.off:
          ## off_hm_loss
          off_hm_loss += self.crit_lms(output['hm_off'], batch['mask'],
                                    batch['ind'], batch['hm_off']) / opt.num_stacks
          ## off_lms_loss                          
          off_lms_loss += self.crit_lms(output['lms21_off'], batch['mask'],
                                    batch['ind'], batch['lms21_off']) / opt.num_stacks

        if opt.reproj_loss:
          params = _tranpose_and_gather_feat(output['params'], ind_pred)
          B, C = params.size(0), params.size(1)
          global_orient_coeff_l_up, pose_coeff_l, betas_coeff_l, global_transl_coeff_l_up, \
            global_orient_coeff_r_up, pose_coeff_r, betas_coeff_r, global_transl_coeff_r_up  = \
            self.render.Split_coeff(params.view(-1, params.size(2)),ind_pred.view(-1))
          hand_verts_pred_l, hand_joints_pred_l = self.render.mano_layer_left(global_orient_coeff_l_up, pose_coeff_l, betas_coeff_l, global_transl_coeff_l_up, side ='left')
          hand_verts_pred_r, hand_joints_pred_r = self.render.mano_layer_right(global_orient_coeff_r_up, pose_coeff_r, betas_coeff_r,global_transl_coeff_r_up, side ='right')    
          hand_verts_r_root, hand_joints_r_root = self.render.mano_layer_right(global_orient_coeff_r_up*0, pose_coeff_r, betas_coeff_r, side ='right')                 
          hand_verts_l_root, hand_joints_l_root = self.render.mano_layer_left(global_orient_coeff_l_up*0, pose_coeff_l, betas_coeff_l, side ='left')
          verts_all_pred = torch.stack((hand_verts_pred_l,hand_verts_pred_r),dim=1)
          joints_all_pred = torch.stack((hand_joints_pred_l,hand_joints_pred_r),dim=1)
          if 'mano_coeff' in batch:
            gt_orient = batch['mano_coeff'][:,:,:3].reshape(-1,3)
            gt_pose = batch['mano_coeff'][:,:,3:48].reshape(-1,45)
            gt_shape = batch['mano_coeff'][:,:,48:58].reshape(-1,10)
            gt_uv_root = batch['mano_coeff'][:,:,-3:].reshape(-1,3)
            xyz_root = self.render.get_uv_root_3d(gt_uv_root,batch['K'])
            # gt_trans = gt_uv_root if gt_uv_root[0,2] < 0 else xyz_root # FreiHAND provide uv_root, HO3D provide xyz_root.

            # hand_verts_gt1, hand_joints_gt, full_pose_gt = self.render.Shape_formation(gt_orient, gt_pose, gt_shape, xyz_root,'gt',True)    
            hand_verts_gt, hand_joints_gt = self.render.mano_layer_right(gt_orient*0, gt_pose, gt_shape, side ='right')
  
            #prepare left gt
            hand_verts_r_gt = hand_verts_gt.clone()#.reshape(B,-1,778,3) 
            hand_joints_r_gt = hand_joints_gt.clone()#.reshape(B,-1,21,3)          
            hand_verts_l_gt = hand_verts_gt.clone()
            hand_joints_l_gt = hand_joints_gt.clone()
            hand_verts_l_gt[...,0] = hand_verts_l_gt[...,0]*-1
            hand_joints_l_gt[...,0] = hand_joints_l_gt[...,0]*-1

          norm_loss = (norm_loss + self.crit_normleft(pose_coeff_l, betas_coeff_l) + \
                        self.crit_norm(pose_coeff_r, betas_coeff_r)).reshape(B,-1).mean(dim=1)
          # address the left flip case:
          if opt.pick_hand:
            idx_verts = handness.reshape(-1,1).unsqueeze(2).expand_as(hand_verts_pred_r)
            verts_all = torch.where(idx_verts==0,hand_verts_pred_l,hand_verts_pred_r)
            idx_joints = handness.reshape(-1,1).unsqueeze(2).expand_as(hand_joints_pred_r)
            joints_all = torch.where(idx_joints==0,hand_joints_pred_l,hand_joints_pred_r)  
            if 'mano_coeff' in batch:
              #prepare gt  
              verts_all_gt = torch.where(idx_verts==0,hand_verts_l_gt,hand_verts_r_gt)
              joints_all_gt = torch.where(idx_joints==0,hand_joints_l_gt,hand_joints_r_gt)  
              verts_pred_root = torch.where(idx_verts==0,hand_verts_l_root,hand_verts_r_root)
              joints_pred_root = torch.where(idx_joints==0,hand_joints_l_root, hand_joints_r_root)  
            ### train artificial case with lms21 only. using the same K.
            if self.opt.default_resolution == 384: 
              init_index_x, init_index_y = (ind_pred%(384//4)*4-192)/192. , (ind_pred//(384//4)*4-192)/192.
              # global_transl_coeff_r_up[:,0] = global_transl_coeff_r_up[:,0] + init_index_x.view(-1)
              # global_transl_coeff_r_up[:,1] = global_transl_coeff_r_up[:,1] + init_index_y.view(-1)
              lms21_pred = self.render.get_Landmarks(joints_all) 
              lms21_gt = batch['lms21'].reshape(-1,21,2)
              reproj_loss_all = ((F.mse_loss(lms21_pred, lms21_gt, reduction='none')) / opt.num_stacks).reshape(B,-1).mean(dim=1) 
              reproj_loss_all = reproj_loss_all * 0.01
              
              if opt.bone_loss:
                # tmp_mask = mask.reshape(-1,21,2)
                j2d_con = torch.ones_like(lms21_gt[:,:,0]).unsqueeze(-1)
                # maybe confidence can be used here.
                bone_direc_loss = bone_direction_loss(lms21_pred, lms21_gt, j2d_con).reshape(B,-1).mean(dim=1)

              if 'mano_coeff' in batch:
                ### gt
                root_gt = joints_all_gt[:, 9:10]
                length_gt = torch.norm(joints_all_gt[:, 9] - joints_all_gt[:, 0], dim=-1)
                joints_all_gt_off = joints_all_gt - root_gt
                verts_all_gt_off = verts_all_gt - root_gt
                #### pred
                root_pred = joints_pred_root[:, 9:10]
                length_pred = torch.norm(joints_pred_root[:, 9] - joints_pred_root[:, 0], dim=-1)
                scale_right = (length_gt / length_pred).unsqueeze(-1).unsqueeze(-1)
                joints_all_off = (joints_pred_root - root_pred) * scale_right
                verts_all_off = (verts_pred_root - root_pred) * scale_right  

                pose_loss = (self.l1_loss(joints_all_off, joints_all_gt_off)).reshape(B,-1).mean(dim=1) *1000
                verts_loss = (self.l1_loss(verts_all_off, verts_all_gt_off)).reshape(B,-1).mean(dim=1) *1000    
                # mano_loss = rotate_loss + manopose_loss
                # shape_loss = (F.mse_loss(gt_shape, betas_coeff_r, reduction='none')).reshape(B,-1).mean(dim=1)     
              if opt.photometric_loss and mode == 'train':
                pre_textures = _tranpose_and_gather_feat(output['texture'], ind_pred)
                device = pre_textures.device
                self.render.gamma = _tranpose_and_gather_feat(output['light'], ind_pred)
                Albedo = pre_textures.view(-1,778,3) #[b, 778, 3]
                texture_mean = torch.tensor([0.45,0.35,0.3]).float().to(device)
                texture_mean = texture_mean.unsqueeze(0).unsqueeze(0).repeat(Albedo.shape[0],Albedo.shape[1],1)#[1, 778, 3]
                Albedo = Albedo + texture_mean
                Texture, lighting = self.render.Illumination(Albedo, verts_all.view(-1,778,3))

                rotShape = verts_all.view(-1, C, 778, 3)
                Texture = Texture.view(-1, C, 778, 3)
                nV = rotShape.size(2)
                Verts, Faces, Textures = [], [], []
                valid = []
                tag_drop = False
                tmp_mask = batch['valid']
                for i in range(len(rotShape)):
                  # detach vertex to avoid influence of color
                  V_ = rotShape[i][tmp_mask[i]]#.detach()
                  if V_.size(0) == 0:
                    # to simplify the code, drop the whole batch with any zero_hand image.
                    Verts = []
                    break
                  valid.append(i)
                  T_ = Texture[i][tmp_mask[i]]
                  # F_ = None
                  # for kk in range(handness.shape[1]):
                  #   if tmp_mask[i,kk]==False:
                  #     continue
                  #   if handness[i,kk]==0:
                  #     F_tmp = torch.from_numpy(self.render.MANO_L.faces.astype(np.int64)).to(device).view(1,-1)
                  #   else:
                  #     F_tmp = torch.from_numpy(self.render.MANO_R.faces.astype(np.int64)).to(device).view(1,-1)
                  #   if F_ is None:
                  #     F_ = torch.from_numpy(self.render.MANO_R.faces.astype(np.int64)).to(device).view(1,-1)
                  #   else:
                  #     F_ = torch.cat((F_,F_tmp),dim=0)
                  F_ = torch.from_numpy(self.render.MANO_R.faces.astype(np.int64)).to(device).view(1,-1)
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
                  rendered, gpu_masks, depth = self.render(meshes)
                  rendered = torch.flip(rendered,[1])
                  gpu_masks = torch.flip(gpu_masks,[1])
                  depth = torch.flip(depth,[1]) 
                  gpu_masks = gpu_masks.detach().float()
                  ret_rendered = rendered[-B:]
                  ret_gpu_masks = gpu_masks[-B:]
                  skin_mask = batch['mask']
                  crit = nn.L1Loss()
                  tmp_seg_loss = crit(gpu_masks, skin_mask[valid])        
                  seg_loss += tmp_seg_loss
                  masks = gpu_masks * skin_mask[valid]
                  photometric_loss += (torch.norm(rendered - batch['render'][valid], p=2, dim=3) * masks).sum() / (masks.sum() + 1e-4) / opt.num_stacks 
                else:
                  tag_drop = True
                  photometric_loss += torch.zeros(1).sum().cuda() / opt.num_stacks 

              # show
              if file_id % 205 == 0:
                save_img_0 = (np.squeeze(batch['image'][0])).detach().cpu().numpy().astype(np.float32)
                if opt.reproj_loss:
                  vis_lms = lms21_pred.reshape(B,-1,21,2)  
                  for k in range(batch['valid'].shape[1]):
                      if batch['valid'][0][k]:
                        drawCirclev2(save_img_0,lms21_gt.reshape(B,-1,21,2)[0][k],(0,0,255),1)
                        drawCirclev2(save_img_0,vis_lms[0][k],(0,255,0),1)  
                        showHandJoints(save_img_0,vis_lms[0][k].detach().cpu().numpy(),'outputs/imgs/kps_bone_pred_{}_{}.jpg'.format(file_id,k))
                  cv2.imwrite('outputs/imgs/image_proj_0_{}.jpg'.format(file_id), save_img_0)
                if opt.photometric_loss and mode == 'train':
                  render_img = rendered[0].detach().cpu().numpy()[:, :, ::-1].astype(np.float32)
                  render_msk = gpu_masks[0].detach().cpu().numpy()
                  cv2.imwrite('outputs/imgs/fitted_{}_{}.jpg'.format(reproj_loss_all,file_id), render_img*255+save_img_0*((1-render_msk).reshape(save_img_0.shape[0],save_img_0.shape[1],1)))
                  cv2.imwrite('outputs/imgs/gpu_masks_{}.png'.format(file_id), render_msk*255)
                  cv2.imwrite('outputs/imgs/rendered_{}.jpg'.format(file_id), render_img*255)
                # # for rendering .obj
                Faces_l = self.render.MANO_L.faces.astype(np.int32)
                Faces_r = self.render.MANO_R.faces.astype(np.int32)
                k = 0 # view k-st mesh.
                vis_verts = verts_all.reshape(-1,778,3)[k].detach().cpu().numpy()
                if opt.photometric_loss and mode == 'train':
                  colors = Textures[0].detach().cpu().numpy()
                  with open('outputs/models/colord_rhands_{}.obj'.format(file_id), 'w') as f:
                    for idx in range(len(vis_verts)):
                      f.write('v %f %f %f %f %f %f\n'%(vis_verts[idx][0],vis_verts[idx][1],vis_verts[idx][2],colors[idx][0],colors[idx][1],colors[idx][2]))
                    for face in Faces_r+1:
                      f.write('f %f %f %f\n'%(face[0],face[1],face[2]))     
                if 'handness' in batch and batch['handness'][0][k]==0: # left
                  with open('outputs/models/lhands_{}.obj'.format(file_id), 'w') as f:
                    for v in vis_verts:
                      f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
                    for face in Faces_l+1:
                      f.write('f %f %f %f\n'%(face[0],face[1],face[2])) 
                  if verts_all_gt is not None:
                    with open('outputs/models/gt_hands_{}.obj'.format(file_id), 'w') as f:
                      for v in verts_all_gt.reshape(-1,778,3)[k]:
                        f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
                      for face in Faces_l+1:
                        f.write('f %f %f %f\n'%(face[0],face[1],face[2]))           
                else:
                  with open('outputs/models/rhands_{}.obj'.format(file_id), 'w') as f:
                    for v in vis_verts:
                      f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
                    for face in Faces_r+1:
                      f.write('f %f %f %f\n'%(face[0],face[1],face[2]))   
                  if verts_all_gt is not None:    
                    with open('outputs/models/gt_hands_{}.obj'.format(file_id), 'w') as f:
                      for v in verts_all_gt.reshape(-1,778,3)[k]:
                        f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
                      for face in Faces_r+1:
                        f.write('f %f %f %f\n'%(face[0],face[1],face[2]))     
              if mode == 'val' or mode == 'test':
                return verts_all, joints_all, verts_all_gt, joints_all_gt, lms21_pred,handness
              else:
                loss_stats = {}
                loss += opt.center_weight * hm_loss
                loss_stats.update({'hm_loss': hm_loss})
                loss += opt.center_weight * pick_loss
                loss_stats.update({'pick_loss': pick_loss})
                loss += opt.reproj_weight * reproj_loss_all
                loss_stats.update({'reproj_loss_all': reproj_loss_all})
                loss += opt.norm_weight * norm_loss
                loss_stats.update({'norm_loss': norm_loss})
                loss += opt.bone_dir_weight * bone_direc_loss
                loss_stats.update({'bone_direc_loss': bone_direc_loss})  
                loss += opt.pose_weight * pose_loss 
                loss_stats.update({'pose_loss': pose_loss})
                loss += opt.pose_weight * verts_loss 
                loss_stats.update({'verts_loss': verts_loss})
                if opt.photometric_loss:
                  loss += opt.photometric_weight * photometric_loss
                  loss_stats.update({'photometric_loss': photometric_loss})
                  loss += opt.seg_weight * seg_loss
                  loss_stats.update({'seg_loss': seg_loss})    
                loss_stats.update({'loss': loss})            
                return loss, loss_stats, ret_rendered, ret_gpu_masks     
                        
          else:
            joints_all = hand_joints_pred_r
            verts_all = hand_verts_pred_r
            if 'mano_coeff' in batch:
              verts_all_gt = hand_verts_r_gt
              joints_all_gt = hand_joints_r_gt 
              joints_pred_root = hand_joints_r_root
              verts_pred_root = hand_verts_r_root
          if 'mano_coeff' in batch:
          # mano_loss
          #   rotate_loss = (F.mse_loss(gt_orient, global_orient_coeff_r_up, reduction='none')).reshape(B,-1).mean(dim=1)  
          #   trans_loss = (F.mse_loss(root_gt, global_transl_coeff_r_up, reduction='none')).reshape(B,-1).mean(dim=1)  
            manopose_loss = (F.mse_loss(gt_pose, pose_coeff_r, reduction='none')).reshape(B,-1).mean(dim=1)  
          #   ### shape_loss
            shape_loss = (F.mse_loss(gt_shape, betas_coeff_r, reduction='none')).reshape(B,-1).mean(dim=1)

          #   # parameters for norm only, which is define from flat_hand_mean other than hand_mean.
          #   norm_loss = norm_loss + self.crit_norm(full_pose_r[:,3:], betas_coeff_r) / opt.num_stacks

          # lms21_pred = self.render.get_Landmarks(joints_all)  
          lms_pred = self.render.get_Landmarks(joints_all)  
          if 'lms' in batch: 
            lms21_gt = batch['lms'].reshape(-1,21,2)
            ### lms_loss
            reproj_loss_all = (F.mse_loss(lms_pred, lms21_gt, reduction='none').reshape(B,-1).mean(dim=1)+ 1e-8).reshape(B,-1).mean(dim=1)   

            if opt.bone_loss:
              # tmp_mask = mask.reshape(-1,21,2)
              j2d_con = torch.ones_like(lms21_gt[:,:,0]).unsqueeze(-1)
              # maybe confidence can be used here.
              bone_direc_loss = bone_direction_loss(lms_pred, lms21_gt, j2d_con).reshape(B,-1).mean(dim=1)

            if 'mano_coeff' in batch:
              ### gt
              root_gt = joints_all_gt[:, 9:10]
              length_gt = torch.norm(joints_all_gt[:, 9] - joints_all_gt[:, 0], dim=-1)
              joints_all_gt_off = joints_all_gt - root_gt
              verts_all_gt_off = verts_all_gt - root_gt
              #### pred
              root_pred = joints_pred_root[:, 9:10]
              length_pred = torch.norm(joints_pred_root[:, 9] - joints_pred_root[:, 0], dim=-1)
              scale_right = (length_gt / length_pred).unsqueeze(-1).unsqueeze(-1)
              joints_all_off = (joints_pred_root - root_pred) * scale_right
              verts_all_off = (verts_pred_root - root_pred) * scale_right  

              pose_loss = (self.l1_loss(joints_all_off, joints_all_gt_off)).reshape(B,-1).mean(dim=1) *1000
              verts_loss = (self.l1_loss(verts_all_off, verts_all_gt_off)).reshape(B,-1).mean(dim=1) *1000
            else: # for RHD who has no mano, only supervised with xyz
              for i in range(joints_all.shape[0]):
                index_root_bone_length = torch.sqrt(torch.sum((joints_all[i,10, :] - joints_all[i,9, :])**2))
                joints_all[i] = (joints_all[i] - joints_all[i,0,:])/index_root_bone_length*batch['joint_scale'][i]
              pose_loss = self.l1_loss(joints_all, batch['xyz'].reshape(-1,21,3)).reshape(B,-1).mean(dim=1)*1000                
          else:
            lms = None 

        if opt.photometric_loss and mode == 'train':
          pre_textures = _tranpose_and_gather_feat(output['texture'], ind_pred)
          device = pre_textures.device
          self.render.gamma = _tranpose_and_gather_feat(output['light'], ind_pred)
          Albedo = pre_textures.view(-1,778,3) #[b, 778, 3]
          texture_mean = torch.tensor([0.45,0.35,0.3]).float().to(device)
          texture_mean = texture_mean.unsqueeze(0).unsqueeze(0).repeat(Albedo.shape[0],Albedo.shape[1],1)#[1, 778, 3]
          Albedo = Albedo + texture_mean
          Texture, lighting = self.render.Illumination(Albedo, verts_all.view(-1,778,3))

          rotShape = verts_all.view(-1, C, 778, 3)
          Texture = Texture.view(-1, C, 778, 3)
          nV = rotShape.size(2)
          Verts, Faces, Textures = [], [], []
          valid = []
          tag_drop = False
          tmp_mask = batch['valid']
          for i in range(len(rotShape)):
            # detach vertex to avoid influence of color
            V_ = rotShape[i][tmp_mask[i]]#.detach()
            if V_.size(0) == 0:
              # to simplify the code, drop the whole batch with any zero_hand image.
              Verts = []
              break
            valid.append(i)
            T_ = Texture[i][tmp_mask[i]]
            F_ = torch.from_numpy(self.render.MANO_R.faces.astype(np.int64)).to(device).view(1,-1)
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
            rendered, gpu_masks, depth = self.render(meshes)
            rendered = torch.flip(rendered,[1])
            gpu_masks = torch.flip(gpu_masks,[1])
            depth = torch.flip(depth,[1]) 
            gpu_masks = gpu_masks.detach().float()
            ret_rendered = rendered[-B:]
            ret_gpu_masks = gpu_masks[-B:]
            skin_mask = batch['mask']
            crit = nn.L1Loss()
            tmp_seg_loss = crit(gpu_masks, skin_mask[valid])        
            seg_loss += tmp_seg_loss
            masks = gpu_masks * skin_mask[valid]
            photometric_loss += (torch.norm(rendered - batch['render'][valid], p=2, dim=3) * masks).sum() / (masks.sum() + 1e-4) / opt.num_stacks 
          else:
            tag_drop = True
            photometric_loss += torch.zeros(1).sum().cuda() / opt.num_stacks

        # Reduce the loss by orders of magnitude, 1e-2
        reproj_loss_all = reproj_loss_all * 0.01
        # if opt.mode == 'train_3d':        
        #   pose_loss = pose_loss * 1000
          # mano_loss = mano_loss * 10
          # verts_loss = verts_loss * 1000


        # show
        if file_id % 200 == 0:
          vis_lms = lms_pred.reshape(B,-1,21,2)
          save_img_0 = (np.squeeze(batch['image'][0]))
          save_img_0 = save_img_0.detach().cpu().numpy().astype(np.float32)
          if False:
            off_hm_pred = _tranpose_and_gather_feat(output['hm_off'], batch['ind']).detach().cpu().numpy()
            off_lms_pred = _tranpose_and_gather_feat(output['lms21_off'], batch['ind']).detach().cpu().numpy()
            ind= batch['ind'].detach().cpu().numpy()
            bit = self.opt.size_train[0] // 8
            ct_int =np.array([ind[0][0]%bit,ind[0][0]//bit])
            lms_pred = (off_lms_pred[0][0].reshape(-1,2) + ct_int)*8 
            ct = (ct_int+off_hm_pred[0][0])*8
            gt_ct= [(batch['ind'][0][0]%bit+batch['hm_off'][0][0][0])*8,(batch['ind'][0][0]//bit+batch['hm_off'][0][0][1])*8]
            cv2.circle(save_img_0, (int(ct[0]),int(ct[1])), 3, (0, 255, 0), 2)
            cv2.circle(save_img_0, (int(gt_ct[0]),int(gt_ct[1])), 3, (0, 0, 255), 2)
            cv2.imwrite('outputs/imgs/ct_{}.png'.format(file_id),save_img_0)
            showHandJoints(save_img_0,lms_pred,'outputs/imgs/off_lms_{}.jpg'.format(file_id))

          if opt.photometric_loss and mode == 'train':
            render_img = rendered[0].detach().cpu().numpy()[:, :, ::-1].astype(np.float32)
            render_msk = gpu_masks[0].detach().cpu().numpy()
            cv2.imwrite('outputs/imgs/fitted_{}_{}.jpg'.format(reproj_loss_all,file_id), render_img*255+save_img_0*((1-render_msk).reshape(save_img_0.shape[0],save_img_0.shape[1],1)))
            cv2.imwrite('outputs/imgs/gpu_masks_{}.png'.format(file_id), render_msk*255)
            cv2.imwrite('outputs/imgs/rendered_{}.jpg'.format(file_id), render_img*255)
              
          if opt.pick_hand:
            for k in range(batch['mask'].shape[1]):
                if batch['mask'][0][k]:
                  if 'lms21' in batch:
                    drawCirclev2(save_img_0,batch['lms'].reshape(B,-1,21,2)[0][k],(0,255,0),1) 
                    showHandJoints(save_img_0,batch['lms'].reshape(B,-1,21,2)[0][k].detach().cpu().numpy(),'outputs/imgs/kps_bone_gt_{}_{}.jpg'.format(file_id,k))
                  drawCirclev2(save_img_0,vis_lms[0][k],(0,0,255),1)  
                  showHandJoints(save_img_0,vis_lms[0][k].detach().cpu().numpy(),'outputs/imgs/kps_bone_pred_{}_{}.jpg'.format(file_id,k))
          else:      
            if 'lms' in batch:         
              showHandJoints(save_img_0,batch['lms'].reshape(-1,21,2)[0].detach().cpu().numpy(),'outputs/imgs/kps_bone_gt_{}.jpg'.format(file_id))
              drawCirclev2(save_img_0,batch['lms'].reshape(-1,21,2)[0],(0,0,255),1)   
            showHandJoints(save_img_0,vis_lms.reshape(-1,21,2)[0].detach().cpu().numpy(),'outputs/imgs/kps_bone_pred_{}.jpg'.format(file_id))
            drawCirclev2(save_img_0,vis_lms[0][0],(255,0,0),1) 
          cv2.imwrite('outputs/imgs/image_proj_0_{}.jpg'.format(file_id), save_img_0)

          # # for rendering .obj
          Faces_l = self.render.MANO_L.faces.astype(np.int32)
          Faces_r = self.render.MANO_R.faces.astype(np.int32)
          k = 0 # view k-st mesh.
          vis_verts = verts_all.reshape(-1,778,3)[k].detach().cpu().numpy()
          if opt.photometric_loss and mode == 'train':
            colors = Textures[0].detach().cpu().numpy()
            with open('outputs/models/colord_rhands_{}.obj'.format(file_id), 'w') as f:
              for idx in range(len(vis_verts)):
                f.write('v %f %f %f %f %f %f\n'%(vis_verts[idx][0],vis_verts[idx][1],vis_verts[idx][2],colors[idx][0],colors[idx][1],colors[idx][2]))
              for face in Faces_r+1:
                f.write('f %f %f %f\n'%(face[0],face[1],face[2]))     
          if 'handness' in batch and batch['handness'][k][0]==0: # left
            with open('outputs/models/lhands_{}.obj'.format(file_id), 'w') as f:
              for v in vis_verts:
                f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
              for face in Faces_l+1:
                f.write('f %f %f %f\n'%(face[0],face[1],face[2])) 
            if verts_all_gt is not None:
              with open('outputs/models/gt_hands_{}.obj'.format(file_id), 'w') as f:
                for v in verts_all_gt.reshape(-1,778,3)[k]:
                  f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
                for face in Faces_l+1:
                  f.write('f %f %f %f\n'%(face[0],face[1],face[2]))           
          else:
            with open('outputs/models/rhands_{}.obj'.format(file_id), 'w') as f:
              for v in vis_verts:
                f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
              for face in Faces_r+1:
                f.write('f %f %f %f\n'%(face[0],face[1],face[2]))   
            if verts_all_gt is not None:    
              with open('outputs/models/gt_hands_{}.obj'.format(file_id), 'w') as f:
                for v in verts_all_gt.reshape(-1,778,3)[k]:
                  f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
                for face in Faces_r+1:
                  f.write('f %f %f %f\n'%(face[0],face[1],face[2]))                    

    loss_stats = {}
    loss += opt.center_weight * hm_loss
    loss_stats.update({'hm_loss': hm_loss})
    if opt.pick_hand:
      loss += 20 * pick_loss
      loss_stats.update({'pick_loss': pick_loss})
    # modify 'not' to center_only for continue
    if not opt.center_only:
      if opt.off:
        loss += opt.off_weight * off_hm_loss
        loss_stats.update({'off_hm_loss': off_hm_loss})
        loss += opt.off_weight * off_lms_loss
        loss_stats.update({'off_lms_loss': off_lms_loss})

      if opt.heatmaps:
        loss += opt.heatmaps_weight * heatmaps_loss
        loss_stats.update({'heatmaps_loss':heatmaps_loss})

      if opt.reproj_loss:
        loss += opt.reproj_weight * reproj_loss_all
        loss_stats.update({'reproj_loss_all': reproj_loss_all})
        loss += opt.norm_weight * norm_loss
        loss_stats.update({'norm_loss': norm_loss})
      if opt.bone_loss:
        loss += opt.bone_dir_weight * bone_direc_loss
        loss_stats.update({'bone_direc_loss': bone_direc_loss})  
      if True: # only show joints loss with weight 0
        if 'mano_coeff' in batch:
          loss += opt.pose_weight * pose_loss *0
          loss_stats.update({'pose_loss': pose_loss})
          loss += opt.pose_weight * verts_loss *0
          loss_stats.update({'verts_loss': verts_loss})
          loss += opt.mano_weight * manopose_loss *0
          loss_stats.update({'mano_loss': manopose_loss})     
          loss += 1 * shape_loss *0
          loss_stats.update({'shape_loss': shape_loss}) 
        else:
          loss += opt.pose_weight * pose_loss *0
          loss_stats.update({'pose_loss': pose_loss})                   
      if opt.photometric_loss:
        loss += opt.photometric_weight * photometric_loss
        loss_stats.update({'photometric_loss': photometric_loss})
        loss += opt.seg_weight * seg_loss
        loss_stats.update({'seg_loss': seg_loss})   
    loss_stats.update({'loss': loss})
    if mode == 'val' or mode == 'test':
      return verts_all, joints_all, verts_all_gt, joints_all_gt, lms_pred,handness
    else:
      return loss, loss_stats, ret_rendered, ret_gpu_masks

  def interhandforward(self, outputs, mode, batch):
    opt = self.opt
    hm_loss, heatmaps_loss = 0, 0
    hand_type_loss = 0
    if opt.reproj_loss:
      reproj_loss, norm_loss = 0, 0
      reproj_loss_all = 0
    if opt.bone_loss:
      bone_loss, bone_dir_loss_all = 0, 0
    if opt.photometric_loss:
      norm_loss, var_loss = 0, 0
      photometric_loss, seg_loss = 0, 0
    if opt.perceptual_loss:
      perceptual_loss = 0
    if opt.gcn_decoder:
      S_loss, gcn_reproj_loss = 0, 0
    if opt.off:
      off_hm_loss, off_lms_loss, wh_loss = 0, 0, 0
    if opt.discrepancy:
      discrepancy_loss = 0
    loss = 0
    
    for s in range(opt.num_stacks):
      output = outputs[s]
      output['hm'] = _sigmoid(output['hm'])
      handness = None

      if mode == 'val' or mode == 'test':
        hms = output['hm'].clone().detach()
        score = 0.5
        hms = _nms(hms, 5)
        K = int((hms[0] > score).float().sum())
        K = 1
        topk_scores, pred_ind_left, topk_ys, topk_xs = _topk(hms[:,:1,:,:], K)  
        topk_scores, pred_ind_right, topk_ys, topk_xs = _topk(hms[:,1:,:,:], K)      
      ind_left = pred_ind_left if mode == 'val' or mode == 'test' else batch['ind'][:,:1]
      ind_right = pred_ind_right if mode == 'val' or mode == 'test' else batch['ind'][:,1:]
      # ind_left = batch['ind'][:,:1]
      # ind_right = batch['ind'][:,1:]      
      ## hm_loss
      hm_loss = hm_loss + self.crit(output['hm'], batch['hm']) / opt.num_stacks
      ## hand_type_loss
      # hand_type_pred_left = _tranpose_and_gather_feat(output['hm'][:,:1,:,:], ind_left).reshape(-1,1)
      # hand_type_pred_right = _tranpose_and_gather_feat(output['hm'][:,1:,:,:], ind_right).reshape(-1,1)
      # hand_type_pred = torch.stack((hand_type_pred_left,hand_type_pred_right),dim=1).reshape(-1,2)
      # hand_type_loss = hand_type_loss + get_hand_type_loss(hand_type_pred, batch['valid']) / opt.num_stacks    

      ret_rendered, ret_gpu_masks = None, None
      file_id = batch['file_id'].detach().cpu().numpy().astype(np.int)[0]
  
      if not opt.center_only:
        if opt.off:
          ## off_hm_loss
          off_hm_loss += self.crit_lms(output['off_hm'], batch['valid'],
                                    batch['ind'], batch['off_hm']) / opt.num_stacks
          ## off_lms_loss                          
          off_lms_loss += self.crit_lms(output['off_lms'], batch['valid'],
                                    batch['ind'], batch['off_lms']) / opt.num_stacks
          ## wh_loss                          
          wh_loss += self.crit_lms(output['wh'], batch['valid'],
                                    batch['ind'], batch['wh']) / opt.num_stacks                                                                   
        if opt.reproj_loss:
          # params = _tranpose_and_gather_feat(output['params'][-1], batch['ind'])
          params_left = _tranpose_and_gather_feat(output['params'][-1], ind_left)
          params_right = _tranpose_and_gather_feat(output['params'][-1], ind_right)
          # off_hm_pred_left = _tranpose_and_gather_feat(output['off_hm'], ind_left) if mode == 'val' or mode == 'test' else batch['off_hm'][:,0,:]
          # off_hm_pred_right = _tranpose_and_gather_feat(output['off_hm'], ind_right) if mode == 'val' or mode == 'test' else batch['off_hm'][:,1,:]
          B, C = params_right.size(0), params_right.size(1)
          theta_left = params_left.view(-1, params_left.size(2))
          theta_right = params_right.view(-1, params_right.size(2))
          global_orient_coeff_l_up =  theta_left[:, :3] 
          pose_coeff_l =  theta_left[:, 3:48] 
          betas_coeff_l =  theta_left[:, 48:58] # TEST with shape   
          global_transl_coeff_l_up = theta_left[:, 58:61] 
          # global_transl_coeff_l_up[:,2] = global_transl_coeff_l_up[:,2] 
          # ---------------left-right------------------
          global_orient_coeff_r_up =  theta_right[:, 61:64]  
          pose_coeff_r =  theta_right[:, 64:109]
          betas_coeff_r =  theta_right[:, 109:119] # TEST with shape
          global_transl_coeff_r_up = theta_right[:, 119:122] 
          
          verts_all_gt = torch.stack((batch['verts_left_gt'],batch['verts_right_gt']),dim=1)
          joints_all_gt = torch.stack((batch['joints_left_gt'],batch['joints_right_gt']),dim=1)

          hand_verts_l, hand_joints_l = self.render.mano_layer_left(global_orient_coeff_l_up, pose_coeff_l, betas_coeff_l, side ='left')
          hand_verts_r, hand_joints_r = self.render.mano_layer_right(global_orient_coeff_r_up, pose_coeff_r, betas_coeff_r, side ='right')    
          verts_all_pred = torch.stack((hand_verts_l,hand_verts_r),dim=1)
          joints_all_pred = torch.stack((hand_joints_l,hand_joints_r),dim=1)

          # parameters for norm only, which is define from flat_hand_mean other than hand_mean.
          norm_loss = (norm_loss + self.crit_normleft(pose_coeff_l, betas_coeff_l) + \
                        self.crit_norm(pose_coeff_r, betas_coeff_r)).reshape(B,-1).mean(dim=1)

          if 'lms_left_gt' in batch:
            lms21_proj_l = self.projection_batch(global_transl_coeff_l_up[:,2], global_transl_coeff_l_up[:,:2], hand_joints_l, img_size=self.opt.size_train[0])
            lms21_proj_r = self.projection_batch(global_transl_coeff_r_up[:,2], global_transl_coeff_r_up[:,:2], hand_joints_r, img_size=self.opt.size_train[0])
            lms21_proj = torch.stack((lms21_proj_l,lms21_proj_r),dim=1)
            lms21_gt = torch.stack((batch['lms_left_gt'],batch['lms_right_gt']),dim=1)

            ### lms_loss
            reproj_loss_all = ((F.mse_loss(lms21_proj, lms21_gt, reduction='none')) / opt.num_stacks).reshape(B,-1).mean(dim=1) 

            if opt.bone_loss:
              # tmp_mask = mask.reshape(-1,21,2)
              j2d_con = torch.ones_like(lms21_proj_l[:,:,0]).unsqueeze(-1)
              # maybe confidence can be used here.
              bone_direc_loss = bone_direction_loss(lms21_proj_l, batch['lms_left_gt'], j2d_con).reshape(B,-1).mean(dim=1) + \
                bone_direction_loss(lms21_proj_r, batch['lms_right_gt'], j2d_con).reshape(B,-1).mean(dim=1)

            ### joints_loss
            if 'joints_left_gt' in batch:
              ### gt
              root_gt = joints_all_gt[:,:, 9:10]
              length_gt = torch.norm(joints_all_gt[:,:, 9] - joints_all_gt[:,:, 0], dim=-1)
              joints_all_gt = joints_all_gt - root_gt
              verts_all_gt = verts_all_gt - root_gt
              #### pred
              root_pred = joints_all_pred[:,:, 9:10]
              length_pred = torch.norm(joints_all_pred[:,:, 9] - joints_all_pred[:,:, 0], dim=-1)
              scale_right = (length_gt / length_pred).unsqueeze(-1).unsqueeze(-1)
              joints_all_pred_off = (joints_all_pred - root_pred) * scale_right
              verts_all_pred_off = (verts_all_pred - root_pred) * scale_right  

              pose_loss = (self.l1_loss(joints_all_pred_off, joints_all_gt)).reshape(B,-1).mean(dim=1) *1000
              verts_loss = (self.l1_loss(verts_all_pred_off, verts_all_gt)).reshape(B,-1).mean(dim=1) *1000 
     
          else:
            lms = None 

        if opt.photometric_loss:
          pre_textures = _tranpose_and_gather_feat(output['texture'], batch['ind'])
          device = pre_textures.device
          self.render.gamma = _tranpose_and_gather_feat(output['light'], batch['ind'])
          Albedo = pre_textures.view(-1,778,3) #[b, 778, 3]
          texture_mean = torch.tensor([0.45,0.35,0.3]).float().to(device)
          texture_mean = texture_mean.unsqueeze(0).unsqueeze(0).repeat(Albedo.shape[0],Albedo.shape[1],1)#[1, 778, 3]
          Albedo = Albedo + texture_mean
          Texture, lighting = self.render.Illumination(Albedo, verts_all_pred.view(-1,778,3))

          rotShape = verts_all_pred.view(-1, C, 778, 3)
          Texture = Texture.view(-1, C, 778, 3)
          nV = rotShape.size(2)
          Verts, Faces, Textures = [], [], []
          valid = []
          tag_drop = False
          tmp_mask = batch['valid'].reshape(-1,1)
          for i in range(len(rotShape)):
            # detach vertex to avoid influence of color
            V_ = rotShape[i][tmp_mask[i]]#.detach()
            if V_.size(0) == 0:
              # to simplify the code, drop the whole batch with any zero_hand image.
              Verts = []
              break
            valid.append(i)
            T_ = Texture[i][tmp_mask[i]]
            F_ = torch.from_numpy(self.render.MANO_R.faces.astype(np.int64)).to(device).view(1,-1)
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
            rendered, gpu_masks, depth = self.render(meshes)
            rendered = torch.flip(rendered,[1])
            gpu_masks = torch.flip(gpu_masks,[1])
            depth = torch.flip(depth,[1]) 
            gpu_masks = gpu_masks.detach().float()
            ret_rendered = rendered[-B:]
            ret_gpu_masks = gpu_masks[-B:]
            skin_mask = batch['skin_mask']
            crit = nn.L1Loss()
            tmp_seg_loss = crit(gpu_masks, skin_mask[valid])        
            seg_loss += tmp_seg_loss
            masks = gpu_masks * skin_mask[valid]
            photometric_loss += (torch.norm(rendered - batch['render'][valid], p=2, dim=3) * masks).sum() / (masks.sum() + 1e-4) / opt.num_stacks 
          else:
            tag_drop = True
            photometric_loss += torch.zeros(1).sum().cuda() / opt.num_stacks

        # Reduce the loss by orders of magnitude, 1e-2
        reproj_loss_all = reproj_loss_all * 0.01
        # if opt.mode == 'train_3d':        
        #   pose_loss = pose_loss * 1000
          # mano_loss = mano_loss * 10
          # verts_loss = verts_loss * 1000


        # show
        # if reproj_loss_all < 1 and torch.norm(lms21_gt[0,0,9] - lms21_gt[0,1,9], dim=-1)<100:
        if file_id % 100 == 0:
          save_img_0 = (np.squeeze(batch['image'][0])).detach().cpu().numpy().astype(np.float32)
          if False:
            wh_pred = _tranpose_and_gather_feat(output['wh'], ind_left).detach().cpu().numpy()
            off_hm_pred = _tranpose_and_gather_feat(output['off_hm'], ind_left).detach().cpu().numpy()
            off_lms_pred = _tranpose_and_gather_feat(output['off_lms'], ind_left).detach().cpu().numpy()
            ind_left = ind_left.detach().cpu().numpy()
            bit = self.opt.size_train[0] // 8
            ct_left =np.array([ind_left[0][0]%bit,ind_left[0][0]//bit])
            lms_pred = (off_lms_pred[0][0].reshape(-1,2) + ct_left)*8 
            ct_left = (ct_left+off_hm_pred[0][0])*8
            gt_ct_left = [(batch['ind'][0][0]%bit+batch['off_hm'][0][0][0])*8,(batch['ind'][0][0]//bit+batch['off_hm'][0][0][1])*8]
            w,h = wh_pred[0][0]*8
            gt_w,gt_h = batch['wh'][0][0]*8
            box_pred = (ct_left[0]-w/2), (ct_left[1]-h/2), (ct_left[0]+w/2), (ct_left[1]+h/2)
            box_gt = (gt_ct_left[0]-gt_w/2), (gt_ct_left[1]-gt_h/2), (gt_ct_left[0]+gt_w/2), (gt_ct_left[1]+gt_h/2)
            iou = get_iou(box_pred,box_gt)
            cat_name = 'left'
            txt = '{}{:.1f}'.format(cat_name, iou)
            cv2.putText(save_img_0, txt, (int(box_pred[0]), int(box_pred[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), thickness=1)
            cv2.rectangle(save_img_0, (int(gt_ct_left[0]-gt_w/2), int(gt_ct_left[1]-gt_h/2)), (int(gt_ct_left[0]+gt_w/2), int(gt_ct_left[1]+gt_h/2)), (0,0,255), 1)   
            cv2.rectangle(save_img_0, (int(ct_left[0]-w/2), int(ct_left[1]-h/2)), (int(ct_left[0]+w/2), int(ct_left[1]+h/2)), (0,255,0), 1)   
            cv2.imwrite('outputs/imgs/evaluation/boxes_{}.png'.format(file_id),save_img_0)
            showHandJoints(save_img_0,lms_pred,'outputs/imgs/evaluation/off_lms_{}.jpg'.format(file_id))

            # save_hm_gt = batch['hm'][0][1].detach().cpu().numpy()*255
            # cv2.imwrite('hm_gt{}.jpg'.format(file_id),save_hm_gt)
            # save_hm = output['hm'][0][1].detach().cpu().numpy()*255
            # cv2.imwrite('hm_pred{}.jpg'.format(file_id),save_hm)

          if opt.reproj_loss:
            vis_lms = lms21_proj.reshape(B,-1,21,2)  
            if opt.photometric_loss and rendered is not None:
              render_img = rendered[0].detach().cpu().numpy()[:, :, ::-1].astype(np.float32)
              render_msk = gpu_masks[0].detach().cpu().numpy()
              cv2.imwrite('outputs/imgs/fitted_{}_{}.jpg'.format(reproj_loss_all,file_id), render_img*255+save_img_0*((1-render_msk).reshape(save_img_0.shape[0],save_img_0.shape[1],1)))
              cv2.imwrite('outputs/imgs/gpu_masks_{}.png'.format(file_id), render_msk*255)
              cv2.imwrite('outputs/imgs/rendered_{}.jpg'.format(file_id), render_img*255)
    
            for k in range(batch['valid'].shape[1]):
                if batch['valid'][0][k]:
                  drawCirclev2(save_img_0,lms21_gt.reshape(B,-1,21,2)[0][k],(0,0,255),1)
                  drawCirclev2(save_img_0,vis_lms[0][k],(0,255,0),1)  
                  # showHandJoints(save_img_0,batch['lms'].reshape(B,-1,21,2)[0][k].detach().cpu().numpy(),'outputs/imgs/kps_bone_gt_{}_{}.jpg'.format(file_id,k))
                  showHandJoints(save_img_0,vis_lms[0][k].detach().cpu().numpy(),'outputs/imgs/kps_bone_pred_{}_{}.jpg'.format(file_id,k))
            if 'mano_coeff' in batch and False:
              lms21_proj_l_gt = self.render.get_Landmarks_new(hand_joints_gt_left,batch['K_new']) 
              lms21_proj_r_gt = self.render.get_Landmarks_new(hand_joints_gt_right,batch['K_new'])   
              showHandJoints(save_img_0,lms21_proj_l_gt[0].detach().cpu().numpy(),'outputs/imgs/kps_bone_pred_mano_l{}.jpg'.format(file_id))
              showHandJoints(save_img_0,lms21_proj_r_gt[0].detach().cpu().numpy(),'outputs/imgs/kps_bone_pred_mano_r{}.jpg'.format(file_id))
  
            cv2.imwrite('outputs/imgs/image_proj_0_{}.jpg'.format(file_id), save_img_0)

            # # for rendering .obj
            Faces_l = self.render.MANO_L.faces.astype(np.int32)
            Faces_r = self.render.MANO_R.faces.astype(np.int32)
            vis_verts = verts_all_pred[0].reshape(-1,778,3).detach().cpu().numpy()
            gt_verts = verts_all_gt[0].reshape(-1,778,3).detach().cpu().numpy()
            if opt.photometric_loss:
              colors = Textures[0].detach().cpu().numpy()
              with open('outputs/models/colord_rhands_{}.obj'.format(file_id), 'w') as f:
                for idx in range(len(vis_verts)):
                  f.write('v %f %f %f %f %f %f\n'%(vis_verts[idx][0],vis_verts[idx][1],vis_verts[idx][2],colors[idx][0],colors[idx][1],colors[idx][2]))
                for face in Faces_r+1:
                  f.write('f %f %f %f\n'%(face[0],face[1],face[2]))     
            if batch['valid'][0][0]==1: # left
              with open('outputs/models/lhands_{}.obj'.format(file_id), 'w') as f:
                for v in vis_verts[0]:
                  f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
                for face in Faces_l+1:
                  f.write('f %f %f %f\n'%(face[0],face[1],face[2])) 
              with open('outputs/models/gt_hands_l{}.obj'.format(file_id), 'w') as f:
                for v in gt_verts.reshape(-1,778,3)[0]:
                  f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
                for face in Faces_l+1:
                  f.write('f %f %f %f\n'%(face[0],face[1],face[2]))                  
            if batch['valid'][0][1]==1: # right
              with open('outputs/models/rhands_{}.obj'.format(file_id), 'w') as f:
                for v in vis_verts[1]:
                  f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
                for face in Faces_r+1:
                  f.write('f %f %f %f\n'%(face[0],face[1],face[2]))       
              with open('outputs/models/gt_hands_r{}.obj'.format(file_id), 'w') as f:
                for v in gt_verts.reshape(-1,778,3)[1]:
                  f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
                for face in Faces_r+1:
                  f.write('f %f %f %f\n'%(face[0],face[1],face[2]))                    

    loss_stats = {}
    loss += opt.center_weight * hm_loss
    loss_stats.update({'hm_loss': hm_loss})
    # loss += opt.center_weight * root_loss
    # loss_stats.update({'root_loss': root_loss})
    # loss += opt.handtype_weight * hand_type_loss
    # loss_stats.update({'hand_type_loss': hand_type_loss})
    # modify 'not' to center_only for continue
    if not opt.center_only:
      if opt.off:
        loss += opt.off_weight * off_hm_loss
        loss_stats.update({'off_hm_loss': off_hm_loss})
        loss += opt.off_weight * off_lms_loss
        loss_stats.update({'off_lms_loss': off_lms_loss})
        loss += opt.wh_weight * wh_loss
        loss_stats.update({'wh_loss': wh_loss})
        
      if opt.heatmaps:
        loss += opt.heatmaps_weight * heatmaps_loss
        loss_stats.update({'heatmaps_loss':heatmaps_loss})

      if opt.reproj_loss:
        loss += opt.reproj_weight * reproj_loss_all
        loss_stats.update({'reproj_loss_all': reproj_loss_all})
        loss += opt.norm_weight * norm_loss *10
        loss_stats.update({'norm_loss': norm_loss})
        if opt.bone_loss:
          loss += opt.bone_dir_weight * bone_direc_loss
          loss_stats.update({'bone_direc_loss': bone_direc_loss})  

        loss += opt.pose_weight * pose_loss *0
        loss_stats.update({'pose_loss': pose_loss}) 
        loss += opt.pose_weight * verts_loss *0
        loss_stats.update({'verts_loss': verts_loss})                 
        # loss += opt.mano_weight * mano_loss 
        # loss_stats.update({'mano_loss': mano_loss})     
        # loss += 1 * shape_loss *0.1
        # loss_stats.update({'shape_loss': shape_loss})   
      if opt.photometric_loss:
        loss += opt.photometric_weight * photometric_loss
        loss_stats.update({'photometric_loss': photometric_loss})
        loss += opt.seg_weight * seg_loss
        loss_stats.update({'seg_loss': seg_loss})   
    loss_stats.update({'loss': loss})
    if mode == 'val' or mode == 'test':
      return verts_all_pred_off, joints_all_pred_off, verts_all_gt, joints_all_gt, lms21_proj,handness
    else:
      return loss, loss_stats, ret_rendered, ret_gpu_masks
    

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
    if self.opt.default_resolution == 384: # for multi-hand case
      loss_stats.append('pick_loss')
      loss_stats.append('reproj_loss_all')
      loss_stats.append('norm_loss')
      loss_stats.append('bone_direc_loss')
      loss_stats.append('pose_loss')
      loss_stats.append('verts_loss')
      if opt.photometric_loss:
        loss_stats.append('photometric_loss')
        loss_stats.append('seg_loss')    
    else:
      if opt.pick_hand:
        loss_stats.append('pick_loss')
      if not opt.center_only:
        if opt.off:
          loss_stats.append('off_hm_loss')
          loss_stats.append('off_lms_loss')      
        if opt.heatmaps:
          loss_stats.append('heatmaps_loss')
        if opt.reproj_loss:
          loss_stats.append('reproj_loss_all')
          loss_stats.append('norm_loss')
        if opt.bone_loss:
          loss_stats.append('bone_direc_loss')
        if True:
          if self.opt.dataset == 'RHD':
            loss_stats.append('pose_loss')
          elif self.opt.dataset == 'InterHandNew':
            loss_stats.append('pose_loss')
            loss_stats.append('verts_loss')
            # loss_stats.append('mano_loss')
            # loss_stats.append('shape_loss')
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
