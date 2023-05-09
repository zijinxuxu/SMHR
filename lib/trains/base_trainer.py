from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import torch
from progress.bar import Bar

from models.data_parallel import DataParallel
from utils.utils import AverageMeter
import cv2
import numpy as np
import os
import json
import pickle
from utils.eval import main as eval_main
from utils.eval import align_w_scale 
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.image import get_affine_transform, affine_transform, affine_transform_array
from torch.nn.parallel import DistributedDataParallel as DDP

# This class can be removed
class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss, opt):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss
        self.opt = opt

    def forward(self, batch, mode):            
        if mode == 'train':
            if 'heatmaps' in batch:
                tmp_heatmaps = batch['heatmaps']
            else:
                tmp_heatmaps = None
            outputs, ind = self.model(batch['input'], tmp_heatmaps, batch['ind'])
            if self.opt.dataset == 'InterHandNew':
                loss, loss_stats, rendered, masks = self.loss.interhandforward(outputs, mode, batch)
            else:
                loss, loss_stats, rendered, masks = self.loss(outputs, mode, batch)
        elif mode == 'val' or mode == 'test':
            # loss, loss_stats, rendered, masks = self.test(outputs, batch)
            outputs, ind = self.model(batch['input'], None, None)
            if self.opt.dataset == 'InterHandNew':
                vertex_pred, joints_pred, vertex_gt, joints_gt, lms21_pred, handness = self.loss.interhandforward(outputs, mode, batch)
            else:            
                vertex_pred, joints_pred, vertex_gt, joints_gt, lms21_pred, handness = self.loss(outputs, mode, batch)
            return vertex_pred, joints_pred, vertex_gt, joints_gt, lms21_pred, handness

        return outputs, loss, loss_stats, rendered, masks


class BaseTrainer(object):
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        if optimizer:
            self.loss_stats, self.loss = self._get_losses(opt)
            self.model_with_loss = ModleWithLoss(model, self.loss, opt)
        self.model = model
        self.optimizer.add_param_group({'params': self.loss.parameters()})

    def set_device(self, gpus, chunk_sizes, device, local_rank):
        if local_rank is not None:
            self.model_with_loss = DDP(
                self.model_with_loss.cuda(), device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
        else:
            if len(gpus) >= 1:
                if self.optimizer:
                    self.model_with_loss = DataParallel(
                        self.model_with_loss, device_ids=gpus,
                        chunk_sizes=chunk_sizes).to(device)
            else:
                if self.optimizer:
                    self.model_with_loss = self.model_with_loss.to(device)
                self.model = self.model.to(device)

    def run_epoch(self, phase, epoch, data_loader, logger=None):
        model_with_loss = self.model_with_loss
        model_with_loss.train()

        torch.cuda.empty_cache()

        opt = self.opt

        results = {}
        data_time, batch_time, ema_time = AverageMeter(), AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = min(len(data_loader), 10000) if opt.num_iters < 0 else opt.num_iters
        num_iters = len(data_loader)
        if opt.local_rank ==0: # only write once
            bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()

        # gradual_lr
        if opt.gradual_lr:
            opt.reproj_weight /= pow(10, 0.02)
            opt.photometric_weight *= pow(10, 0.02)

        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':# and k != 'dataset':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)
            output, loss, loss_stats, rendered, gpu_mask = model_with_loss(batch,'train')
            
            if opt.photometric_loss: # finetune using hard mode sample
                valid_loss, idxs = torch.topk(loss, int(0.7 * loss.size()[0]))    
                loss = torch.mean(valid_loss)
            else:
                loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if opt.local_rank ==0: # only write once
                Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                    epoch, iter_id, num_iters, phase=phase,
                    total=bar.elapsed_td, eta=bar.eta_td)
                for l in avg_loss_stats:
                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch['input'].size(0))
                    Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
                    Bar.suffix = Bar.suffix + '|cur_{} {:.4f} '.format(l, avg_loss_stats[l].val)
                if opt.print_iter > 0:
                    if iter_id % opt.print_iter == 0:
                        print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
                else:
                    bar.next()

            ## TensorboardX
                step = (epoch - 1) * num_iters + iter_id
                if logger is not None:
                    if step % 10 == 0:
                        for k, v in avg_loss_stats.items():
                            logger.scalar_summary('train_{}'.format(k), v.avg, step)
                    if step % 500 == 0 and self.opt.photometric_loss:
                        # print('logger: {}'.format(step))
                        img, chosen_img, gt_img = [], [], []
                        img.append(batch['image'][batch['valid'].sum(dim=1)>0].cpu())
                        if rendered is not None:
                            chosen_img.append(rendered.detach().cpu())
                            if 'Vs' in batch:
                                rendered_gt, rendered_gt_mask = self.render(batch['Vs'], batch['Ts'])
                                gt_img.append(rendered_gt.detach().cpu())
                            img = torch.cat(img, 0)
                            chosen_img = torch.clamp(torch.cat(chosen_img, 0), 0., 1.)
                            if len(gt_img) != 0:
                                gt_img = torch.clamp(torch.cat(gt_img, 0), 0., 1.)
                                t = torch.cat([img, chosen_img, gt_img], 2).permute(0, 3, 1, 2).contiguous()
                            else:
                                t = torch.cat([img, chosen_img], 2).permute(0, 3, 1, 2).contiguous()
                            logger.image_summary('train', t[:4], step)

        if opt.local_rank ==0: # only write once
            bar.finish()
            ret = {k: v.avg for k, v in avg_loss_stats.items()}
            ret['time'] = bar.elapsed_td.total_seconds() / 60.

            return ret, results
        else:
            return None, None

    def _get_losses(self, opt):
        raise NotImplementedError

    def train(self, epoch, data_loader, logger=None):
        return self.run_epoch('train', epoch, data_loader, logger=logger)

    def evaluation(self, eval_loader, logger=None):
        model_with_loss = self.model_with_loss
        model_with_loss.eval()
        if isinstance(model_with_loss, DDP):
            model_with_loss = model_with_loss.module

        xyz_pred_list, verts_pred_list = list(), list()
        bar = Bar("EVAL", max=len(eval_loader))

        if self.opt.input_res == 512 or self.opt.input_res == 384:
            hand_num = 4
        else:
            hand_num = 1   

        mpjpe = [[] for _ in range(21*hand_num)] # treat right and left hand identical 
        mpixel = [[] for _ in range(21*hand_num)]   
        mpvpe = [[] for _ in range(778)] # treat right and left hand identical  
        acc_hand_cls = 0
        valid_hand_type_all = 0
        psnr_all = 0
        psnr_count = len(eval_loader)
        left_joints_loss_all, right_joints_loss_all = 0, 0
        left_verts_loss_all, right_verts_loss_all = 0, 0
        lms_loss_all = 0        
        with torch.no_grad():
            for step, data in enumerate(eval_loader):
                for k in data:
                    if k != 'meta':# and k != 'dataset':
                        data[k] = data[k].to(device=self.opt.device, non_blocking=True)
            
                vertex_pred, joints_pred, vertex_gt, joints_gt, lms21_pred,handness  = model_with_loss(data,'test') 

                xyz_pred_list.append(joints_pred.reshape(-1, 3))
                verts_pred_list.append(vertex_pred.reshape(-1, 3))

                if self.opt.pick_hand:
                    if len(handness[0]) == 1:
                        if data['handness'][0,0] == handness[0,0]:
                            acc_hand_cls += 1
                        valid_hand_type_all +=1
                    else:
                        arr = (data['handness'] - handness).detach().cpu().numpy()
                        num_right = np.count_nonzero(arr==0)
                        acc_hand_cls += num_right
                        valid_hand_type_all +=4


                if self.opt.default_resolution == 384: # multi-hand case
                    lms21_gt = data['lms21'].reshape(-1,21,2)
                    lms_loss = torch.norm((lms21_pred -lms21_gt), dim=-1).detach().cpu().numpy()

                    lms_loss_all = lms_loss_all + lms_loss.mean()/2   
                    bar.suffix = '({batchs}/{size})' .format(batchs=step+1, size=len(eval_loader))
                    bar.next()
                    continue                                       
                if self.opt.dataset == 'InterHandNew':
                    # imgTensors = data[0].cuda()
                    joints_left_gt = joints_gt[:,0,:,:]
                    verts_left_gt = vertex_gt[:,0,:,:]
                    joints_right_gt = joints_gt[:,1,:,:]
                    verts_right_gt = vertex_gt[:,1,:,:]
                    lms21_left_gt = data['lms_left_gt']
                    lms21_right_gt = data['lms_right_gt']

                    if False:
                        img = (np.squeeze(data['image'][0])).detach().cpu().numpy().astype(np.float32)
                        cv2.imwrite('img_orig.jpg',img)

                    # 2. use otherInfo['Manolist] verts
                    verts_left_pred =  vertex_pred[:,0,:,:]
                    verts_right_pred = vertex_pred[:,1,:,:]
                    joints_left_pred =  joints_pred[:,0,:,:]
                    joints_right_pred = joints_pred[:,1,:,:]
                    lms21_left_pred = lms21_pred[:,0,:,:]
                    lms21_right_pred = lms21_pred[:,1,:,:]

                    joint_left_loss = torch.norm((joints_left_pred - joints_left_gt), dim=-1)
                    joint_left_loss = joint_left_loss.detach().cpu().numpy()

                    joint_right_loss = torch.norm((joints_right_pred - joints_right_gt), dim=-1)
                    joint_right_loss = joint_right_loss.detach().cpu().numpy()

                    vert_left_loss = torch.norm((verts_left_pred - verts_left_gt), dim=-1)
                    vert_left_loss = vert_left_loss.detach().cpu().numpy()

                    vert_right_loss = torch.norm((verts_right_pred - verts_right_gt), dim=-1)
                    vert_right_loss = vert_right_loss.detach().cpu().numpy()

                    lms_left_loss = torch.norm((lms21_left_pred -lms21_left_gt), dim=-1).detach().cpu().numpy()
                    lms_right_loss = torch.norm((lms21_right_pred -lms21_right_gt), dim=-1).detach().cpu().numpy()

                    lms_loss_all = lms_loss_all + (lms_left_loss + lms_right_loss).mean()/2
                    left_joints_loss_all = left_joints_loss_all + joint_left_loss.mean()*1000
                    left_verts_loss_all = left_verts_loss_all  + vert_left_loss.mean()*1000
                    right_joints_loss_all = right_joints_loss_all + joint_right_loss.mean()*1000
                    right_verts_loss_all = right_verts_loss_all + vert_right_loss.mean()*1000


                    bar.suffix = '({batchs}/{size})' .format(batchs=step+1, size=len(eval_loader))
                    bar.next()
                    continue        

                bar.suffix = '({batchs}/{size})' .format(batchs=step+1, size=len(eval_loader))
                bar.next()
        bar.finish()

        if self.opt.default_resolution == 384: # multi-hand case
            a = lms_loss_all/len(eval_loader)
            score_path = os.path.join(self.opt.root_dir, 'Multihand-eval.txt')
            with open(score_path, 'a') as fo:
                fo.write('lms: %f , Handedness: %s \n' % (a,str(acc_hand_cls / valid_hand_type_all+1e-3)))
            print('Scores written to: %s' % score_path)             
            return None, None 
        if self.opt.dataset == 'InterHandNew':
            a = lms_loss_all/len(eval_loader)
            b = left_joints_loss_all/len(eval_loader)
            c = left_verts_loss_all/len(eval_loader)
            d = right_joints_loss_all/len(eval_loader)
            e = right_verts_loss_all/len(eval_loader)  
            score_path = os.path.join(self.opt.root_dir, 'InterHandNew-eval.txt')
            with open(score_path, 'a') as fo:
                fo.write('lms: %f, jointl: %f,  jointr: %f, vertl: %f, vertr: %f \n' % (a,b,c,d,e))
            print('Scores written to: %s' % score_path)                     
            return None, None
        
        if self.opt.pick_hand and False:
            eval_summary = 'MPJPE for each joint: \n'
            for j in range(21*hand_num):
                mpjpe[j] = np.mean(np.stack(mpjpe[j]))
                mpixel[j] = np.mean(np.stack(mpixel[j]))
                # mcpixel[j] = np.mean(np.stack(mcpixel[j]))
                # joint_name = self.skeleton[j]['name']
                print('joint_{0}:{1}'.format(j,mpjpe[j])) 
                print('joint_{0}:{1}'.format(j,mpixel[j])) 
                # print('joint_{0}:{1}'.format(j,mcpixel[j])) 
            # print(eval_summary)
            print('MPJPE-joint: %.5f' % (np.mean(mpjpe)))
            print('MPJPE-pix: %.5f' % (np.mean(mpixel)))
            # print('MPJPE-uncrop: %.5f' % (np.mean(mcpixel)))
            print('Handedness accuracy: ' + str(acc_hand_cls / valid_hand_type_all))
        else:
            xyz_pred_list = [x.tolist() for x in xyz_pred_list]
            verts_pred_list = [x.tolist() for x in verts_pred_list]
            # with open(os.path.join(self.opt.root_dir, 'pred' + '.json'), 'w') as fo:
            #     json.dump([xyz_pred_list, verts_pred_list], fo)
            # print('Save json file at ' + os.path.join(self.opt.root_dir, 'pred' + '.json'))
            # pred_path = os.path.join(self.opt.root_dir, 'pred' + '.json')
        if True: # test for evaluation score
            gt_path = os.path.join(self.opt.root_dir, 'data/FreiHAND')
            eval_main(gt_path,xyz_pred_list,verts_pred_list,'./')  
            # score_path = os.path.join(self.opt.root_dir, 'FreiHAND-multi.txt')
            # with open(score_path, 'a') as fo:
            #     fo.write('Handedness accuracy: %s \n' % (str(acc_hand_cls / valid_hand_type_all+1e-3)))
            # print('Scores written to: %s \n' % score_path)     
            return None, None          
        return None, None
    
    def evaluation_H2O(self, eval_loader, logger=None):
        model_with_loss = self.model_with_loss
        model_with_loss.eval()

        torch.cuda.empty_cache()
        xyz_pred_list, verts_pred_list = list(), list()
        xyz_gt_list, verts_gt_list = list(), list()
        H2O_list = {"modality": "RGBD"}
        local_list = {}
        bar = Bar("TEST", max=len(eval_loader))

        hand_num = 2 # or 2. modified according to your model.
        lmpjpe = [[] for _ in range(21*1)] # treat right and left hand identical 
        rmpjpe = [[] for _ in range(21*1)]
        mpix = [[] for _ in range(21*hand_num)]  
        mpvpe = [[] for _ in range(778)] # treat right and left hand identical   
        action_id = 1
        left_joints_loss_all, right_joints_loss_all = 0, 0
        left_verts_loss_all, right_verts_loss_all = 0, 0
        lms_loss_all = 0
        with torch.no_grad():
            for step, data in enumerate(eval_loader):
                for k in data:
                    if k != 'meta':# and k != 'dataset':
                        data[k] = data[k].to(device=self.opt.device, non_blocking=True)

                vertex_pred, joints_pred, vertex_gt, joints_gt, lms21_pred  = model_with_loss(data,'test')         

                if self.opt.dataset == 'InterHandNew':
                    # imgTensors = data[0].cuda()
                    joints_left_gt = joints_gt[:,0,:,:]
                    verts_left_gt = vertex_gt[:,0,:,:]
                    joints_right_gt = joints_gt[:,1,:,:]
                    verts_right_gt = vertex_gt[:,1,:,:]
                    lms21_left_gt = data['lms_left_gt'][:,:,:]
                    lms21_right_gt = data['lms_right_gt'][:,:,:]

                    if False:
                        img = (np.squeeze(data['image'][0])).detach().cpu().numpy().astype(np.float32)
                        cv2.imwrite('img_orig.jpg',img)

                    # 2. use otherInfo['Manolist] verts
                    verts_left_pred =  vertex_pred[:,0,:,:]
                    verts_right_pred = vertex_pred[:,1,:,:]
                    joints_left_pred =  joints_pred[:,0,:,:]
                    joints_right_pred = joints_pred[:,1,:,:]
                    lms21_left_pred = lms21_pred[:,0,:,:]
                    lms21_right_pred = lms21_pred[:,1,:,:]

                    joint_left_loss = torch.norm((joints_left_pred - joints_left_gt), dim=-1)
                    joint_left_loss = joint_left_loss.detach().cpu().numpy()

                    joint_right_loss = torch.norm((joints_right_pred - joints_right_gt), dim=-1)
                    joint_right_loss = joint_right_loss.detach().cpu().numpy()

                    vert_left_loss = torch.norm((verts_left_pred - verts_left_gt), dim=-1)
                    vert_left_loss = vert_left_loss.detach().cpu().numpy()

                    vert_right_loss = torch.norm((verts_right_pred - verts_right_gt), dim=-1)
                    vert_right_loss = vert_right_loss.detach().cpu().numpy()

                    lms_left_loss = torch.norm((lms21_left_pred -lms21_left_gt), dim=-1).detach().cpu().numpy()
                    lms_right_loss = torch.norm((lms21_right_pred -lms21_right_gt), dim=-1).detach().cpu().numpy()

                    lms_loss_all = lms_loss_all + (lms_left_loss + lms_right_loss).mean()/2
                    left_joints_loss_all = left_joints_loss_all + joint_right_loss.mean()*1000
                    left_verts_loss_all = left_verts_loss_all  + vert_right_loss.mean()*1000
                    right_joints_loss_all = right_joints_loss_all + joint_right_loss.mean()*1000
                    right_verts_loss_all = right_verts_loss_all + vert_right_loss.mean()*1000


                    bar.suffix = '({batchs}/{size})' .format(batchs=step+1, size=len(eval_loader))
                    bar.next()
                    continue        

                if data['id'][0] == action_id + 1:
                    H2O_list.update({'{}'.format(action_id): local_list})
                    action_id = action_id + 1
                    local_list = {}
                lms_gt = data['lms'].view_as(lms21_pred)
                if joints_gt is None:
                    joints_gt = data['joints'] 

                frame_num = data['frame_num'][0]
                local_list.update({'{:06d}.txt'.format(frame_num):joints_pred.reshape(-1).tolist()})
                                              
                for i in range(lms_gt.shape[0]):
                    tmp_lms_gt = (lms_gt)[i].reshape(-1,2).cpu().numpy()
                    tmp_lms_pred = (lms21_pred)[i].reshape(-1,2).cpu().numpy()
                    tmp_joints_pred = (joints_pred)[i].reshape(-1,3).cpu().numpy()
                    tmp_verts_pred = vertex_pred[i].reshape(-1,3).cpu().numpy()
                    tmp_joints_gt = joints_gt[i].reshape(-1,3).cpu().numpy()
                    # tmp_xyz_gt = xyz_gt[i][0].reshape(-1,3).cpu().numpy()
                    # tmp_verts_gt = vertex_gt[i].reshape(-1,3).cpu().numpy()
                    index_root_bone_length = np.sqrt(np.sum((tmp_joints_pred[10, :] - tmp_joints_pred[9, :])**2))
                    gt_bone_length = np.sqrt(np.sum((tmp_joints_gt[10, :] - tmp_joints_gt[9, :])**2))
                    # xyz_pred_aligned = (tmp_joints_pred - tmp_joints_pred[9,:])/index_root_bone_length*gt_bone_length*1000
                    # verts_pred_aligned = (tmp_verts_pred - tmp_verts_pred[9,:])/index_root_bone_length*gt_bone_length*1000 

                    if tmp_joints_pred.shape[0] > 21 and tmp_joints_gt.shape[0] > 21:
                        # here we consider RHD two hand case
                        # for j in range(len(data['valid'][i])):
                        #     if data['valid'][i][j] == 0: # no visible hand in left/right
                        #         tmp_joints_gt[j*21:21+j*21,:] = tmp_joints_gt[j*21:21+j*21,:] * 0
                        #         tmp_joints_pred[j*21:21+j*21,:] = tmp_joints_pred[j*21:21+j*21,:] * 0
                        #     else:
                        #         tmp_joints_gt[j*21:21+j*21,:] = (tmp_joints_gt[j*21:21+j*21,:] -tmp_joints_gt[9+j*21,:])*1000
                        #         tmp_joints_pred[j*21:21+j*21,:] = (tmp_joints_pred[j*21:21+j*21,:] -tmp_joints_pred[9+j*21,:])*1000
                        joints_gt_aligned = tmp_joints_gt.copy()*1000
                        xyz_pred_aligned = tmp_joints_pred.copy()*1000
                    else:
                        joints_gt_aligned = (tmp_joints_gt - tmp_joints_gt[9,:])*1000
                    
                    # verts_gt_aligned = (tmp_verts_gt - tmp_verts_gt[0,:])*1000
                    # select one hand to align for InterHand
                    if self.opt.dataset == 'InterHand' and self.opt.task == 'interact':
                        select = int(data['handtype'][0][0].cpu())
                        # if select == 0:
                        #     continue # jump left hand for test
                        hand_num = 1
                        xyz_pred_aligned = xyz_pred_aligned[select*21:21+select*21,:].copy()
                        joints_gt_aligned = joints_gt_aligned[select*21:21+select*21,:].copy()
                        tmp_lms_pred = tmp_lms_pred[select*21:21+select*21,:].copy()
                        tmp_lms_gt = tmp_lms_gt[select*21:21+select*21,:].copy()
                    # xyz_pred_aligned = align_w_scale(joints_gt_aligned, xyz_pred_aligned) 

                    # xyz_pred_aligned[:21,:] = align_w_scale(joints_gt_aligned[:21,:], xyz_pred_aligned[:21,:]) 
                    # xyz_pred_aligned[21:,:] = align_w_scale(joints_gt_aligned[21:,:], xyz_pred_aligned[21:,:]) 
                    # print('R, s, s1, s2, t1, t2',align_w_scale(joints_gt_aligned, xyz_pred_aligned, True))
                    for j in range(tmp_lms_pred.shape[0]):
                        if tmp_lms_gt[j][0] == 0:
                            continue # remove outliers    
                        mpix[j].append(np.sqrt(np.sum((tmp_lms_pred[j] - tmp_lms_gt[j])**2)))
                        if j < 21:             
                            lmpjpe[j].append(np.sqrt(np.sum((xyz_pred_aligned[j] - joints_gt_aligned[j])**2)))
                        else:
                            rmpjpe[j-21].append(np.sqrt(np.sum((xyz_pred_aligned[j] - joints_gt_aligned[j])**2)))
                  
                    # for j in range(778):
                    #     mpvpe[j].append(np.sqrt(np.sum((verts_pred_aligned[j] - verts_gt_aligned[j])**2)))
                    if False:
                        verts_gt_aligned = verts_gt[i][0].reshape(-1,3).cpu().numpy()*1000
                        eval_main(joints_gt_aligned,verts_gt_aligned,xyz_pred_aligned,verts_pred_aligned,'./')                        
                # if args.phase == 'eval':
                #     save_a_image_with_mesh_joints(inv_base_tranmsform(data['img'][0].cpu().numpy())[:, :, ::-1], mask_pred, poly, data['K'][0].cpu().numpy(), vertex, self.faces[0], uv_point_pred[0], vertex2xyz,
                #                               os.path.join(args.out_dir, 'eval', str(step) + '_plot.jpg'))
                bar.suffix = '({batchs}/{size})' .format(batchs=step+1, size=len(eval_loader))
                bar.next()
        bar.finish()

        if self.opt.dataset == 'InterHandNew':
            print(lms_loss_all/len(eval_loader))
            print(left_joints_loss_all/len(eval_loader))
            print(left_verts_loss_all/len(eval_loader))
            print(right_joints_loss_all/len(eval_loader))
            print(right_verts_loss_all/len(eval_loader))            
            return None, None
        # append the last term.
        H2O_list.update({'{}'.format(action_id): local_list})

        if False: # test for evaluation score
            eval_main(xyz_gt_list,verts_gt_list,xyz_pred_list,verts_pred_list,'./')  
            return None, None                      

        eval_summary = 'MPJPE for each joint: \n'
        score_path = os.path.join(self.opt.root_dir, 'H2O-test.txt')   

        if joints_gt is None:
            for j in range(21*hand_num):
                mpix[j] = np.mean(np.stack(mpix[j]))
                # joint_name = self.skeleton[j]['name']
                print('lms_{0}:{1}'.format(j,mpix[j])) 
            # print(eval_summary)
            print('MPJPE_lms: %.2f' % (np.mean(mpix[:63])))
            with open(score_path, 'a') as fo:
                fo.write('UV_mean2d: %f\n' % np.mean(mpix[:63]))
            print('Scores written to: %s' % score_path)

            return None, None

        for j in range(21*hand_num):
            mpix[j] = np.mean(np.stack(mpix[j]))
            print('lms_{0}:{1}'.format(j,mpix[j])) 
        for j in range(21*1):
            lmpjpe[j] = np.mean(np.stack(lmpjpe[j]))
            rmpjpe[j] = np.mean(np.stack(rmpjpe[j]))
            print('ljoint_{0}:{1}'.format(j,lmpjpe[j])) 
            print('rjoint_{0}:{1}'.format(j,rmpjpe[j])) 
        # print(eval_summary)
        print('MPJPE_ljoint: %.2f, MPJPE_rjoint: %.2f' % (np.mean(lmpjpe[:]),np.mean(rmpjpe[:])))
        print('MPJPE_lms: %.2f' % (np.mean(mpix[:42])))
        with open(score_path, 'a') as fo:
            fo.write('UV_mean2d: %f\n' % np.mean(mpix[:42]))
            fo.write('UV_mean3d_left: %f\n, UV_mean3d_right: %f\n' % (np.mean(lmpjpe[:]),np.mean(rmpjpe[:])))
        print('Scores written to: %s' % score_path)

        ### save to json file for submitting.
        # xyz_pred_list = [x.tolist() for x in xyz_pred_list]
        # verts_pred_list = [x.tolist() for x in verts_pred_list]
        with open(os.path.join(self.opt.root_dir, 'hand_poses' + '.json'), 'w') as fo:
            json.dump(H2O_list, fo)
        print('Save json file at ' + os.path.join(self.opt.root_dir, 'hand_poses' + '.json'))
    
        return None, None

    def val(self, test_loader, logger=None):
        model_with_loss = self.model_with_loss
        model_with_loss.eval()

        torch.cuda.empty_cache()
        xyz_pred_list, verts_pred_list = list(), list()
        xyz_gt_list, verts_gt_list = list(), list()
        bar = Bar("TEST", max=len(test_loader))

        hand_num = 1 # or 2. modified according to your model.
        mpjpe = [[] for _ in range(21*hand_num)] # treat right and left hand identical 
        mpix = [[] for _ in range(21*hand_num)]   
        mpvpe = [[] for _ in range(778)] # treat right and left hand identical   
        with torch.no_grad():
            for step, data in enumerate(test_loader):
                for k in data:
                    if k != 'meta':# and k != 'dataset':
                        data[k] = data[k].to(device=self.opt.device, non_blocking=True)

                vertex_pred, joints_pred, vertex_gt, joints_gt, lms21_pred, handness  = model_with_loss(data,'val')    

                # gt_hand_type = data['hand_type'][0][0]
                if 'lms' in data:
                    lms_gt = data['lms'].view_as(lms21_pred)
                else:
                    lms_gt = torch.cat((data['lms_left_gt'],data['lms_right_gt'])).view_as(lms21_pred)
                if joints_gt is None:
                    joints_gt = data['joints'] 
                                              
                for i in range(lms_gt.shape[0]):
                    tmp_lms_gt = (lms_gt)[i].reshape(-1,2).cpu().numpy()
                    tmp_lms_pred = (lms21_pred)[i].reshape(-1,2).cpu().numpy()
                    tmp_joints_pred = (joints_pred)[i].reshape(-1,3).cpu().numpy()
                    # tmp_verts_pred = vertex_pred[i].reshape(-1,3).cpu().numpy()
                    tmp_joints_gt = joints_gt[i].reshape(-1,3).cpu().numpy()
                    # tmp_xyz_gt = xyz_gt[i][0].reshape(-1,3).cpu().numpy()
                    # tmp_verts_gt = vertex_gt[i].reshape(-1,3).cpu().numpy()
                    index_root_bone_length = np.sqrt(np.sum((tmp_joints_pred[10, :] - tmp_joints_pred[9, :])**2))
                    gt_bone_length = np.sqrt(np.sum((tmp_joints_gt[10, :] - tmp_joints_gt[9, :])**2))
                    # xyz_pred_aligned = (tmp_joints_pred - tmp_joints_pred[9,:])/index_root_bone_length*gt_bone_length*1000
                    # verts_pred_aligned = (tmp_verts_pred - tmp_verts_pred[9,:])/index_root_bone_length*gt_bone_length*1000 

                    if tmp_joints_pred.shape[0] > 21 and tmp_joints_gt.shape[0] > 21:
                        # here we consider RHD two hand case
                        # for j in range(len(data['valid'][i])):
                        #     if data['valid'][i][j] == 0: # no visible hand in left/right
                        #         tmp_joints_gt[j*21:21+j*21,:] = tmp_joints_gt[j*21:21+j*21,:] * 0
                        #         tmp_joints_pred[j*21:21+j*21,:] = tmp_joints_pred[j*21:21+j*21,:] * 0
                        #     else:
                        #         tmp_joints_gt[j*21:21+j*21,:] = (tmp_joints_gt[j*21:21+j*21,:] -tmp_joints_gt[9+j*21,:])*1000
                        #         tmp_joints_pred[j*21:21+j*21,:] = (tmp_joints_pred[j*21:21+j*21,:] -tmp_joints_pred[9+j*21,:])*1000
                        joints_gt_aligned = tmp_joints_gt.copy()*1000
                        xyz_pred_aligned = tmp_joints_pred.copy()*1000
                    else:
                        joints_gt_aligned = (tmp_joints_gt - tmp_joints_gt[9,:])*1000
                        xyz_pred_aligned = (tmp_joints_pred - tmp_joints_pred[9,:])*1000
                    
                    # verts_gt_aligned = (tmp_verts_gt - tmp_verts_gt[0,:])*1000
                    # select one hand to align for InterHand
                    if self.opt.dataset == 'InterHand' and self.opt.task == 'interact':
                        select = int(data['handtype'][0][0].cpu())
                        # if select == 0:
                        #     continue # jump left hand for test
                        hand_num = 1
                        xyz_pred_aligned = xyz_pred_aligned[select*21:21+select*21,:].copy()
                        joints_gt_aligned = joints_gt_aligned[select*21:21+select*21,:].copy()
                        tmp_lms_pred = tmp_lms_pred[select*21:21+select*21,:].copy()
                        tmp_lms_gt = tmp_lms_gt[select*21:21+select*21,:].copy()
                    # xyz_pred_aligned = align_w_scale(joints_gt_aligned, xyz_pred_aligned) 
                    # xyz_pred_aligned[:21,:] = align_w_scale(joints_gt_aligned[:21,:], xyz_pred_aligned[:21,:]) 
                    # xyz_pred_aligned[21:,:] = align_w_scale(joints_gt_aligned[21:,:], xyz_pred_aligned[21:,:]) 
                    # print('R, s, s1, s2, t1, t2',align_w_scale(joints_gt_aligned, xyz_pred_aligned, True))
                    for j in range(tmp_lms_pred.shape[0]):
                        if tmp_lms_gt[j][0] == 0:
                            continue # remove outliers                        
                        mpix[j].append(np.sqrt(np.sum((tmp_lms_pred[j] - tmp_lms_gt[j])**2)))
                        mpjpe[j].append(np.sqrt(np.sum((xyz_pred_aligned[j] - joints_gt_aligned[j])**2)))
                  
                    # for j in range(778):
                    #     mpvpe[j].append(np.sqrt(np.sum((verts_pred_aligned[j] - verts_gt_aligned[j])**2)))
                    if False:
                        verts_gt_aligned = verts_gt[i][0].reshape(-1,3).cpu().numpy()*1000
                        eval_main(joints_gt_aligned,verts_gt_aligned,xyz_pred_aligned,verts_pred_aligned,'./')                        
                # if args.phase == 'eval':
                #     save_a_image_with_mesh_joints(inv_base_tranmsform(data['img'][0].cpu().numpy())[:, :, ::-1], mask_pred, poly, data['K'][0].cpu().numpy(), vertex, self.faces[0], uv_point_pred[0], vertex2xyz,
                #                               os.path.join(args.out_dir, 'eval', str(step) + '_plot.jpg'))
                bar.suffix = '({batchs}/{size})' .format(batchs=step+1, size=len(test_loader))
                bar.next()
        bar.finish()

        if False: # test for evaluation score
            eval_main(xyz_gt_list,verts_gt_list,xyz_pred_list,verts_pred_list,'./')  
            return None, None                      

        eval_summary = 'MPJPE for each joint: \n'
        score_path = os.path.join(self.opt.root_dir, 'H2O-try.txt')   
        if self.opt.task == 'artificial': # there may be less than 10 hands in artificial and got error
            hand_num = 3
        if joints_gt is None:
            for j in range(21*hand_num):
                mpix[j] = np.mean(np.stack(mpix[j]))
                # joint_name = self.skeleton[j]['name']
                print('lms_{0}:{1}'.format(j,mpix[j])) 
            # print(eval_summary)
            print('MPJPE_lms: %.2f' % (np.mean(mpix[:63])))
            with open(score_path, 'a') as fo:
                fo.write('UV_mean2d: %f\n' % np.mean(mpix[:63]))
            print('Scores written to: %s' % score_path)

            return None, None

        for j in range(21*hand_num):
            mpjpe[j] = np.mean(np.stack(mpjpe[j]))
            mpix[j] = np.mean(np.stack(mpix[j]))
            # joint_name = self.skeleton[j]['name']
            print('joint_{0}:{1}'.format(j,mpjpe[j])) 
            print('lms_{0}:{1}'.format(j,mpix[j])) 
        # print(eval_summary)
        print('MPJPE_joint: %.2f' % (np.mean(mpjpe[:42])))
        print('MPJPE_lms: %.2f' % (np.mean(mpix[:42])))
        with open(score_path, 'a') as fo:
            fo.write('UV_mean2d: %f\n' % np.mean(mpix[:42]))
            fo.write('UV_mean3d: %f\n' % np.mean(mpjpe[:42]))
        print('Scores written to: %s' % score_path)

        return None, None
