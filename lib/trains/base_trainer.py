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

# This class can be removed
class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch, mode):
        # if 'ind' not in batch:
        #     outputs, ind = self.model(batch['input'], None, None)
        # else:
        #     outputs, ind = self.model(batch['input'], batch['heatmaps'], batch['ind'])
        if mode == 'train':
            if 'heatmaps' in batch:
                tmp_heatmaps = batch['heatmaps']
            else:
                tmp_heatmaps = None
            outputs, ind = self.model(batch['input'], tmp_heatmaps, batch['ind'])
            loss, loss_stats, rendered, masks = self.loss(outputs, batch)
        elif mode == 'test':
            # loss, loss_stats, rendered, masks = self.test(outputs, batch)
            outputs, ind = self.model(batch['input'], None, None)
            vertex_pred, joints_pred, vertex_gt, joints_gt, lms21_pred = self.loss.test(outputs, ind, batch)
            return vertex_pred, joints_pred, vertex_gt, joints_gt, lms21_pred

        return outputs, loss, loss_stats, rendered, masks


class BaseTrainer(object):
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        if optimizer:
            self.loss_stats, self.loss = self._get_losses(opt)
            self.model_with_loss = ModleWithLoss(model, self.loss)
        self.model = model
        self.optimizer.add_param_group({'params': self.loss.parameters()})

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            if self.optimizer:
                self.model_with_loss = DataParallel(
                    self.model_with_loss, device_ids=gpus,
                    chunk_sizes=chunk_sizes).to(device)
        else:
            if self.optimizer:
                self.model_with_loss = self.model_with_loss.to(device)
            self.model = self.model.to(device)

    def run_epoch(self, phase, epoch, data_loader, logger=None):
        pass

    def _get_losses(self, opt):
        raise NotImplementedError

    def train(self, epoch, data_loader, logger=None):
        return self.run_epoch('train', epoch, data_loader, logger=logger)

    def evaluation(self, eval_loader, logger=None):
        pass

    def test(self, test_loader, logger=None):
        pass
