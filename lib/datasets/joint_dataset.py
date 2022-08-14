from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from posixpath import splitdrive

import numpy as np
import os
import pickle
import torch.utils.data as data

dataset_index = {
  'Joint': 0,
  'FreiHAND': 1,
  'HO3D': 2,
  'HO3Dv3': 3,
  'OneHand10K': 4,
  'InterHand': 5,
  'RHD': 6,
  'Others': 7,
}

class BaseDataset(data.Dataset):
    num_classes = 1 # hand
    mean = np.array([0.485, 0.456, 0.406],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(BaseDataset, self).__init__()
        self.split = split
        self.flip_idx = [[0, 1], [3, 4]]
        self.opt = opt
        self.update()

    def update(self):
        self.opt.size_train = self.default_resolution
        self.dataset_index = dataset_index

    def __len__(self):
        pass

class JointDataset(BaseDataset):

    def __init__(self, opt, split):
        super(JointDataset, self).__init__(opt, split)
        self.split = split
        self.min_size, self.max_size = 64, 320

