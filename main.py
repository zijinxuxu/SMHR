from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model
from logger import Logger
from datasets.artificial import ArtificialDataset
from datasets.simplified import SimplifiedDataset
from datasets.interhand import InterHandDataset
from datasets.joint_dataset import JointDataset
from trains.simplified import SimplifiedTrainer
from torch.utils.data.sampler import *
from utils.utils import load_model, save_model
import time

import random
import numpy as np
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler

def get_dataset(task):
  if task == 'simplified':
    class Dataset(JointDataset, SimplifiedDataset):
      pass
  elif task == 'interact':
    class Dataset(JointDataset, InterHandDataset):
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
  seed_torch(opt.seed)
  # torch.manual_seed(opt.seed)
  # torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark
  Dataset = get_dataset(opt.task)
  opt = opts.update_dataset_info_and_set_heads(opt, Dataset)
  # print(opt)

  opt.time_str = time.strftime('%Y-%m-%d-%H-%M') + '-' + opt.extra
  logger = Logger(opt) if opt.local_rank ==0 else None

  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)

  optimizer = torch.optim.Adam(model.parameters(), opt.lr)

  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(model, opt.load_model, optimizer)

  # set dist training
  dist.init_process_group(backend='nccl', init_method='env://')
  torch.cuda.set_device(opt.local_rank)
  global_rank = dist.get_rank()

  Trainer = SimplifiedTrainer
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device,opt.local_rank)
  print('Setting up data...') 

  train_dataset = Dataset(opt, opt.mode)
  train_sampler = DistributedSampler(train_dataset)
  train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    # shuffle=True,
    shuffle=(train_sampler is None), 
    sampler=train_sampler,
    num_workers=opt.num_workers,
    pin_memory=True,
    drop_last=True, # True
  )
  if opt.local_rank ==0: # only write once
    pick_mode = 'val' if opt.task == 'artificial' else 'test'
    test_dataset = Dataset(opt, pick_mode)
    test_loader = torch.utils.data.DataLoader(
      test_dataset,
      batch_size=1,
      shuffle=False,
      num_workers=0,
      pin_memory=False,
      drop_last=False # True
    )
  dist.barrier()
  if opt.mode == 'train' or opt.mode == 'train_3d':
    print('Starting training...')

    start_epoch = opt.start_epoch
    save_dir = opt.save_dir if not opt.debug else opt.debug_dir
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
      train_sampler.set_epoch(epoch)
      if opt.reproj_loss:
        log_dict_train, _ = trainer.train(epoch, train_loader, logger)
      else:
        log_dict_train, _ = trainer.train(epoch, train_loader)
      if opt.local_rank ==0: # only write once
        if opt.reproj_loss and (epoch-1) % 10 == 0:
          log_dict_test, _ = trainer.evaluation(test_loader)

        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
          if not k in ['img', 'rendered', 'masks']:
            logger.write('{} {:8f} | '.format(k, v))

        logger.write('\n')
        save_iter = 5 #if opt.photometric_loss else 10 # according to datasize, ep 150k itera; Inter= 1361k/128=10k=15; HO3d=60k/64=1k=150
        if epoch > 0 and epoch % save_iter == 0:
          save_model(os.path.join(save_dir, 'logs_{}'.format(opt.time_str), 'model_{}.pth'.format(epoch)),
                    epoch, model, optimizer)
          if opt.gcn_decoder:
            save_model(os.path.join(save_dir, 'logs_{}'.format(opt.time_str), 'gcn_{}.pth'.format(epoch)),
                      epoch, trainer.render.gcn_decoder, buffer_remove=True)
      dist.barrier()
      if False:
        lr = opt.lr * 0.96 ** epoch
        print('Drop LR to', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr   
      else:           
        if epoch in opt.lr_step:
          # save_model(os.path.join(save_dir, 'logs_{}'.format(opt.time_str), 'model_{}.pth'.format(epoch)),
          #           epoch, model, optimizer)
          lr = opt.lr * (0.1** (opt.lr_step.index(epoch) + 1))
          print('Drop LR to', lr)
          for param_group in optimizer.param_groups:
              param_group['lr'] = lr
    if logger is not None:
      logger.close()  

  elif opt.mode == 'test':
    if opt.local_rank ==0: # only write once    
      # eval_loader = torch.utils.data.DataLoader(
      #     train_dataset,
      #     batch_size=1,
      #     shuffle=False,
      #     num_workers=0,
      #     pin_memory=False,
      #   )
      if opt.reproj_loss:
        log_dict_train, _ = trainer.evaluation(test_loader)
    dist.barrier()

  elif opt.mode == 'val':
    if opt.local_rank ==0: # only write once
      test_loader = torch.utils.data.DataLoader(
          train_dataset,
          batch_size=opt.batch_size,
          shuffle=False,
          num_workers=0,
          pin_memory=False,
          drop_last=True,
        )
      if opt.reproj_loss:
        log_dict_train, _ = trainer.evaluation(test_loader)
    dist.barrier()
  else:
    raise NotImplementedError

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
  dist.destroy_process_group()



