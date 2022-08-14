from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys


class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    # basic experiment setting
    self.parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
    self.parser.add_argument('--nproc_per_node', default=-1, type=int,
                    help='node rank for distributed training')
    self.parser.add_argument('--node_rank', default=-1, type=int,
                    help='node rank for distributed training')                    
    self.parser.add_argument('task', default='detreg',
                             help='detreg')
    self.parser.add_argument('--dataset', default='Joint',
                             help='widerface, fddb or joint dataset')
    self.parser.add_argument('--exp_id', default='default')
    self.parser.add_argument('--load_model', default='',
                             help='path to pretrained model')                         
    self.parser.add_argument('--using_pca', action='store_true',
                             help='only use pca coeff if set true')    
    self.parser.add_argument('--num_pca_comps', type=int, default=30,
                             help='number of pca main components') 
    self.parser.add_argument('--mode', default='train',
                             help='train, eval, test')
    self.parser.add_argument('--pick_hand', action='store_true',
                             help='which_hand in image, left 0 or right 1')
    self.parser.add_argument('--heatmaps', action='store_true',
                             help='whether use extra heatmap feature')                                                                                  
    self.parser.add_argument('--iterations', action='store_true',
                             help='whether use iterations for theta pred')                              
    self.parser.add_argument('--avg_center', action='store_true',
                             help='whether use avg_center for theta pred')                               
    self.parser.add_argument('--latent_heatmap', action='store_true',
                             help='whether use latent_heatmap for theta pred') 
    self.parser.add_argument('--config_info', default='nothing',
                             help='show config informations for debug')                         
    # system
    self.parser.add_argument('--gpus', default='0',
                             help='-1 for CPU, use comma for multiple gpus')
    self.parser.add_argument('--num_workers', type=int, default=8,
                             help='dataloader threads. 0 for single-thread.')
    self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                             help='disable when the input size is not fixed.')
    self.parser.add_argument('--seed', type=int, default=317,
                             help='random seed')  # from CornerNet

    # log
    self.parser.add_argument('--print_iter', type=int, default=20,
                             help='disable progress bar and print to screen.')

    # model
    self.parser.add_argument('--arch', default='csp_50',
                             help='model architecture. Currently tested'
                                  'res_18 | res_101 | resdcn_18 | resdcn_101 |'
                                  'dlav0_34 | dla_34 | hourglass')
    self.parser.add_argument('--head_conv', type=int, default=256,
                             help='conv layer channels for output head'
                                  '0 for no conv layer'
                                  '-1 for default setting: '
                                  '64 for resnets and 256 for dla.')
    self.parser.add_argument('--down_ratio', type=int, default=8,
                             help='output stride.')

    # train
    self.parser.add_argument('--lr', type=float, default=1e-3,
                             help='learning rate for batch size 32.')
    self.parser.add_argument('--lr_step', type=str, default='30,90,120',
                             help='drop learning rate by 10.')
    self.parser.add_argument('--num_epochs', type=int, default=150,
                             help='total training epochs.')
    self.parser.add_argument('--batch_size', type=int, default=32,
                             help='batch size')
    self.parser.add_argument('--master_batch_size', type=int, default=-1,
                             help='batch size on the master gpu.')
    self.parser.add_argument('--num_iters', type=int, default=-1,
                             help='default: #samples / batch_size.')
    self.parser.add_argument('--K', type=int, default=20,
                             help='max number of objects.')

    # ctdet
    self.parser.add_argument('--reg_loss', default='l1',
                             help='regression loss: sl1 | l1 | l2')
    self.parser.add_argument('--center_weight', type=float, default=20.,
                             help='loss weight for center heatmaps.')
    self.parser.add_argument('--heatmaps_weight', type=float, default=10.,
                             help='loss weight for keypoint heatmaps.')  

    # self.parser.add_argument('--off_weight', type=float, default=1,
    #                          help='loss weight for keypoint local offsets.')
    # self.parser.add_argument('--wh_weight', type=float, default=0.1,
    #                          help='loss weight for bounding box size.')

    self.parser.add_argument('--optimizer', type=str, default='Adam',
                             help='optimizer, SGD or Adam')
    self.parser.add_argument('--extra', type=str, default='',
                             help='extra info to show')
    self.parser.add_argument('--cache_path', type=str, default='data',
                             help='regress w and h together for crowdhuman dataset')
    self.parser.add_argument('--output_path', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
                             help='regress w and h together for crowdhuman dataset')
    self.parser.add_argument('--start_epoch', type=int, default=0,
                             help='start epoch.')

    self.parser.add_argument('--reproj_loss', action='store_true',
                             help='re-projection loss')
    self.parser.add_argument('--reproj_weight', type=float, default=20.,
                             help='re-projection_weight for re-projection loss')
    self.parser.add_argument('--photometric_loss', action='store_true',
                             help='photometric loss')
    self.parser.add_argument('--photometric_weight', type=float, default=1.,
                             help='photometric loss for reconstruction for reconstruction')
    self.parser.add_argument('--seg_weight', type=float, default=20.,
                             help='segmentation loss for reconstruction for reconstruction')                             
    self.parser.add_argument('--bone_loss', action='store_true',
                             help='middle bone scale loss')
    self.parser.add_argument('--bone_weight', type=float, default=1.,
                             help='bone_weight for bone scale loss')
    self.parser.add_argument('--bone_dir_weight', type=float, default=50.,
                             help='bone_dir_weight for bone dir loss')    
    self.parser.add_argument('--pose_weight', type=float, default=50.,
                             help='pose_weight for 3d pose loss') 
    self.parser.add_argument('--mano_weight', type=float, default=10.,
                             help='mano_weight for mano params loss')                                                           
    self.parser.add_argument('--use_skin_only', action='store_true',
                             help='if true, we only use skin area')
    self.parser.add_argument('--var_weight', type=float, default=0.001,
                             help='skin area variance')
    self.parser.add_argument('--norm_weight', type=float, default=100.,
                             help='normalization')
    self.parser.add_argument('--perceptual_loss', action='store_true',
                             help='using facenet to add perceptual loss')
    self.parser.add_argument('--perceptual_weight', type=float, default=0.01,
                             help='loss weight for perceptual loss')
    self.parser.add_argument('--pre_fix', type=str, default='data',
                             help='dataset path')
    self.parser.add_argument('--BFM', type=str, default='BFM/mSEmTFK68etc.chj',
                             help='BFM model file')
    self.parser.add_argument('--cv2_show', action='store_true',
                             help='show using opencv')
    self.parser.add_argument('--debug', action='store_true',
                             help='if debug, create folder in debug')
    self.parser.add_argument('--gradual_lr', action='store_true',
                             help='if true, we will gradually reduce reproj_weight and increase photo_weight,'
                                  'the starting weights of these terms will also be 10, 100 times (larger or smaller).'
                                  'after 10 epochs, the lr will reduce to the default and keep fixed in the'
                                  'remaining training.')

    self.parser.add_argument('--no_det', action='store_true',
                             help='if true, only reconstruction')
    self.parser.add_argument('--switch', action='store_true',
                             help='if true, randomly use gt_fcts or est_fcts')
    self.parser.add_argument('--num_stacks', type=int, default=1,
                             help='whether to refine')

    self.parser.add_argument('--gcn_decoder', action='store_true',
                             help='use gcn decoder')

    # ---------- for coma -------------
    self.parser.add_argument('--n_layers', default=5, type=int,
                             help='number of layers for mesh down(up)-sampling')
    self.parser.add_argument('--in_channels', default=3, type=int,
                             help='dimension of V/T, both all 3')
    # G_z should be doubled, if separately train shape2shape and texture2texture, in parse
    self.parser.add_argument('--z', default=64, type=int,
                             help='dimension of latent vectors')
    self.parser.add_argument('--downsampling_factors', default='4, 4, 4, 4, 4', type=str,
                             help='downsampling factors for each downsampling layer')
    self.parser.add_argument('--num_conv_filters', default='16, 16, 32, 32, 64, 64',
                             help='number of conv_filters of each layer')
    self.parser.add_argument('--polygon_order', default='6, 6, 6, 6, 6, 6', type=str,
                             help='polygon_order of each layer')
    self.parser.add_argument('--template_fname', default='template/template.obj', type=str,
                             help='file path of template')
    self.parser.add_argument('--coma_cache', default='bfm_cache/var.pkl', type=str,
                             help='pre-stored down(up)-sampling cache')
    self.parser.add_argument('--activation', default='LeakyReLU', type=str,
                             help='activation function for gcn, Sine, ReLU or LeakyReLU')
    # by default, it will be 2 networks for shape and texture separately,
    # set no_S2S or no_T2T to train a single shape or texture,
    # set in_channels to 6 will jointly train a shape and texture ae
    self.parser.add_argument('--load_ST_model', default='',
                             help='path to pretrained ST2ST split model')
    self.parser.add_argument('--load_S_model', default='',
                             help='path to pretrained S2S model')
    self.parser.add_argument('--load_T_model', default='',
                             help='path to pretrained T2T model')
    self.parser.add_argument('--load_G_model', default='',
                             help='path to pretrained Generator model')

    # ---------- for mesh loss -------------
    self.parser.add_argument('--gcn_network', default='Coma', type=str,
                             help='gcn_network, coma or GResNet')
    self.parser.add_argument('--center_only', action='store_true',
                             help='only train a center prediction network')
    self.parser.add_argument('--off', action='store_true',
                             help='auxiliary offset regression branch')
    self.parser.add_argument('--discrepancy', action='store_true',
                             help='enforce a discrepancy loss to mitigate the side effect of two hands near to each other')
    self.parser.add_argument('--default_resolution', type=int, default=384,
                             help='the default training resolution, 224 for no_det, 384 for single hand and 512 for multiple hand.')
    self.parser.add_argument('--discrepancy_weight', type=float, default=0.01,
                             help='default discrepancy weight')
    self.parser.add_argument('--brightness', action='store_true',
                             help='color distortion.')

  def parse(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)

    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]

    # opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

    if opt.master_batch_size == -1:
      opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = (opt.batch_size - opt.master_batch_size)
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
      slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
      if i < rest_batch_size % (len(opt.gpus) - 1):
        slave_chunk_size += 1
      opt.chunk_sizes.append(slave_chunk_size)
    print('training chunk_sizes:', opt.chunk_sizes)

    # opt.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    opt.root_dir = opt.output_path
    print(opt.root_dir)
    # opt.data_dir = os.path.join(opt.root_dir, 'data')
    opt.exp_dir = os.path.join(opt.root_dir, 'outputs/logs', opt.task)
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    print('The output will be saved to ', opt.save_dir)

    # if opt.load_model != '':
    #   opt.load_model = os.path.join(opt.save_dir, opt.load_model)
    if opt.load_ST_model != '':
      opt.load_ST_model = os.path.join(opt.save_dir, opt.load_ST_model)
    return opt

  @staticmethod
  def update_dataset_info_and_set_heads(opt, dataset):
    dataset.default_resolution = [opt.default_resolution, opt.default_resolution]
    opt.input_res = dataset.default_resolution[0]
    opt.mean, opt.std = dataset.mean, dataset.std
    opt.num_classes = dataset.num_classes

    if opt.task in ['simplified', 'artificial','interact']:
      opt.heads = {'hm': 1}
      opt.heads.update({'params': (61)*2})
      if opt.photometric_loss:
        opt.heads.update({'texture': 778*3})
        opt.heads.update({'light': 27})
      if opt.pick_hand:
        opt.heads.update({'handmap': 2}) # left 0,right 1
        # opt.heads.update({'keypoints': 21}) 
      # modify 257 channel to 61/46/22 for hands model
      if opt.gcn_decoder:
        opt.heads.update({'gcn_params': 128})
      if opt.off:
        opt.heads.update({'off': 136})
    else:
      assert 0, 'task not defined!'
    print('heads', opt.heads)
    return opt
