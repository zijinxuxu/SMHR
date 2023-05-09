import os
import torch
import torch.nn as nn
import numpy as np
import math
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer import (
  RasterizationSettings,
  MeshRasterizer,
  MeshRenderer,
  SoftSilhouetteShader,
  HardFlatShader,
  SfMPerspectiveCameras,
  SoftPhongShader,
  HardPhongShader,
  HardGouraudShader,
  BlendParams,
  DirectionalLights,
)
from lib.models.hand3d.RenderDepthRgbMask import RenderDepthRgbMask
from lib.models.hand3d.Mano_model import ManoModel
from lib.models.hand3d.Mano_model import to_np
from lib.models.networks.manolayer import ManoLayer, rodrigues_batch

class ManoRender(nn.Module):
  def __init__(self, opt):
    super(ManoRender, self).__init__()
    self.opt = opt
    if type(opt.input_res).__name__ == 'list':
      self.input_res = opt.input_res[0]
    else:
      self.input_res = opt.input_res

    self.rhm_path = os.path.join(os.path.dirname(__file__), 'mano_core/MANO_RIGHT.pkl')
    self.lhm_path = os.path.join(os.path.dirname(__file__), 'mano_core/MANO_LEFT.pkl')
    mano_path = {'left': self.lhm_path, 'right': self.rhm_path}   
    # pca = 30 is saturates as <Learning joint reconstruction of hands and manipulated objects> says
    n_comps = 45 if not opt.using_pca else opt.num_pca_comps # modified to 6 from 30/45 
    # if self.opt.dataset == 'HO3D':
    #   bool_flat_hand_mean = True
    # else:
    bool_flat_hand_mean = False # we reduce it in get_item.
    
        
    self.MANO_R = ManoModel(model_path=self.rhm_path,
                  num_pca_comps=n_comps,
                  flat_hand_mean=bool_flat_hand_mean,
                  use_pca=opt.using_pca)
    self.MANO_GT = ManoModel(model_path=self.rhm_path,
                  num_pca_comps=n_comps,
                  flat_hand_mean=bool_flat_hand_mean,   # False for FreiHAND and True for HO3D
                  use_pca=opt.using_pca) 
    self.MANO_L = ManoModel(model_path=self.lhm_path,
              num_pca_comps=n_comps,
              flat_hand_mean=bool_flat_hand_mean,
              use_pca=opt.using_pca)
    

    self.mano_layer_left = ManoLayer(mano_path['left'], center_idx=None, use_pca=False)
    self.mano_layer_right = ManoLayer(mano_path['right'], center_idx=None, use_pca=False)

    # this is for InterHandNew dataset, shape. H2O dataset may not neccessary.
    # if torch.sum(torch.abs(self.MANO_L.shapedirs[:,0,:] - self.MANO_R.shapedirs[:,0,:])) < 1:
    #     print('Fix shapedirs bug of MANO')
    #     self.MANO_L.shapedirs[:,0,:] *= -1

    weight = np.array([20,20,1,1,1,1,1,1,20,20,
                      1,1,1,1,1,1,20,20,
                      1,1,1,1,1,1,20,20,
                      1,1,1,1,1,1,20,20,
                      1,1,1,1,1,1,20,20], dtype=np.float32)
    self.register_buffer('weighted_lms', torch.from_numpy(weight.reshape([1, 1, 42])))

    ## renderer for visualization
    fx = 512#512 #1015.
    fy = 512#512 #1015.
    self.f = fx

    if type(opt.input_res).__name__ == 'list':
      cx = opt.input_res[0] / 2
      cy = opt.input_res[1] / 2
    else:
      cx = self.input_res / 2
      cy = self.input_res / 2

    self.cx , self.cy = cx, cy
    K = [[fx, 0., cx],
        [0., fy, cy],
        [0., 0., 1.]]
    self.register_buffer('K', torch.FloatTensor(K))
    self.register_buffer('inv_K', torch.inverse(self.K).unsqueeze(0))
    self.K = self.K.unsqueeze(0)
    self.set_Illu_consts()

    # for pytorch3d
    self.t = torch.zeros([1, 3], dtype=torch.float32)
    self.pt = torch.zeros([1, 2], dtype=torch.float32)
    self.fl = 512. * 2 / self.input_res,
    ptR = [[[-1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]]]
    self.ptR = torch.FloatTensor(ptR)

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0, 0, 0))
    raster_settings = RasterizationSettings(
      image_size=self.input_res,
      # blur_radius=1e-6,
      blur_radius=0,
      faces_per_pixel=1,
      max_faces_per_bin=1000000,
    )

    # renderer
    B = opt.batch_size #// len(opt.gpus)
    self.B = B
    cameras = SfMPerspectiveCameras(focal_length=self.fl,
                                    R=self.ptR.expand(B, -1, -1),
                                    device='cuda')
    rasterizer = MeshRasterizer(raster_settings=raster_settings)
    shader_rgb = HardFlatShader(blend_params=blend_params)
    self.renderer = RenderDepthRgbMask(rasterizer, shader_rgb, SoftSilhouetteShader(), cameras, len(opt.gpus) != 1)
    opt.activation = torch.nn.LeakyReLU(0.1) if opt.activation == 'LeakyReLU' else None

  def forward(self, meshes):
    image, mask, fragments = self.renderer(meshes)
    image = image[..., :3]
    mask = (mask[..., -1] > 0).float()
    depth = fragments.zbuf
    return image, mask, depth

  def decode_translation(self, translation, index):
        # estimated a, b, c
    # (c + 10)(a + cx - w/2) / f, (c + 10)(b + h/2 - cy) / f, c + 10 --> t_x, t_y, t_z
    trans = torch.zeros(translation.size()).cuda()
    cx = (index % (self.input_res // self.opt.down_ratio) + 0.5) * self.opt.down_ratio
    cy = (index // (self.input_res // self.opt.down_ratio) + 0.5) * self.opt.down_ratio
    w, h = self.input_res, self.input_res
    trans[:, 0] = (translation[:, 2] + 10) * (translation[:, 0] + cx - w/2) / self.f
    trans[:, 1] = (translation[:, 2] + 10) * (translation[:, 1] + h/2 - cy) / self.f
    trans[:, 2] = translation[:, 2] + 10
    # print(trans.mean(dim=0).detach().cpu().numpy())
    return trans

  def Split_coeff(self, theta, index):
    if self.opt.using_pca:
      num_pca = self.opt.num_pca_comps
      global_orient_coeff_l_up =  theta[:, :3]
      pose_coeff_l =  theta[:, 3:3+num_pca] 
      betas_coeff_l =  theta[:, 3+num_pca:13+num_pca]  # fixed to zero
      global_transl_coeff_l_up = theta[:, 13+num_pca:16+num_pca] / 10 
      global_transl_coeff_l_up[:,2] += 0.6

      global_orient_coeff_r_up =  theta[:, 61:61+3] 
      pose_coeff_r =  theta[:, 61+3:61+3+num_pca] 
      betas_coeff_r =  theta[:, 61+3+num_pca:61+13+num_pca] # fixed to zero
      global_transl_coeff_r_up = theta[:, 61+13+num_pca:61+16+num_pca] / 10 
      global_transl_coeff_r_up[:,2] += 0.6

    else:      
      global_orient_coeff_l_up =  theta[:, :3] 
      pose_coeff_l =  theta[:, 3:48] 
      betas_coeff_l =  theta[:, 48:58] # TEST with shape   
      global_transl_coeff_l_up = theta[:, 58:61] 
      global_transl_coeff_l_up[:,2] = global_transl_coeff_l_up[:,2] + 0.8
      # ---------------left-right------------------
      global_orient_coeff_r_up =  theta[:, 61:64]  
      pose_coeff_r =  theta[:, 64:109]
      betas_coeff_r =  theta[:, 109:119] # TEST with shape
      global_transl_coeff_r_up = theta[:, 119:122] 
      global_transl_coeff_r_up[:,2] = global_transl_coeff_r_up[:,2] + 0.8

    # attention for tranlation:
    # (c + 10)(a + cx - w/2) / f, (c + 10)(b + h/2 - cy) / f, c + 10 --> t_x, t_y, t_z    
    cx = (index % (self.input_res // self.opt.down_ratio)) * self.opt.down_ratio
    cy = (index // (self.input_res // self.opt.down_ratio)) * self.opt.down_ratio
    # cx = (index % (self.input_res // self.opt.down_ratio) + off_hm_pred[:,0]) * self.opt.down_ratio
    # cy = (index // (self.input_res // self.opt.down_ratio) + off_hm_pred[:,1]) * self.opt.down_ratio    
    w, h = self.input_res, self.input_res
    ret_global_transl_coeff_r_up_x = global_transl_coeff_r_up[:, 2] * (global_transl_coeff_r_up[:, 0] + cx - w/2) / self.f
    ret_global_transl_coeff_r_up_y = global_transl_coeff_r_up[:, 2] * (global_transl_coeff_r_up[:, 1] + cy - h/2) / self.f
    ret_global_transl_coeff_r_up = torch.cat((ret_global_transl_coeff_r_up_x.unsqueeze(1), ret_global_transl_coeff_r_up_y.unsqueeze(1), global_transl_coeff_r_up[:,2].unsqueeze(1)),1)
    ret_global_transl_coeff_l_up_x = global_transl_coeff_l_up[:, 2] * (global_transl_coeff_l_up[:, 0] + cx - w/2) / self.f
    ret_global_transl_coeff_l_up_y = global_transl_coeff_l_up[:, 2] * (global_transl_coeff_l_up[:, 1] + cy - h/2) / self.f
    ret_global_transl_coeff_l_up = torch.cat((ret_global_transl_coeff_l_up_x.unsqueeze(1), ret_global_transl_coeff_l_up_y.unsqueeze(1), global_transl_coeff_l_up[:,2].unsqueeze(1)),1)

    if self.opt.task == 'artificial' or True:
      return global_orient_coeff_l_up, pose_coeff_l, betas_coeff_l, ret_global_transl_coeff_l_up, \
          global_orient_coeff_r_up, pose_coeff_r, betas_coeff_r, ret_global_transl_coeff_r_up
    else:
      return global_orient_coeff_l_up, pose_coeff_l, betas_coeff_l, global_transl_coeff_l_up, \
          global_orient_coeff_r_up, pose_coeff_r, betas_coeff_r, global_transl_coeff_r_up

  def get_Landmarks(self, Shape):
    b = Shape.size(0)
    K = self.K.expand(b, 3, 3)
    projection = Shape.bmm(K.transpose(2, 1)).contiguous()
    projection_ = projection[..., :2] / ( projection[..., 2:])
    return projection_

  def get_Landmarks_new(self, Shape, K):
    b = Shape.size(0)
    K = K.reshape(-1, 3, 3)
    assert K.shape[0] == b
    projection = Shape.bmm(K.transpose(2, 1)).contiguous()
    projection_ = projection[..., :2] / ( projection[..., 2:] + 1e-7)
    return projection_       
  
  def get_uv_root_3d(self, gt_uv_scale, gt_K):
    gt_K = gt_K.reshape(gt_uv_scale.shape[0],1,-1)
    focal = 0.5 * (gt_K[:,0,0] + gt_K[:,0,4])
    if gt_K[-1].max()==0:
      print('hello')
    cx = gt_K[:,0,2]
    cy = gt_K[:,0,5]
    gt_uv_root = gt_uv_scale[:,:2]
    gt_scale = gt_uv_scale[:,2]
    device = gt_uv_root.device
    b = gt_uv_root.size(0)
    # init depth to 0.6
    depth = focal / (gt_scale+1e-6)
    X = (gt_uv_root[:,0] - cx) / (focal+1e-6) * depth
    Y = (gt_uv_root[:,1] - cy) / (focal+1e-6) * depth
    return torch.stack([X,Y,depth], 1)

  def Shape_formation(self, global_orient, pose, betas, global_transl, type, wrist_rotate):
    # wrist_rotate = True
    if type == 'gt':
      output = self.MANO_GT(betas=betas,
              global_orient=global_orient,
              hand_pose=pose,
              transl=global_transl,
              return_verts=True,
              return_tips = True,
              return_full_pose = True,
              using_wrist_rotate=wrist_rotate)
    elif type == 'right':
      output = self.MANO_R(betas=betas,
              global_orient=global_orient,
              hand_pose=pose,
              transl=global_transl,
              return_verts=True,
              return_tips = True,
              return_full_pose = True,
              using_wrist_rotate=wrist_rotate)
    else:
      output = self.MANO_L(betas=betas,
              global_orient=global_orient,
              hand_pose=pose,
              transl=global_transl,
              return_verts=True,
              return_tips = True,
              return_full_pose = True,
              using_wrist_rotate=wrist_rotate) 
    
    vertices = output.vertices
    joints = output.joints
    full_pose = output.full_pose
    return vertices, joints, full_pose

  def Illumination(self, Albedo, canoShape):
    face_norm = self.Compute_norm(canoShape)
    face_color, lighting = self.Illumination_layer(Albedo, face_norm, self.gamma)
    return face_color, lighting

  def Compute_norm(self, face_shape):
      device = face_shape.device
      face_id = torch.from_numpy(self.MANO_R.faces.astype(np.int64)).to(device)
      # face_id = self.BFM.tri  # vertex index for each triangle face, with shape [F,3], F is number of faces
      point_id = self.MANO_R.adj_faces  # adjacent face index for each vertex, with shape [N,8], N is number of vertex
      shape = face_shape
      v1 = shape[:, face_id[:, 0], :]
      v2 = shape[:, face_id[:, 1], :]
      v3 = shape[:, face_id[:, 2], :]
      e1 = v1 - v2
      e2 = v2 - v3
      face_norm = e1.cross(e2, dim=2)  # compute normal for each face
      empty = torch.zeros((face_norm.size(0), 1, 3), dtype=face_norm.dtype, device=face_norm.device)

      face_norm = torch.cat((face_norm, empty), 1)  # concat face_normal with a zero vector at the end
      v_norm = face_norm[:, point_id, :].sum(2)  # compute vertex normal using one-ring neighborhood
      # CHJ: not average, directly normalize
      v_norm = v_norm / (v_norm.norm(dim=2).unsqueeze(2) + 1e-8)  # normalize normal vectors

      return v_norm
      
  def set_Illu_consts(self):
    import numpy as np
    a0 = np.pi
    a1 = 2 * np.pi / np.sqrt(3.0)
    a2 = 2 * np.pi / np.sqrt(8.0)
    c0 = 1 / np.sqrt(4 * np.pi)
    c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
    c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
    d0 = 0.5 / np.sqrt(3.0)

    self.illu_consts = [a0, a1, a2, c0, c1, c2, d0]

  def Illumination_layer(self, face_texture, norm, gamma):
    
    n_b, num_vertex, _ = face_texture.size()
    n_v_full = n_b * num_vertex
    gamma = gamma.view(-1, 3, 9).clone()
    gamma[:, :, 0] += 0.8

    gamma = gamma.permute(0, 2, 1).contiguous()

    a0, a1, a2, c0, c1, c2, d0 = self.illu_consts

    Y0 = torch.ones(n_v_full).float() * a0 * c0
    if gamma.is_cuda: Y0 = Y0.cuda()
    norm = norm.view(-1, 3)
    nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
    arrH = []

    arrH.append(Y0)
    arrH.append(-a1 * c1 * ny)
    arrH.append(a1 * c1 * nz)
    arrH.append(-a1 * c1 * nx)
    arrH.append(a2 * c2 * nx * ny)
    arrH.append(-a2 * c2 * ny * nz)
    arrH.append(a2 * c2 * d0 * (3 * nz.pow(2) - 1))
    arrH.append(-a2 * c2 * nx * nz)
    arrH.append(a2 * c2 * 0.5 * (nx.pow(2) - ny.pow(2)))

    H = torch.stack(arrH, 1)
    Y = H.view(n_b, num_vertex, 9)
    # Y shape:[batch,N,9].

    # shape:[batch,N,3]
    lighting = Y.bmm(gamma)

    face_color = face_texture * lighting
    # lighting *= 128
    # print( face_color[0, 5] )
    return face_color, lighting
