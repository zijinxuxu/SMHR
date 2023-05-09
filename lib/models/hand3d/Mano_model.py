# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# from .utils import Mesh,points2sphere, colors

ModelOutput = namedtuple('ModelOutput',
                         ['vertices', 'joints', 'full_pose', 'betas',
                          'global_orient',
                          'hand_pose'
                          ])
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)

TIP_IDS = {
    'mano': {
            'thumb':		744,
            'index':		320,
            'middle':		443,
            'ring':		    555,
            'pinky':		672,
        },
    'ho3d': {
            'thumb':		728,
            'index':		353,
            'middle':		442,
            'ring':		    576,
            'pinky':		694,
        }    
} 

# HO3D is [728, 353, 442, 576, 694]:  # THUMB, INDEX, MIDDLE, RING, PINKY

JOINT_NAMES = [
    'wrist',
    'index1',
    'index2',
    'index3',
    'middle1',
    'middle2',
    'middle3',
    'pinky1',
    'pinky2',
    'pinky3',
    'ring1',
    'ring2',
    'ring3',
    'thumb1',
    'thumb2',
    'thumb3',    
    'thumb_tip',
    'index_tip',
    'middle_tip',
    'ring_tip',
    'pinky_tip',
]


class ManoModel(nn.Module):
    # The hand joints are replaced by MANO
    NUM_BODY_JOINTS = 1
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + NUM_HAND_JOINTS
    NUM_BETAS = 10

    def __init__(self,
                 model_path,
                 is_rhand=True,
                 data_struct=None,
                 create_betas=True,
                 betas=None,
                 create_global_orient=True,
                 global_orient=None,
                 create_transl=True,
                 transl=None,
                 create_hand_pose=True,
                 hand_pose=None,
                 use_pca=True,
                 num_pca_comps=6,
                 flat_hand_mean=False,
                 batch_size=1,
                 joint_mapper=None,
                 v_template=None,
                 dtype=torch.float32,
                 vertex_ids=None,
                 use_compressed=True,
                 ext='pkl',
                 **kwargs):
        ''' MANO model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            create_hand_pose: bool, optional
                Flag for creating a member variable for the pose of the
                hand. (default = True)
            hand_pose: torch.tensor, optional, BxP
                The default value for the left hand pose member variable.
                (default = None)
            num_pca_comps: int, optional
                The number of PCA components to use for each hand.
                (default = 6)
            flat_hand_mean: bool, optional
                If False, then the pose of the hand is initialized to False.
            batch_size: int, optional
                The batch size used for creating the member variables
            dtype: torch.dtype, optional
                The data type for the created variables
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        '''

        self.num_pca_comps = num_pca_comps
        self.flat_hand_mean = flat_hand_mean
        # If no data structure is passed, then load the data from the given
        # model folder
        if data_struct is None:
            # Load the model
            if osp.isdir(model_path):
                model_fn = 'MANO_{}.{ext}'.format('RIGHT' if is_rhand else 'LEFT', ext=ext)
                mano_path = os.path.join(model_path, model_fn)
            else:
                mano_path = model_path
                self.is_rhand = True if 'RIGHT' in os.path.basename(model_path) else False
            assert osp.exists(mano_path), 'Path {} does not exist!'.format(
                mano_path)

            if ext == 'pkl':
                with open(mano_path, 'rb') as mano_file:
                    model_data = pickle.load(mano_file, encoding='latin1')
            elif ext == 'npz':
                model_data = np.load(mano_path, allow_pickle=True)
            else:
                raise ValueError('Unknown extension: {}'.format(ext))
            # for key, val in model_data.items():
            #     print('key: ', key)


        # self.tip_ids = TIP_IDS['mano'] # this is for different tips , if not well, change to false.
        if self.flat_hand_mean == False:
            self.tip_ids = TIP_IDS['mano']
        else:
            self.tip_ids = TIP_IDS['ho3d']
        # self.tip_ids = TIP_IDS['ho3d']

        super(ManoModel, self).__init__()

        self.batch_size = batch_size
        self.dtype = dtype
        self.joint_mapper = joint_mapper

        self.faces = model_data['f']
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))

        if create_betas:
            if betas is None:
                default_betas = torch.zeros([batch_size, self.NUM_BETAS],
                                            dtype=dtype)
            else:
                if 'torch.Tensor' in str(type(betas)):
                    default_betas = betas.clone().detach()
                else:
                    default_betas = torch.tensor(betas,
                                                 dtype=dtype)

            self.register_parameter('betas', nn.Parameter(default_betas,
                                                          requires_grad=True))

        if create_global_orient:
            if global_orient is None:
                default_global_orient = torch.zeros([batch_size, 3],
                                                    dtype=dtype)
            else:
                if torch.is_tensor(global_orient):
                    default_global_orient = global_orient.clone().detach()
                else:
                    default_global_orient = torch.tensor(global_orient,
                                                         dtype=dtype)

            global_orient = nn.Parameter(default_global_orient,requires_grad=True)
            self.register_parameter('global_orient', global_orient)

        if create_transl:
            if transl is None:
                default_transl = torch.zeros([batch_size, 3], dtype=dtype, requires_grad=True)
            else:
                default_transl = torch.tensor(transl, dtype=dtype)
            self.register_parameter('transl', nn.Parameter(default_transl, requires_grad=True))


        if v_template is None:
            v_template = model_data['v_template']
        if not torch.is_tensor(v_template):
            v_template = to_tensor(to_np(v_template), dtype=dtype)
        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(model_data['v_template']), dtype=dtype))

        # The shape components
        shapedirs = model_data['shapedirs']
        # The shape components
        self.register_buffer('shapedirs', to_tensor(to_np(shapedirs), dtype=dtype))

        j_regressor = to_tensor(to_np(model_data['J_regressor']), dtype=dtype)
        self.register_buffer('J_regressor', j_regressor)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = model_data['posedirs'].shape[-1]
        # 207 x 20670
        posedirs = np.reshape(model_data['posedirs'], [-1, num_pose_basis]).T
        self.register_buffer('posedirs',to_tensor(to_np(posedirs), dtype=dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(model_data['kintree_table'][0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights',to_tensor(to_np(model_data['weights']), dtype=dtype))

        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        if self.num_pca_comps ==45:
            self.use_pca = False
        

        hand_components = model_data['hands_components'][:num_pca_comps]

        self.np_hand_components = hand_components

        if self.use_pca:
            self.register_buffer(
                'hand_components',
                torch.tensor(hand_components, dtype=dtype))

        if self.flat_hand_mean:
            hand_mean = np.zeros_like(model_data['hands_mean'])
        else:
            hand_mean = model_data['hands_mean']

        self.register_buffer('hand_mean',
                             to_tensor(hand_mean, dtype=self.dtype))
        self.register_buffer('model_hands_mean',
                             to_tensor(model_data['hands_mean'], dtype=self.dtype))

        # Create the buffers for the pose of the left hand
        hand_pose_dim = num_pca_comps if use_pca else 3 * self.NUM_HAND_JOINTS
        if create_hand_pose:
            if hand_pose is None:
                default_hand_pose = torch.zeros([batch_size, hand_pose_dim],
                                                dtype=dtype)
            else:
                default_hand_pose = torch.tensor(hand_pose, dtype=dtype)

            hand_pose_param = nn.Parameter(default_hand_pose,
                                           requires_grad=True)
            self.register_parameter('hand_pose', hand_pose_param)

        # Create the buffer for the mean pose.
        pose_mean = self.create_mean_pose(model_data,
                                          flat_hand_mean=flat_hand_mean)
        pose_mean_tensor = pose_mean.clone().to(dtype)
        self.register_buffer('pose_mean', pose_mean_tensor)

        # add vertex_adj_faces_buff, 778x8, each vetex 8 neighborhood
        self.adj_faces = self.get_adj_face(self.faces)

    def create_mean_pose(self, model_data, flat_hand_mean=False):
        # Create the array for the mean pose. If flat_hand is false, then use
        # the mean that is given by the data, rather than the flat open hand
        global_orient_mean = torch.zeros([3], dtype=self.dtype)

        pose_mean = torch.cat([global_orient_mean,
                               self.hand_mean], dim=0)
        return pose_mean

    def get_num_verts(self):
        return self.v_template.shape[0]

    def get_num_faces(self):
        return self.faces.shape[0]

    def get_adj_face(self, trifaces):
        vertex_adj_faces = []
        for i in range(self.v_template.shape[0]):
            vertex_adj_faces.append([])
        for fIdx, vIdx in enumerate(trifaces[:,0]):
            vertex_adj_faces[vIdx].append(fIdx)
        for fIdx, vIdx in enumerate(trifaces[:,1]):
            vertex_adj_faces[vIdx].append(fIdx)
        for fIdx, vIdx in enumerate(trifaces[:,2]):
            vertex_adj_faces[vIdx].append(fIdx)
        for i in range(self.v_template.shape[0]):
            num = len(vertex_adj_faces[i])
            for j in range(8):
                if j < num:
                    continue
                vertex_adj_faces[i].append(vertex_adj_faces[i][j-num]) 
        return vertex_adj_faces
        # print(vertex_adj_faces)
        # for fIdx, vIdx in enumerate(trifaces[:,0]):
        #     vertex_adj_faces[vIdx].
        #     vertex_normals[:,vIdx,:] += fIdx
        # for fIdx, vIdx in enumerate(trifaces[:,1]):
        #     vertex_normals[:,vIdx,:] += faceNormals[:,fIdx,:]
        # for fIdx, vIdx in enumerate(trifaces[:,2]):
        #     vertex_normals[:,vIdx,:] += faceNormals[:,fIdx,:]

    def ComputeNormal(self, vertices, trifaces):
        if vertices.shape[0] > 5000:
            print('ComputeNormal: Warning: too big to compute {0}'.format(vertices.shape) )
            return

        #compute vertex Normals for all frames
        U = vertices[:,trifaces[:,1],:] - vertices[:,trifaces[:,0],:]  #frames x faceNum x 3
        V = vertices[:,trifaces[:,2],:] - vertices[:,trifaces[:,1],:]  #frames x faceNum x 3
        originalShape = U.shape  #remember: frames x faceNum x 3

        U = np.reshape(U, [-1,3])
        V = np.reshape(V, [-1,3])
        faceNormals = np.cross(U,V)     #frames x 13776 x 3
        from sklearn.preprocessing import normalize

        if np.isnan(np.max(faceNormals)):
            print('ComputeNormal: Warning nan is detected {0}')
            return
        faceNormals = normalize(faceNormals)

        faceNormals = np.reshape(faceNormals, originalShape)

        if False:        #Slow version
            vertex_normals = np.zeros(vertices.shape) #(frames x 11510) x 3
            for fIdx, vIdx in enumerate(trifaces[:,0]):
                vertex_normals[:,vIdx,:] += faceNormals[:,fIdx,:]
            for fIdx, vIdx in enumerate(trifaces[:,1]):
                vertex_normals[:,vIdx,:] += faceNormals[:,fIdx,:]
            for fIdx, vIdx in enumerate(trifaces[:,2]):
                vertex_normals[:,vIdx,:] += faceNormals[:,fIdx,:]
        else:   #Faster version
            # Computing vertex normals, much faster (and obscure) replacement
            index = np.vstack((np.ravel(trifaces), np.repeat(np.arange(len(trifaces)), 3))).T
            index_sorted = index[index[:,0].argsort()]
            vertex_normals = np.add.reduceat(faceNormals[:,index_sorted[:, 1],:][0],
                np.concatenate(([0], np.cumsum(np.unique(index_sorted[:, 0],
                return_counts=True)[1])[:-1])))[None, :]
            vertex_normals = vertex_normals.astype(np.float64)

        originalShape = vertex_normals.shape
        vertex_normals = np.reshape(vertex_normals, [-1,3])
        vertex_normals = normalize(vertex_normals)
        vertex_normals = np.reshape(vertex_normals,originalShape)

        return vertex_normals

    def extra_repr(self):
        msg = 'Number of betas: {}'.format(self.NUM_BETAS)
        if self.use_pca:
            msg += '\nNumber of PCA components: {}'.format(self.num_pca_comps)
        msg += '\nFlat hand mean: {}'.format(self.flat_hand_mean)
        return msg

    def add_joints(self,vertices,joints, joint_ids = None):
        device = vertices.device
        if joint_ids is None:
            joint_ids = to_tensor(list(self.tip_ids.values()),
                                  dtype=torch.long).to(device)
        extra_joints = torch.index_select(vertices, 1, joint_ids)
        joints = torch.cat([joints, extra_joints], dim=1)
        # !!! the 3d keypoints is not matching the index of 2d keypoints
        idx = np.array([0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20])
        joints = joints[:, idx, :]

        return joints


    def forward(self, betas=None, global_orient=None, hand_pose=None, transl=None,
                return_verts=True, return_tips = False, return_full_pose=False, pose2rot=True,using_wrist_rotate=False,
                **kwargs):
        '''
        '''
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        assert global_orient is not None and betas is not None and hand_pose is not None
        # assert global_orient.requires_grad == True
        global_orient = (global_orient if global_orient is not None else self.global_orient)
        betas = betas if betas is not None else self.betas
        hand_pose = (hand_pose if hand_pose is not None else self.hand_pose)

        # assert transl is not None and transl.requires_grad == True
        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl

        if self.use_pca:
            hand_pose = torch.einsum('bi,ij->bj', [hand_pose, self.hand_components])

        if using_wrist_rotate == False:
            batch_size = max(betas.shape[0], hand_pose.shape[0])
            device = global_orient.device
            wrist_rot = torch.tensor([0.,0.,0.]).expand([batch_size, -1]).to(device)

            # assume wrist fixed, rotate around center. other than wrist.
            full_pose = torch.cat([wrist_rot, 
                                hand_pose], dim=1)
        else:
            full_pose = torch.cat([global_orient,
                                hand_pose], dim=1)
        full_pose += self.pose_mean

        if return_verts:
            vertices, joints = lbs(betas, full_pose, self.v_template,
                                   self.shapedirs, self.posedirs,
                                   self.J_regressor, self.parents,
                                   self.lbs_weights, pose2rot=pose2rot,
                                   dtype=self.dtype)
            
            # modified for fitting FreiHAND gt
            if self.flat_hand_mean == False:
                joints = vertices2joints(self.J_regressor, vertices)
            

            # Add any extra joints that might be needed
            if return_tips:
                joints = self.add_joints(vertices, joints)

            if self.joint_mapper is not None:
                joints = self.joint_mapper(joints)
        if using_wrist_rotate == False:
            # rotate hand on (0,0,0) other than wrist joint.
            Rots = batch_rodrigues(global_orient)
            vertices = torch.matmul(vertices,Rots.transpose(2, 1)) #.contiguous().view(batch_size,-1)
            joints = torch.matmul(joints,Rots.transpose(2, 1)) #.contiguous().view(batch_size,-1)        

        if apply_trans:
            transl = (transl - joints[:,9]).detach()
            joints = joints + transl.unsqueeze(dim=1)
            vertices = vertices + transl.unsqueeze(dim=1)


        output = ModelOutput(vertices=vertices if return_verts else None,
                             joints=joints if return_verts else None,
                             betas=betas,
                             global_orient=global_orient,
                             hand_pose = hand_pose,
                             full_pose=full_pose if return_full_pose else None)

        return output

    # def hand_meshes(self,output, vc= colors['skin']):

    #     vertices = to_np(output.vertices)
    #     if vertices.ndim <3:
    #         vertices = vertices.reshape(-1,778,3)

    #     meshes = []
    #     for v in vertices:
    #         hand_mesh = Mesh(vertices=v, faces=self.faces, vc=vc)
    #         meshes.append(hand_mesh)

    #     return  meshes

    # def joint_meshes(self,output, radius=.002, vc=colors['green']):

    #     joints = to_np(output.joints)
    #     if joints.ndim <3:
    #         joints = joints.reshape(1,-1,3)

    #     meshes = []
    #     for j in joints:
    #         joint_mesh = Mesh(vertices=j, radius=radius, vc=vc)
    #         meshes.append(joint_mesh)

    #     return  meshes


def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = np.array(array.todense())
    elif 'chumpy' in str(type(array)):
        array = np.array(array)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array.astype(dtype)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, pose2rot=True, dtype=torch.float32):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(
            pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed


def vertices2joints(J_regressor, vertices):
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def blend_shapes(betas, shape_disps):
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms