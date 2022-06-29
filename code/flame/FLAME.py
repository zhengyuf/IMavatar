# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

"""
# Copyright (c) ETH Zurich and its affiliates. All rights reserved.
# licensed under the LICENSE file in the root directory of this source tree.
# Code adapted / written by Yufeng Zheng.
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F

from .lbs import *

def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)
def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

class FLAME(nn.Module):
    """
    borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """
    def __init__(self, flame_model_path, n_shape, n_exp, shape_params):
        super(FLAME, self).__init__()
        print("creating the FLAME Decoder")
        with open(flame_model_path, 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)
        factor = 4

        self.dtype = torch.float32
        self.register_buffer('faces_tensor', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(flame_model.v_template) * factor, dtype=self.dtype))
        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:,:,:n_shape], shapedirs[:,:,300:300+50]], 2)
        self.register_buffer('shapedirs', shapedirs * factor)
        self.v_template = self.v_template + torch.einsum('bl,mkl->bmk', [shape_params.cpu(), self.shapedirs[:, :, :n_shape]]).squeeze(0)

        self.canonical_pose = torch.zeros(1, 15).float().cuda()
        self.canonical_pose[:, 6] = 0.2

        self.canonical_exp = torch.zeros(1, n_exp).float().cuda()

        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs) * factor, dtype=self.dtype))
        # 
        self.register_buffer('J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long(); parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype))

        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose,
                                                         requires_grad=False))
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose,
                                                          requires_grad=False))

        self.n_shape = n_shape

    # FLAME mesh morphing
    def forward(self, expression_params, full_pose):
        """
            Input:
                expression_params: N X number of expression parameters
                full_pose: N X number of pose parameters (15)
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = expression_params.shape[0]
        betas = torch.cat([torch.zeros(batch_size, self.n_shape).to(expression_params.device), expression_params], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        vertices, pose_feature, transformations = lbs(betas, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype)

        return vertices, pose_feature, transformations

    def forward_pts(self, pnts_c, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, dtype=torch.float32, mask=None):
        assert len(pnts_c.shape) == 2
        if mask is not None:
            pnts_c = pnts_c[mask]
            betas = betas[mask]
            transformations = transformations[mask]
            pose_feature = pose_feature[mask]
        num_points = pnts_c.shape[0]
        pnts_c_original = inverse_pts(pnts_c, self.canonical_exp.expand(num_points, -1), self.canonical_transformations.expand(num_points, -1, -1, -1), self.canonical_pose_feature.expand(num_points, -1), shapedirs, posedirs, lbs_weights, dtype=dtype)
        pnts_p = forward_pts(pnts_c_original, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, dtype=dtype)
        return pnts_p

    def inverse_skinning_pts(self, pnts_p, transformations, lbs_weights, dtype=torch.float32):
        num_points = pnts_p.shape[0]
        pnts_c_original = inverse_skinning_pts(pnts_p, transformations, lbs_weights, dtype=dtype)
        pnts_c = forward_skinning_pts(pnts_c_original, self.canonical_transformations.expand(num_points, -1, -1, -1), lbs_weights, dtype=dtype, mask=None)
        return pnts_c
