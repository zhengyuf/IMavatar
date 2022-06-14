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

"""
# Copyright (c) ETH Zurich and its affiliates. All rights reserved.
# licensed under the LICENSE file in the root directory of this source tree.
# Code adapted / written by Yufeng Zheng.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import torch
import torch.nn.functional as F

def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, dtype=torch.float32):
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
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        pose_feature: torch.tensor Bx36
        transformations: torch.tensor Bx5x4x4
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    # Add shape contribution
    v_shaped = v_template + torch.einsum('bl,mkl->bmk', [betas, shapedirs])

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)

    rot_mats = batch_rodrigues(
        pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])

    pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
    # (N x P) x (P, V * 3) -> N x V x 3
    pose_offsets = torch.matmul(pose_feature, posedirs) \
        .view(batch_size, -1, 3)

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

    return verts, pose_feature, A


def forward_pts(pnts_c, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, dtype=torch.float32, mask=None):
    # pnts_c: num_points, 3
    assert len(pnts_c.shape) == 2
    if mask is not None:
        pnts_c = pnts_c[mask]
    if pnts_c.shape[0] == 0:
        return pnts_c
    num_points = pnts_c.shape[0]
    device = pnts_c.device

    # Add shape contribution
    pnts_shaped = pnts_c + blend_shapes(betas, shapedirs)

    # Add pose blend shapes
    pose_offsets = pose_correctives(pose_feature, posedirs)
    assert (pose_offsets.shape == pnts_shaped.shape)
    pnts_posed = pose_offsets + pnts_shaped

    return forward_skinning_pts(pnts_posed, transformations, lbs_weights, dtype=dtype)



def forward_skinning_pts(pnts_c, transformations, lbs_weights, dtype=torch.float32, mask=None):
    # pnts_c: num_points, 3
    assert len(pnts_c.shape) == 2
    if mask is not None:
        pnts_c = pnts_c[mask]
    if pnts_c.shape[0] == 0:
        return pnts_c
    # pnts_c: num_points, 3
    num_points = pnts_c.shape[0]
    device = pnts_c.device

    # Do skinning:
    # W is num_points x (J + 1)
    W = lbs_weights
    num_joints = W.shape[-1]
    # T: [num_points, (J + 1)] x [num_points, (J + 1), 16] --> [num_points, 16]
    T = torch.einsum('mj, mjk->mk', [W, transformations.view(-1, num_joints, 16)]).view(num_points, 4, 4)

    homogen_coord = torch.ones([num_points, 1], dtype=dtype, device=device)
    # v_posed_homo: num_points, 4
    v_homo = torch.cat([pnts_c, homogen_coord], dim=1)
    # v_homo: [num_points, 4, 4] x [num_points, 4, 1] --> [num_points, 4, 1]
    v_homo = torch.matmul(T, torch.unsqueeze(v_homo, dim=-1))
    # pnts: [num_points, 3]
    pnts = v_homo[:, :3, 0]

    return pnts


def inverse_skinning_pts(pnts_p, transformations, lbs_weights, dtype=torch.float32):
    # pnts_p: num_points, 3
    assert len(pnts_p.shape) == 2
    if pnts_p.shape[0] == 0:
        return pnts_p
    num_points = pnts_p.shape[0]
    device = pnts_p.device

    pnts_p = pnts_p.reshape(num_points, 3)
    # Do skinning:
    # W is num_points x (J + 1)
    W = lbs_weights
    # T: [num_points, (J + 1)] x [num_points, (J + 1), 16] --> [num_points, 16]
    num_joints = W.shape[-1]
    T = torch.einsum('mj, mjk->mk', [W, transformations.view(-1, num_joints, 16)]).view(num_points, 4, 4)

    homogen_coord = torch.ones([num_points, 1], dtype=dtype, device=device)
    # pnts_p: num_points, 4
    pnts_p = torch.cat([pnts_p, homogen_coord], dim=1)
    # v_homo: [num_points, 4, 4] x [num_points, 4, 1] --> [num_points, 4, 1]
    v_homo = torch.matmul(torch.inverse(T), torch.unsqueeze(pnts_p, dim=-1))
    # pnts: [num_points, 3]
    pnts = v_homo[:, :3, 0]

    return pnts

def inverse_pts(pnts_p, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, dtype=torch.float32):
    # pnts_p: num_points, 3
    assert len(pnts_p.shape) == 2
    pnts_c = inverse_skinning_pts(pnts_p, transformations, lbs_weights)
    pnts_c = pnts_c - blend_shapes(betas, shapedirs)
    pose_offsets = pose_correctives(pose_feature, posedirs)
    assert(pose_offsets.shape == pnts_c.shape)
    pnts_c = pnts_c - pose_offsets
    return pnts_c

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

    # [num_points, 50] x [num_points, 3, 50] --> [num_points, 3]
    if len(shape_disps.shape) == 3:
        blend_shape = torch.einsum('ml,mkl->mk', [betas, shape_disps])
    return blend_shape

def pose_correctives(pose_feature, posedirs):
    # [num_points, 4*9] x [num_points, 4*9, 3] --> [num_points, 3]
    pose_correctives = torch.einsum('mi,mik->mk', [pose_feature, posedirs])
    return pose_correctives


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
    device = rot_vecs.device

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

    # transforms_mat = transform_mat(
    #     rot_mats.view(-1, 3, 3),
    #     rel_joints.view(-1, 3, 1)).view(-1, joints.shape[1], 4, 4)
    transforms_mat = transform_mat(
        rot_mats.view(-1, 3, 3),
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

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms