"""
The code is based on https://github.com/lioryariv/idr
# Copyright (c) ETH Zurich and its affiliates. All rights reserved.
# licensed under the LICENSE file in the root directory of this source tree.
# Code adapted / written by Yufeng Zheng.
"""
import numpy as np
import imageio
import skimage
import cv2
import torch
from torch.nn import functional as F

def load_rgb(path, img_res):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)

    # pixel values between [-1,1]
    img -= 0.5
    img *= 2.
    img = cv2.resize(img, (int(img_res[0]), int(img_res[1])))
    img = img.transpose(2, 0, 1)
    return img

def load_rgb_mask_human(path, img_res):
    # load RGBA images for the makehuman dataset
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)
    img = cv2.resize(img, (int(img_res[0]), int(img_res[1])))

    mask = img[:, :, 3]
    img = img[:, :, :3]
    # pixel values between [-1,1]
    img -= 0.5
    img *= 2.
    img = img.transpose(2, 0, 1)
    return img, mask

def load_semantic(path, img_res):
    img = imageio.imread(path, mode='F')
    h, w = img.shape
    semantics = np.zeros((h, w, 9))
    semantics[:, :, 0] = ((img == 1) + (img == 10) + (img == 8) + (img == 7) + (img == 14)) >= 1 # skin, nose, ears, neck
    semantics[:, :, 1] = ((img == 4) + (img == 5)) >= 1 # left eye, right eye
    semantics[:, :, 2] = ((img == 2) + (img == 3)) >= 1 # left eyebrow, right eyebrow
    semantics[:, :, 3] = ((img == 11) + (img == 6)) >= 1 # mouth interior, eye glasses
    semantics[:, :, 4] = (img == 12) # upper lip
    semantics[:, :, 5] = (img == 13) # lower lip
    semantics[:, :, 6] = ((img == 17) + (img == 9)) >= 1 # hair
    semantics[:, :, 7] = ((img == 15) + (img == 16)) >= 1 # cloth, necklace
    semantics[:, :, 8] = 1. - np.sum(semantics[:, :, :8], 2) # background

    semantics = cv2.resize(semantics, (int(img_res[0]), int(img_res[1])))
    semantics = semantics.transpose(2, 0, 1)
    return semantics

def load_semantic_makehuman(path, img_res):
    img = imageio.imread(path, as_gray=True)
    h, w = img.shape
    semantics = np.zeros((h, w, 9))
    semantics[:, :, 0] = ((img == 101) + (img == 105) + (img == 106) + (img == 107)) >= 1 # skin, nose, ears, neck
    semantics[:, :, 1] = ((img == 103) + (img == 104)) >= 1 # left eye, right eye
    semantics[:, :, 2] = 0 # left eyebrow, right eyebrow
    semantics[:, :, 3] = 0 # mouth interior
    semantics[:, :, 4] = 0 # upper lip
    semantics[:, :, 5] = 0 # lower lip
    semantics[:, :, 6] = (img == 102) # hair
    semantics[:, :, 7] = (img == 200) # cloth
    semantics[:, :, 8] = 1. - np.sum(semantics[:, :, :8], 2) # background

    semantics = cv2.resize(semantics, (int(img_res[0]), int(img_res[1])))
    semantics = semantics.transpose(2, 0, 1)
    return semantics

def visualize_semantics(semantics):
    cmap = torch.from_numpy(np.array([(204, 0, 0), (51, 51, 255), (0, 255, 255), (102, 204, 0), (255, 255, 0),
                     (0, 0, 153), (0, 0, 204), (0, 204, 0), (0, 0, 0)], dtype=np.uint8)).cuda()
    color = torch.sum(semantics[:, :, None] * cmap[None, :, :], 1)
    return color

def load_mask(path, img_res):
    alpha = imageio.imread(path, mode='F')
    alpha = skimage.img_as_float32(alpha)

    alpha = cv2.resize(alpha, (int(img_res[0]), int(img_res[1])))
    object_mask = alpha > 127.5

    return object_mask

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def get_camera_params(uv, pose, intrinsics):
    if pose.shape[1] == 7: #In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:,:4])
        p = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = pose

    batch_size, num_samples, _ = uv.shape
    depth = torch.ones((batch_size, num_samples)).cuda()
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1) * -1.0
    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)
    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)
    return ray_dirs, cam_loc


def get_camera_for_plot(pose):
    if pose.shape[1] == 7: #In case of quaternion vector representation
        cam_loc = pose[:, 4:].detach()
        R = quat_to_rot(pose[:,:4].detach())
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        R = pose[:, :3, :3]
    cam_dir = R[:, :3, 2]
    return cam_loc, cam_dir

def lift(x, y, z, intrinsics):
    # parse intrinsics
    intrinsics = intrinsics.cuda()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)

def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3,3)).cuda()
    qr=q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0]=1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R

def rot_to_quat(R):
    batch_size, _,_ = R.shape
    q = torch.ones((batch_size, 4)).cuda()

    R00 = R[:, 0,0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:,0]=torch.sqrt(1.0+R00+R11+R22)/2
    q[:, 1]=(R21-R12)/(4*q[:,0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q

def get_sphere_intersection(cam_loc, ray_directions, r = 1.0):
    # Input: n_images x 4 x 4 ; n_images x n_rays x 3
    # Output: n_images * n_rays x 2 (close and far) ; n_images * n_rays

    n_imgs, n_pix, _ = ray_directions.shape

    cam_loc = cam_loc.unsqueeze(-1)
    ray_cam_dot = torch.bmm(ray_directions, cam_loc).squeeze()
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2,1) ** 2 - r ** 2)

    under_sqrt = under_sqrt.reshape(-1)
    mask_intersect = under_sqrt > 0

    sphere_intersections = torch.zeros(n_imgs * n_pix, 2).cuda().float()
    sphere_intersections[mask_intersect] = torch.sqrt(under_sqrt[mask_intersect]).unsqueeze(-1) * torch.Tensor([-1, 1]).cuda().float()
    sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[mask_intersect].unsqueeze(-1)

    sphere_intersections = sphere_intersections.reshape(n_imgs, n_pix, 2)
    sphere_intersections = sphere_intersections.clamp_min(0.0)
    mask_intersect = mask_intersect.reshape(n_imgs, n_pix)

    return sphere_intersections, mask_intersect

def get_depth(points, pose):
    ''' Returns depth from 3D points according to camera pose '''
    batch_size, num_samples, _ = points.shape
    if pose.shape[1] == 7:  # In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:, :4])
        pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda().float()
        pose[:, :3, 3] = cam_loc
        pose[:, :3, :3] = R

    points_hom = torch.cat((points, torch.ones((batch_size, num_samples, 1)).cuda()), dim=2)

    # permute for batch matrix product
    points_hom = points_hom.permute(0, 2, 1)

    points_cam = torch.inverse(pose).bmm(points_hom)
    depth = points_cam[:, 2, :][:, :, None] * -1.0
    return depth

