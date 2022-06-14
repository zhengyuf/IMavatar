"""
The code is based on https://github.com/lioryariv/idr.
# Copyright (c) ETH Zurich and its affiliates. All rights reserved.
# licensed under the LICENSE file in the root directory of this source tree.
# Code adapted / written by Yufeng Zheng.
"""

import plotly.graph_objs as go
import plotly.offline as offline
import numpy as np
import torch
from skimage import measure
import torchvision
import trimesh
from PIL import Image
from utils import rend_util
import os
import torch.nn as nn
from utils import mesh_util

def plot(img_index, sdf_function, model_outputs, pose, ground_truth, path, epoch, img_res, plot_nimgs, min_depth, max_depth, res_init, res_up, is_eval=False):
    # arrange data to plot
    batch_size = pose.shape[0]
    num_samples = int(model_outputs['rgb_values'].shape[0] / batch_size)
    network_object_mask = model_outputs['network_object_mask']
    points = model_outputs['points'].reshape(batch_size, num_samples, 3)
    # plot rendered images

    depth = torch.ones(batch_size * num_samples).cuda().float() * max_depth
    depth[network_object_mask] = rend_util.get_depth(points, pose).reshape(-1)[network_object_mask]
    depth = (depth.reshape(batch_size, num_samples, 1) - min_depth) / (max_depth - min_depth)
    if (depth.min() < 0.) or (depth.max() > 1.):
        print("Depth out of range, min: {} and max: {}".format(depth.min(), depth.max()))
        depth = torch.clamp(depth, 0., 1.)

    plot_images(model_outputs, depth, ground_truth, path, epoch, img_index, 1, img_res, batch_size, num_samples, is_eval)
    del depth, points, network_object_mask
    # Generate mesh.
    if is_eval:
        with torch.no_grad():
            import time
            start_time = time.time()
            meshexport = mesh_util.generate_mesh(sdf_function, level_set=0, res_init=res_init, res_up=res_up)
            meshexport.export('{0}/surface_{1}.ply'.format(path, img_index), 'ply')
            print("Plot time per mesh:", time.time() - start_time)
            del meshexport

def plot_depth_maps(depth_maps, path, epoch, img_index, plot_nrow, img_res):
    depth_maps_plot = lin2img(depth_maps, img_res)

    tensor = torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1),
                                         scale_each=True,
                                         normalize=True,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    if not os.path.exists('{0}/depth'.format(path)):
        os.mkdir('{0}/depth'.format(path))
    img.save('{0}/depth/{1}.png'.format(path, img_index))


def plot_image(rgb, path, epoch, img_index, plot_nrow, img_res, type):
    rgb_plot = lin2img(rgb, img_res)

    tensor = torchvision.utils.make_grid(rgb_plot,
                                         scale_each=True,
                                         normalize=True,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    if not os.path.exists('{0}/{1}'.format(path, type)):
        os.mkdir('{0}/{1}'.format(path, type))
    img.save('{0}/{2}/{1}.png'.format(path, img_index, type))


def plot_images(model_outputs, depth_image, ground_truth, path, epoch, img_index, plot_nrow, img_res, batch_size, num_samples, is_eval):
    if 'rgb' in ground_truth:
        rgb_gt = ground_truth['rgb']
        rgb_gt = (rgb_gt.cuda() + 1.) / 2.
    else:
        rgb_gt = None
    rgb_points = model_outputs['rgb_values']
    rgb_points = rgb_points.reshape(batch_size, num_samples, 3)

    normal_points = model_outputs['normal_values']
    normal_points = normal_points.reshape(batch_size, num_samples, 3)

    rgb_points = (rgb_points + 1.) / 2.
    normal_points = (normal_points + 1.) / 2.

    output_vs_gt = rgb_points
    if rgb_gt is not None:
        output_vs_gt = torch.cat((output_vs_gt, rgb_gt, depth_image.repeat(1, 1, 3), normal_points), dim=0)
    else:
        output_vs_gt = torch.cat((output_vs_gt, depth_image.repeat(1, 1, 3), normal_points), dim=0)
    if 'lbs_weight' in model_outputs:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap('Paired')
        red = cmap.colors[5]
        cyan = cmap.colors[3]
        blue = cmap.colors[1]
        pink = [1, 1, 1]

        lbs_points = model_outputs['lbs_weight']
        lbs_points = lbs_points.reshape(batch_size, num_samples, -1)
        if lbs_points.shape[-1] == 5:
            colors = torch.from_numpy(np.stack([np.array(cyan), np.array(blue), np.array(pink), np.array(pink), np.array(pink)])[None, None]).cuda()
        else:
            colors = torch.from_numpy(np.stack([np.array(red), np.array(cyan), np.array(blue), np.array(pink), np.array(pink), np.array(pink)])[None, None]).cuda()

        lbs_points = (colors * lbs_points[:, :, :, None]).sum(2)
        mask = torch.logical_not(model_outputs['network_object_mask'])
        lbs_points[mask[None, ..., None].expand(-1, -1, 3)] = 1.
        output_vs_gt = torch.cat((output_vs_gt, lbs_points), dim=0)
    if 'shapedirs' in model_outputs:
        shapedirs_points = model_outputs['shapedirs']
        shapedirs_points = shapedirs_points.reshape(batch_size, num_samples, 3, 50)[:, :, :, 0] * 50.

        shapedirs_points = (shapedirs_points + 1.) / 2.
        shapedirs_points = torch.clamp(shapedirs_points, 0., 1.)
        output_vs_gt = torch.cat((output_vs_gt, shapedirs_points), dim=0)
    if 'semantics' in ground_truth:
        gt_semantics = ground_truth['semantics'].squeeze(0)
        semantic_gt = rend_util.visualize_semantics(gt_semantics).reshape(batch_size, num_samples, 3)/ 255.
        output_vs_gt = torch.cat((output_vs_gt, semantic_gt), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=output_vs_gt.shape[0]).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    wo_epoch_path = path.replace('/epoch_{}'.format(epoch), '')
    if not os.path.exists('{0}/rendering'.format(wo_epoch_path)):
        os.mkdir('{0}/rendering'.format(wo_epoch_path))
    img.save('{0}/rendering/epoch_{1}_{2}.png'.format(wo_epoch_path, epoch, img_index))
    if is_eval:
        plot_image(rgb_points, path, epoch, img_index, plot_nrow, img_res, 'rgb')
        plot_image(normal_points, path, epoch, img_index, plot_nrow, img_res, 'normal')
    del output_vs_gt

def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])
