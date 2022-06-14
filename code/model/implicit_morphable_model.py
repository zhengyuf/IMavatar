"""
The code is based on https://github.com/lioryariv/idr and https://github.com/xuchen-ethz/snarf
# Copyright (c) ETH Zurich and its affiliates. All rights reserved.
# licensed under the LICENSE file in the root directory of this source tree.
# Code adapted / written by Yufeng Zheng.
"""
import torch
import torch.nn as nn
from utils import rend_util
from utils import general as utils
from model.ray_tracing import RayTracing
from flame.FLAME import FLAME
from pytorch3d import ops
from functools import partial
from model.geometry_network import GeometryNetwork
from model.texture_network import RenderingNetwork
from model.deformer_network import ForwardDeformer

print_flushed = partial(print, flush=True)


class IMavatar(nn.Module):
    def __init__(self, conf, shape_params, gt_w_seg):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        
        self.FLAMEServer = FLAME('./flame/FLAME2020/generic_model.pkl', n_shape=100,
                                 n_exp=conf.get_config('deformer_network').get_int('num_exp'),
                                 shape_params=shape_params).cuda()
        self.FLAMEServer.canonical_verts, self.FLAMEServer.canonical_pose_feature, self.FLAMEServer.canonical_transformations = \
            self.FLAMEServer(expression_params=self.FLAMEServer.canonical_exp, full_pose=self.FLAMEServer.canonical_pose)
        self.FLAMEServer.canonical_verts = self.FLAMEServer.canonical_verts.squeeze(0)

        self.geometry_network = GeometryNetwork(self.feature_vector_size, **conf.get_config('geometry_network'))
        self.deformer_class = conf.get_string('deformer_class').split('.')[-1]
        self.deformer_network = utils.get_class(conf.get_string('deformer_class'))(FLAMEServer=self.FLAMEServer, **conf.get_config('deformer_network'))

        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')

        self.ghostbone = self.deformer_network.ghostbone
        if self.ghostbone:
            self.FLAMEServer.canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float().cuda(), self.FLAMEServer.canonical_transformations], 1)
        self.gt_w_seg = gt_w_seg


    def query_sdf(self, pnts_p, idx, network_condition, pose_feature, betas, transformations):
        if network_condition is not None:
            network_condition = network_condition[idx]
        pose_feature = pose_feature[idx]
        betas = betas[idx]
        transformations = transformations[idx]

        pnts_c, others = self.deformer_network(pnts_p, pose_feature, betas, transformations)
        num_point, num_init, num_dim = pnts_c.shape
        pnts_c = pnts_c.reshape(num_point * num_init, num_dim)
        output = self.geometry_network(pnts_c, network_condition).reshape(num_point, num_init, -1)
        sdf = output[:, :, 0]
        feature = output[:, :, 1:]
        # aggregate occupancy probablities
        mask = others['valid_ids']
        sdf[~mask] = 1.
        sdf, index = torch.min(sdf, dim=1)
        pnts_c = pnts_c.reshape(num_point, num_init, num_dim)

        pnts_c = torch.gather(pnts_c, dim=1, index=index.unsqueeze(-1).unsqueeze(-1).expand(num_point, num_init, num_dim))[:, 0, :]
        feature = torch.gather(feature, dim=1, index=index.unsqueeze(-1).unsqueeze(-1).expand(num_point, num_init, feature.shape[-1]))[:, 0, :]
        mask = torch.gather(mask, dim=1, index=index.unsqueeze(-1).expand(num_point, num_init))[:, 0]

        return sdf, pnts_c, feature, {'mask': mask}

    def forward(self, input, return_sdf=False):

        uv = input["uv"]
        intrinsics = input["intrinsics"]
        cam_pose = input["cam_pose"]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        object_mask = input["object_mask"].reshape(-1) if "object_mask" in input else None
        # conditioning the geometry network on per-frame learnable latent code
        if "latent_code" in input:
            network_condition = input["latent_code"]
        else:
            network_condition = None

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, cam_pose, intrinsics)
        batch_size, num_pixels, _ = ray_dirs.shape
        idx = torch.arange(batch_size).cuda().unsqueeze(1)
        idx = idx.expand(-1, num_pixels)


        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)

        if self.ghostbone:
            # identity transformation for body
            transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)
        self.geometry_network.eval()
        self.deformer_network.eval()
        with torch.no_grad():
            sdf_function = lambda x, idx: self.query_sdf(pnts_p=x,
                                                    idx=idx,
                                                    network_condition=network_condition,
                                                    pose_feature=pose_feature,
                                                    betas=expression,
                                                    transformations=transformations,
                                                    )[0]
            points, network_object_mask, dists = self.ray_tracer(sdf=sdf_function,
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs,
                                                                 idx=idx)
        self.geometry_network.train()
        self.deformer_network.train()


        points = (cam_loc.unsqueeze(1) + dists.detach().reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)

        _, canonical_points, _, others = self.query_sdf(pnts_p=points,
                                                    idx=idx.reshape(-1),
                                                    network_condition=network_condition,
                                                    pose_feature=pose_feature,
                                                    betas=expression,
                                                    transformations=transformations,
                                                    )
        valid_mask = others['mask']
        canonical_points = canonical_points.detach()
        sdf_output = self.geometry_network(self.get_differentiable_non_surface(canonical_points, points, idx.reshape(-1),
                                                                     pose_feature=pose_feature, betas=expression,
                                                                     transformations=transformations), network_condition)[:, :1]
        sdf_output[~valid_mask] = 1

        points = points.detach()

        surface_mask = network_object_mask & object_mask if self.training else network_object_mask
        shapedirs, posedirs, lbs_weight = self.deformer_network.query_weights(canonical_points, mask=surface_mask)
        ray_dirs = ray_dirs.reshape(-1, 3)

        if self.training:
            surface_mask = network_object_mask & object_mask
            surface_canonical_points = canonical_points[surface_mask]
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]

            differentiable_surface_points = self.get_differentiable_x(pnts_c=surface_canonical_points,
                                                                      idx=idx.reshape(-1)[surface_mask],
                                                                          network_condition=network_condition,
                                                                          pose_feature=pose_feature,
                                                                          betas=expression,
                                                                          transformations=transformations,
                                                                          view_dirs=surface_ray_dirs,
                                                                          cam_loc=surface_cam_loc)

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = canonical_points[surface_mask]


        rgb_values = torch.ones_like(points).float().cuda()
        normal_values = torch.ones_like(points).float().cuda()
        if differentiable_surface_points.shape[0] > 0:
            rgb_values[surface_mask], others = self.get_rbg_value(differentiable_surface_points, idx.reshape(-1)[surface_mask], network_condition, pose_feature, expression, transformations, is_training=self.training,
                                                                  jaw_pose=torch.cat([expression, flame_pose[:, 6:9]], dim=1))
            normal_values[surface_mask] = others['normals']

        flame_distance_values = torch.zeros(points.shape[0]).float().cuda()
        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        flame_distance, index_batch, _ = ops.knn_points(differentiable_surface_points.unsqueeze(0), knn_v, K=1, return_nn=True)
        index_batch = index_batch.reshape(-1)
        index_batch_values = torch.ones(points.shape[0]).long().cuda()
        index_batch_values[surface_mask] = index_batch
        flame_distance_values[surface_mask] = flame_distance.squeeze(0).squeeze(-1)

        output = {
            'points': points, # not differentiable
            'rgb_values': rgb_values,
            'normal_values': normal_values,
            'sdf_output': sdf_output,
            'valid_mask': valid_mask,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'expression': expression,
            'flame_pose': flame_pose,
            'cam_pose': cam_pose,
            'index_batch': index_batch_values,
            'flame_distance': flame_distance_values,
            'flame_lbs_weights': self.FLAMEServer.lbs_weights,
            'flame_posedirs': self.FLAMEServer.posedirs,
            'flame_shapedirs': self.FLAMEServer.shapedirs
        }

        if lbs_weight is not None:
            skinning_values = torch.ones(points.shape[0], 6 if self.ghostbone else 5).float().cuda()
            skinning_values[surface_mask] = lbs_weight
            output['lbs_weight'] = skinning_values
        if posedirs is not None:
            posedirs_values = torch.ones(points.shape[0], 36, 3).float().cuda()
            posedirs_values[surface_mask] = posedirs
            output['posedirs'] = posedirs_values
        if shapedirs is not None:
            shapedirs_values = torch.ones(points.shape[0], 3, 50).float().cuda()
            shapedirs_values[surface_mask] = shapedirs
            output['shapedirs'] = shapedirs_values

        if not return_sdf:
            return output
        else:
            return output, sdf_function

    def get_rbg_value(self, points, idx, network_condition, pose_feature, betas, transformations, jaw_pose=None, is_training=True):
        pnts_c = points
        others = {}
        if network_condition is not None:
            network_condition = network_condition[idx]
        pose_feature = pose_feature[idx]
        betas = betas[idx]
        transformations = transformations[idx]
        jaw_pose = jaw_pose[idx]
        _, gradients, feature_vectors = self.forward_gradient(pnts_c, network_condition, pose_feature, betas, transformations, create_graph=is_training, retain_graph=is_training)

        normals = nn.functional.normalize(gradients, dim=-1, eps=1e-6)

        rgb_vals = self.rendering_network(pnts_c, normals, feature_vectors, jaw_pose=jaw_pose)

        others['normals'] = normals

        return rgb_vals, others

    def forward_gradient(self, pnts_c, network_condition, pose_feature, betas, transformations, create_graph=True, retain_graph=True):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)

        pnts_d = self.deformer_network.forward_lbs(pnts_c, pose_feature, betas, transformations)
        num_dim = pnts_d.shape[-1]
        grads = []
        for i in range(num_dim):
            d_out = torch.zeros_like(pnts_d, requires_grad=False, device=pnts_d.device)
            d_out[:, i] = 1
            # d_out = d_out.double()*scale
            grad = torch.autograd.grad(
                outputs=pnts_d,
                inputs=pnts_c,
                grad_outputs=d_out,
                create_graph=create_graph,
                retain_graph=True if i < num_dim - 1 else retain_graph,
                only_inputs=True)[0]
            grads.append(grad)
        grads = torch.stack(grads, dim=-2)
        grads_inv = grads.inverse()

        output = self.geometry_network(pnts_c, network_condition)
        sdf = output[:, :1]
        feature = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=pnts_c,
            grad_outputs=d_output,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True)[0]

        return grads.reshape(grads.shape[0], -1), torch.nn.functional.normalize(
            torch.einsum('bi,bij->bj', gradients, grads_inv), dim=1), feature


    def get_differentiable_x(self, pnts_c, idx, network_condition, pose_feature, betas, transformations, view_dirs, cam_loc):
        # canonical_x : num_points, 3
        # cam_loc: num_points, 3
        # view_dirs: num_points, 3
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c = pnts_c.detach()
        pnts_c.requires_grad_(True)
        if network_condition is not None:
            network_condition = network_condition[idx]
        pose_feature = pose_feature[idx]
        betas = betas[idx]
        transformations = transformations[idx]
        deformed_x = self.deformer_network.forward_lbs(pnts_c, pose_feature, betas, transformations)
        sdf = self.geometry_network(pnts_c, network_condition)[:, 0:1]
        dirs = deformed_x - cam_loc
        cross_product = torch.cross(view_dirs, dirs)
        constant = torch.cat([cross_product[:, 0:2], sdf], dim=1)
        # constant: num_points, 3
        num_dim = constant.shape[-1]
        grads = []
        for i in range(num_dim):
            d_out = torch.zeros_like(constant, requires_grad=False, device=constant.device)
            d_out[:, i] = 1
            grad = torch.autograd.grad(
                outputs=constant,
                inputs=pnts_c,
                grad_outputs=d_out,
                create_graph=False,
                retain_graph=True,
                only_inputs=True)[0]
            grads.append(grad)
        grads = torch.stack(grads, dim=-2)
        grads_inv = grads.inverse()

        differentiable_x = pnts_c.detach() - torch.einsum('bij,bj->bi', grads_inv, constant - constant.detach())
        return differentiable_x

    def get_differentiable_non_surface(self, pnts_c, points, idx, pose_feature, betas, transformations):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()

        pnts_c = pnts_c.detach()
        pnts_c.requires_grad_(True)
        pose_feature = pose_feature[idx]
        betas = betas[idx]
        transformations = transformations[idx]
        deformed_x = self.deformer_network.forward_lbs(pnts_c, pose_feature, betas, transformations)
        # points is differentiable wrt cam_loc and ray_dirs
        constant = deformed_x - points
        # constant: num_points, 3
        num_dim = constant.shape[-1]
        grads = []
        for i in range(num_dim):
            d_out = torch.zeros_like(constant, requires_grad=False, device=constant.device)
            d_out[:, i] = 1
            grad = torch.autograd.grad(
                outputs=constant,
                inputs=pnts_c,
                grad_outputs=d_out,
                create_graph=False,
                retain_graph=True,
                only_inputs=True)[0]
            grads.append(grad)
        grads = torch.stack(grads, dim=-2)
        grads_inv = grads.inverse()

        differentiable_x = pnts_c.detach() - torch.einsum('bij,bj->bi', grads_inv, constant - constant.detach())
        return differentiable_x


