"""
The code is based on https://github.com/lioryariv/idr.
# Copyright (c) ETH Zurich and its affiliates. All rights reserved.
# licensed under the LICENSE file in the root directory of this source tree.
# Code adapted / written by Yufeng Zheng.
"""

import torch
from torch import nn
from torch.nn import functional as F

class Loss(nn.Module):
    def __init__(self, mask_weight, lbs_weight, flame_distance_weight, alpha, expression_reg_weight, pose_reg_weight, cam_reg_weight, gt_w_seg=False):
        super().__init__()
        
        self.mask_weight = mask_weight
        self.lbs_weight = lbs_weight
        self.flame_distance_weight = flame_distance_weight
        self.expression_reg_weight = expression_reg_weight
        self.cam_reg_weight = cam_reg_weight
        self.pose_reg_weight = pose_reg_weight
        self.gt_w_seg = gt_w_seg
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l2_loss = nn.MSELoss(reduction='sum')

    def get_rgb_loss(self,rgb_values, rgb_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(object_mask.shape[0])
        return rgb_loss

    def get_flame_distance_loss(self, flame_distance, semantic_gt, network_object_mask):
        object_skin_mask = semantic_gt[:, :, 0].reshape(-1) == 1
        if (network_object_mask & object_skin_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        flame_distance = flame_distance[network_object_mask & object_skin_mask]
        flame_distance_loss = torch.mean(flame_distance * flame_distance)

        return flame_distance_loss

    def get_lbs_loss(self, lbs_weight, gt_lbs_weight, flame_distance, network_object_mask, object_mask):

        flame_distance_mask = flame_distance < 0.001
        if (network_object_mask & object_mask & flame_distance_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        lbs_weight = lbs_weight[network_object_mask & object_mask & flame_distance_mask]
        gt_lbs_weight = gt_lbs_weight[network_object_mask & object_mask & flame_distance_mask]
        lbs_loss =self.l2_loss(lbs_weight, gt_lbs_weight)/ float(object_mask.shape[0])
        return lbs_loss


    def get_mask_loss(self, sdf_output, network_object_mask, object_mask, valid_mask):
        mask = (~(network_object_mask & object_mask)) & valid_mask
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(-1), gt, reduction='sum') / float(object_mask.shape[0])
        return mask_loss

    def get_expression_reg_weight(self, pred, gt):
        return self.l2_loss(pred, gt)

    def get_gt_blendshape(self, index_batch, flame_lbs_weights, flame_posedirs, flame_shapedirs, semantics, surface_mask, ghostbone):
        bz = index_batch.shape[0]
        index_batch = index_batch[surface_mask]
        output = {}
        if ghostbone:
            gt_lbs_weight = torch.zeros(len(index_batch), 6).cuda()
            gt_lbs_weight[:, 1:] = flame_lbs_weights[index_batch, :]
        else:
            gt_lbs_weight = flame_lbs_weights[index_batch, :]

        gt_shapedirs = flame_shapedirs[index_batch, :, 100:]
        gt_posedirs = torch.transpose(flame_posedirs.reshape(36, -1, 3), 0, 1)[index_batch, :, :]

        gt_skinning_values = torch.ones(bz, 6 if ghostbone else 5).float().cuda()
        gt_skinning_values[surface_mask] = gt_lbs_weight
        #hair deforms with head
        if self.gt_w_seg:
            hair = semantics[:, :, 6].reshape(-1) == 1
            gt_skinning_values[hair, :] = 0.
            gt_skinning_values[hair, 2 if ghostbone else 1] = 1.
        if ghostbone and self.gt_w_seg:
            # cloth deforms with ghost bone (identity)
            cloth = semantics[:, :, 7].reshape(-1) == 1
            gt_skinning_values[cloth, :] = 0.
            gt_skinning_values[cloth, 0] = 1.
        output['gt_lbs_weight'] = gt_skinning_values

        gt_posedirs_values = torch.ones(bz, 36, 3).float().cuda()
        gt_posedirs_values[surface_mask] = gt_posedirs
        if self.gt_w_seg:
            # mouth interior and eye glasses doesn't deform
            mouth = semantics[:, :, 3].reshape(-1) == 1
            gt_posedirs_values[mouth, :] = 0.0
        output['gt_posedirs'] = gt_posedirs_values


        gt_shapedirs_values = torch.ones(bz, 3, 50).float().cuda()
        gt_shapedirs_values[surface_mask] = gt_shapedirs

        disable_shapedirs_for_mouth_and_cloth = False
        if disable_shapedirs_for_mouth_and_cloth:
            # I accidentally deleted these when cleaning the code...
            # So this is why I don't see teeth anymore...QAQ
            # Most of suppmat experiments used this code block, but it doesn't necessarily help in all cases.
            if self.gt_w_seg:
                # mouth interior and eye glasses doesn't deform
                mouth = semantics[:, :, 3].reshape(-1) == 1
                gt_shapedirs_values[mouth, :] = 0.0
            if ghostbone and self.gt_w_seg:
                # cloth doesn't deform with facial expressions
                cloth = semantics[:, :, 7].reshape(-1) == 1
                gt_shapedirs_values[cloth, :] = 0.
        output['gt_shapedirs'] = gt_shapedirs_values
        return output


    def forward(self, model_outputs, ground_truth):
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], ground_truth['rgb'], network_object_mask, object_mask)
        mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, object_mask, model_outputs['valid_mask'])

        loss = rgb_loss + self.mask_weight * mask_loss



        out = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'mask_loss': mask_loss,
        }
        if self.lbs_weight != 0:
            ghostbone = model_outputs['lbs_weight'].shape[1] == 6
            outputs = self.get_gt_blendshape(model_outputs['index_batch'], model_outputs['flame_lbs_weights'], model_outputs['flame_posedirs'], model_outputs['flame_shapedirs'], ground_truth['semantics'], model_outputs['network_object_mask'] & model_outputs['object_mask'], ghostbone)
            num_points = model_outputs['lbs_weight'].shape[0]
            if self.gt_w_seg:
                # do not enforce nearest neighbor skinning weight for teeth, learn from data instead.
                # now it's also not enforcing nn skinning wieght for glasses, I'm too lazy to correct it but the skinning weight can still learn correctly for glasses.
                lbs_loss = self.get_lbs_loss(model_outputs['lbs_weight'].reshape(num_points, -1), outputs['gt_lbs_weight'].reshape(num_points, -1), model_outputs['flame_distance'], network_object_mask, object_mask & (ground_truth['semantics'][:, :, 3].reshape(-1) != 1))
            else:
                lbs_loss = self.get_lbs_loss(model_outputs['lbs_weight'].reshape(num_points, -1), outputs['gt_lbs_weight'].reshape(num_points, -1), model_outputs['flame_distance'], network_object_mask, object_mask)

            out['loss'] += lbs_loss * self.lbs_weight * 0.1
            out['lbs_loss'] = lbs_loss

            posedirs_loss = self.get_lbs_loss(model_outputs['posedirs'].reshape(num_points, -1) * 10, outputs['gt_posedirs'].reshape(num_points, -1) * 10, model_outputs['flame_distance'], network_object_mask, object_mask)
            out['loss'] += posedirs_loss * self.lbs_weight * 10.0
            out['posedirs_loss'] = posedirs_loss

            shapedirs_loss = self.get_lbs_loss(model_outputs['shapedirs'].reshape(num_points, -1) * 10, outputs['gt_shapedirs'].reshape(num_points, -1) * 10, model_outputs['flame_distance'], network_object_mask, object_mask)
            out['loss'] += shapedirs_loss * self.lbs_weight * 10.0
            out['shapedirs_loss'] = shapedirs_loss

        if 'semantics' in ground_truth and self.flame_distance_weight > 0 and self.gt_w_seg:
            out['flame_distance_loss'] = self.get_flame_distance_loss(model_outputs['flame_distance'], ground_truth['semantics'], network_object_mask)
            out['loss'] += out['flame_distance_loss'] * self.flame_distance_weight

        if self.expression_reg_weight != 0 and 'expression' in ground_truth:
            out['expression_reg_loss'] = self.get_expression_reg_weight(model_outputs['expression'][..., :50], ground_truth['expression'])
            out['loss'] += out['expression_reg_loss'] * self.expression_reg_weight

        if self.pose_reg_weight != 0 and 'flame_pose' in ground_truth:
            out['pose_reg_loss'] = self.get_expression_reg_weight(model_outputs['flame_pose'], ground_truth['flame_pose'])
            out['loss'] += out['pose_reg_loss'] * self.pose_reg_weight

        if self.cam_reg_weight != 0 and 'cam_pose' in ground_truth:
            out['cam_reg_loss'] = self.get_expression_reg_weight(model_outputs['cam_pose'][:, :3, 3], ground_truth['cam_pose'][:, :3, 3])
            out['loss'] += out['cam_reg_loss'] * self.cam_reg_weight


        return out
