"""
# The code is based on https://github.com/lioryariv/idr
# Copyright (c) ETH Zurich and its affiliates. All rights reserved.
# licensed under the LICENSE file in the root directory of this source tree.
# Code adapted / written by Yufeng Zheng.
"""
import os
import torch
import numpy as np
import cv2
from utils import rend_util
import json

class FaceDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_folder,
                 subject_name,
                 json_name,
                 sub_dir,
                 img_res,
                 sample_size,
                 subsample=1,
                 use_semantics=False,
                 only_json=False,
                 ):
        """
        sub_dir: list of scripts/testing subdirectories for the subject, e.g. [MVI_1810, MVI_1811]
        Data structure:
            RGB images in data_folder/subject_name/subject_name/sub_dir[i]/image
            foreground masks in data_folder/subject_name/subject_name/sub_dir[i]/mask
            optional semantic masks in data_folder/subject_name/subject_name/sub_dir[i]/semantic
            json files containing FLAME parameters in data_folder/subject_name/subject_name/sub_dir[i]/json_name
        json file structure:
            frames: list of dictionaries, which are structured like:
                file_path: relative path to image
                world_mat: camera extrinsic matrix (world to camera). Camera rotation is actually the same for all frames,
                           since the camera is fixed during capture.
                           The FLAME head is centered at the origin, scaled by 4 times.
                expression: 50 dimension expression parameters
                pose: 15 dimension pose parameters
                flame_keypoints: 2D facial keypoints calculated from FLAME
            shape_params: 100 dimension FLAME shape parameters, shared by all scripts and testing frames of the subject
            intrinsics: camera focal length fx, fy and the offsets of the principal point cx, cy

        img_res: a list containing height and width, e.g. [256, 256] or [512, 512]
        sample_size: number of pixels sampled for each scripts step, set to -1 to render full images.
        subsample: subsampling the images to reduce frame rate, mainly used for inference and evaluation
        use_semantics: whether to use semantic maps as scripts input
        only_json: used for testing, when there is no GT images, masks or semantics
        """
        sub_dir = [str(dir) for dir in sub_dir]
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.sample_size = sample_size
        self.use_semantics = use_semantics

        self.data = {
            "image_paths": [],
            "mask_paths": [],
            "world_mats": [],
            # FLAME expression and pose parameters
            "expressions": [],
            "flame_pose": [],
            # saving image names and subdirectories
            "img_name": [],
            "sub_dir": [],
            "bbox": []
        }
        if self.use_semantics:
            # optionally using semantic maps
            self.data["semantic_paths"] = []

        for dir in sub_dir:
            instance_dir = os.path.join(data_folder, subject_name, subject_name, dir)
            assert os.path.exists(instance_dir), "Data directory is empty"

            cam_file = '{0}/{1}'.format(instance_dir, json_name)

            with open(cam_file, 'r') as f:
                camera_dict = json.load(f)

            for frame in camera_dict['frames']:
                # world to camera matrix
                world_mat = np.array(frame['world_mat']).astype(np.float32)
                # camera to world matrix
                self.data["world_mats"].append(rend_util.load_K_Rt_from_P(None, world_mat)[1])
                self.data["expressions"].append(np.array(frame['expression']).astype(np.float32))
                self.data["flame_pose"].append(np.array(frame['pose']).astype(np.float32))
                image_path = '{0}/{1}.png'.format(instance_dir, frame["file_path"])
                self.data["image_paths"].append(image_path)
                self.data["mask_paths"].append(image_path.replace('image', 'mask'))
                if use_semantics:
                    self.data["semantic_paths"].append(image_path.replace('image', 'semantic'))
                self.data["img_name"].append(int(frame["file_path"].split('/')[-1]))
                self.data["sub_dir"].append(dir)
                self.data["bbox"].append(((np.array(frame['bbox']) + 1.) * np.array([img_res[0],img_res[1],img_res[0],img_res[1]])/ 2).astype(int))

        self.shape_params = torch.tensor(camera_dict['shape_params']).float().unsqueeze(0)
        focal_cxcy = camera_dict['intrinsics']
        # construct intrinsic matrix in pixels
        intrinsics = np.eye(3)
        if focal_cxcy[3] > 1:
            # An old format of my datasets...
            intrinsics[0, 0] = focal_cxcy[0] * img_res[0] / 512
            intrinsics[1, 1] = focal_cxcy[1] * img_res[1] / 512
            intrinsics[0, 2] = focal_cxcy[2] * img_res[0] / 512
            intrinsics[1, 2] = focal_cxcy[3] * img_res[1] / 512
        else:
            intrinsics[0, 0] = focal_cxcy[0] * img_res[0]
            intrinsics[1, 1] = focal_cxcy[1] * img_res[1]
            intrinsics[0, 2] = focal_cxcy[2] * img_res[0]
            intrinsics[1, 2] = focal_cxcy[3] * img_res[1]
        self.intrinsics = intrinsics

        if isinstance(subsample, int) and subsample > 1:
            for k, v in self.data.items():
                self.data[k] = v[::subsample]
        elif isinstance(subsample, list):
            if len(subsample) == 2:
                subsample = list(range(subsample[0], subsample[1]))
            for k, v in self.data.items():
                self.data[k] = [v[s] for s in subsample]

        self.data["expressions"] = torch.from_numpy(np.stack(self.data["expressions"], 0))
        self.data["flame_pose"] = torch.from_numpy(np.stack(self.data["flame_pose"], 0))
        self.data["world_mats"] = torch.from_numpy(np.stack(self.data["world_mats"], 0)).float()
        self.intrinsics = torch.from_numpy(self.intrinsics).float()
        self.only_json = only_json

    def __len__(self):
        return len(self.data["image_paths"])

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "idx": torch.LongTensor([idx]),
            "img_name": torch.LongTensor([self.data["img_name"][idx]]),
            "sub_dir": self.data["sub_dir"][idx],
            "uv": uv,
            "intrinsics": self.intrinsics,
            "expression": self.data["expressions"][idx],
            "flame_pose": self.data["flame_pose"][idx],
            "cam_pose": self.data["world_mats"][idx]
        }
        if not self.only_json:
            sample["object_mask"] = torch.from_numpy(rend_util.load_mask(self.data["mask_paths"][idx], self.img_res).reshape(-1)).bool()
        if not self.only_json:
            ground_truth = {
                "rgb": torch.from_numpy(rend_util.load_rgb(self.data["image_paths"][idx], self.img_res).reshape(3, -1)
                                        .transpose(1, 0)).float() * sample["object_mask"].unsqueeze(1).float()
                                        + (1 - sample["object_mask"].unsqueeze(1).float())
            }
        else:
            ground_truth = {}
        if self.use_semantics:
            if not self.only_json:
                ground_truth["semantics"] = torch.from_numpy(rend_util.load_semantic(self.data["semantic_paths"][idx], self.img_res).reshape(-1, self.img_res[0] * self.img_res[1]).transpose(1, 0)).float()

        if self.only_json:
            assert self.sample_size == -1  # for testing, we need to run all pixels
        if self.sample_size != -1:
            # sampling half of the pixels from entire image
            sampling_idx = torch.randperm(self.total_pixels)[:self.sample_size//2]
            rgb = ground_truth["rgb"]
            mask = sample["object_mask"]
            if self.use_semantics:
                semantic = ground_truth["semantics"]
            ground_truth["rgb"] = rgb[sampling_idx, :]
            sample["object_mask"] = mask[sampling_idx]
            if self.use_semantics:
                ground_truth["semantics"] = semantic[sampling_idx, :]
            sample["uv"] = uv[sampling_idx, :]
            # and half of the pixels from bbox
            bbox = self.data["bbox"][idx]
            bbox_inside = torch.logical_and(torch.logical_and(uv[:, 0] > bbox[0], uv[:, 1] > bbox[1]),
                                            torch.logical_and(uv[:, 0] < bbox[2], uv[:, 1] < bbox[3]))
            uv_bbox = uv[bbox_inside]
            rgb_bbox = rgb[bbox_inside]
            mask_bbox = mask[bbox_inside]
            if self.use_semantics:
                semantic_bbox = semantic[bbox_inside]
            sampling_idx_bbox = torch.randperm(len(uv_bbox))[:self.sample_size - self.sample_size // 2]
            ground_truth["rgb"] = torch.cat([ground_truth["rgb"], rgb_bbox[sampling_idx_bbox, :]], 0)
            sample["object_mask"] = torch.cat([sample["object_mask"], mask_bbox[sampling_idx_bbox]], 0)
            if self.use_semantics:
                ground_truth["semantics"] = torch.cat([ground_truth["semantics"], semantic_bbox[sampling_idx_bbox]], 0)

            sample["uv"] = torch.cat([sample["uv"], uv_bbox[sampling_idx_bbox, :]], 0)
        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns sample, ground_truth as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    try:
                        ret[k] = torch.stack([obj[k] for obj in entry])
                    except:
                        ret[k] = [obj[k] for obj in entry]
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)
