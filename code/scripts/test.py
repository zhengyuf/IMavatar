"""
The code is based on https://github.com/lioryariv/idr.
# Copyright (c) ETH Zurich and its affiliates. All rights reserved.
# licensed under the LICENSE file in the root directory of this source tree.
# Code adapted / written by Yufeng Zheng.
"""

import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
import time

import utils.general as utils
import utils.plots as plt

from functools import partial
print = partial(print, flush=True)
class TestRunner():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.exps_folder_name = self.conf.get_string('train.exps_folder')
        self.subject = self.conf.get_string('dataset.subject_name')
        self.methodname = self.conf.get_string('train.methodname')

        self.expdir = os.path.join(self.exps_folder_name, self.subject, self.methodname)
        train_split_name = utils.get_split_name(self.conf.get_list('dataset.train.sub_dir'))

        self.eval_dir = os.path.join(self.expdir, train_split_name, 'eval')
        self.train_dir = os.path.join(self.expdir, train_split_name, 'train')

        if kwargs['load_path'] != '':
            load_path = kwargs['load_path']
        else:
            load_path = self.train_dir
        assert os.path.exists(load_path)

        utils.mkdir_ifnotexists(self.eval_dir)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                         sample_size=-1,
                                                                                         subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                         json_name=self.conf.get_string('dataset.json_name'),
                                                                                         use_semantics=self.conf.get_bool('loss.gt_w_seg'),
                                                                                         only_json=kwargs['only_json'],
                                                                                         **self.conf.get_config('dataset.test'))

        print('Finish loading data ...')

        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=1, # only support batch_size = 1
                                                           shuffle=False,
                                                           collate_fn=self.plot_dataset.collate_fn
                                                           )

        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'), shape_params=self.plot_dataset.shape_params, gt_w_seg=self.conf.get_bool('loss.gt_w_seg'))
        if torch.cuda.is_available():
            self.model.cuda()
        old_checkpnts_dir = os.path.join(load_path, 'checkpoints')
        assert os.path.exists(old_checkpnts_dir)
        saved_model_state = torch.load(
            os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
        self.model.load_state_dict(saved_model_state["model_state_dict"])
        self.start_epoch = saved_model_state['epoch']
        self.optimize_latent_code = self.conf.get_bool('train.optimize_latent_code')
        if self.optimize_latent_code:
            self.input_params_subdir = "InputParameters"
            data = torch.load(
                os.path.join(old_checkpnts_dir, self.input_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.latent_codes = torch.nn.Embedding(data["latent_codes_state_dict"]['weight'].shape[0], 32, sparse=True).cuda()
            self.latent_codes.load_state_dict(data["latent_codes_state_dict"])
        self.total_pixels = self.plot_dataset.total_pixels
        self.img_res = self.plot_dataset.img_res
        self.plot_conf = self.conf.get_config('plot')

    def run(self):
        self.model.geometry_network.alpha = float(self.start_epoch)
        print_all = True
        self.model.eval()
        eval_iterator = iter(self.plot_dataloader)
        for img_index in range(len(self.plot_dataset)):
            start_time = time.time()
            if img_index >= self.conf.get_int('plot.plot_nimgs') and not print_all:
                break
            indices, model_input, ground_truth = next(eval_iterator)

            for k, v in model_input.items():
                try:
                    model_input[k] = v.cuda()
                except:
                    model_input[k] = v

            for k, v in ground_truth.items():
                try:
                    ground_truth[k] = v.cuda()
                except:
                    ground_truth[k] = v

            if self.optimize_latent_code:
                # use the latent code from the first scripts frame
                model_input['latent_code'] = self.latent_codes(torch.LongTensor([0]).cuda()).squeeze(1).detach()
            split = utils.split_input(model_input, self.total_pixels, n_pixels=min(33000, self.img_res[0] * self.img_res[1]))

            res = []
            for s in split:
                out, sdf_function = self.model(s, return_sdf=True)
                for k, v in out.items():
                    try:
                        out[k] = v.detach()
                    except:
                        out[k] = v
                res.append(out)

            batch_size = model_input['expression'].shape[0]
            model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
            plot_dir = os.path.join(self.eval_dir, model_input['sub_dir'][0], 'epoch_'+str(self.start_epoch))
            img_name = model_input['img_name'][0,0].cpu().numpy()
            utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
            if print_all:
                utils.mkdir_ifnotexists(plot_dir)
            print("Saving image {} into {}".format(img_name, plot_dir))
            plt.plot(img_name,
                     sdf_function,
                     model_outputs,
                     model_input['cam_pose'],
                     ground_truth,
                     plot_dir,
                     self.start_epoch,
                     self.img_res,
                     is_eval=print_all,
                     **self.plot_conf
                     )
            print("Plot time per frame: {}".format(time.time() - start_time))
            del model_outputs, res, ground_truth