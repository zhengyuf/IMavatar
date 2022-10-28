"""
The code is based on https://github.com/lioryariv/idr.
# Copyright (c) ETH Zurich and its affiliates. All rights reserved.
# licensed under the LICENSE file in the root directory of this source tree.
# Code adapted / written by Yufeng Zheng.
 """

import os
from glob import glob
import torch
import math

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def split_input(model_input, total_pixels, n_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    # n_pixels = 10000
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        if 'object_mask' in data and data['object_mask'][0] is not None:
            data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        if 'semantics' in data and data['semantics'][0] is not None:
            data['semantics'] = torch.index_select(model_input['semantics'], 1, indx)
        split.append(data)
    return split

def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            try:
                model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                                 1).reshape(batch_size * total_pixels, -1)
            except:
                model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                                 1).reshape(batch_size, -1)

    return model_outputs

def get_split_name(splits):
    # example splits: [MVI_1812, MVI_1813]
    # example output: MVI_1812+MVI_1813
    name = ''
    for s in splits:
        name += str(s)
        name += '+'
    assert len(name) > 1
    return name[:-1]


