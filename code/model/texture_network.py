import torch
from model.embedder import *
import torch.nn as nn


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            condition_in=53,
            bottleneck=8,
            weight_norm=True,
            multires_view=0,
            multires_pnts=0,
    ):
        super().__init__()

        dims = [d_in + bottleneck + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        self.embedpnts_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        if multires_pnts > 0:
            embedpnts_fn, input_ch_pnts = get_embedder(multires_pnts)
            self.embedpnts_fn = embedpnts_fn
            dims[0] += (input_ch_pnts - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.bottleneck = bottleneck
        if bottleneck != 0:
            self.condition_bottleneck = nn.Linear(condition_in, bottleneck)

    def forward(self, points, normals, feature_vectors, jaw_pose=None):
        if self.embedview_fn is not None:
            normals = self.embedview_fn(normals)

        if self.embedpnts_fn is not None:
            points = self.embedpnts_fn(points)

        num_pixels = int(points.shape[0] / jaw_pose.shape[0])
        jaw_pose = jaw_pose.unsqueeze(1).expand(-1, num_pixels, -1).reshape(points.shape[0], -1)
        if self.bottleneck != 0:
            jaw_pose = self.condition_bottleneck(jaw_pose)
            rendering_input = torch.cat([points, normals, jaw_pose, feature_vectors], dim=-1)
        else:
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)

        return x