import torch
from model.embedder import *
import numpy as np
import torch.nn as nn


class GeometryNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            condition_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            use_freq_band=False
    ):
        super().__init__()

        dims = [d_in + condition_in] + dims + [d_out + feature_vector_size]
        self.condition_in = condition_in
        self.embed_fn = None
        self.multires = multires
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch + condition_in

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                    #torch.nn.init.constant_(lin.bias, 0.0)
                    #torch.nn.init.constant_(lin.bias[0], -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

        if use_freq_band:
            self.alpha = 0
        self.use_freq_band = use_freq_band

    def forward(self, input, condition):
        if self.embed_fn is not None:
            input = self.embed_fn(input)
            if self.use_freq_band:
                weight = [1.]

                def cal_weight(alpha, k):
                    if alpha < k:
                        return 0
                    elif alpha < k+1:
                        return (1 - np.cos((alpha - k) * np.pi))/2
                    else:
                        return 1

                for i in range(self.multires):
                    weight.append(cal_weight(self.alpha, i))
                    weight.append(cal_weight(self.alpha, i))
                weight = torch.Tensor(weight).cuda().float().unsqueeze(0).repeat_interleave(3, 1)
                input = input * weight
        if self.condition_in > 0:
            # Currently only support batch_size=1
            # This is because the current implementation of masking in ray tracing doesn't support other batch sizes.
            num_pixels = int(input.shape[0] / condition.shape[0])
            condition = condition.unsqueeze(1).expand(-1, num_pixels, -1).reshape(-1, self.condition_in)
            input = torch.cat([input, condition], dim=1)
        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x, condition):
        x.requires_grad_(True)
        y = self.forward(x, condition)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)