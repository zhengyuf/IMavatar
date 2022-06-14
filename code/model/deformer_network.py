import torch
from model.broyden import broyden
from model.embedder import *
import numpy as np
import torch.nn as nn


class ForwardDeformer(nn.Module):
    def __init__(self,
                FLAMEServer,
                d_in,
                dims,
                multires,
                num_exp=50,
                weight_norm=True,
                ghostbone=False):
        super().__init__()

        self.FLAMEServer = FLAMEServer
        # pose correctives, expression blendshapes and linear blend skinning weights
        d_out = 36 * 3 + num_exp * 3

        self.num_exp = num_exp
        dims = [d_in] + dims + [d_out]
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.init_bones = [0, 1, 2] if not ghostbone else [0, 2, 3]         # shoulder/identity, head and jaw

        for l in range(0, self.num_layers - 2):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if multires > 0 and l == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.blendshapes = nn.Linear(dims[self.num_layers - 2], d_out)
        self.skinning_linear = nn.Linear(dims[self.num_layers - 2], dims[self.num_layers - 2])
        self.skinning = nn.Linear(dims[self.num_layers - 2], 6 if ghostbone else 5)
        torch.nn.init.constant_(self.skinning_linear.bias, 0.0)
        torch.nn.init.normal_(self.skinning_linear.weight, 0.0, np.sqrt(2) / np.sqrt(dims[self.num_layers - 2]))
        if weight_norm:
            self.skinning_linear = nn.utils.weight_norm(self.skinning_linear)
        # initialize blendshapes to be zero, and skinning weights to be equal for every bone (after softmax activation)
        torch.nn.init.constant_(self.blendshapes.bias, 0.0)
        torch.nn.init.constant_(self.blendshapes.weight, 0.0)
        torch.nn.init.constant_(self.skinning.bias, 0.0)
        torch.nn.init.constant_(self.skinning.weight, 0.0)

        self.ghostbone = ghostbone

    def query_weights(self, pnts_c, mask=None):
        if mask is not None:
            pnts_c = pnts_c[mask]
        if self.embed_fn is not None:
            pnts_c = self.embed_fn(pnts_c)

        x = pnts_c

        for l in range(0, self.num_layers - 2):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            x = self.softplus(x)

        blendshapes = self.blendshapes(x)
        posedirs = blendshapes[:, :36 * 3]
        shapedirs = blendshapes[:, 36 * 3: 36 * 3 + self.num_exp * 3]
        lbs_weight = self.skinning(self.softplus(self.skinning_linear(x)))
        lbs_weights = torch.nn.functional.softmax(20 * lbs_weight, dim=1)
        return shapedirs.reshape(-1, 3, self.num_exp), posedirs.reshape(-1, 4*9, 3), lbs_weights.reshape(-1, 6 if self.ghostbone else 5)

    def forward_lbs(self, pnts_c, pose_feature, betas, transformations, mask=None):
        shapedirs, posedirs, lbs_weights = self.query_weights(pnts_c, mask)
        pts_p = self.FLAMEServer.forward_pts(pnts_c, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, dtype=torch.float32, mask=mask)
        return pts_p

    def init(self, pnts_p, transformations):
        # pnts_p: num_points, 3
        # transformations: num_points, num_joints, 4, 4 (only support image batch size = 1)
        num_points = pnts_p.shape[0]
        num_joint = transformations.shape[1]

        pc_init = []
        for i in self.init_bones:
            w = torch.zeros((num_points, num_joint), device=pnts_p.device)
            w[:, i] = 1

            pc_init.append(self.FLAMEServer.inverse_skinning_pts(pnts_p, transformations, w))

        pc_init = torch.stack(pc_init, dim=1)
        # pc_init: num_batch * num_point, num_init, 3
        return pc_init

    def search(self, xd, xc_init, pose_feature, betas, transformations):
        """Search correspondences.
        Args:
            xd (tensor): deformed points in batch. shape: [N, D]
            xc_init (tensor): deformed points in batch. shape: [N, I, D]
        Returns:
            xc_opt (tensor): canonoical correspondences of xd. shape: [B, N, I, D]
            valid_ids (tensor): identifiers of converged points. [B, N, I]
        """
        # reshape to [B,?,D] for other functions
        n_point, n_init, n_dim = xc_init.shape
        xc_init = xc_init.reshape(n_point * n_init, n_dim)
        xd_tgt = xd.repeat_interleave(n_init, dim=0)
        transformations_repeat = transformations.repeat_interleave(n_init, dim=0)
        pose_feature_repeat = pose_feature.repeat_interleave(n_init, dim=0)
        betas_repeat = betas.repeat_interleave(n_init, dim=0)

        # compute initial approximated jacobians using inverse-skinning
        _, _, w = self.query_weights(xc_init)
        # m: num_points, j: num_joints, b: 1, k: 16 (4x4)

        J_inv = torch.einsum('mj, mjk->mk', [w, transformations_repeat.view(-1, 6 if self.ghostbone else 5, 16)]).view(-1, 4, 4)
        J_inv_init = J_inv[:, :3, :3].inverse()

        # reshape init to [?,D,...] for broyden
        xc_init = xc_init.reshape(-1, n_dim, 1)
        # J_inv_init: num_points, 3, 3

        # construct function for root finding
        def _func(xc_opt, mask=None):
            xc_opt = xc_opt.reshape(n_point * n_init, n_dim)
            xd_opt = self.forward_lbs(xc_opt, pose_feature_repeat, betas_repeat, transformations_repeat, mask=mask)
            error = xd_opt - xd_tgt[mask]
            # reshape to [?,D,1] for boryden
            error = error.unsqueeze(-1)
            return error

        # run broyden without grad
        with torch.no_grad():
            result = broyden(_func, xc_init, J_inv_init)

        # reshape back to [B,N,I,D]
        xc_opt = result["result"].reshape(n_point, n_init, n_dim)
        result["valid_ids"] = result["valid_ids"].reshape(n_point, n_init)

        return xc_opt, result

    def forward(self, xd, pose_feature, betas, transformations):
        """Given deformed point return its canonical correspondence"""
        xc_init = self.init(xd, transformations)

        xc_opt, others = self.search(xd, xc_init, pose_feature, betas, transformations)

        return xc_opt, others
