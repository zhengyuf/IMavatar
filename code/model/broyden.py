"""
The code is based on https://github.com/xuchen-ethz/snarf/blob/main/lib/model/broyden.py
"""

import torch


def broyden(g, x_init, J_inv_init, max_steps=10, cvg_thresh=1e-5, dvg_thresh=1, eps=1e-6):
    """Find roots of the given function g(x) = 0.
    This function is impleneted based on https://github.com/locuslab/deq.
    Tensor shape abbreviation:
        N: number of points
        D: space dimension
    Args:
        g (function): the function of which the roots are to be determined. shape: [N, D, 1]->[N, D, 1]
        x_init (tensor): initial value of the parameters. shape: [N, D, 1]
        J_inv_init (tensor): initial value of the inverse Jacobians. shape: [N, D, D]
        max_steps (int, optional): max number of iterations. Defaults to 50.
        cvg_thresh (float, optional): covergence threshold. Defaults to 1e-5.
        dvg_thresh (float, optional): divergence threshold. Defaults to 1.
        eps (float, optional): a small number added to the denominator to prevent numerical error. Defaults to 1e-6.
    Returns:
        result (tensor): root of the given function. shape: [N, D, 1]
        diff (tensor): corresponding loss. [N]
        valid_ids (tensor): identifiers of converged points. [N]
    """

    # initialization
    x = x_init.clone().detach()
    J_inv = J_inv_init.clone().detach()

    ids_val = torch.ones(x.shape[0]).bool()

    gx = g(x, mask=ids_val)
    update = -J_inv.bmm(gx)

    x_opt = x
    gx_norm_opt = torch.linalg.norm(gx.squeeze(-1), dim=-1)

    delta_gx = torch.zeros_like(gx)
    delta_x = torch.zeros_like(x)

    ids_val = torch.ones_like(gx_norm_opt).bool()
    for step in range(max_steps):
        # update parameter values
        delta_x[ids_val] = update
        x[ids_val] += delta_x[ids_val]
        delta_gx[ids_val] = g(x, mask=ids_val) - gx[ids_val]
        gx[ids_val] += delta_gx[ids_val]

        # store values with minial loss
        gx_norm = torch.linalg.norm(gx.squeeze(-1), dim=-1)
        ids_opt = gx_norm < gx_norm_opt

        gx_norm_opt[ids_opt] = gx_norm.clone().detach()[ids_opt]
        x_opt[ids_opt] = x.clone().detach()[ids_opt]

        # exclude converged and diverged points from future iterations
        ids_val = (gx_norm_opt > cvg_thresh) & (gx_norm < dvg_thresh)
        if ids_val.sum() <= 0:
            break

        # compute paramter update for next iter
        vT = (delta_x[ids_val]).transpose(-1, -2).bmm(J_inv[ids_val])
        a = delta_x[ids_val] - J_inv[ids_val].bmm(delta_gx[ids_val])
        b = vT.bmm(delta_gx[ids_val])
        b[b >= 0] += eps
        b[b < 0] -= eps
        u = a / b
        J_inv[ids_val] += u.bmm(vT)
        update = -J_inv[ids_val].bmm(gx[ids_val])

    return {'result': x_opt, 'diff': gx_norm_opt, 'valid_ids': gx_norm_opt < cvg_thresh}