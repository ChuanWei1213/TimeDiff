"""Differentiable, per-path condition errors in training-label z units.

SPX (default): the condition is computed from feature 0 (the single close log return).
Multi-asset:   pass close_indices (e.g. Dow10: 0, 3, ..., 27); the condition is computed
               from the daily-rebalanced equal-weight portfolio
               r_p = log(mean_i exp(r_i)) = logsumexp(r) - log(n), which is differentiable.
"""
import math

import torch


def condition_consistency_losses(x0, target_z, feature_scaler, condition_scaler, close_indices=(0,)):
    """x0: normalized (B, features, time); target_z: normalized (B, 2).

    Undo the training-only feature scaler, then measure each path's compounded return
    and annualized sample standard deviation. Do not average paths before calculating
    statistics.
    """
    if x0.ndim != 3 or x0.shape[-1] < 2 or target_z.shape != (x0.shape[0], 2):
        raise ValueError('Expected x0 (B, features, time>=2) and conditions (B, 2)')
    idx = list(close_indices)
    mean = x0.new_tensor([feature_scaler['mean'][i] for i in idx]).view(1, -1, 1)
    scale = x0.new_tensor([feature_scaler['scale'][i] for i in idx]).view(1, -1, 1)
    r = x0[:, idx, :] * scale + mean                                   # (B, n_assets, T)
    if len(idx) == 1:
        rp = r[:, 0, :]
    else:
        rp = torch.logsumexp(r, dim=1) - math.log(len(idx))           # (B, T)
    actual = torch.stack((torch.expm1(rp.sum(dim=-1)), rp.std(dim=-1, correction=1) * (252 ** .5)), dim=-1)
    c_mean = actual.new_tensor(condition_scaler['mean'])
    c_scale = actual.new_tensor(condition_scaler['scale'])
    errors = ((actual - c_mean) / c_scale - target_z).square().mean(dim=0)
    return errors[0], errors[1]
