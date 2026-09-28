"""Differentiable, per-path SPX condition errors in training-label z units."""
import torch


def condition_consistency_losses(x0, target_z, feature_scaler, condition_scaler):
    """x0: normalized (B, features, time); target_z: normalized (B, 2).

    Use feature 0 (daily close log return), undo the training-only feature
    scaler, then measure each path's compounded return and annualized sample
    standard deviation. Do not average paths before calculating statistics.
    """
    if x0.ndim != 3 or x0.shape[-1] < 2 or target_z.shape != (x0.shape[0], 2):
        raise ValueError('Expected x0 (B, features, time>=2) and conditions (B, 2)')
    r = x0[:, 0, :] * feature_scaler['scale'][0] + feature_scaler['mean'][0]
    actual = torch.stack((torch.expm1(r.sum(dim=-1)), r.std(dim=-1, correction=1) * (252 ** .5)), dim=-1)
    mean = actual.new_tensor(condition_scaler['mean'])
    scale = actual.new_tensor(condition_scaler['scale'])
    errors = ((actual - mean) / scale - target_z).square().mean(dim=0)
    return errors[0], errors[1]
