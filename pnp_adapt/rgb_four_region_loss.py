import torch
import torch.nn as nn
import torch.nn.functional as F

from stage2_selfsup.four_region_loss import grad4_l1, hp3, hp5
from stage2_selfsup.four_region_prior import luma


def set_ft_train_mode(model: nn.Module):
    model.train()
    if hasattr(model, "VQVAE"):
        model.VQVAE.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()


def compute_rgb_four_region_loss(
    yr: torch.Tensor,
    yr_ref: torch.Tensor,
    y_target: torch.Tensor,
    flake: torch.Tensor,
    cov: torch.Tensor,
    tree: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Model-agnostic RGB-space disjoint 4-region loss for arbitrary restoration backbones."""
    flk = flake.float()
    cv = cov.float()
    tr = tree.float()
    bg = ((1.0 - flk) * (1.0 - cv) * (1.0 - tr)).clamp_min(0.0)

    g_yr = luma(yr.clamp(0.0, 1.0))
    g_tgt = luma(y_target)

    l_flake_y = ((yr - y_target).abs() * flk).sum() / (3.0 * flk.sum() + 1.0)
    l_peak = ((g_yr - g_tgt).clamp_min(0.0) * flk).sum() / (flk.sum() + 1.0)
    l_flake = l_flake_y + 0.9 * l_peak

    cv_norm = 3.0 * cv.sum() + 1.0
    l_cov = (
        ((yr - yr_ref).abs() * cv).sum() / cv_norm
        + 1.5 * ((hp5(yr) - hp5(yr_ref)).abs() * cv).sum() / cv_norm
        + 1.5 * ((hp3(yr) - hp3(yr_ref)).abs() * cv).sum() / cv_norm
    )

    tr_norm = 3.0 * tr.sum() + 1.0
    l_tree = (
        ((yr - yr_ref).abs() * tr).sum() / tr_norm
        + 1.5 * ((hp3(yr) - hp3(yr_ref)).abs() * tr).sum() / tr_norm
        + 0.8 * ((hp5(yr) - hp5(yr_ref)).abs() * tr).sum() / tr_norm
        + 1.2 * (grad4_l1(yr, yr_ref) * tr).sum() / tr_norm
    )

    bg_norm = 3.0 * bg.sum() + 1.0
    l_bg = ((yr - yr_ref).abs() * bg).sum() / bg_norm + 1.2 * ((hp3(yr) - hp3(yr_ref)).abs() * bg).sum() / bg_norm

    l_lf = (F.avg_pool2d(yr, 16) - F.avg_pool2d(yr_ref, 16)).abs().mean()
    l_floor = (g_tgt - 0.015 - g_yr).clamp_min(0.0).mean() + (yr - yr_ref - 0.02).clamp_min(0.0).mean()

    return {
        "flake": l_flake,
        "cov": l_cov,
        "tree": l_tree,
        "bg": l_bg,
        "lf": l_lf,
        "floor": l_floor,
    }
