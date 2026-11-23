import torch
import torch.nn.functional as F

from .four_region_prior import luma


def hp5(z: torch.Tensor) -> torch.Tensor:
    return z - F.avg_pool2d(F.pad(z, (2, 2, 2, 2), mode="replicate"), 5, 1)


def hp3(z: torch.Tensor) -> torch.Tensor:
    return z - F.avg_pool2d(F.pad(z, (1, 1, 1, 1), mode="replicate"), 3, 1)


def grad4_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dh = ((a[..., :, 1:] - a[..., :, :-1]) - (b[..., :, 1:] - b[..., :, :-1])).abs()
    dv = ((a[..., 1:, :] - a[..., :-1, :]) - (b[..., 1:, :] - b[..., :-1, :])).abs()
    dd1 = ((a[..., 1:, 1:] - a[..., :-1, :-1]) - (b[..., 1:, 1:] - b[..., :-1, :-1])).abs()
    dd2 = ((a[..., 1:, :-1] - a[..., :-1, 1:]) - (b[..., 1:, :-1] - b[..., :-1, 1:])).abs()
    return (
        F.pad(dh, (0, 1, 0, 0))
        + F.pad(dv, (0, 0, 0, 1))
        + F.pad(dd1, (0, 1, 0, 1))
        + F.pad(dd2, (1, 0, 0, 1))
    ) * 0.25


def compute_four_region_loss(
    yr: torch.Tensor,
    auxr: dict,
    yr_ref: torch.Tensor,
    aux_ref: dict,
    a_init: torch.Tensor,
    y_target: torch.Tensor,
    a_hat_tot: torch.Tensor,
    flake: torch.Tensor,
    cov: torch.Tensor,
    tree: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Disjoint four-region self-supervised loss (L_flake, L_tree, L_anc, L_lock)."""
    flk = flake.float()
    cv = cov.float()
    tr = tree.float()
    bg = ((1.0 - flk) * (1.0 - cv) * (1.0 - tr)).clamp_min(0.0)

    yc = auxr["y_coarse"].clamp(0.0, 2.0)
    yc_ref = aux_ref["y_coarse"].clamp(0.0, 2.0)
    r_pred, r_ref = auxr["r"], aux_ref["r"]

    g_yr = luma(yr.clamp(0.0, 1.0))
    g_tgt = luma(y_target)

    # L_flake(Omega_flake, Omega): flake suppression + global floor guard
    l_flk_y = ((yr - y_target).abs() * flk).sum() / (3.0 * flk.sum() + 1.0)
    l_flk_yc = ((yc - y_target).abs() * flk).sum() / (3.0 * flk.sum() + 1.0)
    l_peak = ((g_yr - g_tgt).clamp_min(0.0) * flk).sum() / (flk.sum() + 1.0)
    l_floor = (g_tgt - 0.020 - g_yr).clamp_min(0.0).mean()
    loss_flake = l_flk_y + 0.55 * l_flk_yc + 0.95 * l_peak + 2.0 * l_floor

    # L_tree(Omega_tree): semantic canopy sharpness lock
    tr_norm = 3.0 * tr.sum() + 1.0
    l_tr_pix = ((yr - yr_ref).abs() * tr).sum() / tr_norm
    l_tr_hp3 = ((hp3(yr) - hp3(yr_ref)).abs() * tr).sum() / tr_norm
    l_tr_hp5 = ((hp5(yr) - hp5(yr_ref)).abs() * tr).sum() / tr_norm
    l_tr_grad = (grad4_l1(yr, yr_ref) * tr).sum() / tr_norm
    l_tr_yc = (((yc - yc_ref).abs() + 1.2 * (hp3(yc) - hp3(yc_ref)).abs()) * tr).sum() / tr_norm
    l_tr_res = ((r_pred - r_ref).abs() * tr).sum() / tr_norm
    g_ref = luma(yr_ref.clamp(0.0, 1.0))
    l_tr_bright = ((g_ref - g_yr).clamp_min(0.0) * tr).sum() / (tr.sum() + 1.0)
    loss_tree = (
        1.4 * l_tr_pix
        + 1.6 * l_tr_hp3
        + 1.0 * l_tr_hp5
        + 1.2 * l_tr_grad
        + 1.0 * l_tr_yc
        + 1.2 * l_tr_res
        + 2.2 * l_tr_bright
    )

    # L_anc(Omega_cov, Omega_bg): snow cover & clean background radiometric anchor
    cv_norm = 3.0 * cv.sum() + 1.0
    l_cov = (
        ((yr - yr_ref).abs() * cv).sum() / cv_norm
        + ((hp5(yr) - hp5(yr_ref)).abs() * cv).sum() / cv_norm
        + 0.8 * ((hp3(yr) - hp3(yr_ref)).abs() * cv).sum() / cv_norm
    )
    bg_norm = 3.0 * bg.sum() + 1.0
    l_bg = (
        1.6 * ((yr - yr_ref).abs() * bg).sum() / bg_norm
        + ((yc - yc_ref).abs() * bg).sum() / bg_norm
        + 0.8 * ((r_pred - r_ref).abs() * bg).sum() / bg_norm
        + 1.2 * ((hp3(yr) - hp3(yr_ref)).abs() * bg).sum() / bg_norm
    )
    loss_anc = l_cov + l_bg

    # L_lock(a, o_g, g_fill): opacity calibration & coarse fill-gate lock
    a_pred = torch.sigmoid(auxr["logit"])
    a_tgt_flk = torch.maximum(a_init, a_hat_tot)
    l_opa_flk = ((a_pred - a_tgt_flk).abs() * flk).sum() / (flk.sum() + 1.0)
    l_opa_cov = (a_pred * cv).sum() / (cv.sum() + 1.0)
    l_opa_bg = ((a_pred - a_init).abs() * bg).sum() / (bg.sum() + 1.0)
    l_opa_tr = (
        2.0 * ((a_pred - a_init).clamp_min(0.0) * tr).sum() / (tr.sum() + 1.0)
        + 1.2 * ((a_pred - a_init).abs() * tr).sum() / (tr.sum() + 1.0)
    )

    o_g, gate = auxr["o_g"], auxr["gate"]
    o_g_ref, gate_ref = aux_ref["o_g"], aux_ref["gate"]
    non_flk = (1.0 - flk).clamp_min(0.0)
    l_gate_nonflk = (((o_g - o_g_ref).abs() + 2.0 * (gate - gate_ref).abs()) * non_flk).sum() / (non_flk.sum() + 1.0)
    l_gate_tr = 2.2 * ((gate - gate_ref).clamp_min(0.0) * tr).sum() / (tr.sum() + 1.0)
    loss_lock = 1.8 * l_opa_flk + 1.2 * l_opa_cov + 1.5 * l_opa_bg + l_opa_tr + 1.6 * l_gate_nonflk + l_gate_tr

    return {
        "flake": loss_flake,
        "tree": loss_tree,
        "anc": loss_anc,
        "lock": loss_lock,
        "flake_frac": flk.mean(),
        "cov_frac": cv.mean(),
        "tree_frac": tr.mean(),
    }
