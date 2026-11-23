import torch
import torch.nn.functional as F


def _maxpool(z: torch.Tensor, kh: int, kw: int) -> torch.Tensor:
    return F.max_pool2d(F.pad(z, (kw // 2, kw // 2, kh // 2, kh // 2), mode="replicate"), (kh, kw), 1)


def _minpool(z: torch.Tensor, kh: int, kw: int) -> torch.Tensor:
    return -_maxpool(-z, kh, kw)


def open_rect(x: torch.Tensor, kh: int, kw: int) -> torch.Tensor:
    return _maxpool(_minpool(x, kh, kw), kh, kw)


def luma(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def saturation(x: torch.Tensor) -> torch.Tensor:
    mx, mn = x.amax(1, keepdim=True), x.amin(1, keepdim=True)
    return (mx - mn) / mx.clamp_min(1e-3)


def _shift(z: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    p = max(abs(dy), abs(dx))
    if p == 0:
        return z
    zp = F.pad(z, (p, p, p, p), mode="replicate")
    H, W = z.shape[-2:]
    return zp[..., p + dy : p + dy + H, p + dx : p + dx + W]


def fast_axis_bg(g: torch.Tensor, r: int = 4) -> torch.Tensor:
    p0 = torch.maximum(_shift(g, 0, r), _shift(g, 0, -r))
    p1 = torch.maximum(_shift(g, r, 0), _shift(g, -r, 0))
    p2 = torch.maximum(_shift(g, r, r), _shift(g, -r, -r))
    p3 = torch.maximum(_shift(g, r, -r), _shift(g, -r, r))
    M1, m1 = torch.maximum(p0, p1), torch.minimum(p0, p1)
    M2, m2 = torch.maximum(p2, p3), torch.minimum(p2, p3)
    return torch.maximum(torch.minimum(M1, M2), torch.maximum(m1, m2))


def safe_nconv_bg(yr: torch.Tensor, valid: torch.Tensor, k: int = 7) -> torch.Tensor:
    p = k // 2
    num = F.avg_pool2d(F.pad(yr * valid, (p, p, p, p), mode="replicate"), k, 1)
    den = F.avg_pool2d(F.pad(valid, (p, p, p, p), mode="replicate"), k, 1)
    op5 = open_rect(yr, 5, 5)
    avg9 = F.avg_pool2d(F.pad(yr, (4, 4, 4, 4), mode="replicate"), 9, 1)
    fallback = torch.maximum(op5, 0.78 * avg9)
    return torch.where(den > 0.08, num / den.clamp_min(0.08), fallback)


def four_region_targets(
    y_ref: torch.Tensor,
    x: torch.Tensor,
    sem_tree: torch.Tensor,
    thr: float = 0.032,
):
    """Partition Omega = Omega_flake perp Omega_cov perp Omega_tree perp Omega_bg and compute (Y_target, a_hat)."""
    yr = y_ref.clamp(0.0, 1.0)
    g_yr = luma(yr)
    g_x = luma(x.clamp(0.0, 1.0))
    sat = torch.maximum(saturation(yr), saturation(x.clamp(0.0, 1.0)))

    bg_g2 = fast_axis_bg(g_yr, r=2)
    bg_g4 = torch.minimum(bg_g2, fast_axis_bg(g_yr, r=4))
    bg_g6 = torch.minimum(bg_g4, fast_axis_bg(g_yr, r=6))
    bg_x4 = torch.minimum(fast_axis_bg(g_x, r=2), fast_axis_bg(g_x, r=4))

    res2 = (g_yr - bg_g2).clamp_min(0.0)
    res4 = (g_yr - bg_g4).clamp_min(0.0)
    res6 = (g_yr - bg_g6).clamp_min(0.0)
    res_x4 = (g_x - bg_x4).clamp_min(0.0)

    tree = _maxpool((sem_tree.float() > 0.35).float(), 5, 5) > 0

    big = ((open_rect(g_yr, 9, 9) > 0.62) | (open_rect(g_yr, 13, 13) > 0.56)) & (sat < 0.38)
    horiz_ledge = (open_rect(g_yr, 3, 13) > 0.64) & (sat < 0.38)
    cov = (_maxpool((big | horiz_ledge).float(), 5, 5) > 0) & (~tree)

    ortho_open = torch.maximum(
        open_rect(g_yr, 7, 7),
        torch.maximum(open_rect(g_yr, 1, 9), open_rect(g_yr, 9, 1)),
    )
    hi7 = _maxpool(open_rect(g_yr, 3, 3), 7, 7)
    lo5 = _minpool(g_yr, 5, 5)
    sign_or_sharp = (hi7 > 0.62) & (lo5 < 0.18)

    ok = (sat < 0.32) & (~cov) & (~tree)
    is_fine = (ortho_open - bg_g6) < 0.080
    is_fine_46 = is_fine & (~sign_or_sharp) & ((ortho_open - bg_g6) < 0.070)

    peak2 = (res2 >= _maxpool(res2, 3, 3) - 1e-4) & (res2 > thr * 0.85) & ok & is_fine
    peak4 = (
        (res4 >= _maxpool(res4, 3, 3) - 1e-4)
        & ((res4 > thr) | ((res4 > 0.65 * thr) & (res_x4 > thr)))
        & ok
        & is_fine_46
    )
    peak6 = (res6 >= _maxpool(res6, 5, 5) - 1e-4) & (res6 > 1.15 * thr) & ok & is_fine_46

    flake2 = (_maxpool(peak2.float(), 3, 3) > 0) & (res2 > 0.40 * thr) & ok
    flake4 = (_maxpool(peak4.float(), 5, 5) > 0) & (res4 > 0.40 * thr) & ok & (~sign_or_sharp)
    flake6 = (_maxpool(peak6.float(), 7, 7) > 0) & (res6 > 0.45 * thr) & ok & is_fine_46
    flake = (flake2 | flake4 | flake6) & (~cov) & (~tree)

    flk_f = flake.float()
    tf = tree.float()
    valid = (1.0 - _maxpool(flk_f, 3, 3)).clamp_min(0.0) * (1.0 - tf * 0.90)
    bg_rgb_fine = safe_nconv_bg(yr, valid, k=5)
    bg_rgb_med = safe_nconv_bg(yr, valid, k=7)

    w2 = (res2 / (res6 + 1e-4)).clamp(0.0, 1.0)
    bg_rgb = w2 * bg_rgb_fine + (1.0 - w2) * bg_rgb_med

    g_bg_rgb = luma(bg_rgb).clamp_min(1e-3)
    scale_luma = (bg_g6 / g_bg_rgb).clamp(0.55, 1.0)
    target_flake = torch.minimum(yr, torch.maximum(bg_rgb * scale_luma, yr - 0.35))

    alpha_blend = ((res6 - 0.30 * thr) / (0.60 * thr)).clamp(0.0, 1.0) * flk_f * (1.0 - tf)
    y_target = (1.0 - alpha_blend) * yr + alpha_blend * target_flake

    g_tgt = luma(target_flake)
    a_hat = ((g_yr - g_tgt).clamp_min(0.0) / (1.0 - g_tgt + 0.06)).clamp(0.0, 0.92) * flk_f * (1.0 - tf)
    return y_target, a_hat, flake, cov, tree


@torch.no_grad()
def shielded_cutmix(
    x: torch.Tensor,
    p_density: float = 0.0035,
    shield_mask: torch.Tensor | None = None,
):
    B, _, H, W = x.shape
    dev = x.device
    s1 = (torch.rand(B, 1, H, W, device=dev) < p_density).float() * torch.empty(B, 1, H, W, device=dev).uniform_(0.35, 0.88)
    f1 = F.avg_pool2d(F.pad(_maxpool(s1, 3, 3), (1, 1, 1, 1), mode="replicate"), 3, 1) * 1.4

    s2 = (torch.rand(B, 1, H, W, device=dev) < (p_density * 0.45)).float() * torch.empty(B, 1, H, W, device=dev).uniform_(0.40, 0.92)
    f2 = F.avg_pool2d(F.pad(_maxpool(s2, 5, 5), (2, 2, 2, 2), mode="replicate"), 5, 1) * 1.5

    s3 = (torch.rand(B, 1, H, W, device=dev) < (p_density * 0.30)).float() * torch.empty(B, 1, H, W, device=dev).uniform_(0.40, 0.88)
    st = s3 + 0.65 * (_shift(s3, -4, -2) + _shift(s3, -2, -1) + _shift(s3, 2, 1) + _shift(s3, 4, 2))
    f3 = F.avg_pool2d(F.pad(st.clamp(0.0, 1.0), (1, 1, 1, 1), mode="replicate"), 3, 1)

    m_syn = (f1 + f2 + f3).clamp(0.0, 0.90)
    if shield_mask is not None:
        allow = (1.0 - _maxpool(shield_mask.float(), 5, 5)).clamp_min(0.0)
        m_syn = m_syn * allow
    s_col = torch.empty(B, 3, 1, 1, device=dev).uniform_(0.92, 1.00)
    x_syn = (x * (1.0 - m_syn) + s_col * m_syn).clamp(0.0, 1.0)
    return x_syn, m_syn
