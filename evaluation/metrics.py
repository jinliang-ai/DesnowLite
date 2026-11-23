import torch
import torch.nn.functional as F

from stage1_pretrain.losses import gauss_win, ssim_map


def _maxpool(z: torch.Tensor, kh: int, kw: int) -> torch.Tensor:
    return F.max_pool2d(F.pad(z, (kw // 2, kw // 2, kh // 2, kh // 2), mode="replicate"), (kh, kw), 1)


def open_rect(x: torch.Tensor, kh: int, kw: int) -> torch.Tensor:
    return _maxpool(-_maxpool(-x, kh, kw), kh, kw)


def luma(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def saturation(x: torch.Tensor) -> torch.Tensor:
    mx, mn = x.amax(1, keepdim=True), x.amin(1, keepdim=True)
    return (mx - mn) / mx.clamp_min(1e-3)


def accumulation_mask(
    x: torch.Tensor,
    k: int = 21,
    bright: float = 0.7,
    grow: int = 5,
    dens_win: int = 31,
    dens: float = 0.25,
    sat_cover: float = 0.4,
) -> torch.Tensor:
    g = luma(x)
    white = (g > bright) & (saturation(x) < sat_cover)
    big = (open_rect(g, k, k) > bright) & (saturation(x) < sat_cover)
    p = dens_win // 2
    dense = F.avg_pool2d(F.pad(white.float(), (p, p, p, p), mode="replicate"), dens_win, 1) > dens
    cover = big | (dense & white)
    return _maxpool(cover.float(), 2 * grow + 1, 2 * grow + 1) > 0


def eval_flake_support(
    x: torch.Tensor,
    thr: float = 0.05,
    k: int = 11,
    peak_win: int = 5,
    grow: int = 2,
):
    g = luma(x)
    bg = open_rect(g, k, k)
    th = g - bg
    ok = (saturation(x) < 0.2) & ~accumulation_mask(x)
    peak = (th >= _maxpool(th, peak_win, peak_win)) & (th > thr) & ok
    region = _maxpool(peak.float(), 2 * grow + 1, 2 * grow + 1) > 0
    return region & (th > 0.6 * thr) & ok, th, bg


def compute_r_flake(x: torch.Tensor, y: torch.Tensor, flake_mask: torch.Tensor | None = None, k: int = 11) -> float:
    """Contrast-normalized fine-flake removal rate R_flake in [0, 1]."""
    if flake_mask is None:
        flake_mask, _, _ = eval_flake_support(x, k=k)
    gx, gy = luma(x), luma(y)
    tx, ty = gx - open_rect(gx, k, k), gy - open_rect(gy, k, k)
    if flake_mask.sum() == 0:
        return float("nan")
    ref = (~flake_mask) & (tx > 0.03)
    scale = (ty[ref].sum() / tx[ref].sum().clamp_min(1e-6)).clamp(0.3, 3.0) if ref.sum() > 50 else torch.tensor(1.0, device=x.device)
    r = (ty[flake_mask].clamp_min(0).sum() / (tx[flake_mask].sum() * scale).clamp_min(1e-6)).clamp(0, 1)
    return float(1 - r)


def compute_psnr_ssim(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor) -> tuple[float, float]:
    y = (pred_rgb.clamp(0, 1) * 255.0).round() / 255.0
    t = gt_rgb.clamp(0, 1)
    mse = (y - t).square().mean().clamp_min(1e-10)
    psnr = float(-10.0 * torch.log10(mse))
    win = gauss_win(y.device)
    ssim = float(ssim_map(y, t, win).mean())
    return psnr, ssim
