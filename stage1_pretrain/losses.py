import torch
import torch.nn.functional as F


def gauss_win(dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.exp(-(torch.arange(11, dtype=torch.float32, device=dev) - 5) ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    return g.view(1, 1, 1, 11).repeat(3, 1, 1, 1), g.view(1, 1, 11, 1).repeat(3, 1, 1, 1)


def ssim_map(x: torch.Tensor, y: torch.Tensor, win: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    conv = lambda z: F.conv2d(F.conv2d(z, win[0], groups=3), win[1], groups=3)
    mx, my = conv(x), conv(y)
    vx, vy, cxy = conv(x * x) - mx ** 2, conv(y * y) - my ** 2, conv(x * y) - mx * my
    return ((2 * mx * my + 1e-4) * (2 * cxy + 9e-4)) / ((mx ** 2 + my ** 2 + 1e-4) * (vx + vy + 9e-4))


def stage1_criterion(
    y: torch.Tensor,
    aux: dict,
    t: torch.Tensor,
    m: torch.Tensor | None,
    win: tuple[torch.Tensor, torch.Tensor],
    a_w: float = 0.0,
    base_w: float = 0.0,
    psnr_only: bool = False,
) -> torch.Tensor:
    if m is not None and m.shape[1] > 1:
        m = m[:, :1]
    e = y - t
    mse = e.square().mean((1, 2, 3))
    if psnr_only:
        return 0.1 * torch.log(mse + 1e-8).mean()

    loss = torch.sqrt(e.square() + 1e-6).mean()
    loss = loss + 0.5 * F.avg_pool2d(e, 8).abs().mean()
    loss = loss + 0.02 * torch.log(mse + 1e-8).mean()
    loss = loss + 0.05 * (1 - ssim_map(y, t, win).mean())
    if "y_coarse" in aux:
        loss = loss + 0.1 * torch.sqrt((aux["y_coarse"] - t).square() + 1e-6).mean()
    if m is not None and "logit" in aux:
        loss = loss + 0.05 * F.binary_cross_entropy_with_logits(aux["logit"], m)
    if m is not None and "logit" in aux and a_w > 0:
        a = torch.sigmoid(aux["logit"])
        wa = 1 + 2 * ((m > 0.05) & (m <= 0.6)).float()
        loss = loss + a_w * (wa * (a - m).abs()).mean()
        gx = lambda z: z[..., :, 1:] - z[..., :, :-1]
        gy = lambda z: z[..., 1:, :] - z[..., :-1, :]
        loss = loss + 0.25 * a_w * ((gx(a) - gx(m)).abs().mean() + (gy(a) - gy(m)).abs().mean())
    if m is not None and "base" in aux and base_w > 0:
        cw = (m < 0.004).float()
        loss = loss + base_w * ((aux["base"] - t).abs() * cw).sum() / (3 * cw.sum() + 1)
    return loss
