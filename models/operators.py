import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(1, keepdim=True)
        var = (x - mu).square().mean(1, keepdim=True)
        return (x - mu) / torch.sqrt(var + 1e-6) * self.w + self.b


def opening(x: torch.Tensor, k: int) -> torch.Tensor:
    p = k // 2
    mp = lambda z: F.max_pool2d(F.max_pool2d(z, (1, k), 1), (k, 1), 1)
    eroded = -mp(F.pad(-x, (p, p, p, p), mode="replicate"))
    return mp(F.pad(eroded, (p, p, p, p), mode="replicate"))


def snow_prior(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x - opening(x, 3), x - opening(x, 7)], 1)


def nconv(src: torch.Tensor, wv: torch.Tensor, k: int) -> torch.Tensor:
    ap = lambda z: F.avg_pool2d(
        F.avg_pool2d(z, (1, k), 1, (0, k // 2), count_include_pad=False),
        (k, 1), 1, (k // 2, 0), count_include_pad=False,
    )
    return ap(src * wv) / ap(wv)


_interp_cache: dict = {}


def _interp_weights(n_in: int, n_out: int, mode: str, ref: torch.Tensor) -> torch.Tensor:
    key = (n_in, n_out, mode, ref.device, ref.dtype)
    w = _interp_cache.get(key)
    if w is None:
        eye = torch.eye(n_in, device=ref.device, dtype=ref.dtype)[None, None]
        w = F.interpolate(eye, size=(n_out, n_in), mode=mode, align_corners=False)[0, 0]
        _interp_cache[key] = w
    return w


def upsample(x: torch.Tensor, size: tuple[int, int], mode: str = "bicubic") -> torch.Tensor:
    wy = _interp_weights(x.shape[-2], size[0], mode, x)
    wx = _interp_weights(x.shape[-1], size[1], mode, x)
    return wy @ x @ wx.T


def block_pool(z: torch.Tensor, c: int, op: str) -> torch.Tensor:
    B, C, h, w = z.shape
    v = z.reshape(B, C, h // c, c, w // c, c)
    return v.mean((3, 5)) if op == "mean" else v.amax((3, 5))


def cell_stats(low: torch.Tensor) -> torch.Tensor:
    B, _, h, w = low.shape
    mean = F.avg_pool2d(low, 16)
    v = low.reshape(B, 3, h // 16, 16, w // 16, 16).permute(0, 1, 2, 4, 3, 5).reshape(B, 3, h // 16, w // 16, 256)
    v = v.sort(-1).values[..., [5, 25, 128, 230, 250]]
    return torch.cat([v.permute(0, 1, 4, 2, 3).reshape(B, 15, h // 16, w // 16), mean], 1)


def make_low128(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(F.avg_pool2d(x, 2), size=(128, 128), mode="bilinear", align_corners=False)


class Block(nn.Module):
    def __init__(self, c: int, strip: bool = False):
        super().__init__()
        self.n1 = LayerNorm2d(c)
        self.pw1 = nn.Conv2d(c, 2 * c, 1)
        self.dw = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.strip = None
        if strip:
            self.strip = nn.Sequential(
                nn.Conv2d(2 * c, 2 * c, (1, 11), padding=(0, 5), groups=2 * c),
                nn.Conv2d(2 * c, 2 * c, (11, 1), padding=(5, 0), groups=2 * c),
            )
        self.sca = nn.Conv2d(c, c, 1)
        self.pw2 = nn.Conv2d(c, c, 1)
        self.n2 = LayerNorm2d(c)
        self.ffn1 = nn.Conv2d(c, 2 * c, 1)
        self.ffn2 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw1(self.n1(x))
        d = self.dw(y)
        if self.strip is not None:
            d = d + self.strip(y)
        a, b = d.chunk(2, 1)
        y = a * b
        y = y * self.sca(y.mean((2, 3), keepdim=True))
        x = x + self.beta * self.pw2(y)
        a, b = self.ffn1(self.n2(x)).chunk(2, 1)
        return x + self.gamma * self.ffn2(a * b)
