import contextlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .operators import Block, block_pool, cell_stats, make_low128, nconv, opening, snow_prior, upsample


def _small_head(cin: int) -> nn.Sequential:
    h = nn.Sequential(nn.Conv2d(cin, 16, 1), nn.SiLU(), Block(16), nn.Conv2d(16, 1, 3, padding=1))
    nn.init.constant_(h[2].beta, 0.3)
    nn.init.constant_(h[2].gamma, 0.3)
    nn.init.zeros_(h[-1].weight)
    nn.init.zeros_(h[-1].bias)
    return h


class VeilBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.SiLU(), nn.Conv2d(16, 16, 3, 2, 1), nn.SiLU())
        self.head = nn.Sequential(
            nn.Conv2d(40, 32, 1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1, padding_mode="replicate"), nn.SiLU(),
            nn.Conv2d(32, 6, 1),
        )

    def forward(self, low: torch.Tensor) -> torch.Tensor:
        B, _, h, w = low.shape
        dark = -F.max_pool2d(-low.amin(1, keepdim=True), 5, 1, 2)
        bright = low.amax(1, keepdim=True)
        q = lambda z, idx: (
            z.reshape(B, h // 16, 16, w // 16, 16).permute(0, 1, 3, 2, 4)
            .reshape(B, h // 16, w // 16, 256).sort(-1).values[..., idx].permute(0, 3, 1, 2)
        )
        st = [F.avg_pool2d(self.feat(low), 4), cell_stats(low), q(dark, [5, 25, 128]), q(bright, [128, 230, 250])]
        return self.head(torch.cat(st, 1))


class FlakeCal(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(13, 16, 1), nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, padding_mode="replicate"), nn.SiLU(),
            nn.Conv2d(16, 6, 1),
        )

    def forward(self, xp: torch.Tensor, a: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        w = ((a - 0.7) / 0.25).clamp(0, 1)
        cnt = block_pool(w, 64, "mean")
        flake = block_pool(xp * w, 64, "mean") / (cnt + 1e-4)
        dark = -block_pool(-(xp + (a > 0.1).float()), 64, "max")
        feat = torch.cat([flake, cnt, dark, block_pool(alpha, 64, "mean"), block_pool(beta, 64, "mean")], 1)
        return self.head(feat)


class FullImageVeilNet(nn.Module):
    def __init__(self, widths=(16, 24, 32), mid: int = 2):
        super().__init__()
        w0, w1, w2 = widths
        self.stem = nn.Conv2d(10, w0, 3, 2, 1)
        self.enc1 = Block(w0)
        self.down1 = nn.Conv2d(w0, w1, 3, 2, 1)
        self.enc2 = Block(w1)
        self.down2 = nn.Conv2d(w1, w2, 3, 2, 1)
        self.mid = nn.Sequential(*[Block(w2) for _ in range(mid)])
        self.glob_mlp = nn.Sequential(nn.Linear(w2 * 2 + 42, 64), nn.SiLU(), nn.Linear(64, 6))
        self.grid_head = nn.Sequential(
            nn.Conv2d(w2, w1, 3, 1, 1, padding_mode="replicate"), nn.SiLU(), nn.Conv2d(w1, 6, 1)
        )

    def forward(self, low128: torch.Tensor) -> torch.Tensor:
        B = low128.shape[0]
        with torch.no_grad():
            op = opening(low128, 5)
            dark = -F.max_pool2d(-op.amin(1, keepdim=True), 5, 1, 2)
            bright = op.amax(1, keepdim=True)
            yy = torch.linspace(-1, 1, 128, device=low128.device, dtype=low128.dtype).view(1, 1, 128, 1).expand(B, 1, 128, 128)
            xx = torch.linspace(-1, 1, 128, device=low128.device, dtype=low128.dtype).view(1, 1, 1, 128).expand(B, 1, 128, 128)
            inp = torch.cat([low128, op, dark, bright, xx, yy], 1)
            q_src = F.avg_pool2d(torch.cat([low128, op, dark], 1), 4).flatten(2).sort(-1).values
            N = q_src.shape[-1]
            q_idx = [int(p * (N - 1)) for p in (0.01, 0.05, 0.15, 0.50, 0.85, 0.98)]
            qs = q_src[..., q_idx].flatten(1)
        f1 = self.enc1(F.silu(self.stem(inp)))
        f2 = self.enc2(F.silu(self.down1(f1)))
        f3 = self.mid(F.silu(self.down2(f2)))
        g_feat = torch.cat([f3.mean((2, 3)), f3.amax((2, 3)), qs], 1)
        g_ab = self.glob_mlp(g_feat).view(B, 6, 1, 1)
        grid = self.grid_head(f3)
        grid = F.avg_pool2d(F.pad(grid, (1, 1, 1, 1), mode="replicate"), 3, 1)
        return g_ab + (grid - grid.mean((2, 3), keepdim=True))


class FuseA(nn.Sequential):
    def __init__(self, fuse: nn.Sequential, cin: int, inv: bool = False):
        super().__init__(*fuse)
        self.ahead = _small_head(cin + 1)
        self.ahead_fn, self.use_amp, self.extra = None, False, None
        if inv:
            self.ahead_inv = nn.Conv2d(4, 16, 1)
            nn.init.zeros_(self.ahead_inv.weight)
            nn.init.zeros_(self.ahead_inv.bias)

    def _ahead(self, zin: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
        h = self.ahead[0](zin)
        if extra is not None:
            h = h + self.ahead_inv(extra)
        for i in range(1, len(self.ahead)):
            h = self.ahead[i](h)
        return h

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        o = z
        for i in range(4):
            o = self[i](o)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp and z.is_cuda):
            dl = (self.ahead_fn or self._ahead)(torch.cat([z, o[:, 3:4]], 1), self.extra)
        dl = dl.float()
        return torch.cat([o[:, :3], o[:, 3:4] + dl, o[:, 4:]], 1)


class RefineNet(nn.Module):
    def __init__(self, widths=(18, 32, 48), blocks=(1, 2, 2), dec_blocks=(1, 3), stem: int = 12):
        super().__init__()
        self.stem = nn.Conv2d(13, stem, 3, padding=1)
        self.in_proj = nn.Conv2d(4 * stem, widths[0], 1)
        self.enc = nn.ModuleList([nn.Sequential(*[Block(c) for _ in range(n)]) for c, n in zip(widths, blocks)])
        self.down = nn.ModuleList([nn.Conv2d(4 * widths[i], widths[i + 1], 1) for i in range(len(widths) - 1)])
        self.up = nn.ModuleList([nn.Conv2d(widths[i + 1], 4 * widths[i], 1) for i in reversed(range(len(widths) - 1))])
        self.dec = nn.ModuleList([
            nn.Sequential(*[Block(widths[i]) for _ in range(dec_blocks[i])])
            for i in reversed(range(len(widths) - 1))
        ])
        self.out_proj = nn.Conv2d(widths[0], 4 * stem, 1)
        self.out = nn.Conv2d(stem, 3, 3, padding=1)
        self.pix_mlp = nn.Sequential(nn.Conv2d(13, 16, 1), nn.SiLU(), nn.Conv2d(16, 3, 1))
        self.veil_lin = nn.Conv2d(9, 3, 1, bias=False)
        self.body_fn = None

    def body(self, inp: torch.Tensor) -> torch.Tensor:
        s = F.silu(self.stem(inp))
        f = self.in_proj(F.pixel_unshuffle(s, 2))
        skips = []
        for i, blk in enumerate(self.enc):
            f = blk(f)
            if i < len(self.down):
                skips.append(f)
                f = self.down[i](F.pixel_unshuffle(f, 2))
        for up, blk in zip(self.up, self.dec):
            f = blk(F.pixel_shuffle(up(f), 2) + skips.pop())
        return F.pixel_shuffle(self.out_proj(f), 2) + s

    def forward(self, y_coarse, xp, base_full, j_full, a, j1, amp):
        inp = torch.cat([xp, base_full.clamp(0, 1.5), j_full.clamp(0, 1.5), y_coarse.clamp(0, 1.5), a], 1)
        with amp():
            f = (self.body_fn or self.body)(inp)
        res = self.out(F.silu(f.float())) + self.pix_mlp(inp)
        dj = (j_full - j1).clamp(-1.0, 1.0)
        dj_zm = dj - dj.mean((-2, -1), keepdim=True)
        res = res + self.veil_lin(torch.cat([dj, dj * (1.0 - a), dj_zm], 1))
        return y_coarse + res


class LowFreqHead(nn.Module):
    def __init__(self, c: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7, c, 1), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1, padding_mode="replicate"), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=2, dilation=2, groups=c, padding_mode="replicate"), nn.SiLU(),
            nn.Conv2d(c, 6, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, y: torch.Tensor, xp: torch.Tensor, a: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        o = self.net(F.avg_pool2d(torch.cat([y.clamp(0, 1.5), xp, a], 1), 16))
        o = upsample(o, y.shape[-2:])
        if scale != 1.0:
            return y * torch.exp((o[:, :3] * scale).clamp(-1, 1)) + o[:, 3:] * scale
        return y * torch.exp(o[:, :3].clamp(-1, 1)) + o[:, 3:]


class DesnowLiteBase(nn.Module):
    def __init__(
        self,
        widths=(16, 32, 48, 64),
        blocks=(1, 2, 2, 3),
        dec=(1, 1, 2),
        stem: int = 12,
        amp: bool = False,
        fast_grad: bool = True,
        veil_mid: int = 2,
        refine_blocks=(1, 2, 2),
        refine_dec=(1, 3),
    ):
        super().__init__()
        self.factor = 64
        self.amp = amp
        self.fast_grad = fast_grad
        self.veil = VeilBranch()
        self.veilnet = FullImageVeilNet(mid=veil_mid)
        self.flakecal = FlakeCal()
        cin = 9
        self.stem = nn.Conv2d(cin, stem, 3, padding=1)
        self.in_proj = nn.Conv2d(4 * stem, widths[0], 1)
        self.enc = nn.ModuleList([
            nn.Sequential(*[Block(c, strip=i >= len(widths) - 2) for _ in range(n)])
            for i, (c, n) in enumerate(zip(widths, blocks))
        ])
        self.down = nn.ModuleList([nn.Conv2d(4 * widths[i], widths[i + 1], 1) for i in range(len(widths) - 1)])
        self.up = nn.ModuleList([nn.Conv2d(widths[i + 1], 4 * widths[i], 1) for i in reversed(range(len(widths) - 1))])
        self.dec = nn.ModuleList([
            nn.Sequential(*[Block(widths[i]) for _ in range(n)])
            for i, n in zip(reversed(range(len(widths) - 1)), dec)
        ])
        self.out_proj = nn.Conv2d(widths[0], 4 * stem, 1)
        self.k_img = nn.Parameter(torch.tensor(0.986))
        f_c = stem + cin
        self.S = nn.Parameter(torch.ones(1, 3, 1, 1))
        self.nk = 8
        nout = 4 + 3 + 1 + self.nk
        self.fuse = nn.Sequential(
            nn.Conv2d(f_c, f_c, 3, padding=1, groups=f_c),
            nn.Conv2d(f_c, 16, 1),
            nn.SiLU(),
            nn.Conv2d(16, nout, 1),
        )
        self.refine = RefineNet(blocks=refine_blocks, dec_blocks=refine_dec)
        self.lowfreq = LowFreqHead()
        self.unet_fn = None
        self._init_heads(nout)

    def _init_heads(self, nout: int):
        for m in self.modules():
            if isinstance(m, Block) and any(m is b for s in [self.veilnet, self.refine] for b in s.modules()):
                nn.init.constant_(m.beta, 0.3)
                nn.init.constant_(m.gamma, 0.3)
        added = self.refine.dec[0][-1]
        nn.init.zeros_(added.beta)
        nn.init.zeros_(added.gamma)
        for blocks in self.enc[-2:]:
            for b in blocks:
                nn.init.zeros_(b.strip[-1].weight)
                nn.init.zeros_(b.strip[-1].bias)
        for m in (self.veil.head[-1], self.flakecal.head[-1], self.fuse[-1], self.refine.veil_lin):
            nn.init.zeros_(m.weight)
        for m in (self.veil.head[-1], self.flakecal.head[-1], self.fuse[-1]):
            nn.init.zeros_(m.bias)
        for m in (self.refine.out, self.refine.pix_mlp[-1], self.veilnet.grid_head[-1], self.veilnet.glob_mlp[-1]):
            nn.init.normal_(m.weight, std=1e-3)
            nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.fuse[-1].bias[3] = -4.0
            self.fuse[-1].bias[nout - self.nk] = 2.0

    def compile_bodies(self):
        self.unet_fn = torch.compile(self.unet_body)
        self.refine.body_fn = torch.compile(self.refine.body)
        return self

    def _amp_ctx(self, t: torch.Tensor):
        if not self.amp:
            return contextlib.nullcontext()
        return torch.autocast(device_type=t.device.type, dtype=torch.bfloat16)

    def unet_body(self, inp: torch.Tensor) -> torch.Tensor:
        s = F.silu(self.stem(inp))
        f = self.in_proj(F.pixel_unshuffle(s, 2))
        skips = []
        for i, blocks in enumerate(self.enc):
            f = blocks(f)
            if i < len(self.down):
                skips.append(f)
                f = self.down[i](F.pixel_unshuffle(f, 2))
        for up, blocks in zip(self.up, self.dec):
            f = blocks(F.pixel_shuffle(up(f), 2) + skips.pop())
        return F.pixel_shuffle(self.out_proj(f), 2) + s


class DesnowLiteNet(DesnowLiteBase):
    """Physics-guided desnowing network (418,040 parameters in default configuration)."""

    def __init__(
        self,
        a_head: bool = True,
        phys_s: bool = True,
        snow_inv: bool = True,
        amp: bool = False,
        fast_grad: bool = True,
    ):
        super().__init__(amp=amp, fast_grad=fast_grad)
        self.phys_s = phys_s
        self.snow_inv = snow_inv
        cin = self.fuse[0].in_channels
        if a_head:
            self.fuse = FuseA(self.fuse, cin, inv=snow_inv)

    def after_load(self, missing):
        with torch.no_grad():
            for k in missing:
                if (k.endswith(".beta") or k.endswith(".gamma")) and ".ahead." not in k:
                    self.get_parameter(k).zero_()

    def compile_bodies(self):
        super().compile_bodies()
        if isinstance(self.fuse, FuseA):
            self.fuse.__dict__["ahead_fn"] = torch.compile(self.fuse._ahead)
        return self

    def veil_field(self, xp: torch.Tensor, uh: int, uw: int, ctx):
        p0 = self.veil(F.avg_pool2d(xp, 4))
        p0 = F.avg_pool2d(F.pad(p0, (1, 1, 1, 1), mode="replicate"), 3, 1)
        p0_up = upsample(p0, xp.shape[-2:])
        if ctx is not None:
            low128, boxes = ctx
            d16 = self.veilnet(low128)
            g = d16.shape[-1]
            B, _, h, w = xp.shape
            t, l = boxes[:, 0:1].unsqueeze(-1), boxes[:, 1:2].unsqueeze(-1)
            Hf, Wf = boxes[:, 2:3].unsqueeze(-1), boxes[:, 3:4].unsqueeze(-1)
            gy = torch.linspace(0.5, h - 0.5, g, device=xp.device, dtype=xp.dtype).view(1, g, 1).expand(B, g, g)
            gx = torch.linspace(0.5, w - 0.5, g, device=xp.device, dtype=xp.dtype).view(1, 1, g).expand(B, g, g)
            grid = torch.stack([((l + gx) / Wf) * 2 - 1, ((t + gy) / Hf) * 2 - 1], -1)
            d_crop = F.grid_sample(d16, grid, mode="bilinear", padding_mode="border", align_corners=False)
            d_up = upsample(d_crop, (h, w))
        else:
            d16 = self.veilnet(make_low128(xp[..., :uh, :uw]))
            d_unpad = upsample(d16, (uh, uw))
            ph, pw = xp.shape[-2] - uh, xp.shape[-1] - uw
            d_up = F.pad(d_unpad, (0, pw, 0, ph), mode="reflect") if (ph or pw) else d_unpad
        return p0_up, d_up, d16

    def forward(self, x: torch.Tensor, return_aux: bool = False, ctx=None, veil_scale: float = 1.0):
        h, w = x.shape[-2:]
        ph, pw = -h % self.factor, -w % self.factor
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect") if ph or pw else x
        p0_up, d_up, p16 = self.veil_field(xp, h, w, ctx if (ph == 0 and pw == 0) else None)
        if veil_scale != 1.0:
            p0_up, d_up = p0_up * veil_scale, d_up * veil_scale
        alpha0, beta0 = torch.exp(p0_up[:, :3]).clamp(0.15, 1.5), p0_up[:, 3:]
        base0 = (xp - beta0) / alpha0
        src = base0.clamp(0, 1.5)
        prior = snow_prior(src.detach() if self.fast_grad else src)
        inp = torch.cat([src, prior], 1)
        with self._amp_ctx(inp):
            f = (self.unet_fn or self.unet_body)(inp)
        z = torch.cat([f.float(), inp], 1)
        if isinstance(self.fuse, FuseA):
            self.fuse.use_amp = self.amp
            if self.snow_inv:
                lx = torch.log((1 - xp).clamp_min(1e-3))
                lum = lx.mean(1, keepdim=True)
                self.fuse.__dict__["extra"] = torch.cat([lx - lum, lum], 1)
        o = self.fuse(z)
        a = (torch.sigmoid(o[:, 3:4]) * self.k_img).clamp(1e-4, 1 - 1e-4)
        logit = torch.log(a) - torch.log1p(-a)

        d = upsample(self.flakecal(xp, a.detach(), alpha0, beta0), xp.shape[-2:])
        if veil_scale != 1.0:
            d = d * veil_scale
        alpha0, beta0 = (alpha0 * torch.exp(d[:, :3])).clamp(0.15, 1.5), beta0 + d[:, 3:]
        base0 = (xp - beta0) / alpha0
        alpha_full = (alpha0.detach() * torch.exp(d_up[:, :3])).clamp(0.15, 1.5)
        beta_full = beta0.detach() + d_up[:, 3:]
        base_full = (xp - beta_full) / alpha_full

        r = o[:, :3]
        mod = 1 + 0.25 * torch.tanh(o[:, 4:7])
        if self.phys_s:
            S0 = ((1 - beta0) / alpha0).clamp(0.5, 2.0) * mod
            Sf = ((1 - beta_full) / alpha_full).clamp(0.5, 2.0) * mod
        else:
            S0 = Sf = self.S * mod

        wts = torch.softmax(o[:, -self.nk :], 1)
        b0 = base0.detach() if self.fast_grad else base0
        op7, op15 = opening(b0, 7), opening(b0, 15)

        j1 = (base0 - a * S0) * (1 - a) / ((1 - a).square() + 0.01)
        j_full = (base_full - a * Sf) * (1 - a) / ((1 - a).square() + 0.01)
        sv, wv = j1.clamp(0, 1), (1 - a).square() + 1e-4
        if self.fast_grad:
            sv, wv = sv.detach(), wv.detach()
        ls, lw = F.avg_pool2d(sv * wv, 4), F.avg_pool2d(wv, 4)
        fill = wts[:, :1] * base0 + wts[:, 1:2] * op7 + wts[:, 2:3] * op15
        for i, make in enumerate(
            (
                lambda: nconv(sv, wv, 5),
                lambda: nconv(sv, wv, 15),
                lambda: nconv(sv, wv, 31),
                lambda: upsample(nconv(ls / lw, lw, 15), base0.shape[-2:], "bilinear"),
                lambda: upsample(nconv(ls / lw, lw, 31), base0.shape[-2:], "bilinear"),
            ),
            3,
        ):
            fill = fill + wts[:, i : i + 1] * make()
        gate = torch.sigmoid(10 * (a - 0.55) + o[:, 7:8])
        y_coarse = (1 - gate) * j1 + gate * fill + r
        y_full = self.refine(y_coarse, xp, base_full, j_full, a, j1, lambda: self._amp_ctx(inp))
        y_full = self.lowfreq(y_full, xp, a, scale=veil_scale)
        y = y_full[..., :h, :w]

        if return_aux:
            c = lambda t: t[..., :h, :w]
            return y, {
                "y_coarse": c(y_coarse),
                "logit": c(logit),
                "base": c(base_full),
                "p16": p16,
                "y_prev": [],
                "alpha": c(alpha_full),
                "beta": c(beta_full),
                "S": c(Sf),
                "o_g": c(o[:, 7:8]),
                "gate": c(gate),
                "r": c(r),
            }
        return y


ARCHS = {
    "desnowlite": (lambda amp=False, fast_grad=True: DesnowLiteNet(a_head=True, phys_s=True, snow_inv=True, amp=amp, fast_grad=fast_grad), 420_000),
    "desnowlite_no_inv": (lambda amp=False, fast_grad=True: DesnowLiteNet(a_head=True, phys_s=True, snow_inv=False, amp=amp, fast_grad=fast_grad), 420_000),
    "desnowlite_base": (lambda amp=False, fast_grad=True: DesnowLiteNet(a_head=False, phys_s=False, snow_inv=False, amp=amp, fast_grad=fast_grad), 420_000),
}


def build_desnowlite(arch: str = "desnowlite", amp: bool = False, fast_grad: bool = True) -> DesnowLiteNet:
    fn, _ = ARCHS[arch]
    return fn(amp=amp, fast_grad=fast_grad)


def load_desnowlite(ckpt_path: str | Path, device: torch.device | str = "cpu", arch: str = "desnowlite") -> DesnowLiteNet:
    net = build_desnowlite(arch=arch, amp=False, fast_grad=True).to(device).eval()
    st = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    sd = st.get("ema", st.get("net", st)) if isinstance(st, dict) else st
    clean_sd = {k: v for k, v in sd.items() if ".ahead_fn." not in k}
    net.load_state_dict(clean_sd, strict=True)
    return net
