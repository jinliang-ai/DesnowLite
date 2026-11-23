import glob
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.operators import make_low128

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
cv2.setNumThreads(1)


def subdir(root: str, pat: str) -> str | None:
    for d in sorted(os.listdir(root)):
        if os.path.isdir(os.path.join(root, d)) and re.search(pat, d, re.I):
            return os.path.join(root, d)
    return None


def read_rgb(p: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def read_gray(p: str) -> np.ndarray:
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE)


def pairs_from_root(root: str) -> list[tuple[str, str, str | None]]:
    sd = subdir(root, r"snow|synthetic|input")
    gd = subdir(root, r"^(gt|clean|groundtruth)")
    md = subdir(root, r"^mask")
    if sd is None or gd is None:
        raise FileNotFoundError(f"{root}: snow and Gt directories not found")
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    index = lambda d: {stem(p): p for p in glob.glob(f"{d}/*") if p.lower().endswith(IMG_EXTS)} if d else {}
    gt, mk = index(gd), index(md)
    snow = sorted(p for p in glob.glob(f"{sd}/*") if p.lower().endswith(IMG_EXTS) and stem(p) in gt)
    return [(p, gt[stem(p)], mk.get(stem(p))) for p in snow]


def finish_batch(x, y, m, low, t, l, hw, crop):
    B = x.shape[0]
    flip = torch.rand(B, device=x.device) < 0.5
    x[flip], y[flip], low[flip] = x[flip].flip(3), y[flip].flip(3), low[flip].flip(3)
    if m is not None:
        m[flip] = m[flip].flip(3)
    l = torch.where(flip, hw[:, 1] - crop - l, l)
    boxes = torch.stack([t.float(), l.float(), hw[:, 0].float(), hw[:, 1].float()], 1)
    return x, y, m, (low, boxes)


class VramSet:
    def __init__(self, xs, ts, ms, dev: torch.device, mask_mix_prob: float = 0.0):
        self.has_mask = ms is not None
        self.mask_mix_prob = mask_mix_prob
        self.x = torch.from_numpy(np.stack(xs)).to(dev)
        self.t = torch.from_numpy(np.stack(ts)).to(dev)
        self.m = torch.from_numpy(np.stack(ms)).to(dev) if self.has_mask else None
        self.N, self.H, self.W = self.x.shape[:3]
        self.dev = self.index_dev = dev
        self.desc = f"{self.H}x{self.W}"
        low = []
        for i in range(0, self.N, 64):
            low.append(make_low128(self.x[i : i + 64].permute(0, 3, 1, 2).float() / 255).half())
        self.low = torch.cat(low)

    def chunks(self, ids, bs: int):
        return [ids[i : i + bs] for i in range(0, ids.numel(), bs)]

    def full(self, b):
        return self.x[b].permute(0, 3, 1, 2).float() / 255, self.t[b].permute(0, 3, 1, 2).float() / 255

    def mask_mix(self, ids, p: float):
        B = ids.numel()
        x = self.x[ids].permute(0, 3, 1, 2).float().div_(255)
        m = self.m[ids][:, None].float().div_(255)
        sel = torch.rand(B, device=self.dev) < p
        if sel.any():
            d = torch.randint(0, self.N, (int(sel.sum()),), device=self.dev)
            md = self.m[d][:, None].float().div_(255)
            if torch.rand(()) < 0.5:
                md = md.flip(3)
            xs, ms = x[sel], m[sel]
            swap = ms <= 0.8
            v = (xs - ms) / (1 - ms).clamp_min(0.2)
            x_swap, x_over = v * (1 - md) + md, xs * (1 - md) + md
            x[sel] = torch.where(swap, x_swap, x_over).clamp(0, 1).mul(255).round().div(255)
            m[sel] = torch.where(swap, md, 1 - (1 - ms) * (1 - md))
        return x, m, make_low128(x)

    def batch(self, ids, crop: int):
        B = ids.numel()
        t = torch.randint(0, self.H - crop + 1, (B,), device=self.dev)
        l = torch.randint(0, self.W - crop + 1, (B,), device=self.dev)
        r = torch.arange(crop, device=self.dev)
        yy, xx = (t[:, None] + r)[:, :, None], (l[:, None] + r)[:, None, :]
        n = ids[:, None, None]
        hw = torch.tensor([self.H, self.W], device=self.dev).expand(B, 2)
        if self.has_mask and self.mask_mix_prob > 0:
            x, m, low = self.mask_mix(ids, self.mask_mix_prob)
            b = torch.arange(B, device=self.dev)[:, None, None]
            x = x[b, :, yy, xx].permute(0, 3, 1, 2)
            m = m[b, :, yy, xx].permute(0, 3, 1, 2)
            y = self.t[n, yy, xx].permute(0, 3, 1, 2).float().div_(255)
            return finish_batch(x.contiguous(), y, m.contiguous(), low, t, l, hw, crop)
        x = self.x[n, yy, xx].permute(0, 3, 1, 2).float().div_(255)
        y = self.t[n, yy, xx].permute(0, 3, 1, 2).float().div_(255)
        m = self.m[n, yy, xx][:, None].float().div_(255) if self.has_mask else None
        return finish_batch(x, y, m, self.low[ids].float(), t, l, hw, crop)


class HostSet:
    def __init__(self, xs, ts, ms, dev: torch.device, nbuf: int = 3):
        self.has_mask = ms is not None
        self.x = [torch.from_numpy(a) for a in xs]
        self.t = [torch.from_numpy(a) for a in ts]
        self.m = [torch.from_numpy(a) for a in ms] if self.has_mask else None
        self.N = len(xs)
        self.dev, self.index_dev = dev, torch.device("cpu")
        self.hw = torch.tensor([a.shape[:2] for a in xs])
        self.desc = f"{len({tuple(s) for s in self.hw.tolist()})} sizes, min side {int(self.hw.min())}"
        self.low = torch.cat([
            make_low128(self.x[i].to(dev).permute(2, 0, 1)[None].float() / 255).half()
            for i in range(self.N)
        ])
        self.nbuf, self.k, self.bufs, self.evs = nbuf, 0, {}, [None] * nbuf
        self.pool = ThreadPoolExecutor(8)

    def chunks(self, ids, bs: int):
        out, cur = [], []
        for i in ids.tolist():
            if cur and (len(cur) == bs or self.hw[i].tolist() != self.hw[cur[0]].tolist()):
                out.append(cur)
                cur = []
            cur.append(i)
        return out + [cur] if cur else out

    def full(self, b):
        f = lambda src: torch.stack([src[i] for i in b]).to(self.dev).permute(0, 3, 1, 2).float() / 255
        return f(self.x), f(self.t)

    def batch(self, ids, crop: int):
        B = ids.numel()
        hw = self.hw[ids]
        t = (torch.rand(B) * (hw[:, 0] - crop + 1)).long()
        l = (torch.rand(B) * (hw[:, 1] - crop + 1)).long()
        key = (B, crop)
        if key not in self.bufs:
            pin = self.dev.type == "cuda"
            mk = lambda c: [torch.empty(B, crop, crop, *c, dtype=torch.uint8, pin_memory=pin) for _ in range(self.nbuf)]
            self.bufs[key] = (mk((3,)), mk((3,)), mk(()) if self.has_mask else None)
        k = self.k = (self.k + 1) % self.nbuf
        if self.evs[k] is not None:
            self.evs[k].synchronize()
        bx, bt, bm = (z[k] if z else None for z in self.bufs[key])
        il, tl, ll = ids.tolist(), t.tolist(), l.tolist()

        def cut(j):
            i, a, c = il[j], tl[j], ll[j]
            bx[j].copy_(self.x[i][a : a + crop, c : c + crop])
            bt[j].copy_(self.t[i][a : a + crop, c : c + crop])
            if bm is not None:
                bm[j].copy_(self.m[i][a : a + crop, c : c + crop])

        list(self.pool.map(cut, range(B)))
        go = lambda z: z.to(self.dev, non_blocking=True)
        x = go(bx).permute(0, 3, 1, 2).float().div_(255)
        y = go(bt).permute(0, 3, 1, 2).float().div_(255)
        m = go(bm)[:, None].float().div_(255) if bm is not None else None
        if self.dev.type == "cuda":
            self.evs[k] = torch.cuda.Event()
            self.evs[k].record()
        return finish_batch(x, y, m, self.low[go(ids)].float(), go(t), go(l), go(hw), crop)


def load_set(pairs, dev: torch.device, threads: int = 8, limit: int = 0, name: str = "", min_side: int = 0, mask_mix_prob: float = 0.0):
    pairs = pairs[:limit] if limit else pairs
    if not pairs:
        raise FileNotFoundError(f"{name}: empty pair list")
    has_mask = all(m for _, _, m in pairs)
    t0 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        xs = list(ex.map(read_rgb, [p[0] for p in pairs]))
        ts = list(ex.map(read_rgb, [p[1] for p in pairs]))
        ms = list(ex.map(read_gray, [p[2] for p in pairs])) if has_mask else None
    for i, (a, b) in enumerate(zip(xs, ts)):
        if a.shape != b.shape:
            raise ValueError(f"{pairs[i][0]}: input {a.shape} vs gt {b.shape}")
    if min_side:
        keep = [i for i, a in enumerate(xs) if min(a.shape[:2]) >= min_side]
        xs, ts = [xs[i] for i in keep], [ts[i] for i in keep]
        ms = [ms[i] for i in keep] if ms else None
    if len({a.shape for a in xs}) == 1:
        ds = VramSet(xs, ts, ms, dev, mask_mix_prob=mask_mix_prob)
    else:
        ds = HostSet(xs, ts, ms, dev)
    print(f"[{name}] {ds.N} pairs ({ds.desc}, mask={ds.has_mask}) loaded in {time.time() - t0:.1f}s", flush=True)
    return ds
