#!/usr/bin/env python3
import argparse
import copy
import hashlib
import importlib.util
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import load_desnowlite
from pnp_adapt.rgb_four_region_loss import compute_rgb_four_region_loss, set_ft_train_mode
from stage1_pretrain.dataset import pairs_from_root, subdir
from stage1_pretrain.losses import gauss_win, ssim_map
from stage2_selfsup.canopy_detector import Mask2FormerCanopyDetector
from stage2_selfsup.four_region_loss import hp3
from stage2_selfsup.four_region_prior import four_region_targets, shielded_cutmix

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
cv2.setNumThreads(1)


def load_external_backbone(py_file: str, builder_fn: str, ckpt_path: str, dev: torch.device) -> tuple[nn.Module, int]:
    spec = importlib.util.spec_from_file_location("ext_backbone", py_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = getattr(mod, builder_fn)(ckpt_path, dev)
    if isinstance(out, tuple):
        return out[0].to(dev).eval(), int(out[1])
    return out.to(dev).eval(), 16


def forward_padded(model: nn.Module, x: torch.Tensor, pad_mult: int):
    h, w = x.shape[-2:]
    ph, pw = -h % pad_mult, -w % pad_mult
    xp = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
    raw = model(xp)
    if isinstance(raw, (list, tuple)):
        y = raw[-1].float()[..., :h, :w]
        return y, raw
    return raw.float()[..., :h, :w], None


class RgbPairPool:
    def __init__(self, pairs: list[tuple[str, str, str | None]], limit: int = 1200):
        rng = random.Random(42)
        picks = list(pairs)
        rng.shuffle(picks)
        picks = picks[:limit]

        def _read(p):
            s = cv2.cvtColor(cv2.imread(p[0], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            g = cv2.cvtColor(cv2.imread(p[1], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            return s, g

        with ThreadPoolExecutor(16) as ex:
            items = list(ex.map(_read, picks))
        self.S = [it[0] for it in items]
        self.G = [it[1] for it in items]
        self.N = len(items)

    def batch(self, B: int, crop: int, dev: torch.device):
        xs, gs = [], []
        for _ in range(B):
            i = random.randrange(self.N)
            s, g = self.S[i], self.G[i]
            H, W = s.shape[:2]
            t = random.randint(0, max(0, H - crop))
            l = random.randint(0, max(0, W - crop))
            sc = s[t : t + crop, l : l + crop]
            gc = g[t : t + crop, l : l + crop]
            if random.random() < 0.5:
                sc, gc = sc[:, ::-1].copy(), gc[:, ::-1].copy()
            xs.append(torch.from_numpy(sc).permute(2, 0, 1))
            gs.append(torch.from_numpy(gc).permute(2, 0, 1))
        return (
            torch.stack(xs).to(dev, non_blocking=True).float() / 255.0,
            torch.stack(gs).to(dev, non_blocking=True).float() / 255.0,
        )


def load_real_rgb_with_masks(real_dir: Path, ref_model: nn.Module, pad_mult: int, cache_dir: Path, crop: int, dev: torch.device):
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(str(p) for p in real_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)

    def _decode(p: str):
        im = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        s = max(512.0 / min(h, w), float(crop) / min(h, w))
        if max(h, w) * s > 1280:
            s = 1280.0 / max(h, w)
        if abs(s - 1.0) > 1e-3:
            im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
        return p, im

    with ThreadPoolExecutor(16) as ex:
        loaded = dict(ex.map(_decode, files))

    need = []
    masks = {}
    for p, im in loaded.items():
        hk = hashlib.md5(f"{p}_{im.shape[0]}x{im.shape[1]}".encode()).hexdigest()[:16]
        cp = cache_dir / f"{hk}.png"
        if cp.exists():
            m = cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape[:2] == im.shape[:2]:
                masks[p] = m
                continue
        need.append((p, im, cp))

    if need:
        detector = Mask2FormerCanopyDetector(device=dev)
        if dev.type == "cuda":
            detector.ade_m2f.half()
            detector.mv_m2f.half()
        for p, im, cp in need:
            with torch.no_grad():
                x_t = torch.from_numpy(im).to(dev).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                yr_t, _ = forward_padded(ref_model, x_t, pad_mult)
                yr_t = yr_t.clamp(0.0, 1.0)
                if dev.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        cm = detector.detect(yr_t)
                else:
                    cm = detector.detect(yr_t)
                u8 = (cm[0, 0].float().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
                cv2.imwrite(str(cp), u8)
                masks[p] = u8
        del detector
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return [loaded[p] for p in files], [masks[p] for p in files]


def sample_real_crops(imgs, masks, B: int, crop: int, dev: torch.device):
    xs, ms = [], []
    N = len(imgs)
    for _ in range(B):
        i = random.randrange(N)
        im, mk = imgs[i], masks[i]
        H, W = im.shape[:2]
        t = random.randint(0, max(0, H - crop))
        l = random.randint(0, max(0, W - crop))
        ic = im[t : t + crop, l : l + crop]
        mc = mk[t : t + crop, l : l + crop]
        if random.random() < 0.5:
            ic, mc = ic[:, ::-1].copy(), mc[:, ::-1].copy()
        xs.append(torch.from_numpy(ic).permute(2, 0, 1))
        ms.append(torch.from_numpy(mc).unsqueeze(0))
    return (
        torch.stack(xs).to(dev, non_blocking=True).float() / 255.0,
        torch.stack(ms).to(dev, non_blocking=True).float() / 255.0,
    )


def main():
    ap = argparse.ArgumentParser(description="Model-agnostic RGB-space 4-region plug-and-play adaptation")
    ap.add_argument("--backbone", default="desnowlite", help="'desnowlite' or 'external'")
    ap.add_argument("--ext-file", default="", help="Python file defining external backbone builder")
    ap.add_argument("--ext-builder", default="build_model", help="Builder callable name in --ext-file")
    ap.add_argument("--init-ckpt", required=True, help="0-step pretrained checkpoint")
    ap.add_argument("--real-dir", required=True, help="Unlabeled real-world snowfall directory")
    ap.add_argument("--source-root", required=True, help="Paired source replay root (Train/Snow, Train/Gt)")
    ap.add_argument("--mask-cache", default="runs/pnp_canopy_cache")
    ap.add_argument("--out", default="runs/pnp_adapted/adapted_2k.pth")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs-src", type=int, default=8)
    ap.add_argument("--bs-real", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3.0e-5)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=2000)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    if args.backbone == "desnowlite":
        net = load_desnowlite(args.init_ckpt, device=dev)
        pad_mult = 64
    else:
        net, pad_mult = load_external_backbone(args.ext_file, args.ext_builder, args.init_ckpt, dev)

    ref = copy.deepcopy(net).requires_grad_(False).eval()
    ema = copy.deepcopy(net).requires_grad_(False).eval()
    for p in net.parameters():
        p.requires_grad_(True)
    train_params = [p for p in net.parameters() if p.requires_grad]
    all_params = list(net.parameters())
    ema_params = list(ema.parameters())

    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    win = gauss_win(dev)

    src_pairs = pairs_from_root(subdir(args.source_root, r"^train") or args.source_root)
    src_pool = RgbPairPool(src_pairs)
    real_imgs, real_masks = load_real_rgb_with_masks(Path(args.real_dir), ref, pad_mult, Path(args.mask_cache), args.crop, dev)

    sched = lambda s: args.lr * min(1.0, max(0.05, s / 100.0)) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * s / args.steps)))
    set_ft_train_mode(net)
    t0 = time.time()

    for step in range(1, args.steps + 1):
        for pg in opt.param_groups:
            pg["lr"] = sched(step)

        xc, tc = src_pool.batch(args.bs_src, args.crop, dev)
        with torch.no_grad():
            yc_ref, _ = forward_padded(ref, xc, pad_mult)
            yc_ref = yc_ref.clamp(0.0, 1.0)

        xr, sem_tree = sample_real_crops(real_imgs, real_masks, args.bs_real, args.crop, dev)
        with torch.no_grad():
            yr_ref, _ = forward_padded(ref, xr, pad_mult)
            yr_ref = yr_ref.clamp(0.0, 1.0)
            y_target, _, flake, cov, tree = four_region_targets(yr_ref, xr, sem_tree)
            hB = args.bs_real // 2
            shield = tree[hB:] | cov[hB:]
            x_syn, m_syn = shielded_cutmix(xr[hB:], shield_mask=shield)
            xr_in = torch.cat([xr[:hB], x_syn], dim=0)
            syn_flk = (m_syn > 0.04) & (~shield)
            flake_in = torch.cat([flake[:hB], flake[hB:] | syn_flk], dim=0)
            cov_in = torch.cat([cov[:hB], cov[hB:] & (~syn_flk)], dim=0)
            tree_in = torch.cat([tree[:hB], tree[hB:] & (~syn_flk)], dim=0)

        opt.zero_grad(set_to_none=True)
        yc, raw_c = forward_padded(net, xc, pad_mult)
        l_src = (yc - tc).abs().mean() + 0.15 * (1.0 - ssim_map(yc.clamp(0.0, 1.0), tc, win).mean())
        l_lwf = 1.5 * ((yc - yc_ref).abs().mean() + 0.8 * (hp3(yc) - hp3(yc_ref)).abs().mean())

        l_ms = 0.0
        if isinstance(raw_c, (list, tuple)) and len(raw_c) == 3:
            l_ms = 0.1 * (raw_c[0].float() - F.avg_pool2d(tc, 4)).abs().mean() + 0.1 * (raw_c[1].float() - F.avg_pool2d(tc, 2)).abs().mean()

        yr, raw_r = forward_padded(net, xr_in, pad_mult)
        lf = compute_rgb_four_region_loss(yr, yr_ref, y_target, flake_in, cov_in, tree_in)
        if isinstance(raw_r, (list, tuple)) and len(raw_r) == 3:
            l_ms = (
                l_ms
                + 0.08 * (raw_r[0].float() - F.avg_pool2d(yr.detach().clamp(0.0, 1.0), 4)).abs().mean()
                + 0.08 * (raw_r[1].float() - F.avg_pool2d(yr.detach().clamp(0.0, 1.0), 2)).abs().mean()
            )

        loss = l_src + l_lwf + l_ms + (
            3.2 * lf["flake"] + 3.0 * lf["cov"] + 2.8 * lf["tree"] + 2.0 * lf["bg"] + 2.5 * lf["lf"] + 6.0 * lf["floor"]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(ema_params, all_params, 1.0 - args.ema)

        if step % 200 == 0 or step == args.steps:
            print(f"[PnP step {step:4d}/{args.steps}] loss={loss.item():.4f} elapsed={(time.time()-t0)/60.0:.1f}m", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ema": {k: v.cpu() for k, v in ema.state_dict().items()}, "step": args.steps}, str(out_path))
    print(f"Saved adapted checkpoint: {out_path}")


if __name__ == "__main__":
    main()
