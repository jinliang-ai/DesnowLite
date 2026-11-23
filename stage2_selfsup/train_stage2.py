#!/usr/bin/env python3
import argparse
import copy
import hashlib
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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import load_desnowlite
from models.operators import make_low128
from stage1_pretrain.dataset import finish_batch, pairs_from_root, subdir
from stage1_pretrain.losses import gauss_win, ssim_map
from stage2_selfsup.canopy_detector import Mask2FormerCanopyDetector
from stage2_selfsup.four_region_loss import compute_four_region_loss, hp3
from stage2_selfsup.four_region_prior import four_region_targets, shielded_cutmix

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
cv2.setNumThreads(1)
torch.set_num_threads(4)


class CpuPairPool:
    def __init__(self, pairs: list[tuple[str, str, str | None]], dev_init: torch.device, limit: int = 1200):
        rng = random.Random(42)
        picks = list(pairs)
        rng.shuffle(picks)
        picks = picks[:limit]

        def _load_one(p):
            sp, gp, mp = p
            s = cv2.cvtColor(cv2.imread(sp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            g = cv2.cvtColor(cv2.imread(gp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if (mp and os.path.exists(mp)) else np.zeros(s.shape[:2], np.uint8)
            return s, g, m

        with ThreadPoolExecutor(16) as ex:
            items = list(ex.map(_load_one, picks))
        self.S = [it[0] for it in items]
        self.G = [it[1] for it in items]
        self.M = [it[2] for it in items]
        self.N = len(items)
        self.L = []
        with torch.no_grad():
            for s in self.S:
                s_full = torch.from_numpy(s).to(dev_init).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                self.L.append(make_low128(s_full)[0].half().cpu())

    def batch(self, B: int, crop: int, dev: torch.device):
        xs, gs, ms, lows, ts, ls, hws = [], [], [], [], [], [], []
        for _ in range(B):
            i = random.randrange(self.N)
            s, g, m = self.S[i], self.G[i], self.M[i]
            H, W = s.shape[:2]
            t = random.randint(0, max(0, H - crop))
            l = random.randint(0, max(0, W - crop))
            xs.append(torch.from_numpy(s[t : t + crop, l : l + crop]).to(dev, non_blocking=True).permute(2, 0, 1).float() / 255.0)
            gs.append(torch.from_numpy(g[t : t + crop, l : l + crop]).to(dev, non_blocking=True).permute(2, 0, 1).float() / 255.0)
            ms.append(torch.from_numpy(m[t : t + crop, l : l + crop]).to(dev, non_blocking=True).unsqueeze(0).float() / 255.0)
            lows.append(self.L[i].to(dev, non_blocking=True).float())
            ts.append(t)
            ls.append(l)
            hws.append((H, W))
        return finish_batch(
            torch.stack(xs),
            torch.stack(gs),
            torch.stack(ms),
            torch.stack(lows),
            torch.tensor(ts, device=dev),
            torch.tensor(ls, device=dev),
            torch.tensor(hws, device=dev),
            crop,
        )


def freeze_lowfreq_modules(net: torch.nn.Module):
    frozen_prefixes = ("veil.", "veilnet.", "flakecal.", "lowfreq.")
    for name, p in net.named_parameters():
        p.requires_grad_(not name.startswith(frozen_prefixes))


def load_real_pool_with_canopy_cache(
    real_dir: Path,
    ref_net: torch.nn.Module,
    cache_dir: Path,
    crop: int,
    dev: torch.device,
    max_images: int = 0,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(str(p) for p in real_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if max_images > 0:
        files = files[:max_images]
    if not files:
        raise FileNotFoundError(f"No images found in {real_dir}")

    def _decode(p: str):
        im = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        s = 512.0 / min(h, w)
        if max(h, w) * s > 1280:
            s = 1280.0 / max(h, w)
        s = max(s, float(crop) / float(min(h, w)))
        if abs(s - 1.0) > 1e-3:
            im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
        return p, im

    with ThreadPoolExecutor(16) as ex:
        loaded = dict(ex.map(_decode, files))

    need_detect = []
    mask_map: dict[str, np.ndarray] = {}
    for p, im in loaded.items():
        h_key = hashlib.md5(f"{p}_{im.shape[0]}x{im.shape[1]}".encode()).hexdigest()[:16]
        cp = cache_dir / f"{h_key}.png"
        if cp.exists():
            m = cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape[:2] == im.shape[:2]:
                mask_map[p] = m
                continue
        need_detect.append((p, im, cp))

    if need_detect:
        print(f"[Canopy Cache] Extracting offline Omega_tree masks for {len(need_detect)} images...", flush=True)
        detector = Mask2FormerCanopyDetector(device=dev)
        if dev.type == "cuda":
            detector.ade_m2f.half()
            detector.mv_m2f.half()
        for p, im, cp in need_detect:
            with torch.no_grad():
                x_t = torch.from_numpy(im).to(dev).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                yr_t = ref_net(x_t).float().clamp(0.0, 1.0)
                if dev.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        core = detector.detect(yr_t)
                else:
                    core = detector.detect(yr_t)
                cm_u8 = (core[0, 0].float().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
                cv2.imwrite(str(cp), cm_u8)
                mask_map[p] = cm_u8
        del detector
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    R_imgs, R_masks, R_lows = [], [], []
    with torch.no_grad():
        for p in files:
            im = loaded[p]
            x_t = torch.from_numpy(im).to(dev).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            R_imgs.append(im)
            R_masks.append(mask_map[p])
            R_lows.append(make_low128(x_t)[0].cpu())
    return R_imgs, R_masks, R_lows


def sample_real_batch(R_imgs, R_masks, R_lows, B: int, crop: int, dev: torch.device):
    xs, ms, lows, ts, ls, hws = [], [], [], [], [], []
    N = len(R_imgs)
    for _ in range(B):
        i = random.randrange(N)
        im, mk = R_imgs[i], R_masks[i]
        H, W = im.shape[:2]
        t = random.randint(0, max(0, H - crop))
        l = random.randint(0, max(0, W - crop))
        xs.append(torch.from_numpy(im[t : t + crop, l : l + crop]).to(dev, non_blocking=True).permute(2, 0, 1).float() / 255.0)
        ms.append(torch.from_numpy(mk[t : t + crop, l : l + crop]).to(dev, non_blocking=True).unsqueeze(0).float() / 255.0)
        lows.append(R_lows[i].to(dev, non_blocking=True).float())
        ts.append(t)
        ls.append(l)
        hws.append((H, W))
    x = torch.stack(xs)
    m = torch.stack(ms)
    low = torch.stack(lows)
    tt = torch.tensor(ts, device=dev)
    ll = torch.tensor(ls, device=dev)
    hw = torch.tensor(hws, device=dev)
    flip = torch.rand(B, device=dev) < 0.5
    x[flip], m[flip], low[flip] = x[flip].flip(3), m[flip].flip(3), low[flip].flip(3)
    ll = torch.where(flip, hw[:, 1] - crop - ll, ll)
    boxes = torch.stack([tt.float(), ll.float(), hw[:, 0].float(), hw[:, 1].float()], 1)
    return x, m, (low, boxes)


def main():
    ap = argparse.ArgumentParser(description="DesnowLite Stage-2 disjoint 4-region self-supervised adaptation")
    ap.add_argument("--real-dir", required=True, help="Unlabeled real-world snowfall image directory")
    ap.add_argument("--csd-root", required=True, help="Paired CSD root for source replay")
    ap.add_argument("--init-stage1", default=str(ROOT / "checkpoints" / "desnowlite_042m_stage1.pth"))
    ap.add_argument("--mask-cache", default="runs/canopy_mask_cache")
    ap.add_argument("--out", default="runs/stage2_desnowlite/desnowlite_042m_2k.pth")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs-src", type=int, default=8)
    ap.add_argument("--bs-real", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--lambda-flake", type=float, default=3.2)
    ap.add_argument("--lambda-tree", type=float, default=2.8)
    ap.add_argument("--lambda-anc", type=float, default=2.5)
    ap.add_argument("--lambda-lock", type=float, default=1.5)
    ap.add_argument("--max-real", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True

    ref = load_desnowlite(args.init_stage1, device=dev).requires_grad_(False).eval()
    net = copy.deepcopy(ref).train()
    freeze_lowfreq_modules(net)
    ema = copy.deepcopy(net).requires_grad_(False).eval()

    tuned_params = [p for p in net.parameters() if p.requires_grad]
    all_params = list(net.parameters())
    ema_params = list(ema.parameters())
    total_k = sum(p.numel() for p in all_params) / 1e3
    tuned_k = sum(p.numel() for p in tuned_params) / 1e3
    print(f"[DesnowLite Stage-2] Tuned/Total params: {tuned_k:.1f}K / {total_k:.1f}K", flush=True)

    opt = torch.optim.AdamW(tuned_params, lr=args.lr, weight_decay=1e-4)
    win = gauss_win(dev)

    tr_pairs = pairs_from_root(subdir(args.csd_root, r"^train"))
    src_pool = CpuPairPool(tr_pairs, dev_init=dev)
    R_imgs, R_masks, R_lows = load_real_pool_with_canopy_cache(
        Path(args.real_dir), ref, Path(args.mask_cache), args.crop, dev, max_images=args.max_real
    )

    sched = lambda s: args.lr * min(1.0, (s + 1) / 100.0) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * s / args.steps)))
    t0 = time.time()

    for step in range(1, args.steps + 1):
        for pg in opt.param_groups:
            pg["lr"] = sched(step)

        xc, tc, _, ctx_c = src_pool.batch(args.bs_src, args.crop, dev)
        with torch.no_grad():
            yc_ref = ref(xc, ctx=ctx_c).clamp(0.0, 1.0)

        xr, sem_tree, ctx_r = sample_real_batch(R_imgs, R_masks, R_lows, args.bs_real, args.crop, dev)
        with torch.no_grad():
            yr_ref, aux_ref = ref(xr, return_aux=True, ctx=ctx_r)
            yr_ref = yr_ref.clamp(0.0, 1.0)
            a_init = torch.sigmoid(aux_ref["logit"])
            y_target, a_hat, flake, cov, tree = four_region_targets(yr_ref, xr, sem_tree)

            hB = args.bs_real // 2
            if hB > 0:
                shield = tree[hB:] | cov[hB:]
                x_syn, m_syn = shielded_cutmix(xr[hB:], shield_mask=shield)
                xr_in = torch.cat([xr[:hB], x_syn], dim=0)
                syn_flk = (m_syn > 0.04) & (~shield)
                flake_in = torch.cat([flake[:hB], (flake[hB:] | syn_flk) & (~tree[hB:]) & (~cov[hB:])], dim=0)
                a_hat_tot = torch.cat([a_hat[:hB], torch.maximum(a_hat[hB:], m_syn * (~shield).float())], dim=0)
            else:
                xr_in, flake_in, a_hat_tot = xr, flake, a_hat

        opt.zero_grad(set_to_none=True)
        yc = net(xc, ctx=ctx_c)
        l_src = (yc - tc).abs().mean() + 0.15 * (1.0 - ssim_map(yc.clamp(0.0, 1.0), tc, win).mean())
        l_lwf = 1.5 * ((yc - yc_ref).abs().mean() + 0.8 * (hp3(yc) - hp3(yc_ref)).abs().mean())

        yr, auxr = net(xr_in, return_aux=True, ctx=ctx_r)
        lf = compute_four_region_loss(yr, auxr, yr_ref, aux_ref, a_init, y_target, a_hat_tot, flake_in, cov, tree)

        loss = l_src + l_lwf + (
            args.lambda_flake * lf["flake"]
            + args.lambda_tree * lf["tree"]
            + args.lambda_anc * lf["anc"]
            + args.lambda_lock * lf["lock"]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tuned_params, 1.0)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(ema_params, all_params, 1.0 - args.ema)

        if step % 100 == 0 or step == args.steps:
            elapsed_m = (time.time() - t0) / 60.0
            print(
                f"[step {step:4d}/{args.steps}] loss={loss.item():.4f} "
                f"flake={float(lf['flake_frac'])*100:.1f}% tree={float(lf['tree_frac'])*100:.1f}% "
                f"elapsed={elapsed_m:.1f}m",
                flush=True,
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_sd = {k: v.cpu() for k, v in ema.state_dict().items() if ".ahead_fn." not in k}
    torch.save({"ema": clean_sd, "step": args.steps, "arch": "desnowlite"}, str(out_path))
    print(f"Saved standalone checkpoint: {out_path}")


if __name__ == "__main__":
    main()
