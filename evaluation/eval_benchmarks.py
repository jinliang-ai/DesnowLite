#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import compute_psnr_ssim, compute_r_flake, eval_flake_support
from models import load_desnowlite

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def decode_rgb_cap(path: Path, cap: int = 1024) -> np.ndarray:
    im = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    if cap > 0 and max(h, w) > cap:
        s = cap / float(max(h, w))
        im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    h2, w2 = im.shape[0] - (im.shape[0] % 2), im.shape[1] - (im.shape[1] % 2)
    return im[:h2, :w2]


@torch.no_grad()
def evaluate_real_folder(net: torch.nn.Module, data_dir: Path, dev: torch.device, cap: int = 1024, use_nriqa: bool = True) -> dict:
    files = sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(f"No images found in {data_dir}")

    niqe_fn = bris_fn = None
    if use_nriqa:
        import pyiqa
        niqe_fn = pyiqa.create_metric("niqe", device=dev)
        bris_fn = pyiqa.create_metric("brisque", device=dev)

    r_vals, niqe_vals, bris_vals = [], [], []
    for p in files:
        rgb = decode_rgb_cap(p, cap=cap)
        x = torch.from_numpy(rgb).to(dev).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        y = (net(x).float().clamp(0.0, 1.0) * 255.0).round() / 255.0
        flk_mask, _, _ = eval_flake_support(x)
        r = compute_r_flake(x, y, flk_mask)
        if not np.isnan(r):
            r_vals.append(r)
        if niqe_fn is not None and bris_fn is not None:
            niqe_vals.append(float(niqe_fn(y).item()))
            bris_vals.append(float(bris_fn(y).item()))

    res = {
        "n_images": len(files),
        "R_flake": float(np.mean(r_vals)) if r_vals else 0.0,
    }
    if niqe_vals:
        res["NIQE"] = float(np.mean(niqe_vals))
        res["BRISQUE"] = float(np.mean(bris_vals))
    return res


@torch.no_grad()
def evaluate_paired_folder(net: torch.nn.Module, snow_dir: Path, gt_dir: Path, dev: torch.device) -> dict:
    gt_map = {p.stem: p for p in gt_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
    pairs = sorted((p, gt_map[p.stem]) for p in snow_dir.iterdir() if p.is_file() and p.stem in gt_map)
    if not pairs:
        raise FileNotFoundError(f"No matched pairs in {snow_dir} and {gt_dir}")

    psnrs, ssims = [], []
    for sp, gp in pairs:
        s_rgb = cv2.cvtColor(cv2.imread(str(sp), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        g_rgb = cv2.cvtColor(cv2.imread(str(gp), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(s_rgb).to(dev).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = torch.from_numpy(g_rgb).to(dev).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        y = net(x).float().clamp(0.0, 1.0)
        p, s = compute_psnr_ssim(y, t)
        psnrs.append(p)
        ssims.append(s)
    return {"n_pairs": len(pairs), "PSNR": float(np.mean(psnrs)), "SSIM": float(np.mean(ssims))}


def main():
    ap = argparse.ArgumentParser(description="Evaluate DesnowLite on real-world or paired synthetic benchmarks")
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "desnowlite_042m_2k.pth"))
    ap.add_argument("--real-dir", default="", help="Unpaired real-snowfall image folder")
    ap.add_argument("--snow-dir", default="", help="Paired synthetic Snow/ folder")
    ap.add_argument("--gt-dir", default="", help="Paired synthetic Gt/ folder")
    ap.add_argument("--cap", type=int, default=1024)
    ap.add_argument("--no-nriqa", action="store_true")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    net = load_desnowlite(args.ckpt, device=dev)

    results = {"ckpt": args.ckpt, "params": sum(p.numel() for p in net.parameters())}
    if args.real_dir:
        results["real"] = evaluate_real_folder(net, Path(args.real_dir), dev, cap=args.cap, use_nriqa=not args.no_nriqa)
    if args.snow_dir and args.gt_dir:
        results["paired"] = evaluate_paired_folder(net, Path(args.snow_dir), Path(args.gt_dir), dev)

    print(json.dumps(results, indent=2))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
