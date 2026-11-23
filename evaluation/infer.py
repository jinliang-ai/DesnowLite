#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import load_desnowlite

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def select_device(dev_arg: str) -> torch.device:
    if dev_arg != "auto":
        return torch.device(dev_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def infer_tensor(net: torch.nn.Module, x: torch.Tensor, tta_hflip: bool = False, veil_scale: float = 1.0) -> torch.Tensor:
    y = net(x, veil_scale=veil_scale).float().clamp(0.0, 1.0)
    if tta_hflip:
        yf = torch.flip(net(torch.flip(x, dims=[-1]), veil_scale=veil_scale).float().clamp(0.0, 1.0), dims=[-1])
        y = 0.5 * (y + yf)
    return y.clamp(0.0, 1.0)


def process_one(
    net: torch.nn.Module,
    src: Path,
    dst: Path,
    device: torch.device,
    max_side: int = 0,
    tta: bool = False,
    veil_scale: float = 1.0,
    save_compare: bool = False,
) -> float:
    bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {src}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if max_side > 0 and max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        rgb = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)

    x = torch.from_numpy(rgb).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t0 = time.perf_counter()
    y = infer_tensor(net, x, tta_hflip=tta, veil_scale=veil_scale)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    out_rgb = (y[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), out_bgr)

    if save_compare:
        in_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        comp = np.concatenate([in_bgr, out_bgr], axis=1)
        cv2.imwrite(str(dst.with_name(f"{dst.stem}_compare{dst.suffix}")), comp)
    return dt_ms


def main():
    ap = argparse.ArgumentParser(description="DesnowLite standalone inference")
    ap.add_argument("--input", "-i", required=True, help="Input snowy image or directory")
    ap.add_argument("--output", "-o", required=True, help="Output image or directory")
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "desnowlite_042m_2k.pth"))
    ap.add_argument("--arch", default="desnowlite")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-side", type=int, default=0)
    ap.add_argument("--veil-scale", type=float, default=1.0)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--save-compare", action="store_true")
    args = ap.parse_args()

    device = select_device(args.device)
    net = load_desnowlite(args.ckpt, device=device, arch=args.arch)
    nparams = sum(p.numel() for p in net.parameters())
    print(f"[DesnowLite] Loaded {Path(args.ckpt).name} ({nparams:,} params) on {device}")

    inp, out = Path(args.input), Path(args.output)
    if inp.is_file():
        dst = out if out.suffix else (out / inp.name)
        ms = process_one(net, inp, dst, device, args.max_side, args.tta, args.veil_scale, args.save_compare)
        print(f"[Done] {inp.name} -> {dst} ({ms:.1f} ms)")
    elif inp.is_dir():
        files = sorted(p for p in inp.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if not files:
            raise SystemExit(f"No images found in {inp}")
        times = []
        for idx, p in enumerate(files, 1):
            rel = p.relative_to(inp)
            ms = process_one(net, p, out / rel, device, args.max_side, args.tta, args.veil_scale, args.save_compare)
            times.append(ms)
            if idx == 1 or idx % 25 == 0 or idx == len(files):
                print(f"[{idx}/{len(files)}] {rel} ({ms:.1f} ms)")
        print(f"[Done] {len(files)} images -> {out} (mean {np.mean(times):.1f} ms/img)")
    else:
        raise FileNotFoundError(f"Input path not found: {inp}")


if __name__ == "__main__":
    main()
