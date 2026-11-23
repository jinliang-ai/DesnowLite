#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import load_desnowlite

RESOLUTIONS = {
    "480p": (480, 854),
    "720p": (720, 1280),
    "1080p": (1080, 1920),
    "1440p": (1440, 2560),
    "4K": (2160, 3840),
}


@torch.no_grad()
def profile_eager(net: torch.nn.Module, h: int, w: int, dev: torch.device, warmup: int = 10, runs: int = 30):
    x = torch.rand(1, 3, h, w, device=dev)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        _ = net(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = net(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / runs
    vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if dev.type == "cuda" else 0.0
    return ms, vram_mb


@torch.no_grad()
def profile_cuda_graph(net: torch.nn.Module, h: int, w: int, warmup: int = 10, runs: int = 50):
    dev = torch.device("cuda")
    static_x = torch.rand(1, 3, h, w, device=dev)
    for _ in range(warmup):
        _ = net(static_x)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_y = net(static_x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / runs, static_y.shape


def main():
    ap = argparse.ArgumentParser(description="Profile DesnowLite latency and peak memory across resolutions")
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "desnowlite_042m_2k.pth"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--res", nargs="+", default=["480p", "720p", "1080p"])
    ap.add_argument("--cuda-graph", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device)
    net = load_desnowlite(args.ckpt, device=dev)
    print(f"Device: {dev} | Parameters: {sum(p.numel() for p in net.parameters()):,}")

    for name in args.res:
        h, w = RESOLUTIONS[name]
        ms, vram = profile_eager(net, h, w, dev)
        msg = f"[{name:5s} ({w}x{h})] eager={ms:6.2f} ms"
        if dev.type == "cuda":
            msg += f"  peak_vram={vram:6.1f} MB"
            if args.cuda_graph:
                cg_ms, _ = profile_cuda_graph(net, h, w)
                msg += f"  cuda_graph={cg_ms:6.2f} ms"
        print(msg)


if __name__ == "__main__":
    main()
