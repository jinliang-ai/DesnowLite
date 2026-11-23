#!/usr/bin/env python3
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import ARCHS, build_desnowlite
from stage1_pretrain.dataset import load_set, pairs_from_root, subdir
from stage1_pretrain.losses import gauss_win, ssim_map, stage1_criterion

ALL_ARCHS = dict(ARCHS)


def load_partial_weights(net: torch.nn.Module, path: str):
    st = torch.load(path, map_location="cpu", weights_only=True)
    st = st.get("ema", st.get("net", st))
    own = net.state_dict()
    st = {k: v for k, v in st.items() if ".ahead_fn." not in k and (k not in own or own[k].shape == v.shape)}
    return net.load_state_dict(st, strict=False)


def save_atomic(obj, path: str):
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate_paired(net: torch.nn.Module, ds, ids, bs: int, win):
    net.eval()
    pn, ss = [], []
    for b in ds.chunks(ids, bs):
        x, t = ds.full(b)
        y = (net(x).clamp(0, 1) * 255).round() / 255
        pn.append(-10 * torch.log10((y - t).square().mean((1, 2, 3)).clamp_min(1e-10)))
        ss.append(ssim_map(y, t, win).mean((1, 2, 3)))
    pn, ss = torch.cat(pn), torch.cat(ss)
    return pn.mean().item(), ss.mean().item(), pn.tolist(), ss.tolist()


def build_model(args, dev: torch.device, amp: bool) -> torch.nn.Module:
    cls, budget = ALL_ARCHS[args.arch]
    net = cls(amp=amp).to(dev)
    total = sum(p.numel() for p in net.parameters())
    assert total <= budget, f"parameter budget exceeded: {total:,}"
    if args.init:
        res = load_partial_weights(net, args.init)
        if hasattr(net, "after_load"):
            net.after_load(res.missing_keys)
    print(f"[{args.arch}] parameters: {total:,}", flush=True)
    return net


def main():
    ap = argparse.ArgumentParser(description="DesnowLite Stage-1 paired synthetic pretraining")
    ap.add_argument("--data", help="Dataset root containing Train/ and Test/")
    ap.add_argument("--split", help="Split JSON from make_splits.py")
    ap.add_argument("--eval-set", default="test")
    ap.add_argument("--metrics-out", default="")
    ap.add_argument("--arch", choices=list(ALL_ARCHS), default="desnowlite")
    ap.add_argument("--out", default="runs/stage1_desnowlite")
    ap.add_argument("--init", default=None)
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crop", type=int, default=448)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--mse-tail", type=float, default=0.0)
    ap.add_argument("--a-w", type=float, default=0.2)
    ap.add_argument("--base-w", type=float, default=0.1)
    ap.add_argument("--mask-mix", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--val-every", type=int, default=5000)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-test", action="store_true")
    args = ap.parse_args()

    assert args.crop % 64 == 0, "--crop must be divisible by 64"
    assert bool(args.data) != bool(args.split), "specify either --data or --split"
    split = json.load(open(args.split)) if args.split else None

    dev = torch.device(args.device)
    cuda = dev.type == "cuda"
    amp = cuda and not args.no_amp
    torch.manual_seed(args.seed)
    if cuda:
        torch.backends.cudnn.benchmark = True
    win = gauss_win(dev)

    if args.eval_test:
        pairs = split[args.eval_set] if split else pairs_from_root(subdir(args.data, r"^test"))
        ds = load_set(pairs, dev, args.threads, args.limit, args.eval_set)
        net = build_model(args, dev, False)
        pn, ss, pi, si = evaluate_paired(net, ds, torch.arange(ds.N, device=ds.index_dev), args.eval_batch, win)
        print(f"[{args.eval_set}] PSNR={pn:.3f} dB  SSIM={ss:.4f} (N={ds.N})", flush=True)
        if args.metrics_out:
            with open(args.metrics_out, "w") as f:
                json.dump({"psnr": pn, "ssim": ss, "n": ds.N, "per_image": [[p[0], a, b] for p, a, b in zip(pairs, pi, si)]}, f)
        return

    if split:
        ds = load_set(split["train"], dev, args.threads, args.limit, "train", args.crop, args.mask_mix)
        vs = load_set(split["holdout"], dev, args.threads, args.limit, "holdout")
        train_ids = torch.arange(ds.N, device=ds.index_dev)
        val_ids, nv = torch.arange(vs.N, device=vs.index_dev), vs.N
    else:
        tr_pairs = pairs_from_root(subdir(args.data, r"^train"))
        ds = vs = load_set(tr_pairs, dev, args.threads, args.limit, "train", args.crop, args.mask_mix)
        g = torch.Generator().manual_seed(123)
        perm = torch.randperm(ds.N, generator=g).to(ds.index_dev)
        nv = max(1, int(ds.N * args.val_frac))
        val_ids, train_ids = perm[:nv], perm[nv:]

    os.makedirs(args.out, exist_ok=True)
    net = build_model(args, dev, amp)
    ema = copy.deepcopy(net).requires_grad_(False)
    ema.amp = False
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0, fused=cuda)

    step, best = 0, -1.0
    last = f"{args.out}/last.pth"
    if os.path.exists(last):
        st = torch.load(last, map_location=dev, weights_only=True)
        clean = lambda d: {k: v for k, v in d.items() if ".ahead_fn." not in k}
        net.load_state_dict(clean(st["net"]))
        ema.load_state_dict(clean(st["ema"]))
        opt.load_state_dict(st["opt"])
        step, best = st["step"], st["best"]

    if cuda and not args.no_compile:
        net.compile_bodies()

    params, ema_params = list(net.parameters()), list(ema.parameters())
    order, pos = torch.randperm(train_ids.numel(), device=ds.index_dev), 0
    bs = min(args.batch, train_ids.numel())
    sched = lambda s: args.lr * min(1.0, (s + 1) / 500) * (0.005 + 0.995 * 0.5 * (1 + math.cos(math.pi * s / args.steps)))
    t0, s0 = time.time(), step

    while step < args.steps:
        if pos + bs > order.numel():
            order, pos = torch.randperm(train_ids.numel(), device=ds.index_dev), 0
        ids = train_ids[order[pos : pos + bs]]
        pos += bs
        x, t, m, ctx = ds.batch(ids, args.crop)
        for pg in opt.param_groups:
            pg["lr"] = sched(step)

        y, aux = net(x, return_aux=True, ctx=ctx)
        loss = stage1_criterion(
            y, aux, t, m, win,
            a_w=args.a_w,
            base_w=args.base_w,
            psnr_only=step >= args.steps * (1 - args.mse_tail),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=cuda)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(ema_params, params, 1 - args.ema)
        step += 1

        if step % args.log_every == 0:
            lv = loss.item()
            if not math.isfinite(lv):
                raise RuntimeError(f"non-finite loss at step {step}")
            ips = (step - s0) * bs / max(1e-6, time.time() - t0)
            print(f"[step {step}] loss={lv:.4f} lr={sched(step):.2e} speed={ips:.1f} img/s", flush=True)

        if step % args.val_every == 0 or step == args.steps:
            pn, ss, _, _ = evaluate_paired(ema, vs, val_ids, args.eval_batch, win)
            print(f"[val {step}] PSNR={pn:.3f} dB  SSIM={ss:.4f} (N={nv})", flush=True)
            if pn > best:
                best = pn
                save_atomic({"ema": ema.state_dict(), "step": step, "psnr": pn}, f"{args.out}/desnowlite_stage1_best.pth")
            save_atomic({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step, "best": best}, last)


if __name__ == "__main__":
    main()
