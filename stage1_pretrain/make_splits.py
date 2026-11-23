#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
cv2.setNumThreads(1)


def files(d: Path) -> dict[str, Path]:
    return {Path(n).stem: Path(d) / n for n in os.listdir(d) if n.lower().endswith(IMG_EXTS)}


def probe(p: Path):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    th = cv2.resize(im, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return hashlib.md5(im.tobytes()).hexdigest(), th, im.shape[:2]


def full_psnr(a: Path, b: Path) -> float:
    x, y = cv2.imread(str(a)), cv2.imread(str(b))
    if x.shape != y.shape:
        return 0.0
    mse = ((x.astype(np.float64) - y) ** 2).mean()
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i: int, j: int):
        self.p[self.find(i)] = self.find(j)


def group_backgrounds(gts: list[Path], threads: int, k: int = 6, thumb_db: float = 24.0, thr: float = 30.0):
    with ThreadPoolExecutor(threads) as ex:
        info = list(ex.map(probe, gts))
    n = len(gts)
    dsu = DSU(n)
    first = {}
    for i, (h, _, _) in enumerate(info):
        if h in first:
            dsu.union(i, first[h])
        else:
            first[h] = i
    T = np.stack([t for _, t, _ in info])
    sq = (T ** 2).sum(1)
    cand = set()
    for s in range(0, n, 2048):
        qi = np.arange(s, min(n, s + 2048))
        d = (sq[qi, None] + sq[None] - 2 * T[qi] @ T.T).clip(0) / T.shape[1]
        d[np.arange(len(qi)), qi] = np.inf
        nn = np.argpartition(d, min(k, n - 1), 1)[:, :k]
        for r, i in enumerate(qi):
            for j in nn[r]:
                if 10 * np.log10(255 ** 2 / max(d[r, j], 1e-6)) > thumb_db and dsu.find(i) != dsu.find(j):
                    cand.add((min(int(i), int(j)), max(int(i), int(j))))
    cand = sorted(cand)
    with ThreadPoolExecutor(threads) as ex:
        ps = list(ex.map(lambda ij: full_psnr(gts[ij[0]], gts[ij[1]]), cand))
    for (i, j), p in zip(cand, ps):
        if p > thr:
            dsu.union(i, j)
    return [dsu.find(i) for i in range(n)]


def build_csd_split(csd_root: Path, out_json: Path, holdout_frac: float = 0.05, threads: int = 8):
    tr_snow = files(csd_root / "Train" / "Snow")
    tr_gt = files(csd_root / "Train" / "Gt")
    tr_mk = files(csd_root / "Train" / "Mask")
    te_snow = files(csd_root / "Test" / "Snow")
    te_gt = files(csd_root / "Test" / "Gt")
    te_mk = files(csd_root / "Test" / "Mask")

    tr_keys = sorted(set(tr_snow) & set(tr_gt))
    te_keys = sorted(set(te_snow) & set(te_gt))
    gid = group_backgrounds([tr_gt[k] for k in tr_keys], threads=threads)

    rng = random.Random(42)
    groups = sorted(set(gid))
    rng.shuffle(groups)
    by_g = Counter(gid)
    hold_g, cnt = set(), 0
    target = int(len(tr_keys) * holdout_frac)
    for g in groups:
        if cnt >= target:
            break
        hold_g.add(g)
        cnt += by_g[g]

    pack = lambda s, g, m, ks: [[str(s[k]), str(g[k]), str(m[k]) if k in m else None] for k in ks]
    tr_keep = [k for k, g in zip(tr_keys, gid) if g not in hold_g]
    ho_keep = [k for k, g in zip(tr_keys, gid) if g in hold_g]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {
                "train": pack(tr_snow, tr_gt, tr_mk, tr_keep),
                "holdout": pack(tr_snow, tr_gt, tr_mk, ho_keep),
                "test": pack(te_snow, te_gt, te_mk, te_keys),
            },
            f,
            indent=2,
        )
    print(f"Saved {out_json}: train={len(tr_keep)}, holdout={len(ho_keep)}, test={len(te_keys)}")


def main():
    ap = argparse.ArgumentParser(description="Build scene-disjoint CSD splits")
    ap.add_argument("--csd-root", required=True)
    ap.add_argument("--out", default="splits/csd_official.json")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    build_csd_split(Path(args.csd_root), Path(args.out), threads=args.threads)


if __name__ == "__main__":
    main()
