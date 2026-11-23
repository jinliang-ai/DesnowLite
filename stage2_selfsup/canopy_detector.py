import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F

from .four_region_prior import _maxpool, _minpool, _shift, luma, open_rect, saturation

ADE_STRICT_TREE_IDS = [4, 72]
ADE_NON_TREE_FLORA_IDS = [9, 13, 17, 29]
ADE_HARD_OBJ_IDS = [
    0, 1, 3, 5, 6, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26,
    32, 34, 36, 38, 43, 48, 52, 53, 60, 61, 80, 82, 83, 86, 87, 88, 93, 102,
    106, 126, 127, 132, 136,
]

MV_SKY_ID = 27
MV_SNOW_ID = 28
MV_VEG_ID = 30
MV_GROUND_IDS = [2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 23, 24, 28, 29]
MV_HARD_FG_IDS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
]

DIRS8_TAPS9 = [
    [(0, k) for k in range(-4, 5)],
    [(-2, -4), (-2, -3), (-1, -2), (-1, -1), (0, 0), (1, 1), (1, 2), (2, 3), (2, 4)],
    [(k, k) for k in range(-4, 5)],
    [(-4, -2), (-3, -2), (-2, -1), (-1, -1), (0, 0), (1, 1), (2, 1), (3, 2), (4, 2)],
    [(k, 0) for k in range(-4, 5)],
    [(-4, 2), (-3, 2), (-2, 1), (-1, 1), (0, 0), (1, -1), (2, -1), (3, -2), (4, -2)],
    [(k, -k) for k in range(-4, 5)],
    [(-2, 4), (-2, 3), (-1, 2), (-1, 1), (0, 0), (1, -1), (1, -2), (2, -3), (2, -4)],
]
DIRS8_TAPS7 = [taps[1:-1] for taps in DIRS8_TAPS9]


def _box_filter(x: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return x
    k = 2 * r + 1
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="replicate"), kernel_size=k, stride=1)


def _dilate_taps(z: torch.Tensor, taps) -> torch.Tensor:
    out = z
    for dy, dx in taps:
        if dy != 0 or dx != 0:
            out = torch.maximum(out, _shift(z, dy, dx))
    return out


def _open_line_taps(z: torch.Tensor, taps) -> torch.Tensor:
    return _dilate_taps(-_dilate_taps(-z, taps), taps)


def _close_axis5(z: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    taps = [(-2 * dy, -2 * dx), (-dy, -dx), (0, 0), (dy, dx), (2 * dy, 2 * dx)]
    return -_dilate_taps(-_dilate_taps(z, taps), taps)


def _twig_structure_cues(g_yr: torch.Tensor):
    dir_opens9 = [_open_line_taps(g_yr, taps) for taps in DIRS8_TAPS9]
    op_dir8_9 = dir_opens9[0]
    for op_k in dir_opens9[1:]:
        op_dir8_9 = torch.maximum(op_dir8_9, op_k)

    g_struct = torch.minimum(g_yr, _maxpool(op_dir8_9, 3, 3))
    c0 = (_close_axis5(g_struct, 0, 1) - g_struct).clamp_min(0.0)
    c1 = (_close_axis5(g_struct, 1, 0) - g_struct).clamp_min(0.0)
    c2 = (_close_axis5(g_struct, 1, 1) - g_struct).clamp_min(0.0)
    c3 = (_close_axis5(g_struct, 1, -1) - g_struct).clamp_min(0.0)
    perp_for_dir = [
        c1, torch.maximum(c1, c3), c3, torch.maximum(c0, c3),
        c0, torch.maximum(c0, c2), c2, torch.maximum(c1, c2),
    ]
    bth_dirs7 = [_open_line_taps(perp_for_dir[k], DIRS8_TAPS7[k]) for k in range(8)]
    bth_line7 = bth_dirs7[0]
    for bd in bth_dirs7[1:]:
        bth_line7 = torch.maximum(bth_line7, bd)

    bg_up9 = _maxpool(op_dir8_9, 9, 9).clamp_min(0.15)
    rel_bark7 = bth_line7 / bg_up9

    op2d_5 = open_rect(g_yr, 5, 5)
    mean21_2d = F.avg_pool2d(F.pad(op2d_5, (10, 10, 10, 10), mode="replicate"), 21, 1)
    floor15_2d = _minpool(op2d_5, 15, 15)
    not_night = (floor15_2d > 0.055) | (mean21_2d > 0.26)

    bark_core = ((bth_line7 > 0.042) | ((rel_bark7 > 0.130) & (bth_line7 > 0.020))) & not_night
    twig_zone = _maxpool(bark_core.float(), 5, 5) > 0
    return twig_zone, bark_core


class Mask2FormerCanopyDetector(nn.Module):
    """Offline ADE20K + Mapillary Vistas Mask2Former canopy extractor for Omega_tree."""

    def __init__(
        self,
        device: torch.device,
        ade_model_id: str = "facebook/mask2former-swin-large-ade-semantic",
        mv_model_id: str = "facebook/mask2former-swin-large-mapillary-vistas-semantic",
        local_files_only: bool = False,
    ):
        super().__init__()
        from transformers import Mask2FormerForUniversalSegmentation

        self.device = device
        self.ade_m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
            ade_model_id, local_files_only=local_files_only
        ).to(device).eval()
        self.mv_m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
            mv_model_id, local_files_only=local_files_only
        ).to(device).eval()

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.register_buffer("img_mean", mean, persistent=False)
        self.register_buffer("img_std", std, persistent=False)

    @torch.no_grad()
    def _m2f_single_scale(self, x_rgb: torch.Tensor, model: nn.Module, sz: int, out_hw: tuple[int, int]) -> torch.Tensor:
        x_in = F.interpolate(x_rgb.clamp(0.0, 1.0), size=(sz, sz), mode="bilinear", align_corners=False)
        x_norm = (x_in - self.img_mean) / self.img_std
        out = model(pixel_values=x_norm)
        cp = out.class_queries_logits.softmax(dim=-1)[..., :-1]
        mp = F.interpolate(out.masks_queries_logits, size=out_hw, mode="bilinear", align_corners=False).sigmoid()
        scores = torch.einsum("bqc,bqhw->bchw", cp, mp)[0]
        return scores / (scores.sum(dim=0, keepdim=True) + 0.20)

    @torch.no_grad()
    def _m2f_probs(self, x_rgb: torch.Tensor, model: nn.Module) -> torch.Tensor:
        _, _, h, w = x_rgb.shape
        return 0.65 * self._m2f_single_scale(x_rgb, model, 384, (h, w)) + 0.35 * self._m2f_single_scale(x_rgb, model, 512, (h, w))

    @torch.no_grad()
    def detect(self, ref_rgb: torch.Tensor, canopy_close_k: int = 9) -> torch.Tensor:
        _, _, h, w = ref_rgb.shape
        pa = self._m2f_probs(ref_rgb, self.ade_m2f)
        pm = self._m2f_probs(ref_rgb, self.mv_m2f)

        p_ade_tree = pa[ADE_STRICT_TREE_IDS].sum(dim=0, keepdim=True).unsqueeze(0)
        p_ade_low_flora = pa[ADE_NON_TREE_FLORA_IDS].sum(dim=0, keepdim=True).unsqueeze(0)
        p_ade_hard_obj = pa[ADE_HARD_OBJ_IDS].sum(dim=0, keepdim=True).unsqueeze(0)
        p_ade_sky = pa[2:3].unsqueeze(0)

        p_mv_veg = pm[MV_VEG_ID : MV_VEG_ID + 1].unsqueeze(0)
        p_mv_ground = pm[MV_GROUND_IDS].sum(dim=0, keepdim=True).unsqueeze(0)
        p_mv_hard_fg = pm[MV_HARD_FG_IDS].sum(dim=0, keepdim=True).unsqueeze(0)
        p_mv_sky = pm[MV_SKY_ID : MV_SKY_ID + 1].unsqueeze(0)
        p_mv_snow = pm[MV_SNOW_ID : MV_SNOW_ID + 1].unsqueeze(0)

        g = luma(ref_rgb.clamp(0.0, 1.0))
        s_sat = saturation(ref_rgb.clamp(0.0, 1.0))
        twig_zone, bark_core = _twig_structure_cues(g)

        not_pitch_black = _box_filter(open_rect(g, 17, 17), r=15) > 0.026
        g_open5 = open_rect(g, 5, 5)
        gx = F.pad((g_open5[:, :, :, 1:] - g_open5[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        gy = F.pad((g_open5[:, :, 1:, :] - g_open5[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        has_tree_texture = (_box_filter((twig_zone | bark_core).float(), r=9) >= 0.035) | (
            _box_filter((gx + gy > 0.020).float(), r=8) >= 0.07
        )

        not_fire_glow = ~((ref_rgb[:, 0:1] > 0.60) & ((ref_rgb[:, 0:1] - ref_rgb[:, 2:3]) > 0.32))
        not_low_flora = (p_ade_low_flora < 0.26) & (~((p_mv_ground >= 0.38) & (p_ade_tree < 0.42)))
        not_hard_fg = (p_ade_hard_obj < 0.42) & (p_mv_hard_fg < 0.42) & not_fire_glow

        seed_consensus = (p_ade_tree >= 0.30) & (p_mv_veg >= 0.25) & (p_mv_snow < 0.35) & has_tree_texture & not_low_flora & not_hard_fg & not_pitch_black
        seed_frosted = (p_ade_tree >= 0.45) & ((p_mv_veg + p_mv_snow) >= 0.28) & has_tree_texture & (p_ade_low_flora < 0.24) & not_hard_fg & not_pitch_black
        seed_bare = (
            (p_mv_veg >= 0.52)
            & (p_ade_tree >= 0.10)
            & (_box_filter((twig_zone | bark_core).float(), r=9) >= 0.05)
            & (p_ade_low_flora < 0.22)
            & (p_mv_ground < 0.28)
            & (p_ade_hard_obj < 0.35)
            & (p_mv_hard_fg < 0.35)
            & not_fire_glow
            & not_pitch_black
        )

        # Unshield bright achromatic snow clumps resting on branches
        snow_clump_veto = (g > 0.75) & (s_sat < 0.10)
        seed_np = ((seed_consensus | seed_frosted | seed_bare) & (~snow_clump_veto))[0, 0].cpu().numpy().astype(bool)

        min_seed_area = max(96, int(0.0010 * h * w))
        lbl, n_cc = ndi.label(seed_np)
        if n_cc > 0:
            sizes = np.bincount(lbl.ravel())
            keep_ids = np.where(sizes >= min_seed_area)[0]
            keep_ids = keep_ids[keep_ids > 0]
            seed_clean = np.isin(lbl, keep_ids)
        else:
            seed_clean = np.zeros_like(seed_np, dtype=bool)

        if seed_clean.any():
            strict_veto = (
                (p_ade_hard_obj >= 0.45)
                | (p_mv_hard_fg >= 0.45)
                | (p_ade_low_flora >= 0.26)
                | ((p_mv_ground >= 0.38) & (p_ade_tree < 0.38))
                | snow_clump_veto
            )
            grow_allow = (
                (
                    ((p_ade_tree >= 0.20) & (p_mv_veg >= 0.18) & has_tree_texture)
                    | ((p_mv_veg >= 0.45) & (p_ade_tree >= 0.08) & has_tree_texture)
                    | (twig_zone & ((p_ade_tree >= 0.06) | (p_mv_veg >= 0.28)))
                )
                & (~strict_veto)
                & not_pitch_black
            )[0, 0].cpu().numpy().astype(bool)

            near_seed = ndi.binary_dilation(seed_clean, structure=np.ones((21, 21), dtype=bool))
            cand_union = seed_clean | (grow_allow & near_seed)
            lbl_u, n_u = ndi.label(cand_union)
            if n_u > 0:
                valid_u = np.unique(lbl_u[seed_clean])
                valid_u = valid_u[valid_u > 0]
                tree_bin = np.isin(lbl_u, valid_u)
            else:
                tree_bin = seed_clean

            hard_carve = (
                (p_ade_hard_obj >= 0.48) | (p_mv_hard_fg >= 0.48) | (p_ade_low_flora >= 0.32) | snow_clump_veto
            )[0, 0].cpu().numpy()

            se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (canopy_close_k, canopy_close_k))
            closed = cv2.morphologyEx(tree_bin.astype(np.uint8), cv2.MORPH_CLOSE, se_close).astype(bool)
            inv = ~closed
            lbl_h, n_h = ndi.label(inv)
            if n_h > 0:
                border_ids = (
                    set(np.unique(lbl_h[0, :]))
                    | set(np.unique(lbl_h[-1, :]))
                    | set(np.unique(lbl_h[:, 0]))
                    | set(np.unique(lbl_h[:, -1]))
                )
                h_sizes = np.bincount(lbl_h.ravel())
                max_hole = int(0.004 * h * w)
                fill_ids = [idx for idx in range(1, n_h + 1) if idx not in border_ids and h_sizes[idx] <= max_hole]
                if fill_ids:
                    closed = closed | np.isin(lbl_h, fill_ids)
            core_np = (tree_bin | closed) & (~hard_carve)
        else:
            core_np = np.zeros((h, w), dtype=bool)

        return torch.from_numpy(core_np.astype(np.float32)).to(self.device).view(1, 1, h, w)
