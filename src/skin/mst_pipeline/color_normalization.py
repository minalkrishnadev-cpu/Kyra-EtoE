# gray-world white balance + robust skin-color sampling
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.color import rgb2lab


def gray_world_white_balance(image_bgr: np.ndarray) -> np.ndarray:
    # cheap heuristic: scale each channel so its mean matches the overall gray mean
    img = image_bgr.astype(np.float64)
    mean_b, mean_g, mean_r = img[..., 0].mean(), img[..., 1].mean(), img[..., 2].mean()
    gray_mean = (mean_b + mean_g + mean_r) / 3.0

    eps = 1e-6
    scale_b = gray_mean / (mean_b + eps)
    scale_g = gray_mean / (mean_g + eps)
    scale_r = gray_mean / (mean_r + eps)

    img[..., 0] *= scale_b
    img[..., 1] *= scale_g
    img[..., 2] *= scale_r

    return np.clip(img, 0, 255).astype(np.uint8)


@dataclass
class SkinSample:
    lab_median: np.ndarray  # [L, a, b]
    n_pixels_total: int
    n_pixels_used: int  # after exposure filtering
    rejected_fraction: float
    regions_used: list | None = None
    regions_excluded: list | None = None
    region_median_L: dict | None = None


def sample_skin_lab(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    low_pct: float = 5.0,
    high_pct: float = 95.0,
) -> SkinSample:
    # single-mask version, kept for simple use cases
    n_total = int(mask.sum())
    if n_total == 0:
        raise ValueError("Empty skin mask -- no ROI pixels to sample.")

    rgb01 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    lab = rgb2lab(rgb01)

    roi_lab = lab[mask]
    L = roi_lab[:, 0]

    lo, hi = np.percentile(L, [low_pct, high_pct])
    keep = (L >= lo) & (L <= hi)

    used = roi_lab[keep]
    n_used = int(keep.sum())
    if n_used == 0:
        used = roi_lab
        n_used = n_total

    median_lab = np.median(used, axis=0)

    return SkinSample(
        lab_median=median_lab,
        n_pixels_total=n_total,
        n_pixels_used=n_used,
        rejected_fraction=1.0 - (n_used / n_total),
    )


def sample_skin_lab_regions(
    image_bgr: np.ndarray,
    region_masks: dict,
    low_pct: float = 5.0,
    high_pct: float = 95.0,
    outlier_threshold_L: float = 15.0,
) -> SkinSample:
    # per-region version: samples forehead/cheeks separately, drops any
    # region whose median L is an outlier vs the others (hair, shadow, etc.)
    rgb01 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    lab = rgb2lab(rgb01)

    region_lab = {}
    region_median_L = {}
    for name, m in region_masks.items():
        if m.sum() == 0:
            continue
        pixels = lab[m]
        region_lab[name] = pixels
        region_median_L[name] = float(np.median(pixels[:, 0]))

    if not region_lab:
        raise ValueError("All region masks are empty -- no ROI pixels to sample.")

    overall_center = float(np.median(list(region_median_L.values())))
    kept = [
        name for name, med in region_median_L.items()
        if abs(med - overall_center) <= outlier_threshold_L
    ]
    if not kept:
        kept = list(region_lab.keys())  # everything disagrees, keep all rather than fail

    excluded = [name for name in region_lab if name not in kept]

    pooled = np.concatenate([region_lab[name] for name in kept], axis=0)
    n_total = int(pooled.shape[0])

    L = pooled[:, 0]
    lo, hi = np.percentile(L, [low_pct, high_pct])
    keep_mask = (L >= lo) & (L <= hi)
    used = pooled[keep_mask]
    n_used = int(keep_mask.sum())
    if n_used == 0:
        used = pooled
        n_used = n_total

    median_lab = np.median(used, axis=0)

    return SkinSample(
        lab_median=median_lab,
        n_pixels_total=n_total,
        n_pixels_used=n_used,
        rejected_fraction=1.0 - (n_used / n_total),
        regions_used=kept,
        regions_excluded=excluded,
        region_median_L=region_median_L,
    )
