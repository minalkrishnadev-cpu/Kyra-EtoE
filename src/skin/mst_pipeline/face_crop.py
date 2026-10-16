# Crop the head region out of a full-body frame and upscale it so the
# face detector sees a large, well-resolved face instead of a few dozen pixels.
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # (x0, y0, x1, y1) in pixels


def clamp_box(box: Box, image_shape) -> Optional[Box]:
    """Clip a box to the image. Returns None if nothing is left."""
    h, w = image_shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, x1 = max(0, min(w, x0)), max(0, min(w, x1))
    y0, y1 = max(0, min(h, y0)), max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def crop_and_upscale(
    image_bgr: np.ndarray,
    box: Box,
    target_side: int = 512,
    max_scale: float = 5.0,
    min_side: int = 24,
):
    """
    Cut `box` out of the image and enlarge it so its longer side is about
    `target_side` pixels (never more than `max_scale` times, never shrinks).

    Returns (crop_bgr, scale), or None if the box is empty or too small to be useful.
    """
    clamped = clamp_box(box, image_bgr.shape)
    if clamped is None:
        return None
    x0, y0, x1, y1 = clamped
    cw, ch = x1 - x0, y1 - y0
    if min(cw, ch) < min_side:
        return None

    crop = image_bgr[y0:y1, x0:x1]
    scale = min(max_scale, target_side / float(max(cw, ch)))
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        scale = 1.0
        crop = crop.copy()
    return crop, scale
