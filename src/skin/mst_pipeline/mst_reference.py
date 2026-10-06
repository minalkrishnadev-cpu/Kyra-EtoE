# Monk Skin Tone (MST) reference colors + nearest-match classifier
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

# official MST-1..MST-10 hex swatches
MST_HEX = [
    "#f6ede4",  # MST-1
    "#f3e7db",  # MST-2
    "#f7ead0",  # MST-3
    "#eadaba",  # MST-4
    "#d7bd96",  # MST-5
    "#a07e56",  # MST-6
    "#825c43",  # MST-7
    "#604134",  # MST-8
    "#3a312a",  # MST-9
    "#292420",  # MST-10
]


def hex_to_rgb01(hex_code: str) -> np.ndarray:
    hex_code = hex_code.lstrip("#")
    r, g, b = (int(hex_code[i : i + 2], 16) for i in (0, 2, 4))
    return np.array([r, g, b], dtype=np.float64) / 255.0


def _build_reference_lab() -> np.ndarray:
    rgb01 = np.array([hex_to_rgb01(h) for h in MST_HEX]).reshape(1, 10, 3)
    return rgb2lab(rgb01).reshape(10, 3)


MST_REFERENCE_LAB = _build_reference_lab()


@dataclass
class MSTResult:
    mst_category: int  # 1-10
    confidence: float  # 0-1
    distances: dict  # {mst_index: ciede2000 distance}
    sample_lab: tuple  # (L, a, b)
    ita_degrees: float


def classify_lab(sample_lab: np.ndarray) -> MSTResult:
    sample_lab = np.asarray(sample_lab, dtype=np.float64).reshape(1, 3)

    distances = deltaE_ciede2000(np.repeat(sample_lab, 10, axis=0), MST_REFERENCE_LAB)

    order = np.argsort(distances)
    best_idx = order[0]
    second_idx = order[1]
    best_dist = distances[best_idx]
    second_dist = distances[second_idx]

    denom = second_dist + 1e-6
    confidence = float(np.clip(1.0 - (best_dist / denom), 0.0, 1.0))

    L, a, b = sample_lab[0]
    ita = float(np.degrees(np.arctan2(L - 50.0, b)))

    return MSTResult(
        mst_category=int(best_idx) + 1,
        confidence=confidence,
        distances={int(i) + 1: float(d) for i, d in enumerate(distances)},
        sample_lab=(float(L), float(a), float(b)),
        ita_degrees=ita,
    )
