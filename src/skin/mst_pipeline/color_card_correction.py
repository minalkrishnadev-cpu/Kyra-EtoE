# ColorChecker card detection + color correction (needs opencv-contrib-python)
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ColorCardResult:
    found: bool
    corrected_image_bgr: np.ndarray | None  # uint8, same shape as input
    fit_loss: float | None  # lower is better


def detect_and_correct(
    image_bgr: np.ndarray,
    chart_type: int = cv2.mcc.MCC24,
) -> ColorCardResult:
    detector = cv2.mcc.CCheckerDetector_create()
    found = detector.process(image_bgr, chart_type)
    if not found:
        return ColorCardResult(found=False, corrected_image_bgr=None, fit_loss=None)

    checker = detector.getBestColorChecker()
    charts_rgb = np.asarray(checker.getChartsRGB())

    # 24 patches x 3 channels, reshaped to (24, 1, 3) per OpenCV's own example
    src = charts_rgb[:, 1].reshape(24, 1, 3).astype(np.float64) / 255.0

    model = cv2.ccm.ColorCorrectionModel(src, cv2.ccm.COLORCHECKER_MACBETH)
    model.run()
    fit_loss = float(model.getLoss())

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    corrected_rgb = model.infer(img_rgb)
    corrected_rgb = np.clip(corrected_rgb, 0.0, 1.0)
    corrected_bgr = cv2.cvtColor((corrected_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    return ColorCardResult(found=True, corrected_image_bgr=corrected_bgr, fit_loss=fit_loss)
