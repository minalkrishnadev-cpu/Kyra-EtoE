"""
src/height/target_detector.py
=============================
Target detectors for coordinates extraction.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Any, Optional, Tuple


class PersonTopPoint:
    def __init__(self):
        pass

    def detect_top_point(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        from .landmarks import get_head_foot_pixels
        lm = get_head_foot_pixels(frame)
        return lm.get("head") if lm else None

    def detect_base_point(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        from .landmarks import get_head_foot_pixels
        lm = get_head_foot_pixels(frame)
        return lm.get("feet") if lm else None

    def close(self):
        pass


class ROIBoundingBoxDetector:
    def __init__(self):
        self._roi = None

    def select_roi_from_frame(self, frame: np.ndarray, window_name: str = "Select ROI") -> bool:
        roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)
        if roi[2] > 0 and roi[3] > 0:
            self._roi = roi
            return True
        return False

    def roi_pixel_height(self) -> int:
        return self._roi[3] if self._roi else 0

    def detect_top_point(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        if self._roi:
            x, y, w, h = self._roi
            return (int(x + w / 2), y)
        return None

    def detect_base_point(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        if self._roi:
            x, y, w, h = self._roi
            return (int(x + w / 2), y + h)
        return None

    def close(self):
        pass
