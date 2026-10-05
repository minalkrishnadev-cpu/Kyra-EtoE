"""
src/height/height_webcam.py
============================
Single-webcam height estimation using a pre-calibrated scale factor.
Supports loading and saving calibration coefficients to a local JSON file.

Public API
----------
load_calibration_scale() -> float
save_calibration_scale(scale) -> None
get_object_height_webcam(top_pixel, base_pixel, pixels_per_cm) -> float
get_person_height_webcam(frame, user_height_cm=None, pose=None, manual_scale=None) -> dict
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_FALLBACK_HEIGHT_CM = 170.0

# Track the last logged source status to prevent repetitive console warning spam
_last_logged_source: Optional[str] = None


def _log_source_status(source: str, message: str, level: int = logging.INFO) -> None:
    """Log source transitions once at the requested level, and subsequently at DEBUG level."""
    global _last_logged_source
    if source != _last_logged_source:
        _last_logged_source = source
        logger.log(level, message)
    else:
        logger.debug(message)


def load_calibration_data() -> Dict[str, Any]:
    """Load calibration metadata dict from config/height_calibration.json."""
    config_path = os.path.join("config", "height_calibration.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "pixels_per_cm": float(data.get("pixels_per_cm", 12.0)),
                    "calibrated_base_y": data.get("calibrated_base_y")
                }
        except Exception as e:
            logger.warning(f"Failed to load height_calibration.json: {e}")
    return {"pixels_per_cm": 12.0, "calibrated_base_y": None}


def load_calibration_scale() -> float:
    """
    Load pre-calibrated scale factor (pixels/cm) from config/height_calibration.json.
    Returns 12.0 (default fallback) if the configuration file is not found.
    """
    return load_calibration_data()["pixels_per_cm"]


def save_calibration_data(scale: float, base_y: Optional[int] = None) -> None:
    """Save scale factor and calibrated base Y pixel coordinate to config/height_calibration.json."""
    try:
        os.makedirs("config", exist_ok=True)
        config_path = os.path.join("config", "height_calibration.json")
        data = {
            "pixels_per_cm": round(scale, 4),
            "calibrated_base_y": base_y,
            "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Height calibration data saved successfully to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save height calibration data: {e}")


def save_calibration_scale(scale: float) -> None:
    """Legacy wrapper to save scale factor only."""
    save_calibration_data(scale, None)


def get_object_height_webcam(
    top_pixel: Tuple[int, int],
    base_pixel: Tuple[int, int],
    pixels_per_cm: float,
) -> float:
    """Compute the height of an object from pixel coordinates and scale factor."""
    if pixels_per_cm <= 0:
        raise ValueError(f"pixels_per_cm must be positive, got {pixels_per_cm}")
    pixel_height = abs(top_pixel[1] - base_pixel[1])
    return float(pixel_height / pixels_per_cm)


def get_person_height_webcam(
    frame: np.ndarray,
    user_height_cm: Optional[float] = None,
    pose: Any = None,
    manual_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Measure a person's height from a single webcam frame using a pre-calibrated scale.

    Resolution order:
    1. **Landmarks detected in frame**: pixel distance (head top → feet ground point)
       divided by the pre-calibrated scale factor (manual_scale, config file value, or default 12.0)
       → ``source="calibrated_scale"`` (or ``"default_scale"``).
    2. **User-provided height** (``user_height_cm`` is not None): returned
       as-is with no pixel measurement → ``source="user_input"``.
    3. **Neither available**: returns 170.0 cm with a logged warning →
       ``source="default"``.
    """
    from src.height.landmarks import get_head_foot_pixels

    result: Dict[str, Any] = {
        "height_cm": None,
        "source": None,
        "confidence": "low",
        "head_pixel": None,
        "feet_pixel": None,
        "pixels_per_cm": None,
    }

    # Landmark detection
    lm = get_head_foot_pixels(frame, pose=pose)
    if lm:
        result["head_pixel"] = lm.get("head")
        result["feet_pixel"] = lm.get("feet")

    # Resolve scale factor (supplied manual_scale, configuration file scale, or default)
    if manual_scale is not None:
        scale = manual_scale
        is_custom = True
    else:
        scale = load_calibration_scale()
        # If it matches default 12.0, mark as default_scale unless loaded otherwise
        is_custom = os.path.exists(os.path.join("config", "height_calibration.json"))
        
    result["pixels_per_cm"] = round(scale, 2)

    head = result["head_pixel"]
    feet = result["feet_pixel"]

    # Landmark path
    if head is not None and feet is not None:
        pixel_span = abs(head[1] - feet[1])
        # Add 1.0 cm as a residual projection/hair correction factor
        height_cm = (pixel_span / scale) + 1.0
        src = "calibrated_scale" if is_custom else "default_scale"
        result.update({
            "height_cm": round(height_cm, 2),
            "source": src,
            "confidence": (
                "high"   if pixel_span > 0.4 * frame.shape[0] else
                "medium" if pixel_span > 0.2 * frame.shape[0] else
                "low"
            ),
        })
        _log_source_status(src, f"Person height ({src}): {height_cm:.1f} cm [{pixel_span} px / {scale:.2f} px/cm]")
        return result
    else:
        _log_source_status("missing_landmarks", f"Landmarks missing — head={head} feet={feet}. Trying fallback.", level=logging.WARNING)

    # User height path
    if user_height_cm is not None and user_height_cm > 0:
        result.update({
            "height_cm": float(user_height_cm),
            "source": "user_input",
            "confidence": "medium",
        })
        _log_source_status("user_input", f"Person height (user_input): {user_height_cm:.1f} cm")
        return result

    # Default path
    result.update({
        "height_cm": _DEFAULT_FALLBACK_HEIGHT_CM,
        "source": "default",
        "confidence": "low",
    })

    _log_source_status("default", f"Landmarks missing. Returning default height {_DEFAULT_FALLBACK_HEIGHT_CM:.1f} cm.", level=logging.WARNING)
    return result
 
 

if __name__ == "__main__":
    print("height_webcam module loaded successfully.")
    
    # Sanity check on pure-math helper
    height = get_object_height_webcam(
        top_pixel=(320, 50),
        base_pixel=(320, 347),
        pixels_per_cm=10.0,
    )
    print(f"  Mock object height (297 px / 10 px/cm) = {height:.1f} cm  [expected 29.7]")
