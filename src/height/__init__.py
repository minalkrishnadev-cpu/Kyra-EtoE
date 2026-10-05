# src/height package
#

from .height_stereo import detect_floor_plane, get_object_height
from .height_webcam import (
    get_object_height_webcam,
    load_calibration_scale,
    save_calibration_scale,
    load_calibration_data,
    save_calibration_data,
)
from .target_detector import PersonTopPoint, ROIBoundingBoxDetector

from .landmarks import get_head_foot_pixels, draw_landmarks_overlay
from .height_webcam import get_person_height_webcam
from .height_stereo import get_person_height_stereo, smooth_height

# Pipeline symbols are imported lazily via __getattr__ to avoid warnings when pipeline.py runs as __main__
_PIPELINE_EXPORTS = {"measure_height", "measure_person_height"}


def __getattr__(name: str):
    if name in _PIPELINE_EXPORTS:
        from . import pipeline as _pl
        return getattr(_pl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "detect_floor_plane",
    "get_object_height",
    "get_object_height_webcam",
    "load_calibration_scale",
    "save_calibration_scale",
    "load_calibration_data",
    "save_calibration_data",
    "PersonTopPoint",
    "ROIBoundingBoxDetector",
    "measure_height",
    "get_head_foot_pixels",
    "draw_landmarks_overlay",
    "get_person_height_webcam",
    "get_person_height_stereo",
    "smooth_height",
    "measure_person_height",
]
