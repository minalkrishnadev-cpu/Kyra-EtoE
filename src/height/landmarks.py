"""
src/height/landmarks.py
=======================
Extracts body keypoints (head top and averaged ankle ground point) using MediaPipe Pose.
Includes overlay rendering helper functions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

# Cached MediaPipe Pose singleton
_pose_instance = None


def _get_pose_detector() -> Any:
    """Lazy-loads and caches the MediaPipe Pose detector instance."""
    global _pose_instance
    if _pose_instance is None:
        import mediapipe as mp
        _pose_instance = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return _pose_instance


def get_head_foot_pixels(
    frame: np.ndarray,
    pose: Optional[Any] = None
) -> Optional[Dict[str, Tuple[int, int]]]:
    """
    Detect head top and feet ground point using MediaPipe Pose solution.
    
    * Head Top: Nose shifted up vertically by 5% of the frame height.
    * Feet: Mean coordinates of left and right ankle landmarks.
    """
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    detector = pose if pose is not None else _get_pose_detector()

    # Process frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = detector.process(rgb)

    if not res.pose_landmarks:
        return None

    landmarks = res.pose_landmarks.landmark
    nose = landmarks[0]
    left_ankle = landmarks[27]
    right_ankle = landmarks[28]
    left_heel = landmarks[29]
    right_heel = landmarks[30]
    left_toe = landmarks[31]
    right_toe = landmarks[32]

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    # Verify visibility confidences
    threshold = 0.5
    if (nose.visibility < threshold or 
        left_ankle.visibility < threshold or 
        right_ankle.visibility < threshold or
        left_shoulder.visibility < threshold or
        right_shoulder.visibility < threshold):
        return None

    # Compute head top coordinate dynamically based on nose-to-shoulder distance
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
    nose_to_shoulder_px = abs(nose.y - shoulder_y) * h
    head_shift_px = 0.50 * nose_to_shoulder_px

    head_x = int(nose.x * w)
    head_y = int(nose.y * h - head_shift_px)
    head_x = max(0, min(w - 1, head_x))
    head_y = max(0, min(h - 1, head_y))

    # Compute feet ground coordinate (lowest point among ankles, heels, and toes)
    foot_x = int(((left_ankle.x + right_ankle.x) / 2.0) * w)
    max_y = max(
        left_ankle.y, right_ankle.y,
        left_heel.y, right_heel.y,
        left_toe.y, right_toe.y
    )
    foot_y = int(max_y * h)
    foot_x = max(0, min(w - 1, foot_x))
    foot_y = max(0, min(h - 1, foot_y))

    return {
        "head": (head_x, head_y),
        "feet": (foot_x, foot_y)
    }


def draw_landmarks_overlay(
    frame: np.ndarray,
    landmarks: Dict[str, Tuple[int, int]]
) -> np.ndarray:
    """Draw annotations and lines between detected head and feet coordinates."""
    display = frame.copy()
    
    head = landmarks.get("head")
    feet = landmarks.get("feet")

    if head:
        cv2.circle(display, head, 8, (0, 255, 255), -1)
        cv2.putText(display, "HEAD", (head[0] + 12, head[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    if feet:
        cv2.circle(display, feet, 8, (0, 255, 255), -1)
        cv2.putText(display, "FEET", (feet[0] + 12, feet[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    if head and feet:
        cv2.line(display, head, feet, (0, 255, 255), 2)

    return display
