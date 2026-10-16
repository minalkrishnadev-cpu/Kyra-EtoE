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


def _head_box(landmarks, w: int, h: int, vis_threshold: float = 0.5):
    """
    Square box around the head (x0, y0, x1, y1) from the MediaPipe Pose face points
    (0 nose, 1-6 eyes, 7/8 ears, 9/10 mouth). Returns None if the face is not visible enough.
    Used as a hint so face detection can run on an enlarged head crop.
    """
    nose, l_eye, r_eye = landmarks[0], landmarks[2], landmarks[5]
    if min(nose.visibility, l_eye.visibility, r_eye.visibility) < vis_threshold:
        return None

    eye_y = (l_eye.y + r_eye.y) / 2.0 * h
    cx = (l_eye.x + r_eye.x) / 2.0 * w

    # face width: ear-to-ear if both ears are visible, else estimated from the outer eye corners
    face_w = 1.6 * abs(landmarks[3].x - landmarks[6].x) * w
    l_ear, r_ear = landmarks[7], landmarks[8]
    if min(l_ear.visibility, r_ear.visibility) >= vis_threshold:
        face_w = max(face_w, abs(l_ear.x - r_ear.x) * w)
    if face_w < 8:
        return None

    side = 2.4 * face_w          # head + hair + a margin of context
    half = side / 2.0
    box = (cx - half, eye_y - half, cx + half, eye_y + half)
    x0, y0, x1, y1 = box
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def get_head_foot_pixels(
    frame: np.ndarray,
    pose: Optional[Any] = None
) -> Optional[Dict[str, Tuple[int, int]]]:
    """
    Detect head top and feet ground point using MediaPipe Pose solution.
    
    * Head Top: Nose shifted up vertically by 5% of the frame height.
    * Feet: Mean coordinates of left and right ankle landmarks.
    * head_box (optional key): square box around the head, used to enlarge the face
      for skin-tone detection.
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

    result = {
        "head": (head_x, head_y),
        "feet": (foot_x, foot_y)
    }
    head_box = _head_box(landmarks, w, h)
    if head_box is not None:
        result["head_box"] = head_box
    return result


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

    box = landmarks.get("head_box")
    if box:
        cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (255, 200, 0), 1)
        cv2.putText(display, "FACE ROI", (box[0], max(12, box[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)

    return display
