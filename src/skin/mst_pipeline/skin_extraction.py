# Face detection + forehead/cheek ROI extraction (MediaPipe or Haar fallback)
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "face_landmarker.task"
)

# landmark indices into MediaPipe's 468-point face mesh
FOREHEAD_IDX = [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109]
LEFT_CHEEK_IDX = [50, 187, 205, 36, 142, 126, 209, 131, 134]
RIGHT_CHEEK_IDX = [280, 411, 425, 266, 371, 355, 429, 360, 363]


def download_model(dest_path: str = DEFAULT_MODEL_PATH) -> str:
    # one-time model download, needs internet
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, dest_path)
    return dest_path


@dataclass
class ExtractionResult:
    mask: np.ndarray  # bool (H, W), union of all regions
    face_found: bool
    backend: str
    region_masks: dict  # {"forehead": mask, "left_cheek": mask, "right_cheek": mask}


class HaarSkinExtractor:
    # offline fallback, no model download needed
    def __init__(self):
        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml"
        )
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade from {cascade_path}. "
                "Try: pip install \"opencv-python<5\" (or opencv-contrib-python<5)"
            )

    def extract(self, image_bgr: np.ndarray) -> ExtractionResult:
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
        )
        if len(faces) == 0:
            empty = np.zeros((h, w), dtype=bool)
            return ExtractionResult(empty, False, "haar", {"forehead": empty, "left_cheek": empty, "right_cheek": empty})

        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])  # largest face

        forehead_mask = np.zeros((h, w), dtype=bool)
        left_cheek_mask = np.zeros((h, w), dtype=bool)
        right_cheek_mask = np.zeros((h, w), dtype=bool)

        # forehead band, biased toward brow to avoid the hairline
        fh_y0 = fy + int(0.16 * fh)
        fh_y1 = fy + int(0.27 * fh)
        fh_x0 = fx + int(0.28 * fw)
        fh_x1 = fx + int(0.72 * fw)
        forehead_mask[fh_y0:fh_y1, fh_x0:fh_x1] = True

        # cheeks: mid-height, either side of the nose
        ch_y0 = fy + int(0.55 * fh)
        ch_y1 = fy + int(0.75 * fh)
        left_x0 = fx + int(0.08 * fw)
        left_x1 = fx + int(0.32 * fw)
        right_x0 = fx + int(0.68 * fw)
        right_x1 = fx + int(0.92 * fw)
        left_cheek_mask[ch_y0:ch_y1, left_x0:left_x1] = True
        right_cheek_mask[ch_y0:ch_y1, right_x0:right_x1] = True

        region_masks = {
            "forehead": forehead_mask,
            "left_cheek": left_cheek_mask,
            "right_cheek": right_cheek_mask,
        }
        union = forehead_mask | left_cheek_mask | right_cheek_mask
        return ExtractionResult(union, True, "haar", region_masks)


class MediaPipeSkinExtractor:
    # preferred backend, needs download_model() run once first
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found at {model_path}. "
                "Call skin_extraction.download_model() once (requires internet)."
            )

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        self._mp = mp
        self._vision = vision
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def extract(self, image_bgr: np.ndarray) -> ExtractionResult:
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image_rgb)
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            empty = np.zeros((h, w), dtype=bool)
            return ExtractionResult(empty, False, "mediapipe", {"forehead": empty, "left_cheek": empty, "right_cheek": empty})

        landmarks = result.face_landmarks[0]
        pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks])

        region_masks = {}
        for name, idx_group in (
            ("forehead", FOREHEAD_IDX),
            ("left_cheek", LEFT_CHEEK_IDX),
            ("right_cheek", RIGHT_CHEEK_IDX),
        ):
            m = np.zeros((h, w), dtype=np.uint8)
            poly = pts[idx_group].astype(np.int32)
            cv2.fillPoly(m, [poly], 1)
            region_masks[name] = m.astype(bool)

        union = region_masks["forehead"] | region_masks["left_cheek"] | region_masks["right_cheek"]
        return ExtractionResult(union, True, "mediapipe", region_masks)


def get_extractor(prefer_mediapipe: bool = True):
    # tries MediaPipe first, falls back to Haar if the model isn't downloaded
    if prefer_mediapipe:
        try:
            return MediaPipeSkinExtractor()
        except FileNotFoundError:
            pass
    return HaarSkinExtractor()
