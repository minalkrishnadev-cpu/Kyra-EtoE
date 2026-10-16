# top-level pipeline: image -> MST category
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .color_card_correction import detect_and_correct
from .color_normalization import gray_world_white_balance, sample_skin_lab_regions
from .face_crop import crop_and_upscale
from .mst_reference import MSTResult, classify_lab
from .skin_extraction import get_extractor


@dataclass
class PipelineResult:
    mst: MSTResult
    face_found: bool
    backend: str
    n_pixels_used: int
    rejected_fraction: float
    white_balanced: bool
    color_card_used: bool
    color_card_fit_loss: float | None
    regions_used: list
    regions_excluded: list
    # "face_crop" = found on the enlarged head crop, "full_frame" = found on the whole image
    detection_source: str = "full_frame"
    crop_scale: float = 1.0


class MSTPipeline:
    def __init__(
        self,
        prefer_mediapipe: bool = True,
        white_balance: bool = True,
        use_color_card: bool = False,
    ):
        # use_color_card: look for a physical ColorChecker in each photo;
        # falls back to gray_world if none is found (check result.color_card_used)
        self.extractor = get_extractor(prefer_mediapipe=prefer_mediapipe)
        self.white_balance = white_balance
        self.use_color_card = use_color_card

    def classify(self, image_path: str) -> PipelineResult:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.classify_array(image_bgr)

    def _extract_face(self, image_bgr: np.ndarray, face_hint):
        """
        Try the enlarged head crop first (if a head box is known), then the whole image.
        Returns (extraction, image_the_masks_belong_to, source_label, crop_scale).
        """
        if face_hint is not None:
            prepared = crop_and_upscale(image_bgr, face_hint)
            if prepared is not None:
                crop, scale = prepared
                extraction = self.extractor.extract(crop)
                if extraction.face_found:
                    return extraction, crop, "face_crop", scale

        extraction = self.extractor.extract(image_bgr)
        return extraction, image_bgr, "full_frame", 1.0

    def classify_array(self, image_bgr: np.ndarray, face_hint=None) -> PipelineResult:
        """
        face_hint: optional (x0, y0, x1, y1) box around the head in image_bgr
        (e.g. from pose landmarks). When given, face detection runs on that
        region enlarged first, which finds small faces far more reliably.
        """
        color_card_used = False
        color_card_fit_loss = None

        if self.use_color_card:
            card_result = detect_and_correct(image_bgr)
            if card_result.found:
                image_bgr = card_result.corrected_image_bgr
                color_card_used = True
                color_card_fit_loss = card_result.fit_loss

        if self.white_balance and not color_card_used:
            image_bgr = gray_world_white_balance(image_bgr)

        extraction, work_image, source, crop_scale = self._extract_face(image_bgr, face_hint)
        if not extraction.face_found:
            raise RuntimeError(
                "No face detected. Ensure the image contains a clear, "
                "front-facing view of a person's face."
            )

        sample = sample_skin_lab_regions(work_image, extraction.region_masks)
        mst_result = classify_lab(sample.lab_median)

        return PipelineResult(
            mst=mst_result,
            face_found=extraction.face_found,
            backend=extraction.backend,
            n_pixels_used=sample.n_pixels_used,
            rejected_fraction=sample.rejected_fraction,
            white_balanced=self.white_balance and not color_card_used,
            color_card_used=color_card_used,
            color_card_fit_loss=color_card_fit_loss,
            regions_used=sample.regions_used,
            regions_excluded=sample.regions_excluded,
            detection_source=source,
            crop_scale=crop_scale,
        )
