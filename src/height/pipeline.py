"""
src/height/pipeline.py
=======================
Universal height measurement pipeline — orchestrates detectors and geometry solvers.
Supports local calibration from reference object and loading scale factor automatically.
"""

from __future__ import annotations

import os
import sys

# Ensure path includes parent folder for proper package imports when executed directly
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import threading
import argparse
import time
from typing import Any, Dict, Literal, Optional, Tuple

import cv2
import numpy as np


def measure_person_height(
    frame: np.ndarray,
    camera_mode: str,
    point_cloud: Optional[np.ndarray] = None,
    marker_physical_size_cm: float = 10.0,
    user_height_cm: Optional[float] = None,
    manual_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Unified person height measurement entry point.
    """
    from src.height.landmarks import get_head_foot_pixels

    if camera_mode not in ("webcam", "stereo"):
        raise ValueError(f"camera_mode must be 'webcam' or 'stereo', got {camera_mode!r}")

    lm = get_head_foot_pixels(frame) or {}
    head_pixel = lm.get("head")
    feet_pixel = lm.get("feet")

    base = {
        "height_cm": None,
        "source": None,
        "camera_mode": camera_mode,
        "confidence": "low",
        "head_pixel": head_pixel,
        "feet_pixel": feet_pixel,
    }

    if camera_mode == "webcam":
        from src.height.height_webcam import get_person_height_webcam
        result = get_person_height_webcam(
            frame=frame,
            user_height_cm=user_height_cm,
            manual_scale=manual_scale,
        )
        base.update(result)
        base["camera_mode"] = "webcam"
        return base

    # Stereo mode
    if point_cloud is None:
        raise ValueError("point_cloud must be supplied for camera_mode='stereo'.")

    from src.height.height_stereo import detect_floor_plane, get_person_height_stereo
    floor_plane = detect_floor_plane(point_cloud)

    if head_pixel is None:
        import logging as _log
        _log.getLogger(__name__).warning("measure_person_height (stereo): head landmark not detected.")
        base["source"] = "stereo"
        return base

    result = get_person_height_stereo(
        point_cloud=point_cloud,
        head_pixel=head_pixel,
        feet_pixel=feet_pixel,
        floor_plane=floor_plane,
    )
    base.update(result)
    base["camera_mode"] = "stereo"
    return base


def measure_height(
    frame: np.ndarray,
    target_type: Literal["person", "object"],
    camera_mode: Literal["webcam", "stereo"],
    point_cloud: Optional[np.ndarray] = None,
    pixels_per_cm: Optional[float] = None,
    floor_plane: Optional[np.ndarray] = None,
    detector: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Legacy generic measurement resolver.
    """
    if target_type == "person":
        return measure_person_height(
            frame=frame,
            camera_mode=camera_mode,
            point_cloud=point_cloud,
            manual_scale=pixels_per_cm
        )

    # Generic object measurement
    if detector is None:
        from src.height.target_detector import ROIBoundingBoxDetector
        detector = ROIBoundingBoxDetector()

    top = detector.detect_top_point(frame)
    base = detector.detect_base_point(frame)

    result = {
        "height_cm": None,
        "target_type": "object",
        "camera_mode": camera_mode,
        "confidence": "low",
        "source": "webcam",
    }

    if top is None or base is None:
        return result

    if camera_mode == "webcam":
        if pixels_per_cm is None or pixels_per_cm <= 0:
            from src.height.height_webcam import load_calibration_scale
            pixels_per_cm = load_calibration_scale()
        h = abs(top[1] - base[1]) / pixels_per_cm
        result.update({
            "height_cm": round(h, 2),
            "confidence": "high",
            "source": f"manual_box (scale: {pixels_per_cm:.2f} px/cm)"
        })
        return result

    # Stereo mode
    if point_cloud is None:
        raise ValueError("point_cloud must be supplied for stereo mode object measurement.")
    
    from src.height.height_stereo import get_object_height
    h = get_object_height(point_cloud, top, base, floor_plane)
    result.update({
        "height_cm": round(h, 2),
        "confidence": "high" if h > 0 else "low",
        "source": "stereo_depth"
    })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI Helpers
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Camera mounting orientation
# ─────────────────────────────────────────────────────────────────────────────
# The camera is mounted sideways, so every raw frame must be rotated before it
# is shown OR measured. Rotation is applied once, at capture time, inside
# _RotatedCapture.read(). Everything downstream (preview, freeze frame, ROI
# drawing, landmarks, height maths, calibrated_base_y) therefore works purely
# in the rotated (upright) coordinate system.

ROTATION_CHOICES = {
    "none": None,
    "ccw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "cw90": cv2.ROTATE_90_CLOCKWISE,
    "180": cv2.ROTATE_180,
}
DEFAULT_ROTATION = "ccw90"


def rotate_frame(frame: np.ndarray, rotation: str = DEFAULT_ROTATION) -> np.ndarray:
    """Rotate a raw camera frame into the upright viewing orientation."""
    code = ROTATION_CHOICES.get(rotation)
    if code is None or frame is None:
        return frame
    return cv2.rotate(frame, code)


class _RotatedCapture:
    """
    Thin wrapper around cv2.VideoCapture that rotates every frame on read().
    All other attributes/methods (isOpened, release, get, set, ...) pass through.
    """

    def __init__(self, cap: cv2.VideoCapture, rotation: str = DEFAULT_ROTATION):
        self._cap = cap
        self.rotation = rotation

    def read(self):
        ok, frame = self._cap.read()
        if ok and frame is not None:
            frame = rotate_frame(frame, self.rotation)
        return ok, frame

    def __getattr__(self, name):
        return getattr(self._cap, name)


def _open_camera(idx: int, rotation: str = "none"):
    """Open a camera. Pass rotation='ccw90' (etc.) to get upright, rotated frames."""
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    if rotation and rotation != "none":
        return _RotatedCapture(cap, rotation)
    return cap


def _cap_rotation(cap) -> str:
    return getattr(cap, "rotation", "none")


def _check_calibration_orientation(calib_data: Dict[str, Any], rotation: str, frame_shape=None) -> None:
    """Warn loudly if the saved calibration was made in a different orientation / resolution."""
    saved_rot = calib_data.get("rotation")
    if saved_rot is None:
        print("[WARN] height_calibration.json has no 'rotation' field — it was made with the OLD "
              "(unrotated) camera view.\n"
              "       Re-run:  python pipeline.py --mode webcam --target object --ref-object-cm 29.7")
    elif saved_rot != rotation:
        print(f"[WARN] Calibration was saved with rotation='{saved_rot}' but camera is now "
              f"rotation='{rotation}'. Measurements will be wrong — re-run calibration.")
    saved_size = calib_data.get("frame_size")
    if frame_shape is not None and saved_size:
        h, w = frame_shape[:2]
        if [w, h] != list(saved_size):
            print(f"[WARN] Calibration frame size {saved_size[0]}x{saved_size[1]} differs from current "
                  f"{w}x{h}. Re-run calibration for accurate results.")


def _run_calibration(cap: cv2.VideoCapture, ref_cm: float) -> Optional[float]:
    """
    Freeze a camera frame and prompt drawing a box around a vertical reference object
    placed at the standing spot. Saves the scale to config/height_calibration.json.
    """
    print(f"\n=========================================")
    print(f"  SINGLE WEBCAM SPOT CALIBRATION")
    print(f"=========================================")
    print(f"  1. Place an object of KNOWN physical height ({ref_cm} cm)")
    print(f"     vertically exactly on the floor standing spot.")
    print(f"  2. Align the camera preview so the spot is clear.")
    print(f"  3. Press SPACE to freeze the camera frame.")
    print(f"  4. Draw a box from very top to very bottom of the reference object.")
    print(f"  5. Press ENTER to confirm, or ESC to abort.")
    print(f"  (View rotation: {_cap_rotation(cap)} — calibration is saved in this rotated view)")
    print(f"=========================================\n")

    # Camera loop
    cv2.namedWindow("Calibration Frame Capture", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Camera capture failed during calibration.")
            return None
        display = frame.copy()
        cv2.putText(display, f"CALIBRATION: Put {ref_cm}cm object on spot. SPACE=Freeze", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.imshow("Calibration Frame Capture", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow("Calibration Frame Capture")
            print("[INFO] Calibration aborted by user.")
            return None
        if key == 32:  # SPACE
            frozen_frame = frame.copy()
            cv2.destroyWindow("Calibration Frame Capture")
            break

    # ROI selector
    roi_win = f"Draw box around the {ref_cm} cm reference object"
    ref_roi = cv2.selectROI(roi_win, frozen_frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(roi_win)
    
    rx, ry, rw, rh = ref_roi
    if rh == 0:
        print("[WARN] Calibration cancelled. No bounding box drawn.")
        return None

    # frozen_frame is already rotated, so rh / ry are in the rotated (upright) coordinates
    pixels_per_cm = rh / ref_cm
    base_y = ry + rh
    fh, fw = frozen_frame.shape[:2]
    from src.height.height_webcam import save_calibration_data
    save_calibration_data(pixels_per_cm, base_y, rotation=_cap_rotation(cap), frame_size=(fw, fh))
    
    print(f"\n[SUCCESS] Calibration saved!")
    print(f"          Reference Height : {ref_cm} cm")
    print(f"          Pixel Height     : {rh} px")
    print(f"          Base Y Pixel     : {base_y} px")
    print(f"          Rotation         : {_cap_rotation(cap)}")
    print(f"          Frame Size (WxH) : {fw} x {fh}")
    print(f"          Scale Factor     : {pixels_per_cm:.4f} px/cm")
    print(f"          Saved to         : config/height_calibration.json\n")
    return pixels_per_cm


def _run_person_calibration(cap, ref_person_cm: float) -> Optional[float]:
    """Runs interactive calibration using a person of known physical height standing on the spot."""
    from src.height.landmarks import get_head_foot_pixels, draw_landmarks_overlay
    from src.height.height_webcam import save_calibration_data

    print(f"\n=========================================")
    print(f"  SINGLE WEBCAM PERSON CALIBRATION ({ref_person_cm} cm)")
    print(f"=========================================")
    print(f"  1. Stand straight on the marked floor spot.")
    print(f"  2. Make sure your full body is visible in the frame.")
    print(f"  3. Press SPACE to capture and calibrate.")
    print(f"  4. Press ESC or Q to cancel.")
    print(f"=========================================\n")

    cv2.namedWindow("Person Calibration", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Camera capture failed during calibration.")
            return None
        
        display = frame.copy()
        lm = get_head_foot_pixels(frame)
        if lm:
            display = draw_landmarks_overlay(display, lm)
            cv2.putText(display, f"SPACE: Calibrate to {ref_person_cm} cm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(display, "NO PERSON DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

        cv2.imshow("Person Calibration", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow("Person Calibration")
            print("[INFO] Calibration aborted.")
            return None
        
        if key == 32:  # SPACE
            if not lm:
                print("[WARN] Cannot calibrate: no person detected. Keep standing on the spot.")
                continue
            
            head = lm["head"]
            feet = lm["feet"]
            pixel_span = abs(head[1] - feet[1])
            
            # Since our webcam formula is: height_cm = (pixel_span / scale) + 1.0
            # To get exactly the known height, the scale must be:
            # scale = pixel_span / (ref_person_cm - 1.0)
            target_cm = ref_person_cm - 1.0
            if target_cm <= 0:
                print("[ERROR] Invalid reference height.")
                cv2.destroyWindow("Person Calibration")
                return None
                
            pixels_per_cm = pixel_span / target_cm
            fh, fw = frame.shape[:2]
            save_calibration_data(pixels_per_cm, base_y=feet[1],
                                  rotation=_cap_rotation(cap), frame_size=(fw, fh))
            
            cv2.destroyWindow("Person Calibration")
            print(f"\n[SUCCESS] Calibration saved successfully!")
            print(f"          Target Height : {ref_person_cm} cm")
            print(f"          Pixel Span    : {pixel_span} px")
            print(f"          Base Y Pixel  : {feet[1]} px")
            print(f"          Scale Factor  : {pixels_per_cm:.4f} px/cm")
            print(f"          Saved to      : config/height_calibration.json\n")
            return pixels_per_cm

def _make_fit_client(args):
    """Start the body-fit worker (separate Python env) if an HMR python is configured."""
    if getattr(args, "no_fit", False):
        return None
    hmr_python = getattr(args, "hmr_python", None) or os.environ.get("HMR_PYTHON")
    if not hmr_python:
        print("[INFO] Body fit disabled. To enable it pass --hmr-python <path-to-hmr-env-python.exe> "
              "or set the HMR_PYTHON environment variable.")
        return None
    from src.height.fit_client import FitClient
    fit_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fit")
    client = FitClient(hmr_python, fit_dir)
    client.start()
    print(f"[INFO] Body-fit worker starting in the background (log: {client.log_path})")
    return client


def _put_text(img, text, org, scale, colour, thick=1):
    """Text with a dark outline so it stays readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def _draw_fit_panel(display, fit, y0: int = 110) -> None:
    state, msg, result = fit.snapshot()
    if state == "starting":
        head, colour = "FIT: loading models...", (0, 255, 255)
    elif state == "running":
        head, colour = "FIT: running (can take a few seconds)...", (0, 255, 255)
    elif state == "error":
        head, colour = "FIT: unavailable", (0, 0, 255)
    else:
        head, colour = "FIT: ready - press SPACE / ENTER", (0, 255, 0)
    _put_text(display, head, (10, y0), 0.6, colour, 2)
    y = y0 + 24
    if msg:
        _put_text(display, msg[:60], (10, y), 0.5, (0, 165, 255))
        y += 22
    if result:
        _put_text(display, f"Fit @ {result['height_cm']:.1f} cm input:", (10, y), 0.5, (255, 255, 255))
        y += 22
        for name, value in result["measurements"].items():
            _put_text(display, f"{name}: {value:.1f}", (10, y), 0.5, (255, 255, 255))
            y += 20


from src.skin.mst_pipeline.pipeline import MSTPipeline
def _cli_person_webcam(args) -> None:
    """Live-view CLI for single-webcam person height measurement using pre-calibrated scale."""
    import collections
    import threading
    from src.height.landmarks import get_head_foot_pixels, draw_landmarks_overlay
    from src.height.height_webcam import get_person_height_webcam, load_calibration_data

    cam_idx = getattr(args, "left_cam", 0)
    rotation = getattr(args, "rotate", DEFAULT_ROTATION)
    cap = _open_camera(cam_idx, rotation=rotation)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {cam_idx}.")
        return

    # Calibration values were saved in the rotated coordinate system,
    # and cap.read() returns rotated frames, so they line up directly.
    calib_data = load_calibration_data()
    scale = calib_data["pixels_per_cm"]
    calibrated_base_y = calib_data["calibrated_base_y"]
    ok_probe, probe = cap.read()
    _check_calibration_orientation(calib_data, rotation, probe.shape if ok_probe else None)

    fit = _make_fit_client(args)

    manual_scale = getattr(args, "manual_scale", None)
    if manual_scale is not None:
        scale = manual_scale
        scale_label = f"MANUAL ({scale:.2f} px/cm)"
    else:
        scale_label = f"CALIBRATED ({scale:.2f} px/cm)"
    mst_pipeline = MSTPipeline(
        prefer_mediapipe=not getattr(args, "no_mediapipe", False),
        white_balance=False,
        use_color_card=getattr(args, "use_color_card", False),
    )
    saved_mst_result = None
    mst_status_str = "Scanning face..."
    mst_in_progress = False
    def print_mst_summary(result) -> None:
        print(f"Backend used:       {result.backend}")
        print(f"Face found via:     {getattr(result, 'detection_source', 'full_frame')}"
              f" (crop x{getattr(result, 'crop_scale', 1.0):.1f})")
        print(f"MST category:       {result.mst.mst_category}  (1=lightest, 10=darkest)")
        print(f"Confidence:         {result.mst.confidence:.2f}  (separation from runner-up category)")
        print(f"Sample Lab (L,a,b): {tuple(round(x, 2) for x in result.mst.sample_lab)}")
        print(f"ITA (degrees):      {result.mst.ita_degrees:.1f}")
        print(f"Skin pixels used:   {result.n_pixels_used}  (rejected {result.rejected_fraction:.0%} as over/under-exposed)")
        print(f"Face regions used:  {', '.join(result.regions_used)}")
        if getattr(result, "regions_excluded", None):
            print(f"Regions EXCLUDED:   {', '.join(result.regions_excluded)}")
            
        print()
        print("Distance to each MST category (lower = closer):")
        for cat, dist in sorted(result.mst.distances.items()):
            marker = "  <-- best match" if cat == result.mst.mst_category else ""
            print(f"  MST-{cat}: {dist:6.2f}{marker}")
        print()
    def run_mst_async(frame_bgr: np.ndarray, face_hint=None):
        nonlocal saved_mst_result, mst_status_str, mst_in_progress
        try:
            res = mst_pipeline.classify_array(frame_bgr, face_hint=face_hint)
            saved_mst_result = res
            mst_status_str = f"MST-{res.mst.mst_category}"
            print_mst_summary(res)
        except Exception as e:
            mst_status_str = "Scanning face (No face detected)..."
        finally:
            mst_in_progress = False
    print(f"[INFO] Person height live view | Marker-free mode | Scale: {scale_label}")
    print("       Press Q or ESC to quit.")

    history = collections.deque(maxlen=10)

    cv2.namedWindow("KYRA Person Height (webcam)", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lm = get_head_foot_pixels(frame)
        if not mst_in_progress:
            mst_in_progress=True
            head_box = lm.get("head_box") if lm else None
            t=threading.Thread(target=run_mst_async, args=(frame.copy(), head_box),daemon=True)
            t.start()

        display = frame.copy()
        cv2.putText(display, f"Scale: {scale_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        flag=False
        
        # Position helper logic
        if lm:
            display = draw_landmarks_overlay(display, lm)
            feet = lm.get("feet")
            if feet and calibrated_base_y is not None:
                diff = feet[1] - calibrated_base_y
                # If feet coordinate is too large (too low on screen), they are too close
                if diff > 35:
                    pos_txt = "POSITION: STEP BACK"
                    pos_color = (0, 140, 255) # Orange/Red
                # If feet coordinate is too small (too high on screen), they are too far
                elif diff < -35:
                    pos_txt = "POSITION: STEP FORWARD"
                    pos_color = (0, 140, 255) # Orange/Red
                else:
                    pos_txt = "POSITION: OK"
                    pos_color = (0, 255, 0) # Green
                    flag=True
            else:
                pos_txt = "POSITION: OK (Uncalibrated spot)"
                pos_color = (0, 255, 255) # Yellow
        else:
            pos_txt = "POSITION: NO PERSON DETECTED"
            pos_color = (0, 165, 255) # Orange
        cv2.putText(display, pos_txt, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, pos_color, 2, cv2.LINE_AA)

        result = get_person_height_webcam(frame, manual_scale=scale)
        h_cm = result.get("height_cm")
        src = result.get("source", "?")
        # if h_cm is not None and src in ("calibrated_scale", "default_scale"):
        if h_cm is not None and src == "calibrated_scale":
            history.append(h_cm)
        H_frame = display.shape[0]
        if history:
            median_h = float(np.median(list(history)))
            status_colour = (0, 255, 0) if src in ("calibrated_scale", "default_scale") else (0, 165, 255)
            height_txt = f"Height: {median_h:.1f} cm  [{src}]"
            if flag and h_cm is not None and src in ("calibrated_scale", "default_scale"):
                mst_category = saved_mst_result.mst.mst_category if saved_mst_result else "Scanning..."
                print(f"\r  when position is OKKK height_cm={median_h:.1f} | MST={mst_category} | source={src:<18s}", end="", flush=True)

        else:
            status_colour = (0, 165, 255) # Orange
            height_txt = "Height: Detecting..."

        cv2.putText(display, height_txt, (10, H_frame - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_colour, 2, cv2.LINE_AA)         
        # mst_display_str = f"Skin Tone: {mst_status_str}"
        # mst_color = (0, 255, 0) if "MST-" in mst_status_str else (0, 165, 255)
        # cv2.putText(display, mst_category, (10, H_frame - 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, mst_color, 2, cv2.LINE_AA)
        if saved_mst_result is not None:
            mst_display_str = f"Skin Tone: MST-{saved_mst_result.mst.mst_category}"
            mst_color = (0, 255, 0)  # Green when showing a valid cached result
        else:
            mst_display_str = f"Skin Tone: {mst_status_str}"
            mst_color = (0, 165, 255)  # Orange while initially scanning
        cv2.putText(display, mst_display_str, (10, H_frame - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, mst_color, 2, cv2.LINE_AA)
        if fit is not None:
            _draw_fit_panel(display, fit)
        cv2.imshow("KYRA Person Height (webcam)", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (13, 32) and fit is not None:  # ENTER / SPACE -> run body fit
            if not history:
                fit.set_note("Wait for a stable height reading first")
            else:
                fit_height = float(np.median(list(history)))
                if fit.request(frame, fit_height):
                    print(f"\n[FIT] Running body fit with height {fit_height:.1f} cm ...")
                else:
                    fit.set_note("Fit worker not ready / busy")

    print()
    if fit is not None:
        fit.close()
    cap.release()
    cv2.destroyAllWindows()


def _cli_webcam(args: argparse.Namespace) -> None:
    """Universal webcam CLI for measuring objects and persons marker-free."""
    from src.height.target_detector import PersonTopPoint, ROIBoundingBoxDetector
    from src.height.height_webcam import load_calibration_scale

    rotation = getattr(args, "rotate", DEFAULT_ROTATION)
    print(f"[INFO] Opening webcam index {args.left_cam} (rotation: {rotation})...")
    cap = _open_camera(args.left_cam, rotation=rotation)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        sys.exit(1)

    for _ in range(10):
        cap.read()

    # If --ref-object-cm was supplied, run calibration first and exit
    if getattr(args, "ref_object_cm", None):
        _run_calibration(cap, args.ref_object_cm)
        cap.release()
        cv2.destroyAllWindows()
        return

    pixels_per_cm = getattr(args, "manual_scale", None)
    if pixels_per_cm is None:
        pixels_per_cm = load_calibration_scale()
        scale_source = f"calibrated ({pixels_per_cm:.2f} px/cm)"
    else:
        scale_source = f"manual ({pixels_per_cm:.2f} px/cm)"

    detector = ROIBoundingBoxDetector() if args.target == "object" else PersonTopPoint()
    roi_locked = (args.target == "person")

    last_result = None

    cv2.namedWindow("KYRA Universal Height", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        display = frame.copy()
        cv2.putText(display, f"Scale: {scale_source}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

        top = detector.detect_top_point(frame)
        base = detector.detect_base_point(frame)

        if top:
            cv2.circle(display, top, 6, (0, 255, 255), -1)
        if base:
            cv2.circle(display, base, 6, (255, 100, 0), -1)

        H_frame = frame.shape[0]
        if last_result and last_result.get("height_cm"):
            cv2.putText(display, f"Height: {last_result['height_cm']:.1f} cm", (10, H_frame - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 128), 2)

        cv2.imshow("KYRA Universal Height", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == 32:  # SPACE
            if args.target == "object" and not roi_locked:
                ok2, frame2 = cap.read()
                if ok2:
                    if detector.select_roi_from_frame(frame2, "KYRA — Draw box"):
                        roi_locked = True
                continue

            last_result = measure_height(
                frame,
                target_type=args.target,
                camera_mode="webcam",
                pixels_per_cm=pixels_per_cm,
                detector=detector,
            )
            print(f"  ✓ Height: {last_result.get('height_cm')} cm")

    cap.release()
    cv2.destroyAllWindows()


def _cli_stereo(args: argparse.Namespace) -> None:
    try:
        from src.calibration import load_calibration, compute_depth_map
    except ImportError:
        print("[ERROR] Could not import stereo depth functions.")
        return

    calibration = load_calibration(args.calibration)
    # Stereo rigs keep their own calibration; no rotation applied here.
    cap_l = _open_camera(args.left_cam, rotation="none")
    cap_r = _open_camera(args.right_cam, rotation="none")

    if not cap_l.isOpened() or not cap_r.isOpened():
        print("[ERROR] Could not open stereo cameras.")
        return

    print("[INFO] Running stereo view — press SPACE to capture, Q to quit.")
    cv2.namedWindow("KYRA Stereo Height", cv2.WINDOW_NORMAL)
    while True:
        ok_l, left = cap_l.read()
        ok_r, right = cap_r.read()
        if not ok_l or not ok_r:
            break

        display = np.concatenate([left, right], axis=1)
        cv2.putText(display, "SPACE=Capture  Q=Quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("KYRA Stereo Height", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32:
            point_cloud = compute_depth_map(left, right, calibration)
            result = measure_person_height(left, camera_mode="stereo", point_cloud=point_cloud)
            print(f"  ✓ Height: {result.get('height_cm')} cm")

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KYRA Universal Height Module")
    p.add_argument("--target", choices=["person", "object"], default="object")
    p.add_argument("--mode", choices=["webcam", "stereo"], default="webcam")
    p.add_argument("--left-cam", type=int, default=0)
    p.add_argument("--right-cam", type=int, default=1)
    p.add_argument("--calibration", default="config/stereo_calibration.yaml")
    p.add_argument("--manual-scale", type=float, default=None)
    p.add_argument("--ref-object-cm", type=float, default=None)
    p.add_argument("--ref-person-cm", type=float, default=None)
    p.add_argument("--person-mode", action="store_true")
    p.add_argument("--hmr-python", default=None,
                   help="Path to python.exe of the HMR (Python 3.7 + TF 1.15) environment. "
                        "Can also be set with the HMR_PYTHON environment variable.")
    p.add_argument("--no-fit", action="store_true", help="Disable the body-fit worker.")
    p.add_argument("--rotate", choices=list(ROTATION_CHOICES.keys()), default=DEFAULT_ROTATION,
                   help="Rotate camera frames before display AND measurement "
                        "(default: ccw90 = 90° counter-clockwise for the sideways Pi camera mount).")
    # MST Pipeline Flags
    p.add_argument("--no-mediapipe", action="store_true", help="Use Haar cascade fallback.")
    p.add_argument("--no-white-balance", action="store_true", help="Disable white balance.")
    p.add_argument("--use-color-card", action="store_true", help="Enable ColorChecker detection.")
    return p


if __name__ == "__main__":
    print("Person height module loaded successfully")
    print("  Expected accuracy: +/-1-2 cm webcam mode (calibrated), +/-1-2 cm stereo mode")

    args = _build_parser().parse_args()

    if getattr(args, "ref_person_cm", None) is not None:
        cap = _open_camera(args.left_cam, rotation=args.rotate)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera.")
            sys.exit(1)
        for _ in range(10):
            cap.read()
        _run_person_calibration(cap, args.ref_person_cm)
        cap.release()
        cv2.destroyAllWindows()
    elif getattr(args, "person_mode", False) or (args.mode == "webcam" and args.target == "person"):
        _cli_person_webcam(args)
    elif args.mode == "webcam":
        _cli_webcam(args)
    else:
        _cli_stereo(args)
