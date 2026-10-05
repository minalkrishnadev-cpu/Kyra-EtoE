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

def _open_camera(idx: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    return cap


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

    pixels_per_cm = rh / ref_cm
    base_y = ry + rh
    from src.height.height_webcam import save_calibration_data
    save_calibration_data(pixels_per_cm, base_y)
    
    print(f"\n[SUCCESS] Calibration saved!")
    print(f"          Reference Height : {ref_cm} cm")
    print(f"          Pixel Height     : {rh} px")
    print(f"          Base Y Pixel     : {base_y} px")
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
            save_calibration_data(pixels_per_cm, base_y=feet[1])
            
            cv2.destroyWindow("Person Calibration")
            print(f"\n[SUCCESS] Calibration saved successfully!")
            print(f"          Target Height : {ref_person_cm} cm")
            print(f"          Pixel Span    : {pixel_span} px")
            print(f"          Base Y Pixel  : {feet[1]} px")
            print(f"          Scale Factor  : {pixels_per_cm:.4f} px/cm")
            print(f"          Saved to      : config/height_calibration.json\n")
            return pixels_per_cm


def _cli_person_webcam(args) -> None:
    """Live-view CLI for single-webcam person height measurement using pre-calibrated scale."""
    import collections
    from src.height.landmarks import get_head_foot_pixels, draw_landmarks_overlay
    from src.height.height_webcam import get_person_height_webcam, load_calibration_data

    cam_idx = getattr(args, "left_cam", 0)
    cap = _open_camera(cam_idx)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {cam_idx}.")
        return

    calib_data = load_calibration_data()
    scale = calib_data["pixels_per_cm"]
    calibrated_base_y = calib_data["calibrated_base_y"]

    manual_scale = getattr(args, "manual_scale", None)
    if manual_scale is not None:
        scale = manual_scale
        scale_label = f"MANUAL ({scale:.2f} px/cm)"
    else:
        scale_label = f"CALIBRATED ({scale:.2f} px/cm)"

    print(f"[INFO] Person height live view | Marker-free mode | Scale: {scale_label}")
    print("       Press Q or ESC to quit.")

    history = collections.deque(maxlen=10)

    cv2.namedWindow("KYRA Person Height (webcam)", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        display = frame.copy()
        cv2.putText(display, f"Scale: {scale_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

        lm = get_head_foot_pixels(frame)
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

        # Overlay position status at the top-left of the screen
        cv2.putText(display, pos_txt, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, pos_color, 2, cv2.LINE_AA)

        result = get_person_height_webcam(frame, manual_scale=scale)
        h_cm = result.get("height_cm")
        src = result.get("source", "?")

        # Only append to history if we actually measured the height using landmarks
        if h_cm is not None and src in ("calibrated_scale", "default_scale"):
            history.append(h_cm)

        H_frame = display.shape[0]
        if history:
            median_h = float(np.median(list(history)))
            status_colour = (0, 255, 0) if src in ("calibrated_scale", "default_scale") else (0, 165, 255)
            height_txt = f"Height: {median_h:.1f} cm  [{src}]"
            if flag and h_cm is not None and src in ("calibrated_scale", "default_scale"):
                print(f"\r  when position is OKKK height_cm={median_h:.1f}  source={src:<18s}", end="", flush=True)
        else:
            status_colour = (0, 165, 255) # Orange
            height_txt = "Height: Detecting..."

        cv2.putText(display, height_txt, (10, H_frame - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_colour, 2, cv2.LINE_AA)

        cv2.imshow("KYRA Person Height (webcam)", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    print()
    cap.release()
    cv2.destroyAllWindows()


def _cli_webcam(args: argparse.Namespace) -> None:
    """Universal webcam CLI for measuring objects and persons marker-free."""
    from src.height.target_detector import PersonTopPoint, ROIBoundingBoxDetector
    from src.height.height_webcam import load_calibration_scale

    print(f"[INFO] Opening webcam index {args.left_cam}...")
    cap = _open_camera(args.left_cam)
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
    cap_l = _open_camera(args.left_cam)
    cap_r = _open_camera(args.right_cam)

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
    return p


if __name__ == "__main__":
    print("Person height module loaded successfully")
    print("  Expected accuracy: +/-1-2 cm webcam mode (calibrated), +/-1-2 cm stereo mode")

    args = _build_parser().parse_args()

    if getattr(args, "ref_person_cm", None) is not None:
        cap = _open_camera(args.left_cam)
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
