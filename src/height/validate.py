"""
src/height/validate.py
======================
Real-world validation suite for the KYRA Person Height Module.
Validates height accuracy and frame-to-frame stability without markers.

Saves structured session logs to the `src/height/validation_logs/` directory.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Ensure path includes parent folder for proper package imports when executed directly
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.height import (
    measure_person_height,
    get_head_foot_pixels,
    draw_landmarks_overlay
)

logger = logging.getLogger(__name__)


def _open_camera(idx: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    return cap


def _get_log_dir() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, "validation_logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def compute_validation_summary(
    samples: List[Dict[str, Any]],
    known_person_height_cm: float,
    mode_name: str
) -> Dict[str, Any]:
    """Compute statistics across height samples."""
    if not samples:
        return {}

    heights = [s["height_cm"] for s in samples]
    mean_h = float(np.mean(heights))
    std_h = float(np.std(heights))
    abs_error = abs(mean_h - known_person_height_cm)
    pct_error = (abs_error / known_person_height_cm) * 100.0
    passed = abs_error <= 3.0

    return {
        "mode": mode_name,
        "known_height_cm": known_person_height_cm,
        "mean_measured_height_cm": round(mean_h, 2),
        "std_dev_cm": round(std_h, 2),
        "absolute_error_cm": round(abs_error, 2),
        "error_percentage": round(pct_error, 2),
        "passed": passed,
        "n_samples": len(samples)
    }


def print_stylized_summary(summary: Dict[str, Any], mode_label: str) -> None:
    if not summary:
        print("============================")
        print(f"{mode_label.upper()} VALIDATION RESULT")
        print("No valid samples were collected.")
        print("============================")
        return

    passed_str = "PASS (within ±3cm tolerance)" if summary["passed"] else "FAIL (outside ±3cm tolerance)"
    print("============================")
    print(f"{mode_label.upper()} VALIDATION RESULT")
    print(f"Known height:     {summary['known_height_cm']:.1f} cm")
    print(f"Measured (mean):  {summary['mean_measured_height_cm']:.1f} cm")
    print(f"Std deviation:    {summary['std_dev_cm']:.1f} cm")
    print(f"Absolute error:   {summary['absolute_error_cm']:.1f} cm")
    print(f"Result:           {passed_str}")
    print("============================")


def run_webcam_validation(
    known_person_height_cm: float,
    num_samples: int = 20,
    camera_idx: int = 0,
    manual_scale: Optional[float] = None
) -> None:
    """Open the webcam and collect height measurements using the calibrated scale."""
    cap = _open_camera(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {camera_idx}.")
        return

    for _ in range(10):
        cap.read()

    from src.height.height_webcam import load_calibration_data
    calib_data = load_calibration_data()
    resolved_scale = manual_scale if manual_scale is not None else calib_data["pixels_per_cm"]
    calibrated_base_y = calib_data["calibrated_base_y"]
    scale_label = f"MANUAL ({resolved_scale:.2f} px/cm)" if manual_scale is not None else f"CALIBRATED ({resolved_scale:.2f} px/cm)"

    samples: List[Dict[str, Any]] = []
    last_warn_time = 0.0

    print(f"\n[INFO] Starting Webcam Validation.")
    print(f"       Stand on the calibrated spot in full view of the camera.")
    print(f"       Collecting {num_samples} samples. Press Ctrl+C to abort early.\n")

    try:
        cv2.namedWindow("KYRA Height Validation Preview", cv2.WINDOW_NORMAL)
        while len(samples) < num_samples:
            ok, frame = cap.read()
            if not ok:
                break

            lm = get_head_foot_pixels(frame)
            res = measure_person_height(
                frame=frame,
                camera_mode="webcam",
                manual_scale=resolved_scale
            )
            h_cm = res.get("height_cm")
            src = res.get("source")

            display = frame.copy()
            cv2.putText(display, f"Scale: {scale_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

            # Position helper logic
            if lm:
                display = draw_landmarks_overlay(display, lm)
                feet = lm.get("feet")
                if feet and calibrated_base_y is not None:
                    diff = feet[1] - calibrated_base_y
                    if diff > 35:
                        pos_txt = "POSITION: STEP BACK"
                        pos_color = (0, 140, 255)
                    elif diff < -35:
                        pos_txt = "POSITION: STEP FORWARD"
                        pos_color = (0, 140, 255)
                    else:
                        pos_txt = "POSITION: OK"
                        pos_color = (0, 255, 0)
                else:
                    pos_txt = "POSITION: OK (Uncalibrated spot)"
                    pos_color = (0, 255, 255)
            else:
                pos_txt = "POSITION: NO PERSON DETECTED"
                pos_color = (0, 165, 255)

            # Overlay position status
            cv2.putText(display, pos_txt, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, pos_color, 2, cv2.LINE_AA)

            H, W = display.shape[:2]
            cv2.putText(display, f"Webcam Validation | Sample: {len(samples)}/{num_samples}",
                        (10, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            
            if h_cm is not None and src in ("calibrated_scale", "default_scale"):
                color = (0, 255, 0)
                cv2.putText(display, f"Height: {h_cm:.1f} cm ({src})", (10, H - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            else:
                cv2.putText(display, "Height: Detecting...", (10, H - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

            cv2.imshow("KYRA Height Validation Preview", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

            # Accepts calibrated_scale or default_scale measurements
            if h_cm is not None and src in ("calibrated_scale", "default_scale"):
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                sample_idx = len(samples) + 1
                samples.append({
                    "sample_index": sample_idx,
                    "height_cm": h_cm,
                    "source": src,
                    "timestamp": timestamp
                })
                print(f"Sample {sample_idx:02d}/{num_samples}: {h_cm:.2f} cm | source={src} | time={timestamp}")
                time.sleep(0.15)
            else:
                now = time.time()
                if now - last_warn_time > 2.5:
                    print(f"[WARN] Skipping fallback/invalid frame (source: {src}). Keep body landmarks fully visible on calibrated spot.")
                    last_warn_time = now

    except KeyboardInterrupt:
        print("\n[INFO] Collection interrupted.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    summary = compute_validation_summary(samples, known_person_height_cm, "webcam")
    print_stylized_summary(summary, "webcam")

    if samples:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"webcam_{timestamp_str}.json"
        log_path = os.path.join(_get_log_dir(), log_filename)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "samples": samples}, f, indent=4)
        print(f"[INFO] Validation log written to: {log_path}")


def run_stereo_validation(
    known_person_height_cm: float,
    num_samples: int = 20,
    calibration_path: str = "config/stereo_calibration.yaml",
    left_cam_idx: int = 0,
    right_cam_idx: int = 1
) -> None:
    try:
        from src.calibration import load_calibration, compute_depth_map
    except ImportError:
        print("[ERROR] Could not import stereo calibration functions.")
        return

    if not os.path.exists(calibration_path):
        print(f"[ERROR] Calibration file not found at: {calibration_path}")
        return

    calibration = load_calibration(calibration_path)
    cap_l = _open_camera(left_cam_idx)
    cap_r = _open_camera(right_cam_idx)

    if not cap_l.isOpened() or not cap_r.isOpened():
        print("[ERROR] Could not open stereo cameras.")
        return

    for _ in range(10):
        cap_l.read()
        cap_r.read()

    samples: List[Dict[str, Any]] = []
    last_warn_time = 0.0

    print(f"\n[INFO] Starting Stereo Validation.")
    print(f"       Collecting {num_samples} samples. Press Ctrl+C to abort early.\n")

    try:
        cv2.namedWindow("KYRA Stereo Validation Preview", cv2.WINDOW_NORMAL)
        while len(samples) < num_samples:
            ok_l, left = cap_l.read()
            ok_r, right = cap_r.read()
            if not ok_l or not ok_r:
                break

            point_cloud = compute_depth_map(left, right, calibration)
            res = measure_person_height(frame=left, camera_mode="stereo", point_cloud=point_cloud)
            h_cm = res.get("height_cm")
            src = res.get("source")

            display = left.copy()
            lm = get_head_foot_pixels(left)
            if lm:
                display = draw_landmarks_overlay(display, lm)

            display_stereo = np.concatenate([display, right], axis=1)
            cv2.putText(display_stereo, f"Stereo Validation | Sample: {len(samples)}/{num_samples}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if h_cm is not None:
                cv2.putText(display_stereo, f"Height: {h_cm:.1f} cm", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("KYRA Stereo Validation Preview", display_stereo)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

            if src == "stereo" and h_cm is not None:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                sample_idx = len(samples) + 1
                samples.append({
                    "sample_index": sample_idx,
                    "height_cm": h_cm,
                    "source": src,
                    "timestamp": timestamp
                })
                print(f"Sample {sample_idx:02d}/{num_samples}: {h_cm:.2f} cm | time={timestamp}")
                time.sleep(0.15)
            else:
                now = time.time()
                if now - last_warn_time > 2.5:
                    print(f"[WARN] Skipping invalid frame (source: {src}).")
                    last_warn_time = now

    except KeyboardInterrupt:
        print("\n[INFO] Collection interrupted.")

    finally:
        cap_l.release()
        cap_r.release()
        cv2.destroyAllWindows()

    summary = compute_validation_summary(samples, known_person_height_cm, "stereo")
    print_stylized_summary(summary, "stereo")

    if samples:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"stereo_{timestamp_str}.json"
        log_path = os.path.join(_get_log_dir(), log_filename)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "samples": samples}, f, indent=4)
        print(f"[INFO] Validation log written to: {log_path}")


def run_distance_sensitivity_test(
    known_person_height_cm: float,
    camera_idx: int = 0,
    manual_scale: Optional[float] = None
) -> None:
    cap = _open_camera(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera.")
        return

    for _ in range(10):
        cap.read()

    from src.height.height_webcam import load_calibration_scale
    resolved_scale = manual_scale if manual_scale is not None else load_calibration_scale()
    scale_label = f"MANUAL ({resolved_scale:.2f} px/cm)" if manual_scale is not None else f"CALIBRATED ({resolved_scale:.2f} px/cm)"

    distances = ["close (~1m)", "medium (~2m)", "far (~3m)"]
    bucket_results: Dict[str, Any] = {}

    print("\n=========================================")
    print("  DISTANCE SENSITIVITY TEST SESSION")
    print("=========================================\n")

    try:
        for bucket in distances:
            print(f"[ACTION] Prompt: Stand at {bucket.upper()} range.")
            input("         Ensure standing spot is correct. Press ENTER to start capturing... ")

            print(f"[INFO] Collecting 5 samples for {bucket}. Press Ctrl+C to skip/finish...")
            samples = []
            last_warn_time = 0.0

            cv2.namedWindow("Distance Sensitivity Verification Preview", cv2.WINDOW_NORMAL)
            while len(samples) < 5:
                ok, frame = cap.read()
                if not ok:
                    break

                lm = get_head_foot_pixels(frame)
                res = measure_person_height(
                    frame=frame,
                    camera_mode="webcam",
                    manual_scale=resolved_scale
                )
                h_cm = res.get("height_cm")
                src = res.get("source")

                display = frame.copy()
                cv2.putText(display, f"Scale: {scale_label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

                if lm:
                    display = draw_landmarks_overlay(display, lm)

                cv2.putText(display, f"Distance Test: {bucket} | Samples: {len(samples)}/5",
                            (10, display.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Distance Sensitivity Verification Preview", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break

                if h_cm is not None and src in ("calibrated_scale", "default_scale"):
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    sample_idx = len(samples) + 1
                    samples.append({
                        "sample_index": sample_idx,
                        "height_cm": h_cm,
                        "source": src,
                        "timestamp": timestamp
                    })
                    print(f"  [{bucket}] Sample {sample_idx}/5: {h_cm:.2f} cm")
                    time.sleep(0.2)
                else:
                    now = time.time()
                    if now - last_warn_time > 2.0:
                        print("  [WARN] Keep body landmarks fully visible on spot.")
                        last_warn_time = now

            if samples:
                summary = compute_validation_summary(samples, known_person_height_cm, f"distance_{bucket}")
                bucket_results[bucket] = {"summary": summary, "samples": samples}
                print(f"[SUCCESS] Finished {bucket}. Mean={summary['mean_measured_height_cm']} cm\n")
            else:
                print(f"[INFO] No samples collected.\n")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user request.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n=========================================")
    print("  DISTANCE SENSITIVITY TEST SUMMARY")
    print("=========================================")
    for bucket, res_data in bucket_results.items():
        sum_data = res_data["summary"]
        status = "PASS" if sum_data["passed"] else "FAIL"
        print(f"Bucket {bucket:<14s}: Mean={sum_data['mean_measured_height_cm']:.1f} cm | Result={status}")
    print("=========================================\n")

    if bucket_results:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"distance_sensitivity_{timestamp_str}.json"
        log_path = os.path.join(_get_log_dir(), log_filename)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(bucket_results, f, indent=4)
        print(f"[INFO] Distance test log written to: {log_path}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KYRA Height Module — Real-World Validation Suite")
    p.add_argument("--mode", choices=["webcam", "stereo", "distance-test"], required=True)
    p.add_argument("--manual-scale", type=float, default=None)
    p.add_argument("--known-height", type=float, required=True)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--calibration", default="config/stereo_calibration.yaml")
    p.add_argument("--left-cam", type=int, default=0)
    p.add_argument("--right-cam", type=int, default=1)
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if args.mode == "webcam":
        run_webcam_validation(
            known_person_height_cm=args.known_height,
            num_samples=args.samples,
            camera_idx=args.left_cam,
            manual_scale=args.manual_scale
        )
    elif args.mode == "stereo":
        run_stereo_validation(
            known_person_height_cm=args.known_height,
            num_samples=args.samples,
            calibration_path=args.calibration,
            left_cam_idx=args.left_cam,
            right_cam_idx=args.right_cam
        )
    elif args.mode == "distance-test":
        run_distance_sensitivity_test(
            known_person_height_cm=args.known_height,
            camera_idx=args.left_cam,
            manual_scale=args.manual_scale
        )
