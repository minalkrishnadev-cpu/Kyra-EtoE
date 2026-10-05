"""
src/height/height_stereo.py
===========================
Stereo camera height estimation using RANSAC ground plane fitting.

Public API
----------
detect_floor_plane(point_cloud) -> np.ndarray
get_person_height_stereo(point_cloud, head_pixel, feet_pixel, floor_plane) -> dict
get_object_height(point_cloud, top_pixel, base_pixel, floor_plane) -> float
smooth_height(history) -> float
"""

from __future__ import annotations

import collections
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_floor_plane(
    point_cloud: np.ndarray,
    distance_threshold: float = 2.0,
    max_iterations: int = 150
) -> np.ndarray:
    """
    Fit a plane equation (Ax + By + Cz + D = 0) to the lower third of the point cloud
    using RANSAC, representing the ground floor plane.
    """
    H, W, _ = point_cloud.shape
    pts = point_cloud.reshape(-1, 3)
    valid_mask = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & np.isfinite(pts[:, 2])
    pts_valid = pts[valid_mask]

    if len(pts_valid) < 100:
        logger.warning("detect_floor_plane: insufficient valid points, returning horizontal default plane.")
        return np.array([0.0, 1.0, 0.0, 0.0])

    y_coords = pts_valid[:, 1]
    y_threshold = np.percentile(y_coords, 66.0)
    lower_pts = pts_valid[y_coords >= y_threshold]

    if len(lower_pts) < 10:
        lower_pts = pts_valid

    best_plane = np.array([0.0, 1.0, 0.0, 0.0])
    max_inliers = -1

    for _ in range(max_iterations):
        idx = np.random.choice(len(lower_pts), 3, replace=False)
        p1, p2, p3 = lower_pts[idx]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-4:
            continue
        normal = normal / norm

        d = -np.dot(normal, p1)
        plane = np.array([normal[0], normal[1], normal[2], d])

        distances = np.abs(np.dot(lower_pts, normal) + d)
        inliers = np.sum(distances < distance_threshold)

        if inliers > max_inliers:
            max_inliers = inliers
            best_plane = plane

    best_normal = best_plane[:3]
    best_d = best_plane[3]
    distances = np.abs(np.dot(lower_pts, best_normal) + best_d)
    inlier_pts = lower_pts[distances < distance_threshold]

    if len(inlier_pts) >= 3:
        mean = np.mean(inlier_pts, axis=0)
        cov = np.cov((inlier_pts - mean).T)
        u, s, vh = np.linalg.svd(cov)
        normal = u[:, -1]
        normal = normal / np.linalg.norm(normal)
        if normal[1] > 0:
            normal = -normal
        d = -np.dot(normal, mean)
        best_plane = np.array([normal[0], normal[1], normal[2], d])

    return best_plane


def get_person_height_stereo(
    point_cloud: np.ndarray,
    head_pixel: Tuple[int, int],
    feet_pixel: Optional[Tuple[int, int]] = None,
    floor_plane: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Measure person height perpendicularly from head top point to floor plane.
    Applies -1.5 cm correction factor for hair volume.
    """
    result = {
        "height_cm": None,
        "source": "stereo",
        "confidence": "low"
    }

    hx, hy = head_pixel
    H, W = point_cloud.shape[:2]
    win = 3
    x_min, x_max = max(0, hx - win), min(W, hx + win + 1)
    y_min, y_max = max(0, hy - win), min(H, hy + win + 1)
    head_pts = point_cloud[y_min:y_max, x_min:x_max].reshape(-1, 3)
    valid_head_pts = head_pts[np.isfinite(head_pts[:, 2]) & (np.abs(head_pts[:, 2]) < 1000)]

    if len(valid_head_pts) == 0:
        return result

    head_3d = np.mean(valid_head_pts, axis=0)

    if floor_plane is None:
        floor_plane = detect_floor_plane(point_cloud)

    normal = floor_plane[:3]
    d = floor_plane[3]

    perp_dist = np.abs(np.dot(normal, head_3d) + d)
    measured_height = perp_dist - 1.5

    result.update({
        "height_cm": round(measured_height, 2),
        "confidence": "high" if measured_height > 100.0 else "medium"
    })
    return result


def get_object_height(
    point_cloud: np.ndarray,
    top_pixel: Tuple[int, int],
    base_pixel: Tuple[int, int],
    floor_plane: Optional[np.ndarray] = None
) -> float:
    """Measure object height from a 3D point cloud by perpendicular distance to floor."""
    tx, ty = top_pixel
    top_val = point_cloud[ty, tx]
    if not np.all(np.isfinite(top_val)):
        bx, by = base_pixel
        base_val = point_cloud[by, bx]
        if np.all(np.isfinite(base_val)) and np.all(np.isfinite(top_val)):
            return float(np.linalg.norm(top_val - base_val))
        return 0.0

    if floor_plane is None:
        floor_plane = detect_floor_plane(point_cloud)

    normal = floor_plane[:3]
    d = floor_plane[3]

    perp_dist = np.abs(np.dot(normal, top_val) + d)
    return float(perp_dist)


def smooth_height(history_deque: collections.deque) -> float:
    if not history_deque:
        return 0.0
    return float(np.median(list(history_deque)))
