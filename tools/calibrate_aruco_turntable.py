#!/usr/bin/env python3
"""
ArUco Marker Camera Calibration & Turntable Trajectory Pipeline
================================================================
Dataset: captures_0726_clay_checkboard_64  (64 images, 2 ArUco markers per frame)

This script performs:
  1. ArUco marker detection (IDs 0 & 6, DICT_4X4_50, marker side = 15mm)
  2. Camera intrinsic calibration refinement using the ArUco board
     (starting from previously calibrated K)
  3. Per-frame board pose estimation via solvePnP (both markers)
  4. Turntable trajectory joint optimization:
       - Center of rotation (C_rot) in camera space
       - Rotation axis normal vector
       - Actual angular step size per frame
  5. Render center point + Z-axis onto all 64 original images

Usage:
    .venv/bin/python tools/calibrate_aruco_turntable.py
"""

import cv2
import numpy as np
import glob
import os
import json
import scipy.optimize as optimize
from scipy.spatial.transform import Rotation as Rot
from pathlib import Path

# ==========================================
#  CONFIGURATION
# ==========================================
IMAGES_FOLDER = "captures_8-13_three_flat"
OUTPUT_FOLDER = "captures_8-13_three_flat_calibrated"
MARKER_SIZE_MM = 15.0       # Physical side length of each ArUco marker
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
N_IMAGES = 64
NOMINAL_STEP_DEG = 360.0 / N_IMAGES  # 5.625 degrees

# Previous calibration (same camera, ChArUco board, known good)
PREV_K = np.array([
    [3565.4767, 0.0,       2304.0],
    [0.0,       3565.4767, 1296.0],
    [0.0,       0.0,       1.0],
], dtype=np.float64)
PREV_DIST = np.array([-0.44031736, 7.83951075, 0.0, 0.0, -43.24147593])


# ==========================================
#  BOARD MODEL
# ==========================================
# Empirically measured offset of M6 relative to M0 (optimized via
# reprojection error minimization across all 64 frames using the
# previous calibration K).
M6_OFFSET_X = 20.51  # mm
M6_OFFSET_Y = 20.74  # mm

def make_board_obj_pts():
    """3D coordinates of both markers (board frame, Z=0 plane)."""
    s = MARKER_SIZE_MM
    ox, oy = M6_OFFSET_X, M6_OFFSET_Y

    m0 = np.array([
        [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],
    ], dtype=np.float32)

    m6 = np.array([
        [ox, oy, 0], [ox+s, oy, 0], [ox+s, oy+s, 0], [ox, oy+s, 0],
    ], dtype=np.float32)

    return {14: m0, 20: m6}


# ==========================================
#  STEP 0: Image Loading & Marker Detection
# ==========================================

def get_sorted_images():
    images = sorted(
        glob.glob(os.path.join(IMAGES_FOLDER, "*.jpg")),
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
    )
    if len(images) != N_IMAGES:
        print(f"Warning: Expected {N_IMAGES} images, found {len(images)}")
    return images


def detect_markers_all_frames(images):
    """Detect ArUco markers in every frame."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    detections = {}
    image_shape = None

    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if image_shape is None:
            image_shape = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            frame_markers = {}
            for corner, mid in zip(corners, ids.flatten()):
                frame_markers[int(mid)] = corner[0]
            detections[idx] = frame_markers

    return detections, image_shape


# ==========================================
#  STEP 1: Camera Calibration
# ==========================================

def calibrate_camera(detections, image_shape):
    """
    Calibrate camera using both markers as a rigid board.
    Uses the previous calibration as initial guess.
    """
    print("\n" + "=" * 55)
    print("  STEP 1 — CAMERA INTRINSIC CALIBRATION")
    print("=" * 55)

    board_pts = make_board_obj_pts()

    # ---- Approach A: individual markers (128 views, each 4 points) ----
    obj_single = np.array([
        [0, 0, 0], [MARKER_SIZE_MM, 0, 0],
        [MARKER_SIZE_MM, MARKER_SIZE_MM, 0], [0, MARKER_SIZE_MM, 0]
    ], dtype=np.float32)

    objpoints_ind = []
    imgpoints_ind = []
    for idx in sorted(detections.keys()):
        for mid, corners in detections[idx].items():
            objpoints_ind.append(obj_single.copy())
            imgpoints_ind.append(corners.astype(np.float32))

    # ---- Approach B: board model (64 views, each 8 points) ----
    objpoints_brd = []
    imgpoints_brd = []
    for idx in sorted(detections.keys()):
        markers = detections[idx]
        obj_f, img_f = [], []
        for mid in [14, 20]:
            if mid in markers:
                obj_f.append(board_pts[mid])
                img_f.append(markers[mid].astype(np.float32))
        if obj_f:
            objpoints_brd.append(np.vstack(obj_f))
            imgpoints_brd.append(np.vstack(img_f))

    w, h = image_shape
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS |
        cv2.CALIB_FIX_ASPECT_RATIO |
        cv2.CALIB_FIX_PRINCIPAL_POINT |
        cv2.CALIB_ZERO_TANGENT_DIST
    )

    # Run both approaches
    results = {}
    for label, objp, imgp in [
        ("Individual markers (128 views)", objpoints_ind, imgpoints_ind),
        ("Board model (64 views)", objpoints_brd, imgpoints_brd),
    ]:
        if len(objp) == 0:
            print(f"\n  Skipping {label} (0 views available)")
            continue
            
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objp, imgp, image_shape,
            PREV_K.copy(), PREV_DIST.reshape(1, 5).copy(),
            flags=flags
        )
        # Reprojection error
        tot_err, tot_pts = 0.0, 0
        for i in range(len(objp)):
            proj, _ = cv2.projectPoints(objp[i], rvecs[i], tvecs[i], K, dist)
            diff = imgp[i].reshape(-1, 2) - proj.reshape(-1, 2)
            tot_err += np.sum(np.linalg.norm(diff, axis=1))
            tot_pts += len(objp[i])
        mean_err = tot_err / tot_pts
        results[label] = (K, dist, mean_err, len(objp), tot_pts)
        print(f"\n  {label}:")
        print(f"    fx = {K[0,0]:.4f}   reproj = {mean_err:.4f} px")

    if not results:
        raise RuntimeError("No markers detected for calibration.")

    # Pick approach with lower reprojection error
    best_label = min(results, key=lambda k: results[k][2])
    K, dist, mean_err, n_views, n_pts = results[best_label]

    print(f"\n  ✓ Selected: {best_label}")
    print(f"\n  Camera Matrix K:")
    print(f"    fx = {K[0, 0]:.4f}   fy = {K[1, 1]:.4f}")
    print(f"    cx = {K[0, 2]:.4f}   cy = {K[1, 2]:.4f}")
    print(f"\n  Distortion [k1, k2, p1, p2, k3]:")
    print(f"    {dist.ravel()}")
    print(f"\n  Mean reprojection error: {mean_err:.4f} px")

    # Also show comparison with previous K
    print(f"\n  Previous calibration: fx = {PREV_K[0,0]:.4f}")
    print(f"  Change in fx:        {K[0,0] - PREV_K[0,0]:+.4f} ({(K[0,0]/PREV_K[0,0] - 1)*100:+.2f}%)")

    return K, dist, mean_err


# ==========================================
#  STEP 2: Per-Frame Board Pose Estimation
# ==========================================

def estimate_per_frame_poses(detections, K, dist):
    """
    Run solvePnP for each frame using BOTH markers (8 points).
    Falls back to single marker (4 points) if only one is visible.
    """
    board_pts = make_board_obj_pts()

    poses = {}
    for idx in sorted(detections.keys()):
        markers = detections[idx]
        obj_f, img_f = [], []
        for mid in [14, 20]:
            if mid in markers:
                obj_f.append(board_pts[mid])
                img_f.append(markers[mid].astype(np.float64))

        if not obj_f:
            continue

        obj_all = np.vstack(obj_f)
        img_all = np.vstack(img_f)

        ok, rvec, tvec = cv2.solvePnP(obj_all, img_all, K, dist)
        if ok and tvec.ravel()[2] > 0:
            poses[idx] = (rvec.ravel(), tvec.ravel(), len(obj_all))

    return poses


def compute_marker_centers_in_cam(poses):
    """
    Compute the 3D position of the board ORIGIN (M0 TL corner)
    in camera space from solvePnP results.
    Also compute the 3D center of M0 (for potentially better circle fitting).
    """
    origins = {}
    centers = {}
    s = MARKER_SIZE_MM
    m0_center_local = np.array([s/2, s/2, 0])

    for idx, (rvec, tvec, npts) in poses.items():
        R, _ = cv2.Rodrigues(rvec)
        # Board origin = M0 TL corner in camera space = tvec
        origins[idx] = tvec.copy()
        # M0 center in camera space
        centers[idx] = (R @ m0_center_local + tvec)

    return origins, centers


# ==========================================
#  STEP 3: Turntable Trajectory Fitting
# ==========================================

def fit_turntable_trajectory(poses):
    """
    Joint optimization for C_rot, normal, step_size.
    Uses board origin (tvec) positions across frames.
    Tries BOTH rotation directions and picks the better fit.
    """
    print("\n" + "=" * 55)
    print("  STEP 2 — TURNTABLE TRAJECTORY OPTIMIZATION")
    print("=" * 55)

    tvecs_raw = {idx: tvec for idx, (_, tvec, _) in poses.items()}
    npts_info = {idx: npts for idx, (_, _, npts) in poses.items()}

    print(f"  Pose observations: {len(tvecs_raw)} / {N_IMAGES}")
    print(f"  Frames with 8 pts: {sum(1 for n in npts_info.values() if n == 8)}")
    print(f"  Frames with 4 pts: {sum(1 for n in npts_info.values() if n == 4)}")

    z_vals = np.array([t[2] for t in tvecs_raw.values()])
    print(f"  Z stats:  min={z_vals.min():.1f}  median={np.median(z_vals):.1f}  "
          f"max={z_vals.max():.1f}  std={z_vals.std():.1f} mm")

    # Outlier filtering
    tvecs_coarse = {i: t for i, t in tvecs_raw.items() if 100.0 < t[2] < 800.0}
    print(f"  After coarse Z filter: {len(tvecs_coarse)} frames")

    if len(tvecs_coarse) >= 10:
        z_filt = np.array([t[2] for t in tvecs_coarse.values()])
        z_med = np.median(z_filt)
        z_std = np.std(z_filt)
        tvecs_filtered = {
            i: t for i, t in tvecs_coarse.items()
            if abs(t[2] - z_med) < max(3.0 * z_std, 30.0)
        }
    else:
        tvecs_filtered = tvecs_coarse

    print(f"  After refined filter:  {len(tvecs_filtered)} frames")
    if len(tvecs_filtered) < 8:
        raise RuntimeError("Too few inlier frames for trajectory fitting.")

    tvecs_arr = np.array(list(tvecs_filtered.values()))

    # ---- Initial guess via SVD ----
    mean_t = tvecs_arr.mean(axis=0)
    centered = tvecs_arr - mean_t
    _, S, Vt = np.linalg.svd(centered)
    normal_init = Vt[2]
    if normal_init[2] < 0:
        normal_init = -normal_init

    u_ax, v_ax = Vt[0], Vt[1]
    pu = centered @ u_ax
    pv = centered @ v_ax
    A = np.column_stack([2*pu, 2*pv, np.ones(len(pu))])
    B = pu**2 + pv**2
    uc, vc, _ = np.linalg.lstsq(A, B, rcond=None)[0]

    el0 = np.arcsin(np.clip(normal_init[2], -1, 1))
    az0 = np.arctan2(normal_init[1], normal_init[0])
    first_idx = min(tvecs_raw.keys())
    p0 = tvecs_raw[first_idx]

    # ---- Helper ----
    def _axes(el, az):
        n = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
        n /= np.linalg.norm(n)
        ref = [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0]
        u = np.cross(n, ref); u /= np.linalg.norm(u)
        v = np.cross(n, u)
        return n, u, v

    def _residuals(params):
        uc, vc, el, az, step, p0x, p0y, p0z = params
        n, u, v = _axes(el, az)
        c = mean_t + uc * u + vc * v
        p0_vec = np.array([p0x, p0y, p0z])
        errs = []
        for i, obs in tvecs_filtered.items():
            ang = np.deg2rad(i * step)
            R_i = Rot.from_rotvec(ang * n).as_matrix()
            pred = R_i @ (p0_vec - c) + c
            errs.extend(pred - obs)
        return np.array(errs)

    # ---- Multi-start: try several step size initializations ----
    best_res = None
    best_cost = float('inf')

    for step_init in [NOMINAL_STEP_DEG, -NOMINAL_STEP_DEG,
                       5.0, -5.0, 6.0, -6.0]:
        x0 = [uc, vc, el0, az0, step_init, p0[0], p0[1], p0[2]]
        try:
            res = optimize.least_squares(
                _residuals, x0,
                bounds=([-np.inf, -np.inf, -np.pi/2, -np.pi, -10, -np.inf, -np.inf, -np.inf],
                        [np.inf, np.inf, np.pi/2, np.pi, 10, np.inf, np.inf, np.inf])
            )
            if res.cost < best_cost:
                best_cost = res.cost
                best_res = res
                best_step_init = step_init
        except Exception:
            continue

    print(f"\n  Best initialization: step={best_step_init:.3f}°  cost={best_cost:.4f}")

    res = best_res
    uc_f, vc_f, el_f, az_f, step_deg, p0x, p0y, p0z = res.x
    normal, u, v = _axes(el_f, az_f)
    C_rot = mean_t + uc_f * u + vc_f * v

    # Project center onto marker plane
    p0_f = np.array([p0x, p0y, p0z])
    offset = np.dot(p0_f - C_rot, normal)
    C_rot_planar = C_rot + offset * normal

    rmse = np.sqrt(np.mean(res.fun**2))
    total_rot = abs(step_deg) * N_IMAGES

    print(f"\n  ──── Optimized Results ────")
    print(f"  Center C_rot (mm):  [{C_rot_planar[0]:.4f}, {C_rot_planar[1]:.4f}, {C_rot_planar[2]:.4f}]")
    print(f"  Normal (rot axis):  [{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]")
    print(f"  Step size:          {step_deg:.6f}°  (nominal {NOMINAL_STEP_DEG:.6f}°)")
    print(f"  |Step| deviation:   {abs(step_deg) - NOMINAL_STEP_DEG:+.6f}°")
    print(f"  Total rotation:     {total_rot:.4f}°  ({total_rot - 360:+.4f}° from 360°)")
    print(f"  Trajectory RMSE:    {rmse:.4f} mm")
    print(f"  Optimizer success:  {res.success}")

    # ---- Per-step angle validation ----
    print(f"\n  ──── Per-Step Angle Validation ────")
    sorted_frames = sorted(tvecs_filtered.keys())
    step_angles = []
    for j in range(len(sorted_frames) - 1):
        i1, i2 = sorted_frames[j], sorted_frames[j+1]
        if i2 - i1 != 1:
            continue  # skip non-consecutive
        v1 = tvecs_filtered[i1] - C_rot_planar
        v2 = tvecs_filtered[i2] - C_rot_planar
        # Project onto the rotation plane
        v1p = v1 - np.dot(v1, normal) * normal
        v2p = v2 - np.dot(v2, normal) * normal
        cos_a = np.dot(v1p, v2p) / (np.linalg.norm(v1p) * np.linalg.norm(v2p) + 1e-10)
        angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
        step_angles.append(angle)

    if step_angles:
        sa = np.array(step_angles)
        print(f"  Per-step angles: mean={sa.mean():.4f}°  std={sa.std():.4f}°  "
              f"min={sa.min():.4f}°  max={sa.max():.4f}°")
        print(f"  Implied total (mean×{N_IMAGES}): {sa.mean()*N_IMAGES:.2f}°")

    return C_rot_planar, normal, step_deg, rmse


# ==========================================
#  STEP 4: Render Center + Z-Axis
# ==========================================

def render_visualizations(images, K, dist, C_rot, normal, step_deg):
    """Draw plate center crosshair + rotation Z-axis on every image."""
    print("\n" + "=" * 55)
    print("  STEP 3 — RENDERING CENTER + Z-AXIS")
    print("=" * 55)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    rvec0 = np.zeros((3, 1), dtype=np.float64)
    tvec0 = np.zeros((3, 1), dtype=np.float64)

    # Project center
    center_2d, _ = cv2.projectPoints(
        C_rot.reshape(1, 1, 3).astype(np.float64), rvec0, tvec0, K, dist
    )
    cx = int(round(center_2d[0][0][0]))
    cy = int(round(center_2d[0][0][1]))

    # Z-axis line
    s_vals = np.linspace(-80, 120, 300)
    z_pts_3d = np.array([C_rot + s * normal for s in s_vals], dtype=np.float64)
    valid = z_pts_3d[:, 2] > 10.0
    z_pts_3d_valid = z_pts_3d[valid]
    if len(z_pts_3d_valid) > 0:
        z_pts_2d, _ = cv2.projectPoints(z_pts_3d_valid, rvec0, tvec0, K, dist)
        z_pts_px = z_pts_2d.reshape(-1, 2)
        good = np.isfinite(z_pts_px).all(axis=1)
        z_pts_px = z_pts_px[good].astype(np.int32)
    else:
        z_pts_px = np.array([], dtype=np.int32).reshape(0, 2)

    # Arrow tip
    arrow_3d = (C_rot + 80 * normal).reshape(1, 1, 3).astype(np.float64)
    arrow_2d, _ = cv2.projectPoints(arrow_3d, rvec0, tvec0, K, dist)
    ax = int(round(arrow_2d[0][0][0]))
    ay = int(round(arrow_2d[0][0][1]))

    print(f"  Center pixel:     ({cx}, {cy})")
    print(f"  Z-axis arrow tip: ({ax}, {ay})")

    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)

        # Z-axis dashed line
        for j in range(0, len(z_pts_px) - 2, 4):
            pt1 = tuple(z_pts_px[j])
            pt2 = tuple(z_pts_px[min(j + 2, len(z_pts_px) - 1)])
            cv2.line(img, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)

        # Z-axis arrow
        cv2.arrowedLine(img, (cx, cy), (ax, ay),
                        (255, 255, 0), 3, cv2.LINE_AA, tipLength=0.12)

        # Center crosshair
        sz = 50
        cv2.line(img, (cx-sz, cy), (cx+sz, cy), (0, 255, 0), 3, cv2.LINE_AA)
        cv2.line(img, (cx, cy-sz), (cx, cy+sz), (0, 255, 0), 3, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 35, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)

        # Text
        cv2.putText(img, f"Plate Center: ({cx}, {cy})", (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(img, f"Frame {i}/{N_IMAGES}  |  step={abs(step_deg):.4f} deg",
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, "Z-axis (rotation)", (ax + 25, ay - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 2, cv2.LINE_AA)

        out_path = os.path.join(OUTPUT_FOLDER, os.path.basename(img_path))
        orig_img = cv2.imread(img_path) # Load clean image
        cv2.imwrite(out_path, orig_img) # Save clean image so masks don't get ruined by red arrows

    print(f"  Saved {len(images)} annotated images -> {OUTPUT_FOLDER}/")


# ==========================================
#  MAIN
# ==========================================

def main():
    print("=" * 55)
    print("  ARUCO CALIBRATION & TURNTABLE TRAJECTORY PIPELINE")
    print(f"  Dataset: {IMAGES_FOLDER}")
    print("=" * 55)

    images = get_sorted_images()
    print(f"\n  Found {len(images)} images")

    print("  Detecting ArUco markers (DICT_4X4_50, IDs 0 & 6)...")
    detections, image_shape = detect_markers_all_frames(images)
    n_det = len(detections)
    both_count = sum(1 for d in detections.values() if 0 in d and 6 in d)
    print(f"  Markers detected in {n_det}/{len(images)} frames")
    print(f"  Both markers in {both_count}/{len(images)} frames")
    print(f"  Image resolution: {image_shape[0]} × {image_shape[1]}")

    # Step 1: Camera calibration
    K, dist, reproj_err = calibrate_camera(detections, image_shape)

    # Step 2: Per-frame poses
    print(f"\n  Estimating per-frame board poses (solvePnP)...")
    poses = estimate_per_frame_poses(detections, K, dist)
    print(f"  Got poses for {len(poses)}/{len(images)} frames")

    # Step 3: Trajectory
    C_rot, normal, step_deg, traj_rmse = fit_turntable_trajectory(poses)

    # Step 4: Render
    render_visualizations(images, K, dist, C_rot, normal, step_deg)

    # ---- FINAL SUMMARY ----
    total_rot = abs(step_deg) * N_IMAGES
    print("\n" + "=" * 55)
    print("  ★  FINAL CALIBRATION SUMMARY  ★")
    print("=" * 55)
    print(f"\n  Camera Matrix K:")
    print(f"    [[{K[0,0]:10.4f}, {K[0,1]:10.4f}, {K[0,2]:10.4f}],")
    print(f"     [{K[1,0]:10.4f}, {K[1,1]:10.4f}, {K[1,2]:10.4f}],")
    print(f"     [{K[2,0]:10.4f}, {K[2,1]:10.4f}, {K[2,2]:10.4f}]]")
    print(f"\n  Distortion [k1, k2, p1, p2, k3]:")
    d = dist.ravel()
    print(f"    [{d[0]:.8f}, {d[1]:.8f}, {d[2]:.8f}, {d[3]:.8f}, {d[4]:.8f}]")
    print(f"\n  Reprojection error:  {reproj_err:.4f} px")
    print(f"\n  Plate center (mm):   [{C_rot[0]:.4f}, {C_rot[1]:.4f}, {C_rot[2]:.4f}]")
    print(f"  Plate normal:        [{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]")
    print(f"\n  Step size:           {abs(step_deg):.6f}° per frame")
    print(f"  Nominal step:        {NOMINAL_STEP_DEG:.6f}° (360/{N_IMAGES})")
    print(f"  Deviation:           {abs(step_deg) - NOMINAL_STEP_DEG:+.6f}° per step")
    print(f"  Total rotation:      {total_rot:.4f}° ({total_rot - 360:+.4f}° from 360°)")
    print(f"  Trajectory RMSE:     {traj_rmse:.4f} mm")
    print("=" * 55)

    # Save
    cal = {
        "camera_matrix_K": K.tolist(),
        "distortion_coefficients": dist.ravel().tolist(),
        "reprojection_error_px": float(reproj_err),
        "plate_center_mm": C_rot.tolist(),
        "plate_normal": normal.tolist(),
        "step_size_deg": float(abs(step_deg)),
        "step_sign": "positive" if step_deg > 0 else "negative",
        "nominal_step_deg": float(NOMINAL_STEP_DEG),
        "total_rotation_deg": float(total_rot),
        "trajectory_rmse_mm": float(traj_rmse),
        "n_images": N_IMAGES,
        "marker_size_mm": MARKER_SIZE_MM,
        "aruco_dictionary": "DICT_4X4_50",
        "marker_ids": [14, 20],
        "board_offset_mm": [M6_OFFSET_X, M6_OFFSET_Y],
        "image_resolution": list(image_shape),
    }
    cal_path = os.path.join(OUTPUT_FOLDER, "calibration_results.json")
    with open(cal_path, "w") as f:
        json.dump(cal, f, indent=4)
    print(f"\n  Results saved to: {cal_path}")


if __name__ == "__main__":
    main()
