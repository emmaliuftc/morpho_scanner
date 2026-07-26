# Turntable Calibration Report
## Dataset: `captures_0726_clay_checkboard_64` (64 images)

---

## Camera Matrix K & Distortion

| Parameter | Value |
|-----------|-------|
| **fx = fy** | **3593.2937 px** |
| cx | 2304.0000 px (image center) |
| cy | 1296.0000 px (image center) |
| k1 | -0.6727 |
| k2 | 7.4225 |
| k3 | -22.7947 |
| p1, p2 | 0 (fixed) |
| **Reproj. error** | **3.78 px** |

```
K = [[3593.2937,     0.0,  2304.0],
     [    0.0,  3593.2937,  1296.0],
     [    0.0,      0.0,      1.0]]

dist = [-0.67268898, 7.42247568, 0.0, 0.0, -22.79469409]
```

> Calibration uses the previous ChArUco K (fx=3565.48) as initial guess, refined with 128 individual ArUco marker views. Change from previous: **+0.78%** — excellent consistency.

---

## Plate Center & Orientation

| Parameter | Value |
|-----------|-------|
| **Center (mm)** | **[-9.25, -10.13, 240.45]** |
| **Center (px)** | **(2166, 1145)** |
| **Normal (rot axis)** | **[0.003, 0.637, 0.770]** |
| Plate tilt from camera axis | ~39.6° |

---

## Rotation Step Size

| Parameter | Value |
|-----------|-------|
| **Step size** | **6.130° per frame** |
| Nominal (360/64) | 5.625° |
| Deviation | +0.505° per step (+9.0%) |
| **Total rotation (64 steps)** | **392.3°** (+32.3° overshoot) |
| Trajectory RMSE | 0.83 mm |

### Validation (per-step angles from 3D poses)
- Mean: 6.14° ± 0.31° (std)
- Range: [5.31°, 6.87°]

### Independent 2D Verification (no PnP)
Frame **59** returns to within **34 pixels** of frame 0's marker position, confirming:
- 59 × 6.1° ≈ **360°** ✅
- The step size is definitively **~6.1° per frame**
- 64 steps complete **~392°** (one full revolution + 32°)

---

## Method Summary

1. **Marker detection**: ArUco DICT_4X4_50, IDs 0 & 6 detected in all 64/64 frames
2. **Board model**: Two markers as rigid body with empirically optimized offset (20.51, 20.74) mm
3. **K calibration**: 128 individual marker views, previous K as initial guess, fixed aspect/principal point
4. **Pose estimation**: `solvePnP` with 8-point board model per frame
5. **Trajectory fit**: Least-squares optimization (C_rot, normal, step_size) with multi-start, bounded step ∈ [-10°, 10°]
6. **Validation**: Independent 2D ellipse analysis confirms step size without PnP

---

## Files

| File | Description |
|------|-------------|
| `tools/calibrate_aruco_turntable.py` | Full calibration + rendering pipeline |
| `captures_0726_clay_checkboard_64_calibrated/calibration_results.json` | JSON with all calibration values |
| `captures_0726_clay_checkboard_64_calibrated/*.jpg` | 64 annotated images with center + Z-axis |
