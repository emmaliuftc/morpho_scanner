#!/usr/bin/env python3
"""
Generate high-resolution geometric and visual assets for docs/calibration.md.
"""

import os
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rot

os.makedirs("docs/images/calibration", exist_ok=True)

# Load Golden Calibration
with open("captures_8-13_calibration_results/calibration.json") as f:
    calib = json.load(f)

K = np.array(calib["camera_matrix_K"])
dist = np.array(calib["distortion_coefficients"])
C_rot = np.array(calib["plate_center_mm"])
normal = np.array(calib["plate_normal"])
normal = normal / np.linalg.norm(normal)
dist_mm = calib["camera_to_plate_center_distance_mm"]
step_deg = calib["step_size_deg"]
rmse_mm = calib["trajectory_rmse_mm"]

tvecs = []
for frame in calib["frames"]:
    if "board_origin_tvec" in frame:
        tvecs.append(frame["board_origin_tvec"])
tvecs = np.array(tvecs)

# -------------------------------------------------------------
# 1. Figure 1: 3D Turntable Orbit & Camera Geometry
# -------------------------------------------------------------
print("Generating Figure 1: 3D Turntable Orbit & Camera Geometry...")
fig = plt.figure(figsize=(16, 8), facecolor='#0f172a')

# 3D Trajectory View
ax3d = fig.add_subplot(1, 2, 1, projection='3d', facecolor='#0f172a')

# Camera Optical Origin
ax3d.scatter([0], [0], [0], color='#ef4444', s=120, label='Camera Focal Point (0,0,0)', zorder=10)
# Camera Optical Axis Line
ax3d.plot([0, 0], [0, 0], [0, 185], color='#94a3b8', linestyle=':', lw=1.5, label='Optical Axis (Zc)')

# Center of Turntable Plate
ax3d.scatter([C_rot[0]], [C_rot[1]], [C_rot[2]], color='#10b981', s=160, label=f'Plate Center C_rot\n(-3.86, -8.19, 173.86) mm', zorder=10)

# Distance Line
ax3d.plot([0, C_rot[0]], [0, C_rot[1]], [0, C_rot[2]], color='#f59e0b', linestyle='--', lw=2.5,
          label=f'Metric Baseline: 174.10 mm (17.41 cm)')

# Normal vector arrow
arrow_len = 50.0
ax3d.quiver(C_rot[0], C_rot[1], C_rot[2],
            normal[0]*arrow_len, normal[1]*arrow_len, normal[2]*arrow_len,
            color='#38bdf8', lw=3, arrow_length_ratio=0.15, label='Turntable Normal n (Axis)')

# ChArUco 64 detected points
ax3d.scatter(tvecs[:, 0], tvecs[:, 1], tvecs[:, 2], c='#a855f7', s=35, alpha=0.9, label=f'64 ChArUco Poses (RMSE: 0.29 mm)')

# Fitted circular trajectory line
ref_u = np.cross(normal, [1, 0, 0]) if abs(normal[0]) < 0.9 else np.cross(normal, [0, 1, 0])
ref_u = ref_u / np.linalg.norm(ref_u)
ref_v = np.cross(normal, ref_u)
theta_vals = np.linspace(0, 2*np.pi, 200)
orbit_radius = np.mean([np.linalg.norm(t - C_rot - np.dot(t - C_rot, normal)*normal) for t in tvecs])
circle_pts = np.array([C_rot + orbit_radius * (np.cos(th)*ref_u + np.sin(th)*ref_v) for th in theta_vals])
ax3d.plot(circle_pts[:, 0], circle_pts[:, 1], circle_pts[:, 2], color='#c084fc', lw=2, linestyle='-')

# Circular Turntable Disc Mesh
disc_radius = 85.0
theta_disc = np.linspace(0, 2*np.pi, 60)
r_disc = np.linspace(0, disc_radius, 20)
T_mesh, R_mesh = np.meshgrid(theta_disc, r_disc)
disc_x = C_rot[0] + R_mesh * (np.cos(T_mesh)*ref_u[0] + np.sin(T_mesh)*ref_v[0])
disc_y = C_rot[1] + R_mesh * (np.cos(T_mesh)*ref_u[1] + np.sin(T_mesh)*ref_v[1])
disc_z = C_rot[2] + R_mesh * (np.cos(T_mesh)*ref_u[2] + np.sin(T_mesh)*ref_v[2])
ax3d.plot_surface(disc_x, disc_y, disc_z, color='#0284c7', alpha=0.15, edgecolor='none')

ax3d.set_xlabel('Xc (mm)', color='#94a3b8', labelpad=10)
ax3d.set_ylabel('Yc (mm)', color='#94a3b8', labelpad=10)
ax3d.set_zlabel('Zc (mm)', color='#94a3b8', labelpad=10)
ax3d.set_title("3D Camera-to-Turntable Spatial Geometry", color='white', fontsize=14, fontweight='bold', pad=15)
ax3d.view_init(elev=22, azim=-68)
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.tick_params(colors='#94a3b8')
ax3d.legend(loc='upper left', facecolor='#1e293b', edgecolor='#475569', labelcolor='white', fontsize=8.5)

# Subplot 2: 2D Elevation View (Side Projection Yc vs Zc)
ax2d = fig.add_subplot(1, 2, 2, facecolor='#1e293b')

# Plot Camera Origin
ax2d.scatter([0], [0], color='#ef4444', s=160, zorder=5, label='Camera Center (0,0)')
# Optical axis horizontal line
ax2d.axhline(0, color='#94a3b8', linestyle=':', lw=1.5, label='Optical Principal Axis (Yc = 0)')

# Turntable Center
ax2d.scatter([C_rot[2]], [C_rot[1]], color='#10b981', s=200, zorder=5, label=f'Plate Center (Z={C_rot[2]:.1f}, Y={C_rot[1]:.1f})')

# Baseline connecting line
ax2d.plot([0, C_rot[2]], [0, C_rot[1]], color='#f59e0b', linestyle='--', lw=2.5,
          label=f'Total Vector Distance: {dist_mm:.2f} mm ({dist_mm/10.0:.2f} cm)')

# Turntable surface line (slope from normal)
# normal = [nx, ny, nz], in Y-Z plane tangent slope is -nz / ny
tangent_len = 80.0
tan_z = tangent_len * normal[1]
tan_y = -tangent_len * normal[2]
ax2d.plot([C_rot[2] - tan_z, C_rot[2] + tan_z], [C_rot[1] - tan_y, C_rot[1] + tan_y],
          color='#38bdf8', lw=3, label='Turntable Plate Surface (Plane Z=0)')

# Normal vector
ax2d.arrow(C_rot[2], C_rot[1], normal[2]*40, normal[1]*40, color='#0284c7', width=1.0, head_width=3.5, length_includes_head=True)
ax2d.text(C_rot[2] + normal[2]*45, C_rot[1] + normal[1]*45, 'Normal n\n(Tilt 39.8°)', color='#38bdf8', fontsize=10, fontweight='bold')

# ChArUco poses side view
ax2d.scatter(tvecs[:, 2], tvecs[:, 1], c='#c084fc', s=30, alpha=0.8, label='ChArUco Poses')

# Annotations
tilt_angle_deg = np.rad2deg(np.arccos(normal[2]))
ax2d.text(60, -40, f"Elevation Tilt Angle: {tilt_angle_deg:.2f}°\n"
                    f"Camera-to-Plate Distance: {dist_mm:.2f} mm\n"
                    f"Trajectory Residual RMSE: {rmse_mm:.3f} mm\n"
                    f"Motor Step: {step_deg:.3f}°/frame (64 frames)",
          color='white', fontsize=11, family='monospace',
          bbox=dict(boxstyle='round,pad=0.6', facecolor='#0f172a', edgecolor='#10b981', lw=1.5))

ax2d.set_xlabel('Depth Along Optical Axis Zc (mm)', color='#94a3b8', fontsize=11, labelpad=8)
ax2d.set_ylabel('Vertical Offset Yc (mm)', color='#94a3b8', fontsize=11, labelpad=8)
ax2d.set_title("Side-Elevation Projection (Yc vs Zc Plane)", color='white', fontsize=14, fontweight='bold', pad=15)
ax2d.grid(True, linestyle='--', alpha=0.3, color='#64748b')
ax2d.tick_params(colors='#94a3b8')
ax2d.legend(loc='lower left', facecolor='#0f172a', edgecolor='#475569', labelcolor='white', fontsize=9)

plt.suptitle("Golden Calibration Metrology: Rigid Recovery of the 17.41 cm Baseline\nSub-Millimeter Turntable Trajectory and Metric Frame Identification",
             color='white', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig("docs/images/calibration/calibration_3d_orbit_geometry.png", dpi=200, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("Saved docs/images/calibration/calibration_3d_orbit_geometry.png")


# -------------------------------------------------------------
# 2. Figure 2: Real Turntable Frame Annotated Projection
# -------------------------------------------------------------
print("Generating Figure 2: Real Turntable Frame Annotated Projection...")
frame0 = cv2.imread("captures_8-13_calibration_results/capture_0.jpg")
frame0_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)

# Project C_rot and normal arrow
rvec0, tvec0 = np.zeros((3, 1)), np.zeros((3, 1))
center_2d, _ = cv2.projectPoints(C_rot.reshape(1, 1, 3).astype(np.float64), rvec0, tvec0, K, dist)
cx, cy = int(round(center_2d[0][0][0])), int(round(center_2d[0][0][1]))

arrow_3d = (C_rot + 70 * normal).reshape(1, 1, 3).astype(np.float64)
arrow_2d, _ = cv2.projectPoints(arrow_3d, rvec0, tvec0, K, dist)
ax_pt, ay_pt = int(round(arrow_2d[0][0][0])), int(round(arrow_2d[0][0][1]))

# Crop center region (around cx, cy)
crop_w, crop_h = 2400, 1600
x1 = max(0, cx - crop_w // 2)
y1 = max(0, cy - crop_h // 2)
x2 = min(frame0_rgb.shape[1], x1 + crop_w)
y2 = min(frame0_rgb.shape[0], y1 + crop_h)

crop_img = frame0_rgb[y1:y2, x1:x2].copy()

# Adjust coordinates to crop
cx_c = cx - x1
cy_c = cy - y1
ax_c = ax_pt - x1
ay_c = ay_pt - y1

fig, ax = plt.subplots(figsize=(14, 9), facecolor='#0f172a')
ax.imshow(crop_img)

# Annotate on crop
ax.annotate("", xy=(ax_c, ay_c), xytext=(cx_c, cy_c),
            arrowprops=dict(arrowstyle="->,head_width=0.8,head_length=1.2", color='#38bdf8', lw=4))
ax.text(ax_c + 25, ay_c - 15, "Normal Vector n\n(Z=0 Axle)", color='#38bdf8', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f172a', edgecolor='#38bdf8', lw=1.5))

# Target crosshairs
ax.plot([cx_c - 70, cx_c + 70], [cy_c, cy_c], color='#10b981', lw=3)
ax.plot([cx_c, cx_c], [cy_c - 70, cy_c + 70], color='#10b981', lw=3)
circle = plt.Circle((cx_c, cy_c), 50, color='#10b981', fill=False, lw=2.5)
ax.add_patch(circle)
ax.scatter([cx_c], [cy_c], color='#ef4444', s=60, zorder=6)

ax.text(cx_c + 35, cy_c + 55, f"Plate Center C_rot\nPixel: ({cx}, {cy})\nDistance: {dist_mm:.1f} mm ({dist_mm/10.0:.2f} cm)",
        color='#10b981', fontsize=13, fontweight='bold', family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f172a', edgecolor='#10b981', lw=1.5))

# Info banner on top
banner_text = (
    f"GOLDEN CALIBRATION VERIFICATION (Frame 0)\n"
    f"• Camera Intrinsics: fx=fy=3634.69 px, cx=2304.0, cy=1296.0 (Sub-Pixel Reprojection)\n"
    f"• 3D Plate Center C_rot: [-3.86, -8.19, 173.86] mm\n"
    f"• Camera-to-Plate Distance: {dist_mm:.2f} mm (17.41 cm) | Tilt: {tilt_angle_deg:.2f}°\n"
    f"• Trajectory Fitting RMSE: {rmse_mm:.3f} mm across 64 steps (6.35°/step)"
)
ax.text(0.02, 0.96, banner_text, transform=ax.transAxes, va='top',
        fontsize=12, color='white', family='monospace',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#0f172a', edgecolor='#38bdf8', lw=2, alpha=0.9))

ax.axis('off')
plt.title("Physical ChArUco Board Verification: Re-Projected Plate Center & Rotation Normal",
          color='white', fontsize=15, fontweight='bold', pad=12)
plt.tight_layout()
fig.savefig("docs/images/calibration/calibration_frame_projection_annotated.png", dpi=200, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("Saved docs/images/calibration/calibration_frame_projection_annotated.png")

print("All calibration visuals generated successfully!")
