import numpy as np
import sqlite3
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
from PIL import Image
import scipy.ndimage as ndimage
import rembg
import matplotlib.pyplot as plt

workspace = Path("COLMAP/workspace_7-19")
db_path = workspace / "database.db"
raw_dir = Path("captures_7-19")
output_dir = workspace / "visualizations" / "cba_center_pointer"
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Camera Intrinsics
f_val = 3381.185
cx_val = 2276.438
cy_val = 1146.994
K = np.array([
    [f_val, 0, cx_val],
    [0, f_val, cy_val],
    [0, 0, 1]
])
K_inv = np.linalg.inv(K)

print("1. Extracting turntable mask for capture_0.jpg to filter background features...")
img0 = Image.open(raw_dir / "capture_0.jpg")
img0_np = np.array(img0)
session = rembg.new_session("u2net")
rembg_out = rembg.remove(img0, session=session)
alpha0 = np.array(rembg_out.split()[-1])
turntable_mask0 = (alpha0 > 128)

print("2. Loading keypoints and feature tracks from COLMAP database...")
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) // 2147483647
    return image_id1, image_id2

cursor.execute("SELECT image_id, rows, cols, data FROM keypoints")
keypoints = {}
for img_id, rows, cols, blob in cursor.fetchall():
    data = np.frombuffer(blob, dtype=np.float32)
    if len(data) > 0:
        kpts = data.reshape(rows, cols)[:, :2]
        keypoints[img_id] = kpts

cursor.execute("SELECT pair_id, rows, cols, data FROM two_view_geometries WHERE rows > 0")
matches = cursor.fetchall()

parent = {}
def find(i):
    if parent[i] == i: return i
    parent[i] = find(parent[i])
    return parent[i]

def union(i, j):
    root_i, root_j = find(i), find(j)
    if root_i != root_j: parent[root_i] = root_j

for pair_id, rows, cols, blob in matches:
    img_id1, img_id2 = pair_id_to_image_ids(pair_id)
    if len(blob) > 0:
        match_arr = np.frombuffer(blob, dtype=np.uint32).reshape(rows, cols)
        for idx1, idx2 in match_arr:
            feat1, feat2 = (img_id1, idx1), (img_id2, idx2)
            if feat1 not in parent: parent[feat1] = feat1
            if feat2 not in parent: parent[feat2] = feat2
            union(feat1, feat2)

conn.close()

tracks = {}
for feat, root in parent.items():
    root_id = find(root)
    if root_id not in tracks: tracks[root_id] = []
    tracks[root_id].append(feat)

turntable_tracks = []
for t in tracks.values():
    if len(t) >= 10:
        for img_id, kpt_idx in t:
            if img_id == 1:
                u, v = keypoints[1][kpt_idx]
                ix, iy = int(round(u)), int(round(v))
                if 0 <= iy < 2592 and 0 <= ix < 4608 and turntable_mask0[iy, ix]:
                    turntable_tracks.append(t)
                break

selected_tracks = sorted(turntable_tracks, key=lambda t: len(t), reverse=True)[:60]
M_markers = len(selected_tracks)

print(f"Filtered M = {M_markers} turntable feature tracks.")

obs_dict = {}
img0_obs = {}
for j, track in enumerate(selected_tracks):
    for img_id, kpt_idx in track:
        frame_i = img_id - 1
        u_obs, v_obs = keypoints[img_id][kpt_idx]
        obs_dict[(frame_i, j)] = (u_obs, v_obs)
        if frame_i == 0:
            img0_obs[j] = (u_obs, v_obs)

print(f"Total 2D point observations on turntable: {len(obs_dict)}")

# 3. Constrained Bundle Adjustment (CBA) Setup with 3D Points (x_j, y_j, z_j)
N_frames = 32
theta_steps = np.linspace(0, 2 * np.pi, N_frames, endpoint=False)
cz_fixed = 2.60

u_screw, v_screw = 2210.0, 590.0
tx_init = (u_screw - cx_val) / f_val * cz_fixed
ty_init = (v_screw - cy_val) / f_val * cz_fixed

init_rvec = np.array([np.radians(53.7), 0.0, 0.0]) # ~53.7 deg tilt
init_R = Rot.from_rotvec(init_rvec).as_matrix()

init_markers = []
for j in range(M_markers):
    u0, v0 = img0_obs[j]
    ray = K_inv @ np.array([u0, v0, 1.0])
    pts_cam = ray * cz_fixed
    T_init = np.array([tx_init, ty_init, cz_fixed])
    P_local = init_R.T @ (pts_cam - T_init)
    init_markers.extend([P_local[0], P_local[1], P_local[2]])

x0 = np.concatenate([init_rvec, [tx_init, ty_init], init_markers])

print("3. Running Levenberg-Marquardt Constrained Bundle Adjustment with 3D heights...")

def residuals_func(params):
    rvec = params[0:3]
    tx, ty = params[3:5]
    tvec = np.array([tx, ty, cz_fixed])
    R = Rot.from_rotvec(rvec).as_matrix()

    P = np.zeros((M_markers, 3))
    idx = 5
    for j in range(M_markers):
        P[j] = [params[idx], params[idx+1], params[idx+2]]
        idx += 3

    res = []
    for (frame_i, j), (u_obs, v_obs) in obs_dict.items():
        th = theta_steps[frame_i]
        R_z = np.array([
            [np.cos(th), -np.sin(th), 0],
            [np.sin(th),  np.cos(th), 0],
            [0,           0,          1]
        ])

        P_ij = R_z @ P[j]
        X_ij = R @ P_ij + tvec

        u_est = f_val * (X_ij[0] / X_ij[2]) + cx_val
        v_est = f_val * (X_ij[1] / X_ij[2]) + cy_val

        res.append(u_est - u_obs)
        res.append(v_est - v_obs)

    # Prior penalty on rvec to prevent orientation flip
    res.append((rvec[0] - init_rvec[0]) * 100.0)
    res.append((rvec[1] - init_rvec[1]) * 100.0)
    res.append((rvec[2] - init_rvec[2]) * 100.0)

    return np.array(res)

res = least_squares(residuals_func, x0, method='lm', verbose=1)

sol_rvec = res.x[0:3]
sol_tx, sol_ty = res.x[3:5]
sol_tvec = np.array([sol_tx, sol_ty, cz_fixed])
sol_R = Rot.from_rotvec(sol_rvec).as_matrix()

proj_center = K @ sol_tvec
u_center = proj_center[0] / proj_center[2]
v_center = proj_center[1] / proj_center[2]

rmse = np.sqrt(np.mean(res.fun[:-3]**2))
print("\n=== CONSTRAINED BUNDLE ADJUSTMENT RESULTS ===")
print(f"CBA 3D Turntable Center T = [{sol_tvec[0]:.4f}, {sol_tvec[1]:.4f}, {sol_tvec[2]:.4f}]")
print(f"CBA Rotation Vector rvec = [{sol_rvec[0]:.4f}, {sol_rvec[1]:.4f}, {sol_rvec[2]:.4f}]")
print(f"Projected 2D Center     = ({u_center:.2f}, {v_center:.2f})")
print(f"Final Reprojection RMSE  = {rmse:.3f} pixels")

np.savez(workspace / "cba_center_result.npz", T=sol_tvec, R=sol_R, u_center=u_center, v_center=v_center)

# Plot capture_0 visualization
fig, ax = plt.subplots(figsize=(14, 10))
ax.imshow(img0_np)

angles = np.linspace(0, 2 * np.pi, 200)
for j in range(min(25, M_markers)):
    idx = 5 + 3 * j
    P_j = np.array([res.x[idx], res.x[idx+1], res.x[idx+2]])
    r_m = np.linalg.norm(P_j[:2])
    z_m = P_j[2]
    traj_pts = []
    for phi in angles:
        P_phi = np.array([r_m * np.cos(phi), r_m * np.sin(phi), z_m])
        X_phi = sol_R @ P_phi + sol_tvec
        u_t = f_val * (X_phi[0] / X_phi[2]) + cx_val
        v_t = f_val * (X_phi[1] / X_phi[2]) + cy_val
        traj_pts.append([u_t, v_t])
    traj_pts = np.array(traj_pts)
    ax.plot(traj_pts[:, 0], traj_pts[:, 1], 'c-', linewidth=1.5, alpha=0.7)

ax.plot(u_center, v_center, 'r*', markersize=24, label=f'CBA Derived Plate Center ({u_center:.1f}, {v_center:.1f})')
ax.set_title(f"Constrained Bundle Adjustment (CBA) Turntable Center (docs/center_point.md)\nDerived 2D Center: ({u_center:.1f}, {v_center:.1f}) | Reprojection RMSE: {rmse:.2f} px", fontsize=14, fontweight='bold')
ax.legend(fontsize=12)

plt.savefig(output_dir / "cba_derived_center_capture_0.png", bbox_inches='tight', dpi=150)
plt.close()

print(f"Saved visualization: {output_dir / 'cba_derived_center_capture_0.png'}")
