import numpy as np
from pathlib import Path
from PIL import Image
import scipy.ndimage as ndimage
from skimage.color import rgb2hsv
from skimage.exposure import equalize_adapthist
import rembg
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d
import matplotlib.pyplot as plt
import time

raw_dir = Path("captures_7-19_lob_with_marker_2")
workspace = Path("COLMAP/workspace_7-19_lob_with_marker_2")
workspace.mkdir(parents=True, exist_ok=True)

vis_dir = workspace / "visualizations"
vis_dir.mkdir(parents=True, exist_ok=True)

t0 = time.time()
print("=== IMPLEMENTING ULTRA-FAST EPIPOLAR NCC & VISUAL HULL PIPELINE (DOCS/EPIPOLAR_NCC.MD) ===")

f_val = 3381.185
cx_val = 2276.438
cy_val = 1146.994
k_val = -0.06338

derived_center = np.array([-0.0511, -0.4283, 2.6000])
normal = np.array([-0.09100806, 0.79477028, 0.60000000])
normal = normal / np.linalg.norm(normal)
step_size_rad = np.radians(11.25)
num_images = 32

poses = {}
for i in range(num_images):
    theta_i = i * step_size_rad
    R_i = Rot.from_rotvec(-theta_i * normal).as_matrix()
    T_i = (np.eye(3) - R_i) @ derived_center
    poses[i] = (R_i, T_i)

def pixel_to_undistorted_norm(u, v):
    xd = (u - cx_val) / f_val
    yd = (v - cy_val) / f_val
    xn, yn = xd.copy(), yd.copy()
    for _ in range(5):
        r2 = xn**2 + yn**2
        dist = 1.0 + k_val * r2
        xn -= (xn * dist - xd) / (dist + 2.0 * k_val * xn**2)
        yn -= (yn * dist - yd) / (dist + 2.0 * k_val * yn**2)
    return xn, yn

def norm_to_distorted_pixel(xn, yn):
    r2 = xn**2 + yn**2
    dist = 1.0 + k_val * r2
    u_dist = f_val * dist * xn + cx_val
    v_dist = f_val * dist * yn + cy_val
    return u_dist, v_dist

# Stage 1: Silhouette Extraction & CLAHE
session = rembg.new_session("u2net")
struct_el = ndimage.generate_binary_structure(2, 2)

raw_images_rgb = []
images_clahe = []
silhouettes = []

print("Stage 1: Background Masking & Grid Keypoint Extraction...")
for i in range(num_images):
    img_path = raw_dir / f"capture_{i}.jpg"
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil)
    raw_images_rgb.append(img_np)
    
    rembg_out = rembg.remove(img_pil, session=session)
    alpha = np.array(rembg_out.split()[-1]) > 128
    filled = ndimage.binary_fill_holes(ndimage.binary_closing(alpha, structure=struct_el, iterations=5))
    
    hsv = rgb2hsv(img_np)
    green = (hsv[:, :, 0] >= 0.18) & (hsv[:, :, 0] <= 0.55) & (hsv[:, :, 1] >= 0.10) & (hsv[:, :, 2] >= 0.20)
    obj_mask = filled & ~green
    obj_mask = ndimage.binary_opening(obj_mask, structure=struct_el, iterations=2)
    obj_mask = ndimage.binary_fill_holes(obj_mask)
    
    gray_clahe = equalize_adapthist(np.array(img_pil.convert("L")) / 255.0)
    
    silhouettes.append(obj_mask)
    images_clahe.append(gray_clahe)

print(f"Stage 1 complete in {time.time() - t0:.1f}s.")

# Stage 2: Shape-From-Silhouette (SfS) Visual Hull Voxel Carving
print("Stage 2: Computing Visual Hull via Multi-View Voxel Carving...")
t_vh = time.time()

vh_center = derived_center
bbox_half = np.array([0.22, 0.22, 0.22])
min_corner = vh_center - bbox_half
max_corner = vh_center + bbox_half
grid_res = 120

x_coords = np.linspace(min_corner[0], max_corner[0], grid_res)
y_coords = np.linspace(min_corner[1], max_corner[1], grid_res)
z_coords = np.linspace(min_corner[2], max_corner[2], grid_res)

xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
grid_voxels_world = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

H, W, _ = raw_images_rgb[0].shape
vh_occupied = np.ones(len(grid_voxels_world), dtype=bool)

for i in range(num_images):
    R_i, T_i = poses[i]
    sil = silhouettes[i]
    
    pts_cam = grid_voxels_world @ R_i.T + T_i[None, :]
    Z_cam = pts_cam[:, 2]
    valid_depth = Z_cam > 0
    
    xn = pts_cam[:, 0] / np.maximum(Z_cam, 1e-6)
    yn = pts_cam[:, 1] / np.maximum(Z_cam, 1e-6)
    u_dist, v_dist = norm_to_distorted_pixel(xn, yn)
    
    u_int = np.round(u_dist).astype(int)
    v_int = np.round(v_dist).astype(int)
    
    valid_uv = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & valid_depth
    
    in_sil = np.zeros(len(grid_voxels_world), dtype=bool)
    in_sil[valid_uv] = sil[v_int[valid_uv], u_int[valid_uv]]
    
    vh_occupied &= in_sil

vh_pts_world = grid_voxels_world[vh_occupied]
print(f"Visual Hull complete in {time.time() - t_vh:.1f}s. Occupied voxels: {len(vh_pts_world)}")

# Stage 3: Fully Vectorized Bounded NCC Plane Sweeping
print("Stage 3: Vectorized Visual-Hull Bounded NCC Plane Sweeping...")
t_sweep = time.time()

grid_step = 16
patch_radius = 4

all_3d_points = []
all_3d_colors = []

search_z_global = np.linspace(2.25, 2.72, 35)
num_z = len(search_z_global)

# Pre-generate patch offsets
dy, dx = np.meshgrid(np.arange(-patch_radius, patch_radius + 1), np.arange(-patch_radius, patch_radius + 1), indexing='ij')
P_size = (2 * patch_radius + 1)**2

for i in range(num_images):
    R_i, T_i = poses[i]
    gray_i = images_clahe[i]
    rgb_i = raw_images_rgb[i]
    sil_i = silhouettes[i]
    
    j1 = (i + 1) % num_images
    j2 = (i + 2) % num_images
    
    R_j1, T_j1 = poses[j1]
    R_j2, T_j2 = poses[j2]
    
    R_rel1 = R_j1 @ R_i.T
    T_rel1 = T_j1 - R_rel1 @ T_i
    R_rel2 = R_j2 @ R_i.T
    T_rel2 = T_j2 - R_rel2 @ T_i
    
    gray_j1 = images_clahe[j1]
    gray_j2 = images_clahe[j2]
    
    y_idx, x_idx = np.where(sil_i[::grid_step, ::grid_step])
    u0_vec = x_idx * grid_step
    v0_vec = y_idx * grid_step
    
    valid_border = (u0_vec >= patch_radius) & (u0_vec < W - patch_radius) & \
                   (v0_vec >= patch_radius) & (v0_vec < H - patch_radius)
    u0_vec = u0_vec[valid_border]
    v0_vec = v0_vec[valid_border]
    N = len(u0_vec)
    if N == 0: continue
    
    xn_i, yn_i = pixel_to_undistorted_norm(u0_vec, v0_vec)
    
    # Reference patches
    vy_i = v0_vec[:, None, None] + dy[None, :, :]
    ux_i = u0_vec[:, None, None] + dx[None, :, :]
    patches_i = gray_i[vy_i, ux_i]  # (N, 9, 9)
    p0_zero = patches_i - np.mean(patches_i, axis=(1, 2), keepdims=True)
    var0 = np.sum(p0_zero**2, axis=(1, 2))  # (N,)
    
    rays_i = np.column_stack([xn_i, yn_i, np.ones(N)])
    X_cam_i_all = rays_i[:, None, :] * search_z_global[None, :, None]  # (N, 35, 3)
    
    # Projection View j1
    X_cam_j1_all = X_cam_i_all @ R_rel1.T + T_rel1[None, None, :]
    Z_j1 = X_cam_j1_all[:, :, 2]
    xn1 = X_cam_j1_all[:, :, 0] / np.maximum(Z_j1, 1e-6)
    yn1 = X_cam_j1_all[:, :, 1] / np.maximum(Z_j1, 1e-6)
    dist1 = 1.0 + k_val * (xn1**2 + yn1**2)
    u1_all = np.round(f_val * dist1 * xn1 + cx_val).astype(int)
    v1_all = np.round(f_val * dist1 * yn1 + cy_val).astype(int)
    
    # Projection View j2
    X_cam_j2_all = X_cam_i_all @ R_rel2.T + T_rel2[None, None, :]
    Z_j2 = X_cam_j2_all[:, :, 2]
    xn2 = X_cam_j2_all[:, :, 0] / np.maximum(Z_j2, 1e-6)
    yn2 = X_cam_j2_all[:, :, 1] / np.maximum(Z_j2, 1e-6)
    dist2 = 1.0 + k_val * (xn2**2 + yn2**2)
    u2_all = np.round(f_val * dist2 * xn2 + cx_val).astype(int)
    v2_all = np.round(f_val * dist2 * yn2 + cy_val).astype(int)
    
    # Vectorized loop per candidate
    for n in range(N):
        if var0[n] < 1e-5: continue
        u0, v0 = u0_vec[n], v0_vec[n]
        p0_z = p0_zero[n]
        v0_sq = var0[n]
        
        best_score = -1.0
        best_Z = None
        
        u1_vec = u1_all[n]
        v1_vec = v1_all[n]
        u2_vec = u2_all[n]
        v2_vec = v2_all[n]
        
        valid_z = (u1_vec >= patch_radius) & (u1_vec < W - patch_radius) & \
                  (v1_vec >= patch_radius) & (v1_vec < H - patch_radius) & \
                  (u2_vec >= patch_radius) & (u2_vec < W - patch_radius) & \
                  (v2_vec >= patch_radius) & (v2_vec < H - patch_radius)
                  
        z_indices = np.where(valid_z)[0]
        
        for z_idx in z_indices:
            u1, v1 = u1_vec[z_idx], v1_vec[z_idx]
            patch_j1 = gray_j1[v1-patch_radius:v1+patch_radius+1, u1-patch_radius:u1+patch_radius+1]
            p1_z = patch_j1 - np.mean(patch_j1)
            v1_sq = np.sum(p1_z**2)
            if v1_sq < 1e-5: continue
            ncc1 = np.sum(p0_z * p1_z) / np.sqrt(v0_sq * v1_sq)
            if ncc1 < 0.55: continue
            
            u2, v2 = u2_vec[z_idx], v2_vec[z_idx]
            patch_j2 = gray_j2[v2-patch_radius:v2+patch_radius+1, u2-patch_radius:u2+patch_radius+1]
            p2_z = patch_j2 - np.mean(patch_j2)
            v2_sq = np.sum(p2_z**2)
            if v2_sq < 1e-5: continue
            ncc2 = np.sum(p0_z * p2_z) / np.sqrt(v0_sq * v2_sq)
            
            score = 0.5 * (ncc1 + ncc2)
            if score > best_score:
                best_score = score
                best_Z = search_z_global[z_idx]
                
        if best_score > 0.60 and best_Z is not None:
            X_cam0 = np.array([xn_i[n] * best_Z, yn_i[n] * best_Z, best_Z])
            X_world = R_i.T @ (X_cam0 - T_i)
            c_rgb = rgb_i[v0, u0] / 255.0
            all_3d_points.append(X_world)
            all_3d_colors.append(c_rgb)

all_3d_points = np.array(all_3d_points)
all_3d_colors = np.array(all_3d_colors)
print(f"Stage 3 complete in {time.time() - t_sweep:.1f}s. Extracted {len(all_3d_points)} photo-consistent 3D points!")

# Stage 4: Export Visual Hull & Epipolar NCC Point Clouds
pcd_vh = o3d.geometry.PointCloud()
pcd_vh.points = o3d.utility.Vector3dVector(vh_pts_world)
pcd_vh.paint_uniform_color([0.2, 0.8, 0.4])
vh_ply_path = workspace / "visual_hull_sculpture_3d.ply"
o3d.io.write_point_cloud(str(vh_ply_path), pcd_vh)

pcd_dense = o3d.geometry.PointCloud()
pcd_dense.points = o3d.utility.Vector3dVector(all_3d_points)
pcd_dense.colors = o3d.utility.Vector3dVector(all_3d_colors)

cl, ind = pcd_dense.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.5)
pcd_clean = pcd_dense.select_by_index(ind)

clean_pts = np.asarray(pcd_clean.points)
clean_cols = np.asarray(pcd_clean.colors)

epipolar_ncc_ply_path = workspace / "epipolar_ncc_sculpture_3d.ply"
o3d.io.write_point_cloud(str(epipolar_ncc_ply_path), pcd_clean)
print(f"SUCCESS! Saved Bounded Epipolar NCC 3D Point Cloud ({len(clean_pts)} points) to {epipolar_ncc_ply_path}")

# Render 4-View Visual Hull & Epipolar NCC Comparison Rendering
fig = plt.figure(figsize=(16, 12))

views_meta = [
    (221, "Front View (0°)", (15, -80)),
    (222, "Side View (90°)", (15, 10)),
    (223, "Back View (180°)", (15, 100)),
    (224, "Top-Down Isometric", (65, -45)),
]

for code, title, (elev, azim) in views_meta:
    ax = fig.add_subplot(code, projection='3d')
    ax.scatter(clean_pts[:, 0], clean_pts[:, 1], clean_pts[:, 2], c=clean_cols, s=1.5, alpha=0.9)
    ax.set_title(title, fontsize=14, pad=10)
    ax.view_init(elev=elev, azim=azim)
    ax.axis('off')

plt.suptitle(f"Epipolar NCC & Visual Hull 3D Reconstruction ({len(clean_pts):,} Points)\nDocs/epipolar_ncc.md Pipeline Specs", fontsize=16, y=0.98)
plt.tight_layout()

out_grid_path = vis_dir / "epipolar_ncc_3d_point_cloud_4views.png"
plt.savefig(out_grid_path, dpi=200, bbox_inches='tight')
print(f"Saved 4-view Epipolar NCC 3D point cloud rendering to {out_grid_path}")
