import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import scipy.ndimage as ndimage
from skimage.color import rgb2hsv
from skimage.exposure import equalize_adapthist
import rembg
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d
import time

raw_dir = Path("captures_7-19_lob_with_marker_2")
workspace = Path("COLMAP/workspace_7-19_lob_with_marker_2")
workspace.mkdir(parents=True, exist_ok=True)

vis_dir = workspace / "visualizations"
vis_dir.mkdir(parents=True, exist_ok=True)

t0 = time.time()
print("=== STARTING FAST VECTORIZED DENSE GRID RECONSTRUCTION ON 32 CAPTURES ===")

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

session = rembg.new_session("u2net")
struct_el = ndimage.generate_binary_structure(2, 2)
grid_step = 12  # 12px uniform grid spacing (~5,500 grid points / image)

images_clahe = []
grid_kpts_per_image = []

print("Step 1: Fast Background Masking & Keypoint Extraction...")
for i in range(num_images):
    img_path = raw_dir / f"capture_{i}.jpg"
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil)
    
    rembg_out = rembg.remove(img_pil, session=session)
    alpha = np.array(rembg_out.split()[-1]) > 128
    filled = ndimage.binary_fill_holes(ndimage.binary_closing(alpha, structure=struct_el, iterations=5))
    
    hsv = rgb2hsv(img_np)
    green = (hsv[:, :, 0] >= 0.18) & (hsv[:, :, 0] <= 0.55) & (hsv[:, :, 1] >= 0.10) & (hsv[:, :, 2] >= 0.20)
    obj_mask = filled & ~green
    obj_mask = ndimage.binary_opening(obj_mask, structure=struct_el, iterations=2)
    obj_mask = ndimage.binary_fill_holes(obj_mask)
    
    gray_clahe = equalize_adapthist(np.array(img_pil.convert("L")) / 255.0)
    
    y_idx, x_idx = np.where(obj_mask[::grid_step, ::grid_step])
    u0 = x_idx * grid_step
    v0 = y_idx * grid_step
    kpts = np.column_stack([u0, v0])
    
    images_clahe.append(gray_clahe)
    grid_kpts_per_image.append(kpts)

print(f"Masking complete in {time.time() - t0:.1f}s.")

# Vectorized Epipolar Matcher across adjacent pairs
patch_radius = 4
patch_size = 2 * patch_radius + 1  # 9x9 patch
search_range_z = np.linspace(2.20, 2.75, 50)  # 50 depth planes

all_3d_points = []
all_3d_colors = []

img0_rgb = np.array(Image.open(raw_dir / "capture_0.jpg").convert("RGB"))
H, W, _ = img0_rgb.shape

print("Step 2: Vectorized Epipolar Patch Matching across 32 sequential pairs...")
t_match = time.time()

for i in range(num_images):
    j = (i + 1) % num_images
    
    R_i, T_i = poses[i]
    R_j, T_j = poses[j]
    
    R_rel = R_j @ R_i.T
    T_rel = T_j - R_rel @ T_i
    
    gray_i = images_clahe[i]
    gray_j = images_clahe[j]
    kpts_i = grid_kpts_per_image[i]
    
    # Filter points near borders
    valid_mask = (kpts_i[:, 0] >= patch_radius) & (kpts_i[:, 0] < W - patch_radius) & \
                 (kpts_i[:, 1] >= patch_radius) & (kpts_i[:, 1] < H - patch_radius)
    kpts_valid = kpts_i[valid_mask]
    N = len(kpts_valid)
    if N == 0: continue
    
    u0_vec = kpts_valid[:, 0]
    v0_vec = kpts_valid[:, 1]
    
    # Extract 9x9 patches for view i: shape (N, 9, 9)
    dy, dx = np.meshgrid(np.arange(-patch_radius, patch_radius + 1), np.arange(-patch_radius, patch_radius + 1), indexing='ij')
    
    vy_i = v0_vec[:, None, None] + dy[None, :, :]
    ux_i = u0_vec[:, None, None] + dx[None, :, :]
    patches_i = gray_i[vy_i, ux_i]  # (N, 9, 9)
    
    p0_zero = patches_i - np.mean(patches_i, axis=(1, 2), keepdims=True)
    var0 = np.sum(p0_zero**2, axis=(1, 2))  # (N,)
    valid_var = var0 > 1e-5
    
    # Compute 3D hypotheses for all depth planes Z: shape (N, 50, 3)
    xn_i = (u0_vec - cx_val) / f_val
    yn_i = (v0_vec - cy_val) / f_val
    
    # Rays: shape (N, 3)
    rays_i = np.column_stack([xn_i, yn_i, np.ones(N)])
    
    # X_cam_i for all Z: shape (N, 50, 3)
    X_cam_i_all = rays_i[:, None, :] * search_range_z[None, :, None]
    
    # Transform to View j: X_cam_j = X_cam_i @ R_rel.T + T_rel
    X_cam_j_all = X_cam_i_all @ R_rel.T + T_rel[None, None, :]  # (N, 50, 3)
    
    Z_j = X_cam_j_all[:, :, 2]
    valid_depth = Z_j > 0
    
    xn1 = X_cam_j_all[:, :, 0] / np.maximum(Z_j, 1e-6)
    yn1 = X_cam_j_all[:, :, 1] / np.maximum(Z_j, 1e-6)
    dist1 = 1.0 + k_val * (xn1**2 + yn1**2)
    
    u1_all = np.round(f_val * dist1 * xn1 + cx_val).astype(int)  # (N, 50)
    v1_all = np.round(f_val * dist1 * yn1 + cy_val).astype(int)  # (N, 50)
    
    valid_uv1 = (u1_all >= patch_radius) & (u1_all < W - patch_radius) & \
                (v1_all >= patch_radius) & (v1_all < H - patch_radius) & valid_depth
                
    # Evaluate patch NCC for each valid depth hypothesis
    for n in range(N):
        if not valid_var[n]: continue
        
        u0, v0 = u0_vec[n], v0_vec[n]
        p0_z = p0_zero[n]
        v0_sq = var0[n]
        
        best_ncc = -1.0
        best_Z = None
        
        for z_idx in range(len(search_range_z)):
            if not valid_uv1[n, z_idx]: continue
            
            u1 = u1_all[n, z_idx]
            v1 = v1_all[n, z_idx]
            
            patch_j = gray_j[v1-patch_radius:v1+patch_radius+1, u1-patch_radius:u1+patch_radius+1]
            p1_z = patch_j - np.mean(patch_j)
            v1_sq = np.sum(p1_z**2)
            if v1_sq < 1e-5: continue
            
            ncc = np.sum(p0_z * p1_z) / np.sqrt(v0_sq * v1_sq)
            if ncc > best_ncc:
                best_ncc = ncc
                best_Z = search_range_z[z_idx]
                
        if best_ncc > 0.55 and best_Z is not None:
            X_cam0 = np.array([xn_i[n] * best_Z, yn_i[n] * best_Z, best_Z])
            X_world = R_i.T @ (X_cam0 - T_i)
            all_3d_points.append(X_world)
            
            if i == 0:
                c = img0_rgb[v0, u0] / 255.0
            else:
                c = [0.2, 0.7, 0.9]
            all_3d_colors.append(c)

all_3d_points = np.array(all_3d_points)
all_3d_colors = np.array(all_3d_colors)
print(f"Matching complete in {time.time() - t_match:.1f}s. Extracted {len(all_3d_points)} 3D surface points!")

# Clean 3D Point Cloud via Open3D Statistical Outlier Removal
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_3d_points)
pcd.colors = o3d.utility.Vector3dVector(all_3d_colors)

cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.8)
pcd_clean = pcd.select_by_index(ind)

final_ply_path = workspace / "dense_grid_sculpture_3d.ply"
o3d.io.write_point_cloud(str(final_ply_path), pcd_clean)
print(f"SUCCESS! Saved clean 3D dense point cloud ({len(pcd_clean.points)} points) to {final_ply_path}")

# Step 3: Render 3D Point Cloud Overlay onto capture_0.jpg
clean_pts = np.asarray(pcd_clean.points)
img0_overlay = Image.open(raw_dir / "capture_0.jpg").convert("RGB")
draw = ImageDraw.Draw(img0_overlay)

proj_count = 0
for pt in clean_pts:
    X, Y, Z = pt
    if Z <= 0: continue
    xn, yn = X / Z, Y / Z
    dist = 1.0 + k_val * (xn**2 + yn**2)
    u_proj = int(round(f_val * dist * xn + cx_val))
    v_proj = int(round(f_val * dist * yn + cy_val))
    
    if 0 <= u_proj < W and 0 <= v_proj < H:
        draw.ellipse((u_proj - 2, v_proj - 2, u_proj + 2, v_proj + 2), fill="lime", outline="darkgreen")
        proj_count += 1

overlay_out_path = vis_dir / "dense_3d_pcd_overlay_capture_0.jpg"
img0_overlay.save(overlay_out_path, quality=95)
print(f"Saved 3D Point Cloud Overlay onto capture_0.jpg ({proj_count} points) to {overlay_out_path}")
