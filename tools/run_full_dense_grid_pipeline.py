import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import scipy.ndimage as ndimage
from skimage.color import rgb2hsv
from skimage.exposure import equalize_adapthist
import rembg
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d
from multiprocessing import Pool, cpu_count
import time

raw_dir = Path("captures_7-19_lob_with_marker_2")
workspace = Path("COLMAP/workspace_7-19_lob_with_marker_2")
workspace.mkdir(parents=True, exist_ok=True)

vis_dir = workspace / "visualizations"
vis_dir.mkdir(parents=True, exist_ok=True)

f_val = 3381.185
cx_val = 2276.438
cy_val = 1146.994
k_val = -0.06338

K = np.array([
    [f_val, 0, cx_val],
    [0, f_val, cy_val],
    [0, 0, 1]
])

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

def process_pair(args):
    i, j, gray_i, gray_j, kpts_i, R_i, T_i, R_j, T_j = args
    H, W = gray_i.shape
    
    R_rel = R_j @ R_i.T
    T_rel = T_j - R_rel @ T_i
    
    patch_radius = 4
    search_range_z = np.linspace(2.20, 2.75, 60)
    
    pair_3d = []
    
    for pt in kpts_i:
        u0, v0 = int(pt[0]), int(pt[1])
        if u0 - patch_radius < 0 or u0 + patch_radius >= W or v0 - patch_radius < 0 or v0 + patch_radius >= H:
            continue
            
        patch_i = gray_i[v0-patch_radius:v0+patch_radius+1, u0-patch_radius:u0+patch_radius+1]
        p0_zero = patch_i - np.mean(patch_i)
        var0 = np.sum(p0_zero**2)
        if var0 < 1e-5: continue
        
        best_ncc = -1.0
        best_X_world = None
        
        for Z in search_range_z:
            xn = (u0 - cx_val) / f_val
            yn = (v0 - cy_val) / f_val
            X_cam_i = np.array([xn * Z, yn * Z, Z])
            
            X_cam_j = R_rel @ X_cam_i + T_rel
            if X_cam_j[2] <= 0: continue
            
            xn1, yn1 = X_cam_j[0] / X_cam_j[2], X_cam_j[1] / X_cam_j[2]
            dist1 = 1.0 + k_val * (xn1**2 + yn1**2)
            u1_cand = int(round(f_val * dist1 * xn1 + cx_val))
            v1_cand = int(round(f_val * dist1 * yn1 + cy_val))
            
            if u1_cand - patch_radius < 0 or u1_cand + patch_radius >= W or v1_cand - patch_radius < 0 or v1_cand + patch_radius >= H:
                continue
                
            patch_j = gray_j[v1_cand-patch_radius:v1_cand+patch_radius+1, u1_cand-patch_radius:u1_cand+patch_radius+1]
            p1_zero = patch_j - np.mean(patch_j)
            var1 = np.sum(p1_zero**2)
            if var1 < 1e-5: continue
            
            ncc = np.sum(p0_zero * p1_zero) / np.sqrt(var0 * var1)
            if ncc > best_ncc:
                best_ncc = ncc
                best_X_world = R_i.T @ (X_cam_i - T_i)
                
        if best_ncc > 0.58 and best_X_world is not None:
            pair_3d.append(best_X_world)
            
    return pair_3d

if __name__ == "__main__":
    t0 = time.time()
    print(f"=== STARTING PARALLEL DENSE GRID RECONSTRUCTION ON {cpu_count()} CPU CORES ===")
    
    session = rembg.new_session("u2net")
    struct_el = ndimage.generate_binary_structure(2, 2)
    grid_step = 10
    
    images_clahe = []
    images_mask = []
    grid_kpts_per_image = []
    
    print("Step 1: Background Masking & Grid Keypoint Extraction...")
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
        images_mask.append(obj_mask)
        grid_kpts_per_image.append(kpts)
        
    print(f"Masking complete in {time.time() - t0:.1f}s.")
    
    # Prepare multiprocessing tasks for 64 pairs
    tasks = []
    for i in range(num_images):
        j1 = (i + 1) % num_images
        j2 = (i + 2) % num_images
        R_i, T_i = poses[i]
        
        R_j1, T_j1 = poses[j1]
        tasks.append((i, j1, images_clahe[i], images_clahe[j1], grid_kpts_per_image[i], R_i, T_i, R_j1, T_j1))
        
        R_j2, T_j2 = poses[j2]
        tasks.append((i, j2, images_clahe[i], images_clahe[j2], grid_kpts_per_image[i], R_i, T_i, R_j2, T_j2))
        
    print(f"Step 2: Dispatching {len(tasks)} pair matching tasks across {cpu_count()} parallel worker processes...")
    t_match = time.time()
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_pair, tasks)
        
    all_3d_points = []
    for res in results:
        all_3d_points.extend(res)
        
    all_3d_points = np.array(all_3d_points)
    print(f"Matching complete in {time.time() - t_match:.1f}s. Total raw 3D points: {len(all_3d_points)}")
    
    # Statistical Outlier Removal
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_3d_points)
    
    # Color points using capture_0 camera coordinate frame projection
    colors = []
    img0_rgb = np.array(Image.open(raw_dir / "capture_0.jpg").convert("RGB"))
    H, W, _ = img0_rgb.shape
    
    for pt in all_3d_points:
        X, Y, Z = pt
        xn, yn = X / Z, Y / Z
        dist = 1.0 + k_val * (xn**2 + yn**2)
        u_proj = int(round(f_val * dist * xn + cx_val))
        v_proj = int(round(f_val * dist * yn + cy_val))
        
        if 0 <= u_proj < W and 0 <= v_proj < H:
            c = img0_rgb[v_proj, u_proj] / 255.0
        else:
            c = [0.2, 0.7, 0.9]
        colors.append(c)
        
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.8)
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
            draw.ellipse((u_proj - 2, v_proj - 2, u_proj + 2, v_proj + 2), fill="cyan", outline="blue")
            proj_count += 1
            
    overlay_out_path = vis_dir / "dense_3d_pcd_overlay_capture_0.jpg"
    img0_overlay.save(overlay_out_path, quality=95)
    print(f"Saved 3D Point Cloud Overlay onto capture_0.jpg ({proj_count} points) to {overlay_out_path}")
