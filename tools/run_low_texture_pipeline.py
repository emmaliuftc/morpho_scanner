import pycolmap
import numpy as np
from pathlib import Path
import shutil
import sqlite3
import struct
import logging
from PIL import Image
import scipy.ndimage as ndimage
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
import rembg
from skimage.color import rgb2hsv
import open3d as o3d

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    raw_dir = Path("captures_7-19_lob_with_marker")
    workspace = Path("COLMAP/workspace_7-19_lob_with_marker_low_texture")
    
    workspace.mkdir(parents=True, exist_ok=True)
    image_dir = workspace / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = workspace / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # PHASE 1: Background Masking (docs/low_texture.md)
    # -------------------------------------------------------------
    existing_images = list(image_dir.glob("capture_*.jpg"))
    if len(existing_images) == 32:
        logging.info("Phase 1: Reusing existing 32 masked images from workspace/images.")
    else:
        logging.info("Phase 1: Background Masking - Masking out static background & mint green plate...")
        session = rembg.new_session("u2net")
        struct_el = ndimage.generate_binary_structure(2, 2)
        
        for i in range(32):
            img_name = f"capture_{i}.jpg"
            raw_path = raw_dir / img_name
            if not raw_path.exists(): continue
                
            img = Image.open(raw_path)
            img_np = np.array(img)
            
            rembg_out = rembg.remove(img, session=session)
            alpha = np.array(rembg_out.split()[-1])
            turntable_mask = (alpha > 128)
            
            filled_plate = ndimage.binary_fill_holes(ndimage.binary_closing(turntable_mask, structure=struct_el, iterations=5))
            
            hsv = rgb2hsv(img_np)
            hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            green_mask = (hue >= 0.18) & (hue <= 0.55) & (sat >= 0.10) & (val >= 0.20)
            
            object_mask = filled_plate & ~green_mask
            object_mask = ndimage.binary_opening(object_mask, structure=struct_el, iterations=2)
            object_mask = ndimage.binary_closing(object_mask, structure=struct_el, iterations=3)
            object_mask = ndimage.binary_fill_holes(object_mask)
            
            processed_img = img_np.copy()
            processed_img[~object_mask] = [255, 255, 255]
            Image.fromarray(processed_img).save(image_dir / img_name, quality=95)
            
        logging.info("Phase 1 Complete: Masked images saved to workspace/images.")
    
    # -------------------------------------------------------------
    # PHASE 2 & 3: Low-Texture SIFT Extraction (feat_options.sift.peak_threshold) & Pose Injection
    # -------------------------------------------------------------
    logging.info("Phase 3: Low-Texture Feature Extraction Tuning (Lower peak_threshold to 0.002)...")
    db_masked_path = workspace / "database.db"
    if db_masked_path.exists():
        db_masked_path.unlink()
        
    feat_options = pycolmap.FeatureExtractionOptions()
    feat_options.sift.peak_threshold = 0.002  # Lower peak threshold for high sensitivity on low texture
    feat_options.sift.edge_threshold = 15.0
    feat_options.sift.max_num_features = 8192

    pycolmap.extract_features(
        database_path=str(db_masked_path),
        image_path=str(image_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,
        extraction_options=feat_options
    )
    
    f_val = 3381.185
    cx_val = 2276.438
    cy_val = 1146.994
    k_val = -0.06338
    
    conn = sqlite3.connect(str(db_masked_path))
    cursor = conn.cursor()
    params_blob = struct.pack("dddd", f_val, cx_val, cy_val, k_val)
    cursor.execute("UPDATE cameras SET model = 2, width = 4608, height = 2592, params = ? WHERE camera_id = 1", (params_blob,))
    cursor.execute("UPDATE images SET camera_id = 1")
    conn.commit()
    conn.close()
    
    logging.info("Exhaustive Feature Matching...")
    pycolmap.match_exhaustive(database_path=str(db_masked_path))
    
    db = pycolmap.Database.open(str(db_masked_path))
    db_images = db.read_all_images()
    name_to_id_masked = {img.name: img.image_id for img in db_images}
    db.close()
    
    # Pose Injection using pre-calculated CBA 3D center
    logging.info("Phase 2: Pose Injection - Locking camera extrinsics to CBA turntable poses...")
    derived_plate_center_3d = np.array([-0.0511, -0.4283, 2.6000])
    normal = np.array([-0.09100806, 0.79477028, 0.60000000])
    normal = normal / np.linalg.norm(normal)
    step_size_rad = np.radians(11.25)
    
    sparse_masked_ideal = workspace / "sparse_ideal"
    sparse_masked_ideal.mkdir(exist_ok=True, parents=True)
    
    with open(sparse_masked_ideal / "cameras.txt", "w") as f_cam:
        f_cam.write(f"1 SIMPLE_RADIAL 4608 2592 {f_val} {cx_val} {cy_val} {k_val}\n")
        
    with open(sparse_masked_ideal / "images.txt", "w") as f_img:
        for i in range(32):
            name = f"capture_{i}.jpg"
            if name not in name_to_id_masked:
                continue
            img_id = name_to_id_masked[name]
            
            theta = i * step_size_rad
            R_i = Rot.from_rotvec(-theta * normal).as_matrix()
            T_i = (np.eye(3) - R_i) @ derived_plate_center_3d
            
            q = Rot.from_matrix(R_i).as_quat()
            qw, qx, qy, qz = q[3], q[0], q[1], q[2]
            
            f_img.write(f"{img_id} {qw} {qx} {qy} {qz} {T_i[0]} {T_i[1]} {T_i[2]} 1 {name}\n\n")
            
    with open(sparse_masked_ideal / "points3D.txt", "w") as f_pts:
        f_pts.write("")
        
    # Triangulation
    logging.info("Triangulating tuned low-texture feature points...")
    sparse_masked_triangulated = workspace / "sparse_ideal_triangulated"
    triangulated_masked = pycolmap.triangulate_points(
        reconstruction=pycolmap.Reconstruction(str(sparse_masked_ideal)),
        database_path=str(db_masked_path),
        image_path=str(image_dir),
        output_path=str(sparse_masked_triangulated),
        clear_points=True
    )
    triangulated_masked.write(str(sparse_masked_triangulated))
    triangulated_masked.export_PLY(str(visualization_dir / "sparse_model.ply"))
    
    points3d = triangulated_masked.points3D
    pcd_pts = np.array([pt.xyz for pt in points3d.values()])
    pcd_rgb = np.array([pt.color for pt in points3d.values()]) / 255.0
    
    logging.info(f"Triangulated {len(pcd_pts)} 3D points on low-texture surface!")

    # Filter sculpture points
    vec_from_center = pcd_pts - derived_plate_center_3d
    dist_from_axis = np.linalg.norm(vec_from_center - np.outer(vec_from_center @ normal, normal), axis=1)
    mask_obj = (dist_from_axis <= 0.25) & (pcd_pts[:, 2] >= 2.20) & (pcd_pts[:, 2] <= 2.80)
    
    pts_clean = pcd_pts[mask_obj]
    colors_clean = pcd_rgb[mask_obj]
    
    pcd_clean = o3d.geometry.PointCloud()
    pcd_clean.points = o3d.utility.Vector3dVector(pts_clean)
    pcd_clean.colors = o3d.utility.Vector3dVector(colors_clean)
    
    final_ply = workspace / "object_dense_final.ply"
    o3d.io.write_point_cloud(str(final_ply), pcd_clean)
    
    # Render 3D Plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts_clean[:, 0], pts_clean[:, 1], pts_clean[:, 2], c=colors_clean, s=12, alpha=0.95)
    ax.scatter([derived_plate_center_3d[0]], [derived_plate_center_3d[1]], [derived_plate_center_3d[2]], color='lime', s=150, marker='*', label='Turntable Center')
    
    z_axle_line = np.array([derived_plate_center_3d + s * normal for s in np.linspace(-0.2, 0.6, 50)])
    ax.plot(z_axle_line[:, 0], z_axle_line[:, 1], z_axle_line[:, 2], color='cyan', linewidth=3, label='Rotation Z-Axle')
    
    ax.set_title(f"Tuned Low-Texture 3D Point Cloud (docs/low_texture.md)\n{len(pts_clean)} Triangulated Sculpture Points", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.savefig(visualization_dir / "low_texture_lob_rendered_3d.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    # Render Overlay on capture_0.jpg
    img0_np = np.array(Image.open(raw_dir / "capture_0.jpg"))
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.imshow(img0_np)
    
    proj_x, proj_y, proj_c = [], [], []
    for k in range(len(pts_clean)):
        X, Y, Z = pts_clean[k, 0], pts_clean[k, 1], pts_clean[k, 2]
        if Z <= 0: continue
        xn, yn = X / Z, Y / Z
        dist = 1.0 + k_val * (xn**2 + yn**2)
        u = f_val * dist * xn + cx_val
        v = f_val * dist * yn + cy_val
        if 0 <= u < 4608 and 0 <= v < 2592:
            proj_x.append(u)
            proj_y.append(v)
            proj_c.append(colors_clean[k])
            
    ax.scatter(proj_x, proj_y, c=proj_c, s=25, edgecolors='black', linewidths=0.3, alpha=0.9)
    ax.set_title(f"Tuned Low-Texture Point Cloud Overlay on capture_0.jpg ({len(proj_x)} Points)", fontsize=15, fontweight='bold')
    plt.savefig(visualization_dir / "low_texture_pcd_overlay_capture_0.png", bbox_inches='tight', dpi=150)
    plt.close()

    logging.info(f"SUCCESS: Low-Texture pipeline fully executed. Filtered point cloud: {len(pts_clean)} points.")

if __name__ == "__main__":
    main()
