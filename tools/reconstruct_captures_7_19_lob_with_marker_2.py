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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    raw_dir = Path("captures_7-19_lob_with_marker_2")
    workspace = Path("COLMAP/workspace_7-19_lob_with_marker_2")
    
    workspace.mkdir(parents=True, exist_ok=True)
    image_dir = workspace / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = workspace / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    # 1. Background Masking (reuse existing)
    existing_images = list(image_dir.glob("capture_*.jpg"))
    if len(existing_images) == 32:
        logging.info("Stage 1: Reusing existing 32 masked images from workspace/images.")
    else:
        logging.info("Stage 1: Masking out static background and green plate for captures_7-19_lob_with_marker_2...")
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
            
        logging.info("Stage 1 Complete: Masked images saved.")
    
    # 2. Ultra-High Sensitivity SIFT Feature Extraction
    db_masked_path = workspace / "database.db"
    if not db_masked_path.exists():
        logging.info("Stage 2: Running ULTRA-HIGH SIFT Feature Detection (peak_threshold=0.0005, max_features=32768)...")
        feat_options = pycolmap.FeatureExtractionOptions()
        feat_options.sift.peak_threshold = 0.0005
        feat_options.sift.edge_threshold = 30.0
        feat_options.sift.max_num_features = 32768

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
    
    cursor.execute("SELECT image_id, rows FROM keypoints")
    kpt_counts = cursor.fetchall()
    total_kpts = sum(c[1] for c in kpt_counts)
    avg_kpts = total_kpts / len(kpt_counts)
    logging.info(f"ULTRA SIFT EXTRACTION STATS: {total_kpts} total keypoints across 32 images (Average: {avg_kpts:.1f} keypoints/image)!")
    conn.close()
    
    # 3. Exhaustive Feature Matching with Relaxed Ratio Threshold
    logging.info("Stage 3: Running Exhaustive Feature Matching (max_ratio=0.90)...")
    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.sift.max_ratio = 0.90
    
    pycolmap.match_exhaustive(
        database_path=str(db_masked_path),
        matching_options=matching_options
    )
    
    conn = sqlite3.connect(str(db_masked_path))
    cursor = conn.cursor()
    cursor.execute("SELECT count(*), sum(rows) FROM two_view_geometries WHERE rows > 0")
    num_pairs, num_matches = cursor.fetchone()
    logging.info(f"ULTRA MATCHING STATS: Found {num_matches} inlier feature matches across {num_pairs} validated image pairs!")
    conn.close()
    
    db = pycolmap.Database.open(str(db_masked_path))
    db_images = db.read_all_images()
    name_to_id_masked = {img.name: img.image_id for img in db_images}
    db.close()
    
    # 4. Pose Injection & Triangulation
    logging.info("Stage 4: Injecting CBA turntable camera poses and triangulating dense 3D point cloud...")
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
            if name not in name_to_id_masked: continue
            img_id = name_to_id_masked[name]
            
            theta = i * step_size_rad
            R_i = Rot.from_rotvec(-theta * normal).as_matrix()
            T_i = (np.eye(3) - R_i) @ derived_plate_center_3d
            
            q = Rot.from_matrix(R_i).as_quat()
            qw, qx, qy, qz = q[3], q[0], q[1], q[2]
            f_img.write(f"{img_id} {qw} {qx} {qy} {qz} {T_i[0]} {T_i[1]} {T_i[2]} 1 {name}\n\n")
            
    with open(sparse_masked_ideal / "points3D.txt", "w") as f_pts:
        f_pts.write("")
        
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
    logging.info(f"COLMAP TRIANGULATION STATS: Triangulated {len(pcd_pts)} total 3D points!")

if __name__ == "__main__":
    main()
