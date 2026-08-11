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
    workspace = Path("COLMAP/workspace_7-19_lob_with_marker")
    
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    
    image_dir = workspace / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    
    visualization_dir = workspace / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stage 1: Robust rembg + HSV Masking (Preserving raw colors, NO CLAHE)
    logging.info("Copying images from captures_7-19_lob_with_marker and generating masks...")
    logging.info("Stage 1: Performing robust rembg + HSV masking (No CLAHE filter)...")
    
    session = rembg.new_session("u2net")
    struct_el = ndimage.generate_binary_structure(2, 2)
    
    for i in range(32):
        img_name = f"capture_{i}.jpg"
        raw_path = raw_dir / img_name
        if not raw_path.exists():
            continue
            
        img = Image.open(raw_path)
        img_np = np.array(img)
        
        # A. U2Net background removal to isolate turntable (object + plate + rim)
        rembg_out = rembg.remove(img, session=session)
        alpha = np.array(rembg_out.split()[-1])
        turntable_mask = (alpha > 128)
        
        # B. Close small gaps and fill holes to get full turntable region
        filled_plate = ndimage.binary_fill_holes(ndimage.binary_closing(turntable_mask, structure=struct_el, iterations=5))
        
        # C. HSV Hue Segmentation to isolate mint green plate
        hsv = rgb2hsv(img_np)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        
        # Mint green plate hue is ~0.18 to 0.55
        green_mask = (hue >= 0.18) & (hue <= 0.55) & (sat >= 0.10) & (val >= 0.20)
        
        # Object mask = filled_plate & ~green_mask
        object_mask = filled_plate & ~green_mask
        
        # Clean up mask with morphological operations
        object_mask = ndimage.binary_opening(object_mask, structure=struct_el, iterations=2)
        object_mask = ndimage.binary_closing(object_mask, structure=struct_el, iterations=3)
        object_mask = ndimage.binary_fill_holes(object_mask)
        
        # Apply mask to image (Background -> Pure White)
        processed_img = img_np.copy()
        processed_img[~object_mask] = [255, 255, 255]
        
        # Save processed image
        Image.fromarray(processed_img).save(image_dir / img_name, quality=95)
        
    logging.info("Robust rembg + HSV masking complete (pure raw colors preserved).")
    
    # 2. Turntable Center & Normal (Physical screw center: u=2210, v=590)
    derived_plate_center_3d = np.array([-0.0511, -0.4283, 2.6000])
    normal = np.array([-0.09100806, 0.79477028, 0.60000000])
    normal = normal / np.linalg.norm(normal)
    
    f_val = 3381.185
    cx_val = 2276.438
    cy_val = 1146.994
    k_val = -0.06338
    
    # 3. SIFT Extraction on Masked Images
    logging.info("Stage 3: Extracting SIFT features on masked images...")
    db_masked_path = workspace / "database.db"
    if db_masked_path.exists():
        db_masked_path.unlink()
        
    pycolmap.extract_features(database_path=str(db_masked_path), image_path=str(image_dir), camera_mode=pycolmap.CameraMode.SINGLE)
    
    conn = sqlite3.connect(str(db_masked_path))
    cursor = conn.cursor()
    params_blob = struct.pack("dddd", f_val, cx_val, cy_val, k_val)
    cursor.execute("UPDATE cameras SET model = 2, width = 4608, height = 2592, params = ? WHERE camera_id = 1", (params_blob,))
    cursor.execute("UPDATE images SET camera_id = 1")
    conn.commit()
    conn.close()
    
    logging.info("Matching features exhaustively...")
    pycolmap.match_exhaustive(database_path=str(db_masked_path))
    
    db = pycolmap.Database.open(str(db_masked_path))
    db_images = db.read_all_images()
    name_to_id_masked = {img.name: img.image_id for img in db_images}
    db.close()
    
    # 4. Generate ideal camera poses with 11.25 degree step size
    logging.info("Stage 4: Generating ideal camera poses with 11.25 degree step size around optimal center...")
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
        
    # 5. Triangulate object points
    logging.info("Stage 5: Triangulating object 3D points...")
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
    
    # 6. Project center and Z-axle on raw images
    logging.info("Stage 6: Projecting Z-axle and derived center onto raw images...")
    plate_center_viz = visualization_dir / "plate_center_fit"
    plate_center_viz.mkdir(exist_ok=True, parents=True)
    z_axle_viz = visualization_dir / "z_axle_projections"
    z_axle_viz.mkdir(exist_ok=True, parents=True)
    
    for img_id, img_obj in sorted(triangulated_masked.images.items(), key=lambda x: x[1].name):
        name = img_obj.name
        raw_image_path = raw_dir / name
        if not raw_image_path.exists():
            continue
            
        img_data = plt.imread(str(raw_image_path))
        R_cam = img_obj.cam_from_world().rotation.matrix()
        T_cam = img_obj.cam_from_world().translation
        
        # Project center
        pt_center_cam = R_cam @ derived_plate_center_3d + T_cam
        xn_c = pt_center_cam[0] / pt_center_cam[2]
        yn_c = pt_center_cam[1] / pt_center_cam[2]
        r2_c = xn_c**2 + yn_c**2
        dist_c = 1.0 + k_val * r2_c
        px_c = f_val * dist_c * xn_c + cx_val
        py_c = f_val * dist_c * yn_c + cy_val
        
        # Render Center Projection
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img_data)
        ax.plot(px_c, py_c, 'ro', markersize=6)
        circle = plt.Circle((px_c, py_c), 30, color='lime', fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.plot([px_c - 50, px_c + 50], [py_c, py_c], color='lime', linewidth=2)
        ax.plot([px_c, px_c], [py_c - 50, py_c + 50], color='lime', linewidth=2)
        ax.text(px_c + 40, py_c - 40, f"Derived Center ({int(px_c)}, {int(py_c)})", 
                color='lime', fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
        ax.axis('off')
        plt.savefig(str(plate_center_viz / f"center_{name}"), bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        
        # Render Z-axle Projection
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img_data)
        z_pts_3d = [derived_plate_center_3d + s * normal for s in np.linspace(-0.5, 1.5, 100)]
        z_proj_x, z_proj_y = [], []
        for p3d in z_pts_3d:
            pc = R_cam @ p3d + T_cam
            xn = pc[0] / pc[2]
            yn = pc[1] / pc[2]
            r2 = xn**2 + yn**2
            dist = 1.0 + k_val * r2
            z_proj_x.append(f_val * dist * xn + cx_val)
            z_proj_y.append(f_val * dist * yn + cy_val)
            
        ax.plot(z_proj_x, z_proj_y, 'cyan', linewidth=3, label='Rotation Z-Axle')
        ax.plot(px_c, py_c, 'ro', markersize=8, label='Turntable Center')
        ax.axis('off')
        plt.savefig(str(z_axle_viz / f"z_axle_{name}"), bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()

    # 7. Dense Reconstruction & Surface Reconstruction
    logging.info("Stage 7: Running surface reconstructions...")
    points3d = triangulated_masked.points3D
    pcd_pts = np.array([pt.xyz for pt in points3d.values()])
    pcd_rgb = np.array([pt.color for pt in points3d.values()]) / 255.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_pts)
    pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
    
    # Save raw point cloud
    raw_ply = workspace / "object_raw.ply"
    o3d.io.write_point_cloud(str(raw_ply), pcd)
    
    # Filter points around the turntable center radius r <= 1.2
    logging.info("Stage 8: Post-processing and filtering point cloud...")
    d_center = np.linalg.norm(pcd_pts - derived_plate_center_3d, axis=1)
    mask_radius = d_center <= 1.2
    
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(pcd_pts[mask_radius])
    pcd_filtered.colors = o3d.utility.Vector3dVector(pcd_rgb[mask_radius])
    
    final_ply = workspace / "object_final.ply"
    o3d.io.write_point_cloud(str(final_ply), pcd_filtered)
    
    # 8. Render 3D Visualizations
    logging.info("Stage 9: Rendering final 3D visualizations...")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    pts_f = pcd_pts[mask_radius]
    rgb_f = pcd_rgb[mask_radius]
    
    ax.scatter(pts_f[:, 0], pts_f[:, 1], pts_f[:, 2], c=rgb_f, s=15, alpha=0.9)
    ax.scatter([derived_plate_center_3d[0]], [derived_plate_center_3d[1]], [derived_plate_center_3d[2]], color='red', s=100, marker='*', label='Turntable Center')
    
    # Plot Z-axle
    z_axle_line = np.array([derived_plate_center_3d + s * normal for s in np.linspace(-0.3, 0.8, 50)])
    ax.plot(z_axle_line[:, 0], z_axle_line[:, 1], z_axle_line[:, 2], color='cyan', linewidth=3, label='Rotation Axis')
    
    ax.set_title("3D Reconstructed Object from captures_7-19_lob_with_marker", fontsize=14, fontweight='bold')
    ax.legend()
    plt.savefig(visualization_dir / "object_rendered_3d.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    logging.info(f"SYSTEM SUCCESS: E2E Pipeline completed successfully for workspace_7-19_lob_with_marker.")

if __name__ == "__main__":
    main()
