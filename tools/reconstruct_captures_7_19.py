import os
import sys
import shutil
import logging
import sqlite3
import struct
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import pycolmap
from scipy.spatial.transform import Rotation as Rot
import scipy.ndimage as ndimage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    import open3d as o3d
except ImportError:
    o3d = None

def main():
    workspace = Path("COLMAP/workspace_7-19")
    source_dir = Path("captures_7-19")
    
    raw_dir = workspace / "raw_images"
    image_dir = workspace / "images"
    mask_dir = workspace / "masks"
    sparse_dir = workspace / "sparse"
    output_dir_center = workspace / "visualizations" / "plate_center_fit"
    output_dir_z_axle = workspace / "visualizations" / "z_axle_projections"
    visualization_dir = workspace / "visualizations"
    
    for d in [raw_dir, image_dir, mask_dir, sparse_dir, output_dir_center, output_dir_z_axle, visualization_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    logging.info(f"Copying images from {source_dir} to {raw_dir}...")
    for img_file in source_dir.glob("*.jpg"):
        target = raw_dir / img_file.name
        if not target.exists():
            shutil.copy2(img_file, target)
            
    # 1. Robust rembg + HSV Masking (No CLAHE filter)
    logging.info("Stage 1: Performing robust rembg + HSV masking (No CLAHE filter)...")
    import rembg
    import skimage.color as color
    
    session = rembg.new_session("u2net")
    struct_el = ndimage.generate_binary_structure(2, 2)
    raw_images = sorted(list(raw_dir.glob("*.jpg")))
    
    for img_path in raw_images:
        input_image = Image.open(img_path)
        img_np = np.array(input_image)
        
        # rembg to isolate turntable foreground
        rembg_out = rembg.remove(input_image, session=session)
        alpha = np.array(rembg_out.split()[-1])
        turntable_mask = alpha > 128
        turntable_mask_filled = ndimage.binary_fill_holes(ndimage.binary_closing(turntable_mask, structure=struct_el, iterations=5))
        
        # HSV color space to distinguish mint green plate vs orange Lego
        img_float = img_np.astype(float) / 255.0
        img_hsv = color.rgb2hsv(img_float)
        
        H = img_hsv[:, :, 0]
        S = img_hsv[:, :, 1]
        V = img_hsv[:, :, 2]
        
        # Mint green plate has Hue between 0.18 and 0.55
        is_mint_green = (H >= 0.18) & (H <= 0.55) & (S >= 0.05) & (V >= 0.2)
        
        # Lego mask is inside turntable, but NOT mint green plate
        lego_mask = turntable_mask_filled & (~is_mint_green)
        lego_mask_clean = ndimage.binary_opening(lego_mask, structure=struct_el, iterations=3)
        lego_mask_clean = ndimage.binary_closing(lego_mask_clean, structure=struct_el, iterations=5)
        
        whitened_np = img_np.copy()
        whitened_np[~lego_mask_clean] = [255, 255, 255]
        
        whitened_img = Image.fromarray(whitened_np)
        whitened_img.save(image_dir / img_path.name, quality=95)
        
        mask_uint8 = (lego_mask_clean * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_uint8)
        mask_img.save(mask_dir / f"{img_path.name}.png")
        
    logging.info("Robust rembg + HSV masking complete (pure raw colors preserved).")
    
    # 2. Physical screw / Lego 3D center in camera 0 frame (u=2210, v=590)
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
        
    # 5. Triangulate Lego block points
    logging.info("Stage 5: Triangulating Lego block 3D points...")
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
    for img_id, img in sorted(triangulated_masked.images.items(), key=lambda x: x[1].name):
        name = img.name
        image_path = raw_dir / name
        if not image_path.exists():
            continue
        img_data = plt.imread(str(image_path))
        R = img.cam_from_world().rotation.matrix()
        T = img.cam_from_world().translation
        
        point_cam = R @ derived_plate_center_3d + T
        x_n = point_cam[0] / point_cam[2]
        y_n = point_cam[1] / point_cam[2]
        r2 = x_n**2 + y_n**2
        distortion = 1.0 + k_val * r2
        px = f_val * distortion * x_n + cx_val
        py = f_val * distortion * y_n + cy_val
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img_data)
        ax.plot(px, py, 'ro', markersize=4)
        circle = plt.Circle((px, py), 30, color='lime', fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.plot([px - 50, px + 50], [py, py], color='lime', linewidth=2)
        ax.plot([px, px], [py - 50, py + 50], color='lime', linewidth=2)
        ax.text(px + 40, py - 40, f"Derived Center ({int(px)}, {int(py)})", color='lime', fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
        ax.axis('off')
        plt.savefig(str(output_dir_center / f"center_{name}"), bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        
        s_vals = np.linspace(-2.0, 0.5, 100)
        line_pts_3d = np.array([derived_plate_center_3d + s * normal for s in s_vals])
        px_coords, py_coords = [], []
        for pt_3d in line_pts_3d:
            pt_cam = R @ pt_3d + T
            xn = pt_cam[0] / pt_cam[2]
            yn = pt_cam[1] / pt_cam[2]
            r2_pt = xn**2 + yn**2
            dist_pt = 1.0 + k_val * r2_pt
            px_pt = f_val * dist_pt * xn + cx_val
            py_pt = f_val * dist_pt * yn + cy_val
            px_coords.append(px_pt)
            py_coords.append(py_pt)
            
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img_data)
        ax.plot(px_coords, py_coords, color='cyan', linestyle='--', linewidth=3)
        ax.plot(px, py, 'ro', markersize=5)
        circle = plt.Circle((px, py), 30, color='lime', fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.plot([px - 50, px + 50], [py, py], color='lime', linewidth=2)
        ax.plot([px, px], [py - 50, py + 50], color='lime', linewidth=2)
        ax.axis('off')
        plt.savefig(str(output_dir_z_axle / f"z_axle_{name}"), bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        
    # 7. Surface Reconstruction
    logging.info("Stage 7: Running surface reconstructions...")
    if o3d is None:
        logging.error("Open3D not available.")
        sys.exit(1)
        
    pcd = o3d.io.read_point_cloud(str(visualization_dir / "sparse_model.ply"))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(10)
    
    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7)
    o3d.io.write_triangle_mesh(str(visualization_dir / "poisson_mesh.ply"), poisson_mesh)
    
    # 8. Post-processing with Derived Center Cylinder Crop
    logging.info("Stage 8: Post-processing and cylinder crop...")
    pcd_mesh_vertices = o3d.geometry.PointCloud()
    pcd_mesh_vertices.points = poisson_mesh.vertices
    if poisson_mesh.has_vertex_colors():
        pcd_mesh_vertices.colors = poisson_mesh.vertex_colors
    if poisson_mesh.has_vertex_normals():
        pcd_mesh_vertices.normals = poisson_mesh.vertex_normals
        
    pts = np.asarray(pcd_mesh_vertices.points)
    V_crop = pts - derived_plate_center_3d
    h_crop = V_crop @ normal
    U_crop = V_crop - np.outer(h_crop, normal)
    r_crop = np.linalg.norm(U_crop, axis=1)
    
    keep_mask_crop = (r_crop < 0.8) & (h_crop < 0.2)
    
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(pts[keep_mask_crop])
    if pcd_mesh_vertices.has_colors():
        filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd_mesh_vertices.colors)[keep_mask_crop])
    if pcd_mesh_vertices.has_normals():
        filtered_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd_mesh_vertices.normals)[keep_mask_crop])
        
    cleaned_pcd, ind = filtered_pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
    bbox = cleaned_pcd.get_oriented_bounding_box(robust=True)
    bbox.scale(0.95, bbox.get_center())
    final_pcd = cleaned_pcd.crop(bbox)
    final_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    
    final_path = workspace / "lego_block_final.ply"
    o3d.io.write_point_cloud(str(final_path), final_pcd)
    
    # 9. Render final 3D visualization
    logging.info("Stage 9: Rendering final 3D visualizations...")
    final_pts = np.asarray(final_pcd.points)
    final_colors = np.asarray(final_pcd.colors) if final_pcd.has_colors() else None
    if final_colors is not None:
        final_colors = np.clip(final_colors, 0.0, 1.0)
        
    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(final_pts[:, 0], final_pts[:, 1], final_pts[:, 2], 
               c=final_colors if final_colors is not None else 'blue', s=3, depthshade=True)
    ax1.set_title("3D Perspective View")
    
    ax2 = fig.add_subplot(222)
    ax2.scatter(final_pts[:, 0], final_pts[:, 1], c=final_colors if final_colors is not None else 'blue', s=3)
    ax2.set_title("Top-Down View (X vs Y)")
    ax2.grid(True)
    
    ax3 = fig.add_subplot(223)
    ax3.scatter(final_pts[:, 0], final_pts[:, 2], c=final_colors if final_colors is not None else 'blue', s=3)
    ax3.set_title("Front View (X vs Z)")
    ax3.grid(True)
    
    ax4 = fig.add_subplot(224)
    ax4.scatter(final_pts[:, 1], final_pts[:, 2], c=final_colors if final_colors is not None else 'blue', s=3)
    ax4.set_title("Side View (Y vs Z)")
    ax4.grid(True)
    
    plt.savefig(str(visualization_dir / "final_point_cloud_projections.png"), bbox_inches='tight', dpi=150)
    plt.close()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    poisson_vertices = np.asarray(poisson_mesh.vertices)
    poisson_triangles = np.asarray(poisson_mesh.triangles)
    
    V_mesh = poisson_vertices - derived_plate_center_3d
    h_mesh = V_mesh @ normal
    U_mesh = V_mesh - np.outer(h_mesh, normal)
    r_mesh = np.linalg.norm(U_mesh, axis=1)
    
    keep_mask_mesh = (r_mesh < 0.8) & (h_mesh < 0.2)
    kept_indices = np.where(keep_mask_mesh)[0]
    vertex_map = {old: new for new, old in enumerate(kept_indices)}
    
    filtered_triangles = []
    for tri in poisson_triangles:
        if tri[0] in vertex_map and tri[1] in vertex_map and tri[2] in vertex_map:
            filtered_triangles.append([vertex_map[tri[0]], vertex_map[tri[1]], vertex_map[tri[2]]])
            
    filtered_vertices = poisson_vertices[keep_mask_mesh]
    filtered_triangles = np.array(filtered_triangles)
    
    if len(filtered_triangles) > 0:
        ax.plot_trisurf(filtered_vertices[:, 0], filtered_vertices[:, 1], filtered_vertices[:, 2], 
                        triangles=filtered_triangles, color='crimson', edgecolor='none', alpha=0.9, shade=True)
    else:
        ax.scatter(final_pts[:, 0], final_pts[:, 1], final_pts[:, 2], c='crimson', s=40, depthshade=True)
        
    ax.set_title("3D Reconstructed Lego Block Mesh", fontsize=16, fontweight='bold')
    ax.view_init(elev=30, azim=45)
    
    max_range = np.array([filtered_vertices[:, 0].max() - filtered_vertices[:, 0].min(), 
                          filtered_vertices[:, 1].max() - filtered_vertices[:, 1].min(), 
                          filtered_vertices[:, 2].max() - filtered_vertices[:, 2].min()]).max() / 2.0
    mid_x = (filtered_vertices[:, 0].max() + filtered_vertices[:, 0].min()) * 0.5
    mid_y = (filtered_vertices[:, 1].max() + filtered_vertices[:, 1].min()) * 0.5
    mid_z = (filtered_vertices[:, 2].max() + filtered_vertices[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.savefig(str(visualization_dir / "lego_rendered_3d.png"), bbox_inches='tight', dpi=150)
    plt.close()
    
    logging.info("SYSTEM SUCCESS: E2E Pipeline completed successfully for workspace_7-19.")
    print(f"\nReconstruction complete!")
    print(f"  Final point cloud:   {final_path}")
    print(f"  Sparse point count:  {triangulated_masked.num_points3D()}")
    print(f"  Final point count:   {len(final_pts)}")

if __name__ == "__main__":
    main()
