import os
import sys
import glob
import subprocess
import json
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

def run_rerun_0726():
    VENV_PYTHON = ".venv_nerf/bin/python"
    VENV_NS_TRAIN = ".venv_nerf/bin/ns-train"
    VENV_NS_EXPORT = ".venv_nerf/bin/ns-export"
    
    DATASET_DIR = "captures_0726_nerf_dataset"
    TRANSFORMS_FILE = os.path.join(DATASET_DIR, "transforms_8.json")
    RERUN_OUT_DIR = "/home/coding/github/morpho_scanner/0813_0726_cerf_rerun"
    CALIB_JSON = "/home/coding/github/morpho_scanner/captures_0726_clay_checkboard_64_calibrated/calibration_results.json"
    INPUT_FOLDER = "/home/coding/github/morpho_scanner/captures_0726_clay_checkboard_64"
    MASKS_DIR = "/home/coding/github/morpho_scanner/captures_0726_clay_checkboard_64_tsdf/masks"
    
    timestamp = "2026-08-13_183217"
    
    print("==========================================================================")
    print("       0726 CLAY LOBES NeRF 2000-STEP GPU TRAINING & EXACT POSTPROCESS   ")
    print("==========================================================================")
    print(f"Dataset Directory: {DATASET_DIR}")
    print(f"Transforms JSON: {TRANSFORMS_FILE}")
    print(f"Output Directory: {RERUN_OUT_DIR}")
    
    # Locate existing config.yml
    config_files = glob.glob(os.path.join(RERUN_OUT_DIR, "**", "config.yml"), recursive=True)
    if not config_files:
        print("❌ Could not locate config.yml in rerun output directory.")
        return

    config_yaml = sorted(config_files)[-1]
    run_dir = os.path.dirname(config_yaml)
    print(f"\n✅ Config found: {config_yaml}")

    # 1. Export raw point cloud using ns-export (exact original 0726 settings)
    cmd_export_pcd = [
        VENV_NS_EXPORT, "pointcloud",
        "--load-config", config_yaml,
        "--output-dir", RERUN_OUT_DIR,
        "--num-points", "100000",
        "--remove-outliers", "True",
        "--normal-method", "open3d",
        "--save-world-frame", "False"
    ]
    print("\n📦 Step 1: Exporting 0726 Clay Lobes 3D Point Cloud from GPU NeRF...")
    subprocess.run(cmd_export_pcd, check=True)

    raw_ply_path = os.path.join(RERUN_OUT_DIR, "point_cloud.ply")

    # Load calibration parameters
    with open(CALIB_JSON, "r") as f:
        cal_data = json.load(f)

    K_cal = np.array(cal_data["camera_matrix_K"], dtype=np.float64)
    dist_cal = np.array(cal_data["distortion_coefficients"], dtype=np.float64)
    C_rot = np.array(cal_data["plate_center_mm"], dtype=np.float64)
    normal = np.array(cal_data["plate_normal"], dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    step_size_deg = float(cal_data["step_size_deg"])

    # Basis setup
    ref = np.array([1.0, 0.0, 0.0])
    x_col = ref - np.dot(ref, normal) * normal
    x_col = x_col / np.linalg.norm(x_col)
    y_col = np.cross(normal, x_col)
    R_cam = np.column_stack((x_col, y_col, normal))

    def get_camera_pose_for_frame(i):
        angle_deg = i * step_size_deg
        angle_rad = np.deg2rad(angle_deg)
        R_i = Rot.from_rotvec(-angle_rad * normal).as_matrix()
        R_eff = np.dot(R_i, R_cam)
        t_eff = C_rot
        return R_eff, t_eff

    # 2. Read raw exported point cloud
    pcd = o3d.io.read_point_cloud(raw_ply_path)
    pts_opengl = np.asarray(pcd.points)
    colors_raw = np.asarray(pcd.colors)
    N = len(pts_opengl)
    print(f"✅ Loaded {N} raw points from NeRF export.")

    # Smart scale detection
    scale = 150.0 if np.max(np.abs(pts_opengl)) < 1.0 else 1.0
    pts_opencv_mm = pts_opengl * scale
    print(f"   Applied Coordinate Scale: {scale}x")

    # 3. Filter points with Visual Hull Silhouettes across 64 frames
    print("\n🔍 Step 2: Filtering point cloud with visual hull masks...")
    image_paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")),
                         key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

    valid_counts = np.zeros(N, dtype=int)
    W, H = 4608, 2592

    for idx, img_path in enumerate(image_paths):
        mask_path = os.path.join(MASKS_DIR, f"mask_{idx:02d}.png")
        if not os.path.exists(mask_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        R, t = get_camera_pose_for_frame(idx)

        pts_cam = np.dot(pts_opencv_mm, R.T) + t.reshape(1, 3)
        zc = pts_cam[:, 2]
        zc_safe = np.where(zc > 1e-5, zc, 1e-5)

        u = np.round(K_cal[0, 0] * pts_cam[:, 0] / zc_safe + K_cal[0, 2]).astype(int)
        v = np.round(K_cal[1, 1] * pts_cam[:, 1] / zc_safe + K_cal[1, 2]).astype(int)

        valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (zc > 0)
        inside_mask = np.zeros(N, dtype=bool)
        inside_mask[valid_uv] = mask[v[valid_uv], u[valid_uv]] > 0
        valid_counts += inside_mask.astype(int)

    keep_mask = valid_counts >= 15
    keep_indices = np.where(keep_mask)[0]
    print(f"✅ Kept {len(keep_indices)} / {N} points after visual hull consistency filter.")

    pcd_filtered = pcd.select_by_index(keep_indices)
    filtered_ply_path = os.path.join(RERUN_OUT_DIR, "point_cloud_filtered.ply")
    o3d.io.write_point_cloud(filtered_ply_path, pcd_filtered)
    print(f"Saved filtered PLY to: {filtered_ply_path}")

    # 4. Poisson Surface Reconstruction
    print("\n📐 Step 3: Running Poisson surface reconstruction...")
    pcd_filtered.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_filtered, depth=8)

    densities = np.asarray(densities)
    if len(densities) > 0:
        vertices_to_remove = densities < np.percentile(densities, 5)
        mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh_ply_path = os.path.join(RERUN_OUT_DIR, "mesh.ply")
    mesh_stl_path = os.path.join(RERUN_OUT_DIR, "mesh.stl")
    mesh_obj_path = os.path.join(RERUN_OUT_DIR, "mesh.obj")

    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(mesh_ply_path, mesh)

    # Save millimeter version of mesh for CAD (stl/obj)
    vertices_opengl = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertices_mm = vertices_opengl * scale

    mesh_mm = o3d.geometry.TriangleMesh()
    mesh_mm.vertices = o3d.utility.Vector3dVector(vertices_mm)
    mesh_mm.triangles = o3d.utility.Vector3iVector(triangles)
    mesh_mm.compute_vertex_normals()
    o3d.io.write_triangle_mesh(mesh_stl_path, mesh_mm)
    o3d.io.write_triangle_mesh(mesh_obj_path, mesh_mm)

    print(f"✅ Mesh created with {len(vertices_opengl)} vertices and {len(triangles)} triangles. Saved to {mesh_ply_path}")

    # 5. Project Mesh Wireframe Overlay onto Original Images
    print("\n📸 Step 4: Projecting NeRF mesh wireframe overlays onto capture images...")
    vertices_opencv_mm = vertices_opengl * scale

    edges = set()
    for tri in triangles:
        edges.add(tuple(sorted((tri[0], tri[1]))))
        edges.add(tuple(sorted((tri[1], tri[2]))))
        edges.add(tuple(sorted((tri[2], tri[0]))))

    for idx in [0, 16, 32, 48]:
        if idx < len(image_paths):
            img_path = image_paths[idx]
            raw_img = cv2.imread(img_path)
            img = cv2.undistort(raw_img, K_cal, dist_cal, None, K_cal)

            R, t = get_camera_pose_for_frame(idx)
            pts_cam = np.dot(vertices_opencv_mm, R.T) + t.reshape(1, 3)
            zc = pts_cam[:, 2]
            zc_safe = np.where(zc > 1e-5, zc, 1e-5)

            u = np.round(K_cal[0, 0] * pts_cam[:, 0] / zc_safe + K_cal[0, 2]).astype(int)
            v = np.round(K_cal[1, 1] * pts_cam[:, 1] / zc_safe + K_cal[1, 2]).astype(int)

            overlay = img.copy()
            for p1, p2 in edges:
                if (zc[p1] > 0 and zc[p2] > 0 and
                    0 <= u[p1] < W and 0 <= v[p1] < H and
                    0 <= u[p2] < W and 0 <= v[p2] < H):
                    cv2.line(overlay, (u[p1], v[p1]), (u[p2], v[p2]), (0, 255, 255), 2)

            blended = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
            blended_small = cv2.resize(blended, (1152, 648))

            out_proj_path = os.path.join(RERUN_OUT_DIR, f"mesh_projection_{idx:02d}.jpg")
            cv2.imwrite(out_proj_path, blended_small)
            print(f"Saved projected mesh overlay to {out_proj_path}")

    # 6. Render 3D Perspective Preview
    print("\n🎨 Step 5: Rendering 3D perspective view of NeRF mesh...")
    fig = plt.figure(figsize=(10, 8), facecolor='#111116')
    ax = fig.add_subplot(111, projection='3d', facecolor='#111116')
    verts = vertices_mm[triangles]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    poly = Poly3DCollection(verts, facecolors='#00E664', edgecolors='#005724', linewidths=0.2, alpha=0.85)
    ax.add_collection3d(poly)
    ax.set_xlim(vertices_mm[:, 0].min(), vertices_mm[:, 0].max())
    ax.set_ylim(vertices_mm[:, 1].min(), vertices_mm[:, 1].max())
    ax.set_zlim(vertices_mm[:, 2].min(), vertices_mm[:, 2].max())
    ax.set_title("0726 Clay Lobes: GPU Rerun Poisson Mesh (Step 1999)", color='white', fontsize=14, pad=15)
    ax.set_xlabel("X (mm)", color='white')
    ax.set_ylabel("Y (mm)", color='white')
    ax.set_zlabel("Z (mm)", color='white')
    ax.tick_params(colors='white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.tight_layout()
    preview_img = os.path.join(RERUN_OUT_DIR, "nerf_mesh_3d_preview.png")
    plt.savefig(preview_img, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Saved 3D perspective preview to {preview_img}")

    # 7. Copy all updated output files into run directory
    subprocess.run(f"cp -v {RERUN_OUT_DIR}/mesh.* {run_dir}/", shell=True, check=True)
    subprocess.run(f"cp -v {RERUN_OUT_DIR}/point_cloud*.ply {run_dir}/", shell=True, check=True)
    subprocess.run(f"cp -v {RERUN_OUT_DIR}/mesh_projection_*.jpg {run_dir}/", shell=True, check=True)
    subprocess.run(f"cp -v {RERUN_OUT_DIR}/nerf_mesh_3d_preview.png {run_dir}/", shell=True, check=True)

    print("\n🎉 GPU RERUN POST-PROCESSING FINISHED SUCCESSFULLY!")

if __name__ == "__main__":
    run_rerun_0726()
