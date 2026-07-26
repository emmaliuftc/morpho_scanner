import cv2
import numpy as np
import glob
import os
import sys
import json
from scipy import ndimage
from scipy.spatial.transform import Rotation as Rot
from skimage import measure
import open3d as o3d

# Ensure tools directory is in Python path
sys.path.append("tools")
from silhouette_extractor import SilhouetteExtractor

# ==========================================
# 1. PIPELINE CONFIGURATION & CALIBRATION LOAD
# ==========================================
INPUT_FOLDER = "captures_0726_clay_checkboard_64"
CALIB_JSON = "captures_0726_clay_checkboard_64_calibrated/calibration_results.json"
OUTPUT_FOLDER = "captures_0726_clay_checkboard_64_tsdf"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
masks_dir = os.path.join(OUTPUT_FOLDER, "masks")
os.makedirs(masks_dir, exist_ok=True)

# Load JSON calibration data
with open(CALIB_JSON, "r") as f:
    cal_data = json.load(f)

K_cal = np.array(cal_data["camera_matrix_K"], dtype=np.float32)
dist_cal = np.array(cal_data["distortion_coefficients"], dtype=np.float32)
C_rot = np.array(cal_data["plate_center_mm"], dtype=np.float32)
normal = np.array(cal_data["plate_normal"], dtype=np.float32)
normal = normal / np.linalg.norm(normal)
step_size_deg = float(cal_data["step_size_deg"])
NUM_IMAGES = int(cal_data["n_images"])

CENTER_2D = (2166, 1145)
PLATE_RADIUS_PIXELS = 1100

# TSDF Hyperparameters
GRID_SIZE_MM = 140.0
GRID_RESOLUTION = 256
TRUNC_MARGIN = 15.0
MAX_WEIGHT = 50.0

# 2. Camera coordinate basis
ref = np.array([1.0, 0.0, 0.0])
x_col = ref - np.dot(ref, normal) * normal
x_col = x_col / np.linalg.norm(x_col)
y_col = np.cross(normal, x_col)
R_cam = np.column_stack((x_col, y_col, normal))

def get_camera_pose_for_frame(frame_index):
    angle_deg = frame_index * step_size_deg
    angle_rad = np.deg2rad(angle_deg)
    R_i = Rot.from_rotvec(-angle_rad * normal).as_matrix()
    R_eff = np.dot(R_i, R_cam)
    t_eff = C_rot
    return R_eff, t_eff

# ==========================================
# 2. TSDF VOLUME CLASS
# ==========================================
class TSDFVolume0726:
    def __init__(self, resolution, size_mm, K_cal):
        self.res = resolution
        self.size = size_mm
        self.voxel_size = size_mm / resolution
        self.K = K_cal
        
        print(f"Initializing {resolution}^3 TSDF Grid ({size_mm}mm across, {self.voxel_size:.2f}mm/voxel)...")
        
        self.tsdf = np.ones((resolution, resolution, resolution), dtype=np.float32)
        self.weights = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        
        half_size = size_mm / 2.0
        x = np.linspace(-half_size, half_size, resolution)
        y = np.linspace(-half_size, half_size, resolution)
        z = np.linspace(-15.0, size_mm - 15.0, resolution)
        
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        self.voxel_coords = np.vstack((xv.ravel(), yv.ravel(), zv.ravel()))
        self.num_voxels = self.voxel_coords.shape[1]

    def initialize_with_visual_hull(self, image_paths, extractor, dist_cal):
        print("\n--- INITIALIZING TSDF VOLUME WITH VISUAL HULL (VOXEL CARVING) ---")
        self.tsdf.fill(-1.0)
        self.weights.fill(1.0)
        
        height, width = 2592, 4608
        voxel_in_background = np.zeros(self.num_voxels, dtype=bool)
        
        # Spatial radial constraint to cleanly remove the distant checkerboard
        # Clay is at origin; keep voxels within 55mm radius
        dist_from_origin = np.linalg.norm(self.voxel_coords, axis=0)
        distant_voxels = dist_from_origin > 55.0
        voxel_in_background[distant_voxels] = True
        
        for i, path in enumerate(image_paths):
            if (i % 8) == 0 or i == len(image_paths) - 1:
                print(f"Carving visual hull frame {i+1}/{len(image_paths)}: {os.path.basename(path)}")
                
            raw_img = cv2.imread(path)
            img = cv2.undistort(raw_img, self.K, dist_cal, None, self.K)
            
            mask_path = os.path.join(masks_dir, f"mask_{i:02d}.png")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                mask = extractor.get_silhouette_mask(img, CENTER_2D, PLATE_RADIUS_PIXELS)
                cv2.imwrite(mask_path, mask)
                
            R, t = get_camera_pose_for_frame(i)
            
            cam_pts = np.dot(R, self.voxel_coords) + t.reshape(3, 1)
            voxel_depths = cam_pts[2, :]
            valid_z = voxel_depths > 1.0
            
            u = (self.K[0,0] * cam_pts[0, :] / voxel_depths) + self.K[0,2]
            v = (self.K[1,1] * cam_pts[1, :] / voxel_depths) + self.K[1,2]
            u = np.round(u).astype(int)
            v = np.round(v).astype(int)
            
            valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            valid_mask = valid_z & valid_uv
            
            check_u = u[valid_mask]
            check_v = v[valid_mask]
            pixel_values = mask[check_v, check_u]
            
            hit_bg = pixel_values == 0
            bg_indices = np.where(valid_mask)[0][hit_bg]
            voxel_in_background[bg_indices] = True
            
            outside_indices = np.where(~valid_mask)[0]
            voxel_in_background[outside_indices] = True
            
        self.tsdf.ravel()[voxel_in_background] = 1.0
        self.weights.ravel()[voxel_in_background] = MAX_WEIGHT
        print(f"Visual Hull Initialization Complete. Solid voxels remaining: {np.sum(~voxel_in_background)}")

    def extract_mesh(self, output_path, image_paths, extractor, dist_cal):
        print("\n--- EXTRACTING 3D MESH USING MARCHING CUBES ---")
        try:
            verts, faces, _, _ = measure.marching_cubes(self.tsdf, level=0.0, spacing=(self.voxel_size, self.voxel_size, self.voxel_size))
            
            half_size = self.size / 2.0
            verts[:, 0] -= half_size
            verts[:, 1] -= half_size
            verts[:, 2] -= 15.0
            
            # Color mesh vertices from input images
            colors = self.color_mesh(verts, image_paths, dist_cal)
            
            # Save PLY
            ply_path = output_path.replace(".obj", ".ply")
            with open(ply_path, 'w') as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {len(verts)}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                f.write(f"element face {len(faces)}\n")
                f.write("property list uchar int vertex_indices\nend_header\n")
                for v, c in zip(verts, colors):
                    f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
                for face in faces:
                    f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
            print(f"Successfully saved colored PLY mesh to: {ply_path}")
            
            # Save OBJ
            with open(output_path, 'w') as f:
                for v, c in zip(verts, colors):
                    f.write(f"v {v[0]} {v[1]} {v[2]} {c[0]/255.0:.4f} {c[1]/255.0:.4f} {c[2]/255.0:.4f}\n")
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            print(f"Successfully saved colored OBJ mesh to: {output_path}")
            
            return verts, faces
            
        except ValueError as e:
            print(f"Mesh Extraction Failed: {e}")
            return None, None

    def color_mesh(self, verts, image_paths, dist_cal):
        num_verts = len(verts)
        vert_colors = np.zeros((num_verts, 3), dtype=np.float32)
        vert_weights = np.zeros(num_verts, dtype=np.float32)
        
        for i, path in enumerate(image_paths):
            img = cv2.undistort(cv2.imread(path), self.K, dist_cal)
            mask_path = os.path.join(masks_dir, f"mask_{i:02d}.png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            R, t = get_camera_pose_for_frame(i)
            
            cam_pts = np.dot(R, verts.T) + t.reshape(3, 1)
            voxel_depths = cam_pts[2, :]
            valid_z = voxel_depths > 1.0
            
            u = (self.K[0,0] * cam_pts[0, :] / voxel_depths) + self.K[0,2]
            v = (self.K[1,1] * cam_pts[1, :] / voxel_depths) + self.K[1,2]
            u = np.round(u).astype(int)
            v = np.round(v).astype(int)
            
            height, width = img.shape[:2]
            valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            valid_mask = valid_z & valid_uv
            
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                continue
                
            v_idx = v[valid_indices]
            u_idx = u[valid_indices]
            inside_silhouette = mask[v_idx, u_idx] > 0
            visible_indices = valid_indices[inside_silhouette]
            
            if len(visible_indices) == 0:
                continue
                
            vis_v = v[visible_indices]
            vis_u = u[visible_indices]
            
            colors = img[vis_v, vis_u, ::-1] # BGR to RGB
            vert_colors[visible_indices] += colors
            vert_weights[visible_indices] += 1.0
            
        final_colors = np.zeros((num_verts, 3), dtype=np.float32)
        has_weight = vert_weights > 0
        final_colors[has_weight] = vert_colors[has_weight] / vert_weights[has_weight, None]
        final_colors[~has_weight] = [128.0, 128.0, 128.0]
        return final_colors.astype(np.uint8)

# ==========================================
# 3. MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    print("=== RUNNING TSDF FUSION ON NEW 64-IMAGE DATASET ===")
    
    image_paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")),
                         key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    print(f"Found {len(image_paths)} input images.")
    
    extractor = SilhouetteExtractor()
    
    volume = TSDFVolume0726(GRID_RESOLUTION, GRID_SIZE_MM, K_cal)
    
    # 1. Run visual hull carving initialization (masks out green plate & checkerboard)
    volume.initialize_with_visual_hull(image_paths, extractor, dist_cal)
    
    # 2. Extract Mesh
    mesh_path = os.path.join(OUTPUT_FOLDER, "clay_tsdf_mesh.obj")
    verts, faces = volume.extract_mesh(mesh_path, image_paths, extractor, dist_cal)
    
    if verts is not None and len(verts) > 0:
        # 3. Project Mesh Wireframe onto Original Images (Views 0, 16, 32, 48)
        print("\n--- PROJECTING MESH WIREFRAME OVERLAYS ONTO ORIGINAL IMAGES ---")
        W, H = 4608, 2592
        edges = set()
        for tri in faces:
            edges.add(tuple(sorted((tri[0], tri[1]))))
            edges.add(tuple(sorted((tri[1], tri[2]))))
            edges.add(tuple(sorted((tri[2], tri[0]))))
            
        for idx in [0, 16, 32, 48]:
            if idx < len(image_paths):
                img_path = image_paths[idx]
                raw_img = cv2.imread(img_path)
                img = cv2.undistort(raw_img, K_cal, dist_cal, None, K_cal)
                
                R, t = get_camera_pose_for_frame(idx)
                pts_cam = np.dot(R, verts.T) + t.reshape(3, 1)
                zc = pts_cam[2, :]
                zc_safe = np.where(zc > 1e-5, zc, 1e-5)
                
                u = np.round(K_cal[0,0] * pts_cam[0, :] / zc_safe + K_cal[0,2]).astype(int)
                v = np.round(K_cal[1,1] * pts_cam[1, :] / zc_safe + K_cal[1,2]).astype(int)
                
                overlay = img.copy()
                for p1, p2 in edges:
                    if (zc[p1] > 0 and zc[p2] > 0 and
                        0 <= u[p1] < W and 0 <= v[p1] < H and
                        0 <= u[p2] < W and 0 <= v[p2] < H):
                        cv2.line(overlay, (u[p1], v[p1]), (u[p2], v[p2]), (0, 255, 255), 2) # Cyan wireframe
                        
                blended = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
                blended_small = cv2.resize(blended, (1152, 648))
                
                out_proj_path = os.path.join(OUTPUT_FOLDER, f"tsdf_mesh_projection_{idx:02d}.jpg")
                cv2.imwrite(out_proj_path, blended_small)
                print(f"Saved projected mesh overlay to {out_proj_path}")
                
        # 4. Render 3D Perspective View of Mesh with Open3D
        print("\n--- RENDERING 3D PERSPECTIVE VIEW OF TSDF MESH ---")
        ply_path = mesh_path.replace(".obj", ".ply")
        mesh_o3d = o3d.io.read_triangle_mesh(ply_path)
        mesh_o3d.compute_vertex_normals()
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1280, height=960)
        vis.add_geometry(mesh_o3d)
        
        ctr = vis.get_view_control()
        ctr.set_zoom(0.7)
        ctr.set_front([0.5, -0.5, -0.7])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        
        vis.poll_events()
        vis.update_renderer()
        render_path = os.path.join(OUTPUT_FOLDER, "tsdf_mesh_3d_preview.png")
        vis.capture_screen_image(render_path, do_render=True)
        vis.destroy_window()
        print(f"Saved 3D perspective render to {render_path}")

    print("\n✅ TSDF PIPELINE FOR 64 IMAGES FINISHED SUCCESSFULLY!")
