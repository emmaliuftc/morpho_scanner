import os
import cv2
import json
import numpy as np
from scipy.spatial.transform import Rotation as Rot

INPUT_DIR = "captures_0726_clay_checkboard_64_calibrated"
OUTPUT_DIR = "optimization_0813"
MESH_STAGE3 = os.path.join(OUTPUT_DIR, "sh_parameterized_surface.obj")
MESH_STAGE4 = os.path.join(OUTPUT_DIR, "sh_parameterized_surface_refined.obj")

def load_calibration():
    calib_json_path = os.path.join(INPUT_DIR, "calibration_results.json")
    with open(calib_json_path, "r") as f:
        cal_data = json.load(f)
    
    K_cal = np.array(cal_data["camera_matrix_K"], dtype=np.float64)
    dist_cal = np.array(cal_data["distortion_coefficients"], dtype=np.float64)
    C_rot = np.array(cal_data["plate_center_mm"], dtype=np.float64)
    normal = np.array(cal_data["plate_normal"], dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    step_size_deg = float(cal_data["step_size_deg"])
    
    return K_cal, dist_cal, C_rot, normal, step_size_deg

def load_mesh(obj_path):
    verts = []
    faces = []
    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.strip().split()[1:]])
            elif line.startswith("f "):
                faces.append([int(x.split("/")[0])-1 for x in line.strip().split()[1:]])
    return np.array(verts), np.array(faces)

def project_mesh_to_image(img_path, verts_local, faces, K_cal, dist_cal, C_rot, normal, step_size_deg, frame_idx, base_color=(0, 255, 0)):
    # 1. Undistort Image
    raw_img = cv2.imread(img_path)
    img_undist = cv2.undistort(raw_img, K_cal, dist_cal, None, K_cal)
    
    # 2. Local Basis
    ref = np.array([1.0, 0.0, 0.0])
    x_col = ref - np.dot(ref, normal) * normal
    x_col = x_col / np.linalg.norm(x_col)
    y_col = np.cross(normal, x_col)
    R_cam = np.column_stack((x_col, y_col, normal))
    
    # 3. Transform Local to Camera 0
    verts_cam0 = np.dot(verts_local, R_cam.T) + C_rot
    
    # 4. Transform Camera 0 to Camera i (simulate turntable rotation)
    angle_rad = np.deg2rad(frame_idx * step_size_deg)
    R_i = Rot.from_rotvec(-angle_rad * normal).as_matrix()
    verts_cam_i = np.dot(verts_cam0 - C_rot, R_i.T) + C_rot
    
    # 5. Project to 2D
    u = (K_cal[0, 0] * verts_cam_i[:, 0] / verts_cam_i[:, 2]) + K_cal[0, 2]
    v = (K_cal[1, 1] * verts_cam_i[:, 1] / verts_cam_i[:, 2]) + K_cal[1, 2]
    
    # 6. Painter's Algorithm + Flat Shading
    # Get 3D coordinates of face vertices
    face_verts = verts_cam_i[faces] # (num_faces, 3, 3)
    
    # Calculate face normals
    v0 = face_verts[:, 0, :]
    v1 = face_verts[:, 1, :]
    v2 = face_verts[:, 2, :]
    vec1 = v1 - v0
    vec2 = v2 - v0
    face_normals = np.cross(vec1, vec2)
    norms = np.linalg.norm(face_normals, axis=1)
    # Avoid division by zero
    norms[norms == 0] = 1e-6
    face_normals = face_normals / norms[:, np.newaxis]
    
    # Simple directional light (pointing from camera to object)
    light_dir = np.array([0.0, 0.0, 1.0])
    # Dot product for diffuse shading
    shading = np.clip(np.dot(face_normals, light_dir), 0.2, 1.0)
    
    # Calculate average Z depth for sorting (Painter's Algorithm)
    depths = np.mean(face_verts[:, :, 2], axis=1)
    
    # Sort faces from furthest (highest Z) to closest (lowest Z)
    sort_idx = np.argsort(depths)[::-1]
    sorted_faces = faces[sort_idx]
    sorted_shading = shading[sort_idx]
    sorted_normals = face_normals[sort_idx]
    
    # Create a blank canvas for the solid mesh
    mesh_canvas = np.zeros_like(img_undist)
    height, width = img_undist.shape[:2]
    
    for i, face in enumerate(sorted_faces):
        # Back-face culling: skip faces pointing away from camera
        if sorted_normals[i, 2] > 0:
            continue
            
        pts_u = u[face]
        pts_v = v[face]
        
        # Check bounds roughly
        if np.any(pts_u < 0) or np.any(pts_u >= width) or np.any(pts_v < 0) or np.any(pts_v >= height):
            continue
            
        pts = np.column_stack((pts_u, pts_v)).astype(np.int32)
        
        # Apply shading to base color
        shade = sorted_shading[i]
        c = (int(base_color[0]*shade), int(base_color[1]*shade), int(base_color[2]*shade))
        
        cv2.fillPoly(mesh_canvas, [pts], c)
        
    # We want to see the mesh underneath the skin.
    # We will blend the original image and the mesh canvas.
    # Where the mesh canvas is completely black, we just keep the original image.
    mask = np.any(mesh_canvas > 0, axis=-1)
    
    result = img_undist.copy()
    
    # Alpha = 0.4 means 40% original photograph (transparent skin), 60% solid 3D mesh
    alpha_skin = 0.4
    result[mask] = (img_undist[mask] * alpha_skin + mesh_canvas[mask] * (1.0 - alpha_skin)).astype(np.uint8)
    
    return result

def main():
    K_cal, dist_cal, C_rot, normal, step_size_deg = load_calibration()
    
    verts_stage3, faces_stage3 = load_mesh(MESH_STAGE3)
    verts_stage4, faces_stage4 = load_mesh(MESH_STAGE4)
    MESH_STAGE5 = os.path.join(OUTPUT_DIR, "sh_stage5_loftr.obj")
    verts_stage5, faces_stage5 = load_mesh(MESH_STAGE5)
    MESH_STAGE6 = os.path.join(OUTPUT_DIR, "sh_stage6_zcut.obj")
    verts_stage6, faces_stage6 = load_mesh(MESH_STAGE6)
    
    frames_to_render = [0, 16]
    
    for idx in frames_to_render:
        img_path = os.path.join(INPUT_DIR, f"capture_{idx}.jpg")
        
        # Render Stage 3 (Green)
        img_s3 = project_mesh_to_image(img_path, verts_stage3, faces_stage3, K_cal, dist_cal, C_rot, normal, step_size_deg, idx, base_color=(0, 255, 0))
        out_s3 = os.path.join(OUTPUT_DIR, f"overlay_solid_stage3_frame_{idx}.jpg")
        cv2.imwrite(out_s3, img_s3)
        print(f"Saved {out_s3}")
        
        # Render Stage 4 (Red)
        img_s4 = project_mesh_to_image(img_path, verts_stage4, faces_stage4, K_cal, dist_cal, C_rot, normal, step_size_deg, idx, base_color=(0, 0, 255))
        out_s4 = os.path.join(OUTPUT_DIR, f"overlay_solid_stage4_frame_{idx}.jpg")
        cv2.imwrite(out_s4, img_s4)
        print(f"Saved {out_s4}")
        
        # Render Stage 5 (Blue)
        img_s5 = project_mesh_to_image(img_path, verts_stage5, faces_stage5, K_cal, dist_cal, C_rot, normal, step_size_deg, idx, base_color=(255, 0, 0))
        out_s5 = os.path.join(OUTPUT_DIR, f"overlay_solid_stage5_frame_{idx}.jpg")
        cv2.imwrite(out_s5, img_s5)
        print(f"Saved {out_s5}")
        
        # Render Stage 6 (Yellow)
        img_s6 = project_mesh_to_image(img_path, verts_stage6, faces_stage6, K_cal, dist_cal, C_rot, normal, step_size_deg, idx, base_color=(0, 255, 255))
        out_s6 = os.path.join(OUTPUT_DIR, f"overlay_solid_stage6_frame_{idx}.jpg")
        cv2.imwrite(out_s6, img_s6)
        print(f"Saved {out_s6}")
        
        # Combined Split View (Side by Side)
        h, w = img_s3.shape[:2]
        # Text annotations
        cv2.putText(img_s3, "Stage 3 (Unrefined SH)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 8)
        cv2.putText(img_s4, "Stage 4 (PyTorch Refined)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8)
        cv2.putText(img_s5, "Stage 5 (LoFTR Constrained)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 8)
        cv2.putText(img_s6, "Stage 6 (Silhouette/Z-Cut)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 8)
        
        # Downscale for easier viewing
        scale = 0.25
        img_s3_small = cv2.resize(img_s3, (int(w*scale), int(h*scale)))
        img_s4_small = cv2.resize(img_s4, (int(w*scale), int(h*scale)))
        img_s5_small = cv2.resize(img_s5, (int(w*scale), int(h*scale)))
        img_s6_small = cv2.resize(img_s6, (int(w*scale), int(h*scale)))
        
        combined = np.hstack((img_s3_small, img_s4_small, img_s5_small, img_s6_small))
        out_combined = os.path.join(OUTPUT_DIR, f"overlay_combined_frame_{idx}.jpg")
        cv2.imwrite(out_combined, combined)
        print(f"Saved combined overlay: {out_combined}")

if __name__ == "__main__":
    main()
