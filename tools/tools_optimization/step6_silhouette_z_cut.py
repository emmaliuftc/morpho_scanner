import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import sph_harm_y
from scipy.spatial.transform import Rotation as Rot

INPUT_DIR = "captures_0726_clay_checkboard_64_calibrated"
MASKS_DIR = "optimization_0813/masks"
OUTPUT_DIR = "optimization_0813"

GRID_SIZE_MM = 120.0
MAX_DEGREE = 15
NUM_SAMPLES_THETA = 60
NUM_SAMPLES_PHI = 120
N_CAMERAS = 64

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
    width = cal_data["image_resolution"][0]
    height = cal_data["image_resolution"][1]
    
    return K_cal, dist_cal, C_rot, normal, step_size_deg, width, height

def get_projection_matrix(K_cal, C_rot, normal, step_size_deg, frame_idx):
    ref = np.array([1.0, 0.0, 0.0])
    x_col = ref - np.dot(ref, normal) * normal
    x_col = x_col / np.linalg.norm(x_col)
    y_col = np.cross(normal, x_col)
    R_cam = np.column_stack((x_col, y_col, normal))
    
    angle_rad = np.deg2rad(frame_idx * step_size_deg)
    R_i = Rot.from_rotvec(-angle_rad * normal).as_matrix()
    
    R_total = R_i @ R_cam
    T_total = C_rot.reshape(3, 1)
    
    RT = np.hstack((R_total, T_total))
    P = K_cal @ RT
    return P

def real_sph_harm(l, m, theta, phi):
    Y = sph_harm_y(l, abs(m), theta, phi)
    if m < 0:
        return np.sqrt(2) * ((-1)**m) * Y.imag
    elif m == 0:
        return Y.real
    else:
        return np.sqrt(2) * ((-1)**m) * Y.real

def get_basis_matrix(l_max, theta, phi):
    num_coeffs = (l_max + 1)**2
    num_points = len(theta)
    Y_mat = np.zeros((num_points, num_coeffs))
    
    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Y_mat[:, idx] = real_sph_harm(l, m, theta, phi)
            idx += 1
    return Y_mat

class Phase6Optimizer(nn.Module):
    def __init__(self, init_coeffs, centroid, sdt_tensor, proj_matrices, masks_tensor, device, orig_w, orig_h):
        super().__init__()
        self.coeffs = nn.Parameter(torch.tensor(init_coeffs, dtype=torch.float32, device=device))
        self.centroid = torch.tensor(centroid, dtype=torch.float32, device=device)
        self.sdt_tensor = sdt_tensor.to(device)
        self.proj_matrices = torch.tensor(proj_matrices, dtype=torch.float32, device=device) # (64, 3, 4)
        self.masks_tensor = masks_tensor.to(device) # (64, 1, H, W)
        self.orig_w = orig_w
        self.orig_h = orig_h
        
        lap_weights = np.zeros((MAX_DEGREE + 1)**2)
        idx = 0
        for l in range(MAX_DEGREE + 1):
            for m in range(-l, l + 1):
                lap_weights[idx] = l * (l + 1)
                idx += 1
        self.lap_weights = torch.tensor(lap_weights, dtype=torch.float32, device=device)
        
        # Dense Grid for optimization
        theta_1d = np.linspace(0.001, np.pi - 0.001, NUM_SAMPLES_THETA)
        phi_1d = np.linspace(0, 2*np.pi, NUM_SAMPLES_PHI, endpoint=False)
        tm, pm = np.meshgrid(theta_1d, phi_1d, indexing='ij')
        
        self.theta_flat = torch.tensor(tm.ravel(), dtype=torch.float32, device=device)
        self.phi_flat = torch.tensor(pm.ravel(), dtype=torch.float32, device=device)
        Y_mat_dense = get_basis_matrix(MAX_DEGREE, self.theta_flat.cpu().numpy(), self.phi_flat.cpu().numpy())
        self.Y_mat_dense = torch.tensor(Y_mat_dense, dtype=torch.float32, device=device)

    def forward(self):
        radii_dense = torch.matmul(self.Y_mat_dense, self.coeffs)
        dx = torch.sin(self.theta_flat) * torch.cos(self.phi_flat)
        dy = torch.sin(self.theta_flat) * torch.sin(self.phi_flat)
        dz = torch.cos(self.theta_flat)
        
        px = self.centroid[0] + radii_dense * dx
        py = self.centroid[1] + radii_dense * dy
        pz = self.centroid[2] + radii_dense * dz
        
        # 1. SDT Alignment Loss (Anchor against drift)
        pts_3d = torch.stack([px, py, pz], dim=-1)
        norm_x = pts_3d[:, 0] / (GRID_SIZE_MM / 2.0)
        norm_y = pts_3d[:, 1] / (GRID_SIZE_MM / 2.0)
        norm_z = (pts_3d[:, 2] - 45.0) / (GRID_SIZE_MM / 2.0)
        
        grid = torch.stack([norm_x, norm_y, norm_z], dim=-1).view(1, 1, 1, -1, 3)
        sdt_sampled = F.grid_sample(self.sdt_tensor, grid, mode='bilinear', align_corners=True).squeeze()
        loss_sdt = torch.mean(sdt_sampled**2)
        
        # 2. Laplacian Regularization
        loss_lap = torch.sum(self.lap_weights * (self.coeffs**2))
        
        # 3. Z-Level Floor Cut Loss (Penalize z < 4.0 mm)
        # We heavily penalize any geometry that goes below the turntable plate!
        # Increased threshold to 4.0mm above the mathematically ideal 0.0 to account for the sagging 'waist'.
        loss_z = torch.sum(F.relu(4.0 - pz)**2)
        
        # 4. Silhouette Projection Loss
        # Randomly sample a few cameras per step to save memory
        num_cams = self.proj_matrices.shape[0]
        cam_indices = torch.randint(0, num_cams, (8,))
        
        pts_4d = torch.stack([px, py, pz, torch.ones_like(px)], dim=-1) # (N, 4)
        loss_mask = 0.0
        
        for idx in cam_indices:
            P = self.proj_matrices[idx] # (3, 4)
            # Project to image plane
            pts_2d_hom = torch.matmul(pts_4d, P.T) # (N, 3)
            
            # Avoid division by zero
            w = pts_2d_hom[:, 2] + 1e-6
            u = pts_2d_hom[:, 0] / w
            v = pts_2d_hom[:, 1] / w
            
            # Normalize to [-1, 1] for grid_sample
            u_norm = (u / (self.orig_w - 1)) * 2.0 - 1.0
            v_norm = (v / (self.orig_h - 1)) * 2.0 - 1.0
            uv_grid = torch.stack([u_norm, v_norm], dim=-1).view(1, 1, -1, 2)
            
            mask_slice = self.masks_tensor[idx:idx+1] # (1, 1, H, W)
            # Sample mask value (0 = background, 1 = object)
            sampled_mask = F.grid_sample(mask_slice, uv_grid, mode='bilinear', align_corners=True, padding_mode='zeros').squeeze()
            
            # If the point projects to background (mask == 0), penalize it!
            # We penalize points proportional to their distance from the mask boundary (1 - sampled_mask is 1.0 in background)
            # The further it goes, the more we penalize it? No, just penalize the boolean area.
            # Using (1.0 - sampled_mask) pushes the radii inwards when it hits background.
            loss_mask += torch.sum((1.0 - sampled_mask)**2)
            
        loss_mask = loss_mask / len(cam_indices)
        
        total_loss = loss_sdt + 1e-4 * loss_lap + 1.0 * loss_z + 1.0 * loss_mask
        
        return total_loss, loss_sdt, loss_z, loss_mask

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Phase 6 on device: {device}")
    
    print("Loading Phase 5 parameters...")
    init_coeffs = np.load(os.path.join(OUTPUT_DIR, "sh_coefficients_stage5.npy"))
    centroid = np.load(os.path.join(OUTPUT_DIR, "sh_centroid.npy"))
    sdt_mm = np.load(os.path.join(OUTPUT_DIR, "sdt_volume.npy"))
    
    sdt_zyx = np.transpose(sdt_mm, (2, 1, 0))
    sdt_tensor = torch.tensor(sdt_zyx, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    K_cal, dist_cal, C_rot, normal, step_size_deg, width, height = load_calibration()
    
    # Preload projection matrices and masks
    proj_matrices = []
    masks = []
    print("Preloading projection matrices and 2D silhouettes...")
    
    for i in range(N_CAMERAS):
        P = get_projection_matrix(K_cal, C_rot, normal, step_size_deg, i)
        proj_matrices.append(P)
        
        mask_path = os.path.join(MASKS_DIR, f"mask_{i:02d}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # We can downscale the mask dramatically for speed, but let's keep it fairly large for accuracy
        # Downscale by 4 for VRAM efficiency (1152 x 648)
        mask_small = cv2.resize(mask, (width // 4, height // 4), interpolation=cv2.INTER_NEAREST)
        masks.append(mask_small / 255.0)
        
    masks_array = np.array(masks, dtype=np.float32) # (64, H/4, W/4)
    masks_tensor = torch.tensor(masks_array).unsqueeze(1) # (64, 1, H/4, W/4)
    
    optimizer_module = Phase6Optimizer(init_coeffs, centroid, sdt_tensor, proj_matrices, masks_tensor, device, width, height)
    optim = torch.optim.Adam(optimizer_module.parameters(), lr=0.05)
    
    print("Beginning Phase 6: Z-Cut and Silhouette Refinement...")
    for epoch in range(150):
        optim.zero_grad()
        loss, l_sdt, l_z, l_mask = optimizer_module()
        loss.backward()
        optim.step()
        
        if epoch % 20 == 0 or epoch == 149:
            print(f"Epoch {epoch:03d} | Total: {loss.item():.4f} | SDT: {l_sdt.item():.4f} | Z-Cut: {l_z.item():.4f} | Mask: {l_mask.item():.4f}")
            
    opt_coeffs = np.array(optimizer_module.coeffs.detach().cpu().tolist())
    np.save(os.path.join(OUTPUT_DIR, "sh_coefficients_stage6.npy"), opt_coeffs)
    
    # Render final mesh
    Y_mat_dense = get_basis_matrix(MAX_DEGREE, optimizer_module.theta_flat.cpu().numpy(), optimizer_module.phi_flat.cpu().numpy())
    radii_mesh = np.dot(Y_mat_dense, opt_coeffs).reshape((NUM_SAMPLES_THETA, NUM_SAMPLES_PHI))
    
    tm = optimizer_module.theta_flat.cpu().numpy().reshape((NUM_SAMPLES_THETA, NUM_SAMPLES_PHI))
    pm = optimizer_module.phi_flat.cpu().numpy().reshape((NUM_SAMPLES_THETA, NUM_SAMPLES_PHI))
    
    x_sh = centroid[0] + radii_mesh * np.sin(tm) * np.cos(pm)
    y_sh = centroid[1] + radii_mesh * np.sin(tm) * np.sin(pm)
    z_sh = centroid[2] + radii_mesh * np.cos(tm)
    
    verts = np.column_stack((x_sh.ravel(), y_sh.ravel(), z_sh.ravel()))
    faces = []
    rows, cols = tm.shape
    for i in range(rows - 1):
        for j in range(cols - 1):
            p1 = i * cols + j
            p2 = p1 + 1
            p3 = (i + 1) * cols + j
            p4 = p3 + 1
            faces.append([p1, p2, p3])
            faces.append([p2, p4, p3])
    for i in range(rows - 1):
        p1 = i * cols + (cols - 1)
        p2 = i * cols + 0
        p3 = (i + 1) * cols + (cols - 1)
        p4 = (i + 1) * cols + 0
        faces.append([p1, p2, p3])
        faces.append([p2, p4, p3])
            
    obj_path = os.path.join(OUTPUT_DIR, "sh_stage6_zcut.obj")
    with open(obj_path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
    print(f"Phase 6 Complete! Model saved to {obj_path}")

if __name__ == "__main__":
    main()
