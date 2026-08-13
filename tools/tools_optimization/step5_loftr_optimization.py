import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import sph_harm_y
from scipy.spatial.transform import Rotation as Rot

try:
    import kornia
    import kornia.feature as KF
except ImportError:
    print("Please install kornia first.")
    exit(1)

INPUT_DIR = "captures_0726_clay_checkboard_64_calibrated"
OUTPUT_DIR = "optimization_0813"
GRID_RESOLUTION = 256
GRID_SIZE_MM = 120.0
MAX_DEGREE = 15
NUM_SAMPLES_THETA = 60
NUM_SAMPLES_PHI = 120

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

def get_projection_matrix(K_cal, C_rot, normal, step_size_deg, frame_idx):
    ref = np.array([1.0, 0.0, 0.0])
    x_col = ref - np.dot(ref, normal) * normal
    x_col = x_col / np.linalg.norm(x_col)
    y_col = np.cross(normal, x_col)
    R_cam = np.column_stack((x_col, y_col, normal))
    
    angle_rad = np.deg2rad(frame_idx * step_size_deg)
    R_i = Rot.from_rotvec(-angle_rad * normal).as_matrix()
    
    # We want world (Local) to Camera transformation
    # v_cam_i = R_i @ (R_cam @ v_local) + C_rot
    # Actually wait.
    # The previous script did: verts_cam_i = R_i @ (R_cam @ verts_local) + C_rot
    # This means Camera is at Origin looking down Z, and the object is translated by C_rot.
    # Therefore, the transformation from Local to Camera_i is:
    # X_cam = R_i @ R_cam @ X_local + C_rot
    # R_total = R_i @ R_cam
    R_total = R_i @ R_cam
    T_total = C_rot.reshape(3, 1)
    
    RT = np.hstack((R_total, T_total))
    P = K_cal @ RT
    return P, R_total, T_total

def extract_loftr_matches(img0_path, img1_path, K, dist, device):
    # Load and undistort
    img0 = cv2.undistort(cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE), K, dist)
    img1 = cv2.undistort(cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE), K, dist)
    
    # Kornia LoFTR expects tensor (B, 1, H, W) scaled to [0, 1]
    t0 = kornia.utils.image_to_tensor(img0).float().to(device) / 255.0
    t1 = kornia.utils.image_to_tensor(img1).float().to(device) / 255.0
    
    h, w = t0.shape[1:]
    scale = 1600.0 / max(h, w)
    new_h = int(np.round(h * scale / 8.0) * 8)
    new_w = int(np.round(w * scale / 8.0) * 8)
    t0 = F.interpolate(t0.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
    t1 = F.interpolate(t1.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    matcher = KF.LoFTR(pretrained='outdoor').to(device)
    matcher.eval()
    
    with torch.no_grad():
        input_dict = {"image0": t0, "image1": t1}
        correspondences = matcher(input_dict)
    
    # Extract matches and scale back to original resolution
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    confidence = correspondences['confidence'].cpu().numpy()
    
    scale_x = w / new_w
    scale_y = h / new_h
    mkpts0[:, 0] *= scale_x
    mkpts0[:, 1] *= scale_y
    mkpts1[:, 0] *= scale_x
    mkpts1[:, 1] *= scale_y
    
    # Filter by confidence
    valid = confidence > 0.5
    return mkpts0[valid], mkpts1[valid]

def triangulate_points(P0, P1, pts0, pts1):
    pts4D = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T)
    pts3D = pts4D[:3, :] / pts4D[3, :]
    return pts3D.T

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

class Phase5Optimizer(nn.Module):
    def __init__(self, init_coeffs, centroid, sdt_tensor, loftr_pts_local, device):
        super().__init__()
        self.coeffs = nn.Parameter(torch.tensor(init_coeffs, dtype=torch.float32, device=device))
        self.centroid = torch.tensor(centroid, dtype=torch.float32, device=device)
        self.sdt_tensor = sdt_tensor.to(device)
        
        # Spectral Regularization Weights
        lap_weights = np.zeros((MAX_DEGREE + 1)**2)
        idx = 0
        for l in range(MAX_DEGREE + 1):
            for m in range(-l, l + 1):
                lap_weights[idx] = l * (l + 1)
                idx += 1
        self.lap_weights = torch.tensor(lap_weights, dtype=torch.float32, device=device)
        
        # Prepare LoFTR points for radial loss
        loftr_tensor = torch.tensor(loftr_pts_local, dtype=torch.float32, device=device)
        # Shift relative to centroid
        rel_pts = loftr_tensor - self.centroid
        
        # Convert to Spherical (r, theta, phi)
        # r = sqrt(x^2 + y^2 + z^2)
        self.r_true = torch.sqrt(torch.sum(rel_pts**2, dim=1))
        
        # theta = acos(z / r)
        self.theta_true = torch.acos(rel_pts[:, 2] / self.r_true)
        
        # phi = atan2(y, x)
        phi = torch.atan2(rel_pts[:, 1], rel_pts[:, 0])
        self.phi_true = torch.where(phi < 0, phi + 2 * np.pi, phi)
        
        # Precompute Y_mat specifically for the LoFTR points!
        Y_mat_loftr = get_basis_matrix(MAX_DEGREE, self.theta_true.cpu().numpy(), self.phi_true.cpu().numpy())
        self.Y_mat_loftr = torch.tensor(Y_mat_loftr, dtype=torch.float32, device=device)
        
        # Precompute dense Y_mat for SDT Alignment loss
        theta_1d = np.linspace(0.001, np.pi - 0.001, NUM_SAMPLES_THETA)
        phi_1d = np.linspace(0, 2*np.pi, NUM_SAMPLES_PHI, endpoint=False)
        tm, pm = np.meshgrid(theta_1d, phi_1d, indexing='ij')
        
        self.theta_flat = torch.tensor(tm.ravel(), dtype=torch.float32, device=device)
        self.phi_flat = torch.tensor(pm.ravel(), dtype=torch.float32, device=device)
        Y_mat_dense = get_basis_matrix(MAX_DEGREE, self.theta_flat.cpu().numpy(), self.phi_flat.cpu().numpy())
        self.Y_mat_dense = torch.tensor(Y_mat_dense, dtype=torch.float32, device=device)

    def forward(self):
        # 1. SDT Alignment Loss (Dense Grid)
        radii_dense = torch.matmul(self.Y_mat_dense, self.coeffs)
        dx = torch.sin(self.theta_flat) * torch.cos(self.phi_flat)
        dy = torch.sin(self.theta_flat) * torch.sin(self.phi_flat)
        dz = torch.cos(self.theta_flat)
        
        px = self.centroid[0] + radii_dense * dx
        py = self.centroid[1] + radii_dense * dy
        pz = self.centroid[2] + radii_dense * dz
        pts_3d = torch.stack([px, py, pz], dim=-1)
        
        norm_x = pts_3d[:, 0] / (GRID_SIZE_MM / 2.0)
        norm_y = pts_3d[:, 1] / (GRID_SIZE_MM / 2.0)
        norm_z = (pts_3d[:, 2] - 45.0) / (GRID_SIZE_MM / 2.0)
        
        grid = torch.stack([norm_x, norm_y, norm_z], dim=-1).view(1, 1, 1, -1, 3)
        sdt_sampled = F.grid_sample(self.sdt_tensor, grid, mode='bilinear', align_corners=True).squeeze()
        loss_sdt = torch.mean(sdt_sampled**2)
        
        # 2. LoFTR Feature Keypoint Loss
        # The predicted radius at the LoFTR angles
        r_pred = torch.matmul(self.Y_mat_loftr, self.coeffs)
        # Minimize the difference between predicted SH radius and triangulated 3D radius
        loss_loftr = torch.mean((r_pred - self.r_true)**2)
        
        # 3. Laplacian Regularization
        loss_lap = torch.sum(self.lap_weights * (self.coeffs**2))
        
        # Weighted Composite Loss
        total_loss = loss_sdt + 1.0 * loss_loftr + 1e-4 * loss_lap
        
        return total_loss, loss_sdt, loss_loftr

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    print("Loading Phase 4 parameters...")
    init_coeffs = np.load(os.path.join(OUTPUT_DIR, "sh_coefficients_optimized.npy"))
    centroid = np.load(os.path.join(OUTPUT_DIR, "sh_centroid.npy"))
    sdt_mm = np.load(os.path.join(OUTPUT_DIR, "sdt_volume.npy"))
    
    sdt_zyx = np.transpose(sdt_mm, (2, 1, 0))
    sdt_tensor = torch.tensor(sdt_zyx, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    K_cal, dist_cal, C_rot, normal, step_size_deg = load_calibration()
    
    print("Extracting dense Transformer matches via LoFTR across 360 degrees...")
    all_valid_pts3D = []
    
    for i in range(0, 60, 8):
        imgA_path = os.path.join(INPUT_DIR, f"capture_{i}.jpg")
        imgB_path = os.path.join(INPUT_DIR, f"capture_{i+4}.jpg")
        
        if not os.path.exists(imgA_path) or not os.path.exists(imgB_path):
            continue
            
        mkptsA, mkptsB = extract_loftr_matches(imgA_path, imgB_path, K_cal, dist_cal, device)
        if len(mkptsA) == 0:
            continue
            
        PA, _, _ = get_projection_matrix(K_cal, C_rot, normal, step_size_deg, i)
        PB, _, _ = get_projection_matrix(K_cal, C_rot, normal, step_size_deg, i+4)
        
        pts3D_local = triangulate_points(PA, PB, mkptsA, mkptsB)
        dist_to_centroid = np.linalg.norm(pts3D_local - centroid, axis=1)
        valid_mask = dist_to_centroid < 60.0
        
        valid_pts = pts3D_local[valid_mask]
        print(f"Pair ({i}, {i+4}): Retained {len(valid_pts)} 3D points.")
        all_valid_pts3D.append(valid_pts)
        
    if not all_valid_pts3D:
        print("Not enough LoFTR matches overall.")
        return
        
    valid_pts3D = np.vstack(all_valid_pts3D)
    print(f"Total Retained: {len(valid_pts3D)} triangulated 3D surface points within bounds across all angles.")
    
    # Setup Optimizer
    optimizer_module = Phase5Optimizer(init_coeffs, centroid, sdt_tensor, valid_pts3D, device)
    optim = torch.optim.Adam(optimizer_module.parameters(), lr=0.1)
    
    print("Beginning PyTorch Joint Optimization with LoFTR Keypoint Constraints...")
    for epoch in range(200):
        optim.zero_grad()
        loss, l_sdt, l_loftr = optimizer_module()
        loss.backward()
        optim.step()
        
        if epoch % 25 == 0 or epoch == 199:
            print(f"Epoch {epoch:03d} | Total: {loss.item():.4f} | SDT: {l_sdt.item():.4f} | LoFTR: {l_loftr.item():.4f}")
            
    # Export Optimized Parameters
    opt_coeffs = np.array(optimizer_module.coeffs.detach().cpu().tolist())
    np.save(os.path.join(OUTPUT_DIR, "sh_coefficients_stage5.npy"), opt_coeffs)
    
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
            
    obj_path = os.path.join(OUTPUT_DIR, "sh_stage5_loftr.obj")
    with open(obj_path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
    print(f"Phase 5 Complete! Model constrained by LoFTR dense features saved to {obj_path}")

if __name__ == "__main__":
    main()
