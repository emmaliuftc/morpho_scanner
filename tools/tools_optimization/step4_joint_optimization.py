import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import sph_harm_y

OUTPUT_DIR = "optimization_0813_three_flat"
GRID_RESOLUTION = 256
GRID_SIZE_MM = 120.0
MAX_DEGREE = 30
NUM_SAMPLES_THETA = 150
NUM_SAMPLES_PHI = 300

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

def get_laplacian_weights(l_max):
    """
    Returns the spectral Laplacian penalty weights l*(l+1) for each coefficient.
    """
    num_coeffs = (l_max + 1)**2
    weights = np.zeros(num_coeffs)
    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            weights[idx] = l * (l + 1)
            idx += 1
    return weights

class JointMorphologyOptimizer(nn.Module):
    def __init__(self, init_coeffs, Y_mat, centroid, sdt_tensor, laplacian_weights):
        super().__init__()
        # Learnable SH parameters
        self.coeffs = nn.Parameter(torch.tensor(init_coeffs, dtype=torch.float32))
        
        # Fixed basis matrix
        self.Y_mat = torch.tensor(Y_mat, dtype=torch.float32)
        
        # Fixed spatial geometry
        self.centroid = torch.tensor(centroid, dtype=torch.float32)
        
        # Continuous Implicit Field Volume (1, 1, D, H, W)
        self.sdt_tensor = sdt_tensor
        
        # Spectral Regularization Weights
        self.lap_weights = torch.tensor(laplacian_weights, dtype=torch.float32)
        
        # Pre-compute spherical angles for XYZ projection
        num_pts = Y_mat.shape[0]
        # We need the theta/phi used to generate Y_mat. 
        # But we can just pass them as buffers if we want, or generate them.
        # Actually, we can pass them in to avoid recomputing.

    def forward(self, angles_tensor):
        """
        angles_tensor: (N, 2) where col 0 is theta, col 1 is phi
        """
        # 1. Analytically reconstruct radii from SH coefficients
        # radii = Y_mat * coeffs
        radii = torch.matmul(self.Y_mat, self.coeffs)
        
        # 2. Project Spherical to Cartesian 3D Coordinates
        theta = angles_tensor[:, 0]
        phi = angles_tensor[:, 1]
        
        dx = torch.sin(theta) * torch.cos(phi)
        dy = torch.sin(theta) * torch.sin(phi)
        dz = torch.cos(theta)
        
        # points = centroid + r * directions
        points_x = self.centroid[0] + radii * dx
        points_y = self.centroid[1] + radii * dy
        points_z = self.centroid[2] + radii * dz
        
        pts_3d = torch.stack([points_x, points_y, points_z], dim=-1) # (N, 3)
        
        # 3. Convert absolute physical coordinates to GridSample [-1, 1] normalized coordinates
        # Grid range is [-60, 60] with Z shifted [-15, 105].
        # In step 2, grid was generated via:
        # x: [-60, 60], y: [-60, 60], z: [-15, 105]
        # grid_sample expects (x, y, z) in [-1, 1]
        norm_x = pts_3d[:, 0] / (GRID_SIZE_MM / 2.0)
        norm_y = pts_3d[:, 1] / (GRID_SIZE_MM / 2.0)
        # Z is centered at 45 ( (105 + -15)/2 = 45 )
        norm_z = (pts_3d[:, 2] - 45.0) / (GRID_SIZE_MM / 2.0)
        
        # F.grid_sample expects input (N, C, D, H, W) and grid (N, D_out, H_out, W_out, 3)
        # We have N points. We can treat them as a 1D spatial grid: (1, 1, 1, N, 3)
        grid = torch.stack([norm_x, norm_y, norm_z], dim=-1)
        grid = grid.view(1, 1, 1, -1, 3)
        
        # Sample SDT Field
        sdt_sampled = F.grid_sample(self.sdt_tensor, grid, mode='bilinear', align_corners=True)
        # sdt_sampled shape: (1, 1, 1, 1, N)
        sdt_values = sdt_sampled.squeeze() # (N,)
        
        # 4. Compute composite Loss
        # a) SDT Alignment Loss: minimize distance to the implicit surface (SDT = 0)
        loss_sdt = torch.mean(sdt_values**2)
        
        # b) Laplacian Spectral Regularization Loss: penalize high-frequency oscillations
        loss_lap = torch.sum(self.lap_weights * (self.coeffs**2))
        
        # Note: Feature Keypoint Loss (LoFTR) is bypassed here as the biological 
        # structure is textureless and unlabelled.
        
        total_loss = loss_sdt + 1e-5 * loss_lap
        
        return total_loss, loss_sdt, loss_lap

def main():
    print("Loading initial SH parameters and Phase 2 Implicit SDT Field...")
    init_coeffs = np.load(os.path.join(OUTPUT_DIR, "sh_coefficients.npy"))
    centroid = np.load(os.path.join(OUTPUT_DIR, "sh_centroid.npy"))
    sdt_mm = np.load(os.path.join(OUTPUT_DIR, "sdt_volume.npy"))
    
    # Prepare PyTorch Tensors
    # sdt_mm is (X, Y, Z). grid_sample expects (N, C, D, H, W) = (1, 1, Z, Y, X)
    # Wait, grid_sample expects coordinates as (x, y, z) corresponding to the last 3 dimensions (W, H, D).
    # So if tensor is (Z, Y, X), grid_sample maps grid_x to the last dim (X), grid_y to Y, grid_z to Z.
    # Our sdt_mm was generated from meshgrid(x, y, z, indexing='ij') -> (X, Y, Z).
    # We must transpose to (Z, Y, X) so grid_sample's (x, y, z) matches correctly.
    sdt_zyx = np.transpose(sdt_mm, (2, 1, 0))
    sdt_tensor = torch.tensor(sdt_zyx, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Generate sampling grid
    theta_1d = np.linspace(0.001, np.pi - 0.001, NUM_SAMPLES_THETA)
    phi_1d = np.linspace(0, 2*np.pi, NUM_SAMPLES_PHI, endpoint=False)
    tm, pm = np.meshgrid(theta_1d, phi_1d, indexing='ij')
    
    theta_flat = tm.ravel()
    phi_flat = pm.ravel()
    angles_tensor = torch.tensor(np.column_stack((theta_flat, phi_flat)), dtype=torch.float32)
    
    Y_mat = get_basis_matrix(MAX_DEGREE, theta_flat, phi_flat)
    lap_weights = get_laplacian_weights(MAX_DEGREE)
    
    optimizer_module = JointMorphologyOptimizer(init_coeffs, Y_mat, centroid, sdt_tensor, lap_weights)
    
    # Adam Optimizer
    optim = torch.optim.Adam(optimizer_module.parameters(), lr=0.5)
    
    print("Beginning PyTorch Joint Optimization Loop (Auto-Differentiation)...")
    for epoch in range(150):
        optim.zero_grad()
        loss, l_sdt, l_lap = optimizer_module(angles_tensor)
        loss.backward()
        optim.step()
        
        if epoch % 25 == 0 or epoch == 149:
            print(f"Epoch {epoch:03d} | Total Loss: {loss.item():.4f} | SDT Loss: {l_sdt.item():.4f}")
            
    # Export Optimized Parameters
    opt_coeffs = np.array(optimizer_module.coeffs.detach().cpu().tolist())
    np.save(os.path.join(OUTPUT_DIR, "sh_coefficients_optimized.npy"), opt_coeffs)
    
    # Export refined mesh
    radii_mesh = np.dot(Y_mat, opt_coeffs).reshape(tm.shape)
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
            
    obj_path = os.path.join(OUTPUT_DIR, "sh_parameterized_surface_refined.obj")
    with open(obj_path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
    print(f"Phase 4 Complete! Highly Refined PyTorch Model saved to {obj_path}")

if __name__ == "__main__":
    main()
