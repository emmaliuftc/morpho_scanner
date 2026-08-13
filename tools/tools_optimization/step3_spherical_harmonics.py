import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import sph_harm_y
from skimage import measure
import matplotlib.pyplot as plt

OUTPUT_DIR = "optimization_0813"
GRID_RESOLUTION = 256
GRID_SIZE_MM = 120.0
MAX_DEGREE = 15 # SH max degree L
NUM_SAMPLES_THETA = 100
NUM_SAMPLES_PHI = 200

def real_sph_harm(l, m, theta, phi):
    """
    Compute real spherical harmonic Y_l^m(theta, phi)
    theta: polar angle [0, pi]
    phi: azimuthal angle [0, 2pi]
    """
    Y = sph_harm_y(l, abs(m), theta, phi)
    if m < 0:
        return np.sqrt(2) * ((-1)**m) * Y.imag
    elif m == 0:
        return Y.real
    else:
        return np.sqrt(2) * ((-1)**m) * Y.real

def get_basis_matrix(l_max, theta, phi):
    """
    Constructs the design matrix for SH fitting.
    """
    num_coeffs = (l_max + 1)**2
    num_points = len(theta)
    Y_mat = np.zeros((num_points, num_coeffs))
    
    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Y_mat[:, idx] = real_sph_harm(l, m, theta, phi)
            idx += 1
    return Y_mat

def bisection_ray_cast(interpolator, centroid, directions, r_min=0.0, r_max=60.0, iters=20):
    """
    Finds the exact zero-crossing (r) along each ray using bisection.
    """
    r_low = np.full(len(directions), r_min)
    r_high = np.full(len(directions), r_max)
    
    for _ in range(iters):
        r_mid = (r_low + r_high) / 2.0
        points = centroid + r_mid[:, np.newaxis] * directions
        # Sample SDT
        sdt_vals = interpolator(points)
        # Assuming SDT < 0 inside, > 0 outside
        inside_mask = sdt_vals < 0
        r_low[inside_mask] = r_mid[inside_mask]
        r_high[~inside_mask] = r_mid[~inside_mask]
        
    return (r_low + r_high) / 2.0

def main():
    print("Loading SDT volume from Phase 2...")
    sdt_mm = np.load(os.path.join(OUTPUT_DIR, "sdt_volume.npy"))
    
    half_size = GRID_SIZE_MM / 2.0
    x = np.linspace(-half_size, half_size, GRID_RESOLUTION)
    y = np.linspace(-half_size, half_size, GRID_RESOLUTION)
    z = np.linspace(-15.0, GRID_SIZE_MM - 15.0, GRID_RESOLUTION)
    
    print("Creating Continuous SDT Field (RegularGridInterpolator)...")
    sdt_interp = RegularGridInterpolator((x, y, z), sdt_mm, bounds_error=False, fill_value=GRID_SIZE_MM)
    
    # Calculate Centroid
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    inside_mask = sdt_mm < 0
    if not np.any(inside_mask):
        print("Error: No object found in SDT (all values > 0).")
        return
        
    cx = np.mean(xx[inside_mask])
    cy = np.mean(yy[inside_mask])
    cz = np.mean(zz[inside_mask])
    centroid = np.array([cx, cy, cz])
    print(f"Volumetric Centroid calculated at: {centroid} mm")
    
    # Create Spherical Ray Grid
    print(f"Casting rays to map surface (theta={NUM_SAMPLES_THETA}, phi={NUM_SAMPLES_PHI})...")
    theta_1d = np.linspace(0.001, np.pi - 0.001, NUM_SAMPLES_THETA)
    phi_1d = np.linspace(0, 2*np.pi, NUM_SAMPLES_PHI, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta_1d, phi_1d, indexing='ij')
    
    theta_flat = theta_grid.ravel()
    phi_flat = phi_grid.ravel()
    
    # Cartesian ray directions
    dx = np.sin(theta_flat) * np.cos(phi_flat)
    dy = np.sin(theta_flat) * np.sin(phi_flat)
    dz = np.cos(theta_flat)
    directions = np.column_stack((dx, dy, dz))
    
    # Cast rays to find exact boundary
    radii = bisection_ray_cast(sdt_interp, centroid, directions)
    
    # Filter out rays that never hit the boundary (r remains near r_max)
    valid_mask = radii < 59.0
    valid_radii = radii[valid_mask]
    valid_theta = theta_flat[valid_mask]
    valid_phi = phi_flat[valid_mask]
    print(f"Successfully hit boundary on {len(valid_radii)}/{len(radii)} rays.")
    
    # Parameter Fitting (Spherical Harmonics)
    print(f"Fitting Spherical Harmonics (Max Degree = {MAX_DEGREE})...")
    Y_mat = get_basis_matrix(MAX_DEGREE, valid_theta, valid_phi)
    
    # Solve Weighted Least Squares system: Y_mat * coeffs = radii
    # Standard OLS for simplicity here
    coeffs, residuals, rank, s = np.linalg.lstsq(Y_mat, valid_radii, rcond=None)
    
    num_coeffs = len(coeffs)
    print(f"Solved for {num_coeffs} SH coefficients.")
    
    # Save coefficients
    coeffs_path = os.path.join(OUTPUT_DIR, "sh_coefficients.npy")
    np.save(coeffs_path, coeffs)
    
    # Save centroid
    np.save(os.path.join(OUTPUT_DIR, "sh_centroid.npy"), centroid)
    print(f"Saved parameterization to {OUTPUT_DIR}/sh_coefficients.npy")
    
    # Reconstruct surface from SH parameters for verification
    print("Reconstructing mathematical surface from parameters...")
    # Generate high-res mesh
    theta_mesh = np.linspace(0, np.pi, 100)
    phi_mesh = np.linspace(0, 2*np.pi, 200)
    tm, pm = np.meshgrid(theta_mesh, phi_mesh, indexing='ij')
    
    Y_mesh = get_basis_matrix(MAX_DEGREE, tm.ravel(), pm.ravel())
    radii_mesh = np.dot(Y_mesh, coeffs)
    radii_mesh = radii_mesh.reshape(tm.shape)
    
    # Spherical to Cartesian
    x_sh = centroid[0] + radii_mesh * np.sin(tm) * np.cos(pm)
    y_sh = centroid[1] + radii_mesh * np.sin(tm) * np.sin(pm)
    z_sh = centroid[2] + radii_mesh * np.cos(tm)
    
    # Export mesh as .obj for visualization
    # We create faces for the spherical grid
    verts = np.column_stack((x_sh.ravel(), y_sh.ravel(), z_sh.ravel()))
    faces = []
    rows, cols = tm.shape
    for i in range(rows - 1):
        for j in range(cols - 1):
            p1 = i * cols + j
            p2 = p1 + 1
            p3 = (i + 1) * cols + j
            p4 = p3 + 1
            # Triangle 1
            faces.append([p1, p2, p3])
            # Triangle 2
            faces.append([p2, p4, p3])
            
    # Connect the azimuthal wrap-around
    for i in range(rows - 1):
        p1 = i * cols + (cols - 1)
        p2 = i * cols + 0
        p3 = (i + 1) * cols + (cols - 1)
        p4 = (i + 1) * cols + 0
        faces.append([p1, p2, p3])
        faces.append([p2, p4, p3])
            
    obj_path = os.path.join(OUTPUT_DIR, "sh_parameterized_surface.obj")
    with open(obj_path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
    print(f"Phase 3 Complete! Parameterized geometry saved as {obj_path}")

if __name__ == "__main__":
    main()
