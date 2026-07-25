import pycolmap
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
import shutil
import sys

# Paths
workspace = Path("COLMAP/workspace")
db_path = workspace / "database.db"
image_dir = workspace / "images"
sparse_in = workspace / "sparse/1"
sparse_out = workspace / "sparse_ideal_triangulated"

# Intrinsics
if not sparse_in.exists():
    print(f"Error: Reference reconstruction not found at {sparse_in}")
    sys.exit(1)
    
recon_ref = pycolmap.Reconstruction(str(sparse_in))
camera_ref = recon_ref.cameras[1]
f_val, cx_val, cy_val, k_val = camera_ref.params

# Camera centers and normal fit
centers = np.array([img.projection_center() for img in recon_ref.images.values()])
center_mean = np.mean(centers, axis=0)
U, S, Vt = np.linalg.svd(centers - center_mean)
normal = Vt[2, :]
if normal[2] < 0: normal = -normal

# Base pose
img_0 = recon_ref.find_image_with_name("capture_0.jpg")
C0 = img_0.projection_center()
R0 = img_0.cam_from_world().rotation.matrix()

# Derived center
derived_plate_center_3d = np.array([0.8568, 1.5746, 2.7133])

# Run for 11.25 degrees
step_11_25 = np.radians(11.25)
print(f"Testing triangulation with step size = 11.25 degrees...")

# Generate ideal sparse folder
sparse_ideal_tmp = workspace / "sparse_tmp_11_25"
if sparse_ideal_tmp.exists():
    shutil.rmtree(sparse_ideal_tmp)
sparse_ideal_tmp.mkdir(exist_ok=True, parents=True)

cam_line = f"1 SIMPLE_RADIAL 4608 2592 {f_val} {cx_val} {cy_val} {k_val}"
with open(sparse_ideal_tmp / "cameras.txt", "w") as f_cam:
    f_cam.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    f_cam.write(cam_line + "\n")

# Read image name to ID mapping from database
if not db_path.exists():
    print(f"Error: Database not found at {db_path}")
    sys.exit(1)
    
db = pycolmap.Database.open(str(db_path))
db_images = db.read_all_images()
name_to_id = {img.name: img.image_id for img in db_images}
db.close()

with open(sparse_ideal_tmp / "images.txt", "w") as f_img:
    f_img.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n\n")
    for i in range(32):
        name = f"capture_{i}.jpg"
        if name not in name_to_id:
            continue
        img_id = name_to_id[name]
        
        theta = i * step_11_25
        r_theta = Rot.from_rotvec(theta * normal)
        R_theta = r_theta.as_matrix()
        
        pos = derived_plate_center_3d + R_theta @ (C0 - derived_plate_center_3d)
        r_theta_inv = Rot.from_rotvec(-theta * normal)
        R = R0 @ r_theta_inv.as_matrix()
        
        q = Rot.from_matrix(R).as_quat()
        qw, qx, qy, qz = q[3], q[0], q[1], q[2]
        T = -R @ pos
        
        f_img.write(f"{img_id} {qw} {qx} {qy} {qz} {T[0]} {T[1]} {T[2]} 1 {name}\n\n")

with open(sparse_ideal_tmp / "points3D.txt", "w") as f_pts:
    f_pts.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")

# Triangulate
recon_11_25 = pycolmap.Reconstruction(str(sparse_ideal_tmp))
triangulated = pycolmap.triangulate_points(
    reconstruction=recon_11_25,
    database_path=str(db_path),
    image_path=str(image_dir),
    output_path=str(sparse_out),
    clear_points=True
)

print(f"\n=== Triangulation Complete (11.25°) ===")
print(f"Registered Images: {triangulated.num_reg_images()} / {triangulated.num_images()}")
print(f"3D Points: {triangulated.num_points3D()}")

# Clean up
shutil.rmtree(sparse_ideal_tmp)
