import open3d as o3d
import numpy as np
import cv2
import json

print("Loading point cloud...")
pcd = o3d.io.read_point_cloud("exports/three_flat_pc_cropped/point_cloud_filtered.ply")
points = np.asarray(pcd.points)

print("Loading transforms...")
with open("data/three_flat_real_calib/transforms.json", "r") as f:
    transforms = json.load(f)

frame = transforms["frames"][0]
c2w = np.array(frame["transform_matrix"])
# The NeRF c2w has the OpenGL flip (Y and Z negated). Let's undo it to get OpenCV c2w.
flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])
c2w_cv = c2w @ flip_yz
w2c_cv = np.linalg.inv(c2w_cv)

print("Loading image...")
# Load the actual image
img_path = "data/three_flat_real_calib/" + frame["file_path"]
img = cv2.imread(img_path)

fl_x = transforms["fl_x"]
fl_y = transforms["fl_y"]
cx = transforms["cx"]
cy = transforms["cy"]

K = np.array([
    [fl_x, 0, cx],
    [0, fl_y, cy],
    [0, 0, 1]
])

print("Projecting points...")
rvec, _ = cv2.Rodrigues(w2c_cv[:3, :3])
tvec = w2c_cv[:3, 3]

# The point cloud is in NeRF world coordinates. 
# We need to remember that during export, the scale factor of 150.0 was baked in!
# Wait, NeRF exports the point cloud in the SCALED coordinate system.
# So the points are in the scaled world coordinates.
# But w2c_cv is ALSO in scaled world coordinates (because c2w[:3,3] was divided by 150).
# So we can just project them directly!
img_pts, _ = cv2.projectPoints(points, rvec, tvec, K, np.zeros(5))
img_pts = img_pts.reshape(-1, 2).astype(np.int32)

print("Drawing...")
for pt in img_pts:
    x, y = pt
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        img[y, x] = (0, 0, 255) # Red points

cv2.imwrite("/tmp/gemini_projection.png", img)
print("Saved /tmp/gemini_projection.png")
