import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("exports/three_flat_aligned_pc_cropped/point_cloud.ply")
pts = np.asarray(pcd.points)

print("Total points:", len(pts))
for axis, name in enumerate(['X', 'Y', 'Z']):
    print(f"--- {name} Axis ---")
    print(f"Min: {pts[:, axis].min():.3f}, Max: {pts[:, axis].max():.3f}")
    print(f"Points > 0: {np.sum(pts[:, axis] > 0)}")
    print(f"Points < 0: {np.sum(pts[:, axis] < 0)}")
