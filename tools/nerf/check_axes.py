import open3d as o3d
import numpy as np
pcd = o3d.io.read_point_cloud("exports/three_flat_aligned_pc_cropped/point_cloud_filtered.ply")
pts = np.asarray(pcd.points)
print(f"X range: {pts[:,0].min():.3f} to {pts[:,0].max():.3f}")
print(f"Y range: {pts[:,1].min():.3f} to {pts[:,1].max():.3f}")
print(f"Z range: {pts[:,2].min():.3f} to {pts[:,2].max():.3f}")
