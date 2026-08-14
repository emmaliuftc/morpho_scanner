import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("captures_0726_nerf_dataset/point_cloud.ply")
pts = np.asarray(pcd.points)
print("0726 Raw Z range:", pts[:,2].min(), "to", pts[:,2].max())
print("0726 Raw Points > 0 (Under table):", np.sum(pts[:,2] > 0))

pcd_filtered = o3d.io.read_point_cloud("captures_0726_nerf_dataset/point_cloud_filtered.ply")
pts_filtered = np.asarray(pcd_filtered.points)
print("0726 Filtered Z range:", pts_filtered[:,2].min(), "to", pts_filtered[:,2].max())
print("0726 Filtered Points > 0 (Under table):", np.sum(pts_filtered[:,2] > 0))
