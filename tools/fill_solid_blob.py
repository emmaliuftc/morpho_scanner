import trimesh
import numpy as np
import open3d as o3d
import argparse
from scipy.spatial import cKDTree

def make_solid_blob(mesh_path, out_ply_path, voxel_pitch=0.01, color_mode="nearest"):
    print(f"Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path)
    
    # We need a watertight mesh for inside/outside test to work perfectly.
    # The Poisson mesh from our pipeline is already watertight.
    print(f"Mesh is watertight: {mesh.is_watertight}")
    
    print(f"Voxelizing mesh with pitch={voxel_pitch}...")
    # Voxelize the surface
    surface_voxels = mesh.voxelized(pitch=voxel_pitch)
    
    print("Filling interior volume...")
    # Fill the inside to make it a solid blob
    solid_voxels = surface_voxels.fill()
    
    # Get all the XYZ coordinates of the solid voxels
    points = solid_voxels.points
    print(f"Generated {len(points)} solid points inside the volume.")
    
    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if color_mode == "black":
        print("Coloring solid interior black...")
        colors = np.zeros((len(points), 3))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif color_mode == "nearest":
        print("Projecting surface colors into the solid interior...")
        # Get surface colors
        o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)
        surface_points = np.asarray(o3d_mesh.vertices)
        surface_colors = np.asarray(o3d_mesh.vertex_colors)
        
        # Build KDTree to find nearest surface point for every voxel
        tree = cKDTree(surface_points)
        print("Querying KDTree...")
        _, idxs = tree.query(points)
        
        voxel_colors = surface_colors[idxs]
        pcd.colors = o3d.utility.Vector3dVector(voxel_colors)
        
    print(f"Saving solid binary blob to {out_ply_path}...")
    o3d.io.write_point_cloud(out_ply_path, pcd)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Input watertight mesh (.ply)")
    parser.add_argument("--out", required=True, help="Output solid pointcloud (.ply)")
    parser.add_argument("--pitch", type=float, default=0.01, help="Voxel size (resolution)")
    parser.add_argument("--color", choices=["black", "nearest"], default="nearest", help="How to color the interior")
    args = parser.parse_args()
    
    make_solid_blob(args.mesh, args.out, voxel_pitch=args.pitch, color_mode=args.color)
