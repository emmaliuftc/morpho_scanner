import trimesh
import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes
import argparse

def create_golden_contour(stl_path, out_ply_path):
    print(f"Loading {stl_path}...")
    mesh = trimesh.load(stl_path)
    
    print("Shifting STL down by 5mm in Z...")
    mesh.apply_translation([0.0, 0.0, -5.0])
    
    # We don't scale by 150 because it's already in mm
    # Define the quantization grid boundaries
    x_range = [-26.0, 26.0]
    y_range = [-26.0, 26.0]
    z_range = [-6.0, 21.0]
    bins = [100, 100, 100]
    
    print("Generating 100x100x100 grid coordinates...")
    # Generate the center coordinate of each voxel
    x = np.linspace(x_range[0], x_range[1], bins[0], endpoint=False) + (x_range[1]-x_range[0])/(2*bins[0])
    y = np.linspace(y_range[0], y_range[1], bins[1], endpoint=False) + (y_range[1]-y_range[0])/(2*bins[1])
    z = np.linspace(z_range[0], z_range[1], bins[2], endpoint=False) + (z_range[1]-z_range[0])/(2*bins[2])
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    query_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    print("Checking which voxel centers are inside the STL volume (this might take a few seconds)...")
    # check inside
    contains = mesh.contains(query_points)
    
    volume = contains.reshape((bins[0], bins[1], bins[2])).astype(float)
    
    print("Running Marching Cubes to extract the 3D contour...")
    try:
        verts, faces, normals, values = marching_cubes(volume, level=0.5)
    except ValueError as e:
        print("Error running marching cubes.", e)
        return

    # Map voxel coordinates back to physical mm space
    x_spacing = (x_range[1] - x_range[0]) / bins[0]
    y_spacing = (y_range[1] - y_range[0]) / bins[1]
    z_spacing = (z_range[1] - z_range[0]) / bins[2]
    
    verts_mm = np.zeros_like(verts)
    verts_mm[:, 0] = verts[:, 0] * x_spacing + x_range[0]
    verts_mm[:, 1] = verts[:, 1] * y_spacing + y_range[0]
    verts_mm[:, 2] = verts[:, 2] * z_spacing + z_range[0]
    
    print(f"Extracted a contour mesh with {len(verts_mm)} vertices and {len(faces)} faces.")
    
    # Save as Open3D TriangleMesh
    out_mesh = o3d.geometry.TriangleMesh()
    out_mesh.vertices = o3d.utility.Vector3dVector(verts_mm)
    out_mesh.triangles = o3d.utility.Vector3iVector(faces)
    out_mesh.compute_vertex_normals()
    
    # Paint it gold
    out_mesh.paint_uniform_color([1.0, 0.84, 0.0])
    
    o3d.io.write_triangle_mesh(out_ply_path, out_mesh)
    print(f"Saved golden contour mesh to {out_ply_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    create_golden_contour(args.input, args.output)
