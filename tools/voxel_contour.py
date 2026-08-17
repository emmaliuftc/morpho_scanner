import open3d as o3d
import numpy as np
import argparse
from skimage.measure import marching_cubes

def extract_voxel_contour(ply_path, out_ply_path):
    print(f"Loading {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    
    # 1. Scale from NeRF units (if 1 unit = 150mm) to mm
    points_mm = points * 150.0
    
    # 2. Define the quantization grid boundaries
    # X: [-26, 26], Y: [-26, 26], Z: [-6, 21]
    x_range = [-26.0, 26.0]
    y_range = [-26.0, 26.0]
    z_range = [-6.0, 21.0]
    bins = [100, 100, 100]
    
    print("Quantizing points into a 100x100x100 voxel grid...")
    # np.histogramdd creates a 3D histogram (occupancy grid)
    H, edges = np.histogramdd(
        points_mm, 
        bins=bins, 
        range=[x_range, y_range, z_range]
    )
    
    # Create a binary volume: 1 if voxel contains points, 0 otherwise
    volume = (H > 0).astype(float)
    
    # Optional: Fill internal holes in the volume if needed using morphological operations
    # (Since the input is already a solid extruded pointcloud, this might not be necessary)
    
    print("Running Marching Cubes to extract the 3D contour...")
    # marching_cubes returns vertices, faces, normals, and values
    # level=0.5 finds the contour exactly halfway between 0 (empty) and 1 (filled)
    try:
        verts, faces, normals, values = marching_cubes(volume, level=0.5)
    except ValueError as e:
        print("Error running marching cubes. The volume might be completely empty or fully solid at the boundaries.", e)
        return

    # 3. Map voxel coordinates (0 to 100) back to physical mm space
    x_spacing = (x_range[1] - x_range[0]) / bins[0]
    y_spacing = (y_range[1] - y_range[0]) / bins[1]
    z_spacing = (z_range[1] - z_range[0]) / bins[2]
    
    verts_mm = np.zeros_like(verts)
    verts_mm[:, 0] = verts[:, 0] * x_spacing + x_range[0]
    verts_mm[:, 1] = verts[:, 1] * y_spacing + y_range[0]
    verts_mm[:, 2] = verts[:, 2] * z_spacing + z_range[0]
    
    print(f"Extracted a contour mesh with {len(verts_mm)} vertices and {len(faces)} faces.")
    
    # 4. Save as an Open3D TriangleMesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_mm)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    
    # Paint it an easy-to-see color (e.g. green)
    mesh.paint_uniform_color([0.1, 0.8, 0.1])
    
    o3d.io.write_triangle_mesh(out_ply_path, mesh)
    print(f"Saved contour mesh to {out_ply_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input PLY pointcloud")
    parser.add_argument("--output", required=True, help="Output PLY contour mesh")
    args = parser.parse_args()
    
    extract_voxel_contour(args.input, args.output)
