import open3d as o3d
import argparse

def main():
    parser = argparse.ArgumentParser(description="Apply Taubin Smoothing to a Mesh")
    parser.add_argument("--input", type=str, required=True, help="Input .ply mesh")
    parser.add_argument("--output", type=str, required=True, help="Output smoothed .ply mesh")
    parser.add_argument("--iterations", type=int, default=15, help="Number of Taubin iterations")
    args = parser.parse_args()

    print(f"Loading mesh from {args.input}...")
    mesh = o3d.io.read_triangle_mesh(args.input)
    
    print(f"Applying Taubin 3D Smoothing ({args.iterations} iterations)...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=args.iterations)
    mesh.compute_vertex_normals()
    
    print(f"Saving smoothed mesh to {args.output}...")
    o3d.io.write_triangle_mesh(args.output, mesh)
    print("Done!")

if __name__ == "__main__":
    main()
