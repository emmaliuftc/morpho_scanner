import open3d as o3d
from pathlib import Path

ply_path = Path("COLMAP/workspace_7-19_lob_with_marker_2/epipolar_ncc_sculpture_3d.ply")

if not ply_path.exists():
    print(f"File not found: {ply_path}")
else:
    print(f"Loading 3D Point Cloud from {ply_path}...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(f"Loaded {len(pcd.points):,} points.")
    print("Opening interactive Open3D window (Left click: rotate, Right click: translate, Scroll: zoom)...")
    o3d.visualization.draw_geometries([pcd], window_name="Clay Sculpture 3D Point Cloud (Epipolar NCC)", width=1280, height=960)
