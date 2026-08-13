import open3d as o3d
import numpy as np

def analyze_pcd(filepath, name):
    try:
        pcd = o3d.io.read_point_cloud(filepath)
        pts = np.asarray(pcd.points)
        num_points = len(pts)
        if num_points == 0:
            return {"name": name, "num_points": 0}
        
        min_bound = pts.min(axis=0)
        max_bound = pts.max(axis=0)
        mean_pt = pts.mean(axis=0)
        p1 = np.percentile(pts, 1, axis=0)
        p99 = np.percentile(pts, 99, axis=0)
        
        return {
            "name": name,
            "num_points": num_points,
            "min": min_bound,
            "max": max_bound,
            "mean": mean_pt,
            "p1": p1,
            "p99": p99
        }
    except Exception as e:
        return {"name": name, "error": str(e)}

def format_vec3(v):
    return f"[{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]"

if __name__ == "__main__":
    pcd_cpu_path = "/home/coding/github/morpho_scanner/captures_0726_nerf_dataset/point_cloud.ply"
    pcd_gpu_path = "/home/coding/github/morpho_scanner/0813_0726_cerf_rerun_aligned_exact58/point_cloud.ply"
    
    res_cpu = analyze_pcd(pcd_cpu_path, "Original 0726 CPU (point_cloud.ply)")
    res_gpu = analyze_pcd(pcd_gpu_path, "Latest 0813 GPU (point_cloud.ply)")
    
    with open("/home/coding/github/morpho_scanner/docs/compare_cpu_vs_gpu.md", "a") as f:
        f.write("\n\n## Point Cloud (.ply) Geometric Comparison\n\n")
        f.write("This table compares the raw geometry bounding boxes exported by `ns-export` for both models before post-processing filtering.\n\n")
        
        f.write(f"| Metric | {res_cpu['name']} | {res_gpu['name']} |\n")
        f.write("|--------|----------------------------------------|---------------------------------------|\n")
        f.write(f"| **Num Points** | {res_cpu['num_points']:,} | {res_gpu['num_points']:,} |\n")
        f.write(f"| **Min Bound**  | `{format_vec3(res_cpu['min'])}` | `{format_vec3(res_gpu['min'])}` |\n")
        f.write(f"| **Max Bound**  | `{format_vec3(res_cpu['max'])}` | `{format_vec3(res_gpu['max'])}` |\n")
        f.write(f"| **Mean Center**| `{format_vec3(res_cpu['mean'])}` | `{format_vec3(res_gpu['mean'])}` |\n")
        f.write(f"| **1st Percentile** | `{format_vec3(res_cpu['p1'])}` | `{format_vec3(res_gpu['p1'])}` |\n")
        f.write(f"| **99th Percentile**| `{format_vec3(res_cpu['p99'])}` | `{format_vec3(res_gpu['p99'])}` |\n")
        
    print("Point cloud comparison successfully appended to docs/compare_cpu_vs_gpu.md")
