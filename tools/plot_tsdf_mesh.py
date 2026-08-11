import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

def plot_mesh(obj_path, output_path):
    print(f"Reading mesh from {obj_path}...")
    vertices = []
    colors = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(parts) >= 7:
                    colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
                else:
                    colors.append([0.5, 0.5, 0.5])
                
    vertices = np.array(vertices)
    colors = np.array(colors)
    print(f"Loaded {len(vertices)} vertices with colors.")
    
    if len(vertices) == 0:
        print("Error: No vertices found in OBJ file.")
        return
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot using the loaded RGB colors
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c=colors, s=1.5, alpha=0.8)
    
    ax.set_title("TSDF 3D Reconstruction with Surface Colors", fontsize=16)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    
    # Equal aspect ratio trick for 3D plot
    max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(), 
                          vertices[:, 1].max()-vertices[:, 1].min(), 
                          vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
                          
    mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Adjust view angle for a nice perspective
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved mesh preview to {output_path}")

if __name__ == "__main__":
    plot_mesh("captures_7-25_tsdf/clay_tsdf_mesh.obj", "captures_7-25_tsdf/mesh_preview.png")
