import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_cameras(json_path, output_path, title):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    xs, ys, zs = [], [], []
    for frame in data['frames']:
        mat = np.array(frame['transform_matrix'])
        xs.append(mat[0, 3])
        ys.append(mat[1, 3])
        zs.append(mat[2, 3])
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot cameras
    ax.scatter(xs, ys, zs, c='b', marker='o', label='Cameras')
    
    # Highlight Frame 0
    ax.scatter([xs[0]], [ys[0]], [zs[0]], c='r', marker='*', s=200, label='Frame 0 (Start)')
    
    # Plot Origin (Object Center)
    ax.scatter([0], [0], [0], c='k', marker='x', s=100, label='Object Center')
    
    # Draw lines from origin to cameras to visualize the "cone"
    for x, y, z in zip(xs, ys, zs):
        ax.plot([0, x], [0, y], [0, z], color='gray', alpha=0.3)
        
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Ensure axes are equally scaled so it looks like a circle, not an ellipse
    max_range = np.array([max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)]).max() / 2.0
    mid_x = (max(xs)+min(xs)) * 0.5
    mid_y = (max(ys)+min(ys)) * 0.5
    mid_z = (max(zs)+min(zs)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # For NeRF (OpenGL) Y is up. Let's adjust viewing angle so it's intuitive.
    # In matplotlib 3d, Z is usually up.
    # If Y is up in our world, let's set the view so Y points up on the screen.
    ax.view_init(elev=20, azim=45)
    
    ax.legend()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    
    plot_cameras(args.json, args.out, args.title)
