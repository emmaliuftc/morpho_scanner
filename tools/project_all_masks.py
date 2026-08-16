import os
import json
import cv2
import numpy as np
import open3d as o3d
import argparse

def project_all_masks(base_dir, pointcloud_file, alpha=1.0):
    transforms_path = os.path.join(base_dir, "transforms.json")
    mesh_path = os.path.join(base_dir, pointcloud_file)
    
    with open(transforms_path, 'r') as f:
        meta = json.load(f)
        
    width = meta['w']
    height = meta['h']
    K = np.array([
        [meta['fl_x'], 0, meta['cx']],
        [0, meta['fl_y'], meta['cy']],
        [0, 0, 1]
    ])
    
    print(f"Loading point cloud from {mesh_path}...")
    pcd = o3d.io.read_point_cloud(mesh_path)
    vertices = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255.0 if pcd.has_colors() else None
    
    pts_homo = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    
    out_dir = os.path.join(base_dir, "projections_side_by_side")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
    
    for frame in meta['frames']:
        file_path = frame['file_path']
        basename = os.path.basename(file_path)
        if basename.startswith("preview_"):
            basename = basename.replace("preview_", "")
            
        mask_path = os.path.join(base_dir, "masks_4", "mask_" + basename)
        orig_path = os.path.join(base_dir, "images_4", basename)
        
        if not os.path.exists(mask_path) or not os.path.exists(orig_path):
            print(f"Skipping {basename} - missing mask or image")
            continue
            
        bg_img = cv2.imread(mask_path)
        orig_img = cv2.imread(orig_path)
        
        if bg_img.shape[0] != height or bg_img.shape[1] != width:
            bg_img = cv2.resize(bg_img, (width, height))
            orig_img = cv2.resize(orig_img, (width, height))
            
        c2w = np.array(frame['transform_matrix'])
        w2c = np.linalg.inv(c2w)
        
        pts_cam = (w2c @ pts_homo.T).T
        pts_cam = pts_cam[:, :3]
        pts_cam[:, 1] *= -1
        pts_cam[:, 2] *= -1
        
        zc = pts_cam[:, 2]
        
        u = np.round(K[0, 0] * pts_cam[:, 0] / np.maximum(zc, 1e-5) + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * pts_cam[:, 1] / np.maximum(zc, 1e-5) + K[1, 2]).astype(int)
        
        overlay = bg_img.copy()
        for i in range(len(vertices)):
            if zc[i] > 0:
                color = (0, 255, 0)
                if colors is not None:
                    color = (int(colors[i][2]), int(colors[i][1]), int(colors[i][0]))
                # Make sure points are in bounds
                if 0 <= u[i] < width and 0 <= v[i] < height:
                    cv2.circle(overlay, (u[i], v[i]), radius=1, color=color, thickness=-1)
                
        result = cv2.addWeighted(bg_img, 1.0 - alpha, overlay, alpha, 0)
        
        # Create side-by-side comparison
        # Projected mask on left, original image on right
        side_by_side = np.hstack((result, orig_img))
        
        out_path = os.path.join(out_dir, "compare_" + basename)
        cv2.imwrite(out_path, side_by_side)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Base session directory")
    parser.add_argument("--ply", required=True, help="Point cloud file inside the directory")
    args = parser.parse_args()
    
    project_all_masks(args.dir, args.ply)
