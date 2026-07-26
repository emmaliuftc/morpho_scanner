import os
import sys
import logging
from pathlib import Path
import pycolmap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def visualize_sparse_reconstruction(workspace_dir: str):
    workspace = Path(workspace_dir)
    sparse_model_dir = workspace / "sparse" / "0"
    output_dir = workspace / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    if not sparse_model_dir.exists():
        logging.error(f"Sparse model directory not found at {sparse_model_dir}. Have you run Stage 4?")
        sys.exit(1)
        
    # Load the reconstruction
    logging.info(f"Loading sparse reconstruction from {sparse_model_dir}...")
    reconstruction = pycolmap.Reconstruction()
    reconstruction.read(str(sparse_model_dir))
    
    logging.info("Sparse Reconstruction Summary:")
    logging.info(f"  Registered images: {reconstruction.num_reg_images()} / {reconstruction.num_images()}")
    logging.info(f"  3D points: {reconstruction.num_points3D()}")
    
    # Export 3D points to PLY format
    ply_output_path = output_dir / "sparse_model.ply"
    logging.info(f"Exporting sparse point cloud to PLY format at {ply_output_path}...")
    reconstruction.export_PLY(str(ply_output_path))
    logging.info("Export completed successfully.")

def main():
    workspace_path = "./workspace"
    if len(sys.argv) > 1:
        workspace_path = sys.argv[1]
        
    visualize_sparse_reconstruction(workspace_path)

if __name__ == "__main__":
    main()
