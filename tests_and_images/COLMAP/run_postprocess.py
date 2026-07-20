import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 2. Run plot_sparse_points.py
print("\nRunning plot_sparse_points.py...")
import plot_sparse_points
plot_sparse_points.generate_projections("COLMAP/workspace")

# 3. Run surface_reconstruction.py
print("\nRunning surface_reconstruction.py...")
import surface_reconstruction
surface_reconstruction.reconstruct_surface("COLMAP/workspace")

# 4. Run stage_6_poisson_input.py
print("\nRunning stage_6_poisson_input.py...")
import stage_6_poisson_input
from stage_1 import LegoReconstructionPipeline
pipeline = LegoReconstructionPipeline("COLMAP/workspace", init_workspace=False)
stage_6_poisson_input.stage_6_poisson_input(pipeline)

# 5. Run plot_final_points.py
print("\nRunning plot_final_points.py...")
import plot_final_points
plot_final_points.generate_final_projections("COLMAP/workspace")

print("\nPost-processing pipeline complete successfully!")
