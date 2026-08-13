import os
import subprocess
import time

RERUN_OUT_DIR = "/home/coding/github/morpho_scanner/0813_0726_cerf_rerun_10k"
DATASET_DIR = "captures_0726_nerf_dataset"
timestamp = "10k_run"
VENV_NS_EXPORT = ".venv_nerf/bin/ns-export"
CONFIG_YML = os.path.join(RERUN_OUT_DIR, "nerfacto", timestamp, "config.yml")

cmd_export = [
    VENV_NS_EXPORT, "pointcloud",
    "--load-config", CONFIG_YML,
    "--output-dir", RERUN_OUT_DIR,
    "--num-points", "100000",
    "--remove-outliers", "True",
    "--normal-method", "open3d",
    "--save-world-frame", "False"
]
print("Exporting Point Cloud...")
subprocess.run(cmd_export, check=True)

print("Finished exporting point cloud! Please run the rest of the post processing script manually.")
