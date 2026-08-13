import torch
import sys

def get_shapes(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    shapes = {}
    if 'pipeline' in ckpt:
        for k, v in ckpt['pipeline'].items():
            if isinstance(v, torch.Tensor):
                shapes[k] = list(v.shape)
    return shapes

if __name__ == "__main__":
    ckpt1 = "/home/coding/github/morpho_scanner/outputs/captures_0726_nerf_dataset/nerfacto/2026-07-27_045135/nerfstudio_models/step-000001999.ckpt"
    ckpt2 = "/home/coding/github/morpho_scanner/0813_0726_cerf_rerun/captures_0726_nerf_dataset/nerfacto/2026-08-13_183217/nerfstudio_models/step-000001999.ckpt"
    
    shapes1 = get_shapes(ckpt1)
    shapes2 = get_shapes(ckpt2)
    
    all_keys = set(shapes1.keys()).union(set(shapes2.keys()))
    print(f"{'Key':<60} | {'Original (CPU)':<20} | {'New GPU':<20}")
    print("-" * 105)
    
    for k in sorted(all_keys):
        s1 = str(shapes1.get(k, "MISSING"))
        s2 = str(shapes2.get(k, "MISSING"))
        if s1 != s2:
            print(f"{k[:58]:<60} | {s1:<20} | {s2:<20}")
