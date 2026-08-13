import torch
import sys
import numpy as np

def compare_ckpts(ckpt1_path, ckpt2_path):
    print(f"Loading Checkpoint 1 (CPU): {ckpt1_path}")
    ckpt1 = torch.load(ckpt1_path, map_location="cpu")
    
    print(f"Loading Checkpoint 2 (GPU): {ckpt2_path}")
    ckpt2 = torch.load(ckpt2_path, map_location="cpu")
    
    state1 = ckpt1.get('pipeline', {})
    state2 = ckpt2.get('pipeline', {})
    
    common_keys = set(state1.keys()).intersection(set(state2.keys()))
    print(f"\nFound {len(common_keys)} common tensors.")
    
    print(f"\n{'Tensor Name':<65} | {'Shape':<15} | {'MAE':<12} | {'Cosine Sim':<12}")
    print("-" * 110)
    
    total_mae = 0.0
    total_elements = 0
    
    for k in sorted(common_keys):
        t1 = state1[k]
        t2 = state2[k]
        
        if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
            continue
            
        if t1.shape != t2.shape:
            continue
            
        if not t1.is_floating_point() or not t2.is_floating_point():
            continue
            
        mae = torch.mean(torch.abs(t1 - t2)).item()
        
        t1_flat = t1.flatten()
        t2_flat = t2.flatten()
        
        if torch.norm(t1_flat) > 0 and torch.norm(t2_flat) > 0:
            cos_sim = torch.nn.functional.cosine_similarity(t1_flat.unsqueeze(0), t2_flat.unsqueeze(0)).item()
        else:
            cos_sim = float('nan')
            
        total_mae += torch.sum(torch.abs(t1 - t2)).item()
        total_elements += t1.numel()
        
        print(f"{k[:63]:<65} | {str(list(t1.shape)):<15} | {mae:<12.6f} | {cos_sim:<12.4f}")
        
    overall_mae = total_mae / total_elements if total_elements > 0 else 0
    print("-" * 110)
    print(f"OVERALL MEAN ABSOLUTE ERROR ACROSS ALL MATCHING WEIGHTS: {overall_mae:.6f}")

if __name__ == "__main__":
    ckpt_cpu = "/home/coding/github/morpho_scanner/outputs/captures_0726_nerf_dataset/nerfacto/2026-07-27_045135/nerfstudio_models/step-000001999.ckpt"
    ckpt_gpu_exact58 = "/home/coding/github/morpho_scanner/0813_0726_cerf_rerun_aligned_exact58/captures_0726_nerf_dataset/nerfacto/aligned_exact58_run/nerfstudio_models/step-000001999.ckpt"
    
    compare_ckpts(ckpt_cpu, ckpt_gpu_exact58)
