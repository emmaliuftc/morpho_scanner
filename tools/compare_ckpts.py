import torch
import sys

def summarize_checkpoint(ckpt_path):
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"  Step: {ckpt.get('step', 'N/A')}")
    if 'pipeline' in ckpt:
        state_dict = ckpt['pipeline']
        print(f"  Pipeline Keys: {len(state_dict)}")
        
        # Calculate some summary statistics
        total_params = 0
        total_norm = 0.0
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                total_params += v.numel()
                total_norm += torch.norm(v).item() ** 2
                
        total_norm = total_norm ** 0.5
        print(f"  Total Floating Point Parameters: {total_params}")
        print(f"  L2 Norm of weights: {total_norm:.4f}")
        
        # Check specific layers if it's nerfacto
        if '_model.field.mlp_base.layers.0.weight' in state_dict:
            layer = state_dict['_model.field.mlp_base.layers.0.weight']
            print(f"  MLP Base Layer 0 shape: {layer.shape}, mean: {layer.mean().item():.6f}")
    else:
        print("  No 'pipeline' state_dict found.")
    print("-" * 50)

if __name__ == "__main__":
    ckpt1 = "/home/coding/github/morpho_scanner/outputs/captures_0726_nerf_dataset/nerfacto/2026-07-27_045135/nerfstudio_models/step-000001999.ckpt"
    ckpt2 = "/home/coding/github/morpho_scanner/0813_0726_cerf_rerun/captures_0726_nerf_dataset/nerfacto/2026-08-13_183217/nerfstudio_models/step-000001999.ckpt"
    
    summarize_checkpoint(ckpt1)
    summarize_checkpoint(ckpt2)
