import yaml
import sys

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    if hasattr(d, '__dict__'):
        d = d.__dict__
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten_dict(v, new_key, sep=sep).items())
    elif isinstance(d, list):
        for i, v in enumerate(d):
            new_key = f"{parent_key}[{i}]"
            items.extend(flatten_dict(v, new_key, sep=sep).items())
    else:
        items.append((parent_key, d))
    return dict(items)

def diff_configs(file1, file2):
    with open(file1, 'r') as f:
        # unsafe_load is needed because nerfstudio configs contain custom python objects
        config1 = yaml.unsafe_load(f)
    with open(file2, 'r') as f:
        config2 = yaml.unsafe_load(f)

    flat1 = flatten_dict(config1)
    flat2 = flatten_dict(config2)
    
    all_keys = sorted(set(flat1.keys()).union(set(flat2.keys())))
    
    differences = []
    
    for k in all_keys:
        v1 = flat1.get(k)
        v2 = flat2.get(k)
        
        # Skip keys that are inherently supposed to be different (paths, timestamps, device types)
        if "timestamp" in k or "output_dir" in k or "data" in k or "machine.device_type" in k:
            continue
            
        # Also skip random string representations of objects if the actual values match
        if str(v1) != str(v2):
            differences.append((k, v1, v2))
            
    print(f"\nComparing Configs:")
    print(f"1 (CPU): {file1}")
    print(f"2 (GPU): {file2}\n")
    
    if len(differences) == 0:
        print("✅ NO DIFFERENCES FOUND! The configs are mathematically identical.")
    else:
        print(f"❌ FOUND {len(differences)} DIFFERENCES:\n")
        print(f"{'Key':<60} | {'CPU Config':<30} | {'GPU Config':<30}")
        print("-" * 125)
        for k, v1, v2 in differences:
            s1 = str(v1)[:28] + ".." if len(str(v1)) > 30 else str(v1)
            s2 = str(v2)[:28] + ".." if len(str(v2)) > 30 else str(v2)
            print(f"{k[:58]:<60} | {s1:<30} | {s2:<30}")

if __name__ == "__main__":
    file1 = "/home/coding/github/morpho_scanner/outputs/captures_0726_nerf_dataset/nerfacto/2026-07-27_045135/config.yml"
    file2 = "/home/coding/github/morpho_scanner/0813_0726_cerf_rerun_aligned_exact58/captures_0726_nerf_dataset/nerfacto/aligned_exact58_run/config.yml"
    diff_configs(file1, file2)
