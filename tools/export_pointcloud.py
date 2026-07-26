import torch
import sys

# Monkey-patch torch.load to globally force weights_only=False
# This bypasses PyTorch 2.6+ unpickling security restrictions for trusted local checkpoints
_orig_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# Invoke the standard nerfstudio exporter entrypoint
from nerfstudio.scripts.exporter import entrypoint

if __name__ == "__main__":
    entrypoint()
