import subprocess
import sys
import torch

def test_gpu():
    print("=" * 60)
    print(" 🛠️  GPU SETUP & DIAGNOSTIC TEST")
    print("=" * 60)

    # 1. Check PCI hardware
    print("\n1. Checking PCI Hardware (lspci):")
    try:
        res = subprocess.run(["lspci", "-v", "-s", "00:03.0"], capture_output=True, text=True)
        print(res.stdout if res.stdout else "No 00:03.0 device found.")
    except Exception as e:
        print(f"Error running lspci: {e}")

    # 2. Check /dev/nvidia devices
    print("\n2. Checking /dev/nvidia* device nodes:")
    import glob
    devs = glob.glob("/dev/nvidia*")
    print(f"Found /dev/nvidia nodes: {devs}")

    # 3. Check PyTorch CUDA status
    print("\n3. Checking PyTorch CUDA Status:")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Built-in Version: {torch.version.cuda}")
    print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU Count: {torch.cuda.device_count()}")
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        
        # Test basic tensor computation on CUDA GPU
        print("\n4. Running CUDA GPU Tensor Computation Test:")
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.matmul(x, x)
        print(f"   ✅ SUCCESS! CUDA Matrix Multiplication Output Shape: {y.shape} on {y.device}")
    else:
        print("\n❌ PyTorch CUDA is currently NOT available.")
        print("   Cause: NVIDIA kernel driver module (/dev/nvidia0) is not loaded in the container kernel.")

if __name__ == '__main__':
    test_gpu()
