import os
import torch
import sys

def check_gpu_status():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    print("\n--- CUDA Availability ---")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nCUDA Device {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA is not available. Checking for possible issues...")
        
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            print(f"CUDA_VISIBLE_DEVICES is set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
            if os.environ['CUDA_VISIBLE_DEVICES'] == '-1' or os.environ['CUDA_VISIBLE_DEVICES'] == '':
                print("  This environment variable is hiding all GPUs from PyTorch.")
        
        print("\nPossible solutions:")
        print("1. Make sure NVIDIA drivers are installed correctly")
        print("2. Check that CUDA toolkit is installed and matches your PyTorch version")
        print("3. Ensure CUDA_VISIBLE_DEVICES is not set to hide GPUs")
    
    print("\n--- MPS Availability (Apple Silicon) ---")
    print(f"torch.backends.mps.is_available(): {torch.backends.mps.is_available()}")
    print(f"torch.backends.mps.is_built(): {torch.backends.mps.is_built()}")
    
    print("\n--- Environment Variables ---")
    for var in ['CUDA_VISIBLE_DEVICES', 'PYTORCH_ENABLE_MPS_FALLBACK', 
                'PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'PYTORCH_MPS_ENABLE_ASYNC_GPU_COPIES']:
        if var in os.environ:
            print(f"{var}: {os.environ[var]}")
        else:
            print(f"{var}: Not set")

if __name__ == "__main__":
    check_gpu_status()
