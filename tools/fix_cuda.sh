#!/bin/bash

# This script helps diagnose and fix CUDA issues

echo "===== CUDA Environment Diagnostic ====="
echo "Checking NVIDIA drivers..."

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers found. Running nvidia-smi:"
    nvidia-smi
else
    echo "ERROR: nvidia-smi not found. NVIDIA drivers may not be installed correctly."
    echo "Install NVIDIA drivers with: sudo apt install nvidia-driver-535"
    exit 1
fi

# Check CUDA version
echo -e "\nChecking CUDA installation..."
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "CUDA version (nvcc): $NVCC_VERSION"
    
    # Check PyTorch CUDA compatibility
    PYTORCH_CUDA=$(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'Not available')")
    echo "PyTorch CUDA version: $PYTORCH_CUDA"
    
    if [[ "$PYTORCH_CUDA" == "Not available" ]]; then
        echo "PyTorch cannot access CUDA. Checking compatibility..."
        
        # Get PyTorch build version
        PYTORCH_BUILD=$(python -c "import torch; print(torch.__version__)")
        echo "PyTorch version: $PYTORCH_BUILD"
        
        if [[ "$PYTORCH_BUILD" == *"cu124"* ]]; then
            echo "Your PyTorch was built for CUDA 12.4"
            if [[ "$NVCC_VERSION" != "12.4"* ]]; then
                echo "MISMATCH: System CUDA ($NVCC_VERSION) doesn't match PyTorch's CUDA (12.4)"
                echo "Options:"
                echo "1. Install CUDA 12.4 toolkit"
                echo "2. Reinstall PyTorch with the correct CUDA version: pip install torch==2.6.0+cu121"
            fi
        elif [[ "$PYTORCH_BUILD" == *"cu121"* ]]; then
            echo "Your PyTorch was built for CUDA 12.1"
        else
            echo "Your PyTorch CUDA version couldn't be determined from build string"
        fi
    fi
else
    echo "CUDA toolkit not found. Install with: sudo apt install cuda-toolkit-12-4"
fi