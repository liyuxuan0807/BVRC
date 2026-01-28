# Installation Guide

## Quick Start

### 1. Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n bvrc python=3.9
conda activate bvrc

# Or using venv
python -m venv bvrc_env
source bvrc_env/bin/activate  # On Linux/Mac
# or
bvrc_env\Scripts\activate  # On Windows
```

### 2. Install PyTorch

Choose the appropriate command based on your system:

#### CUDA 11.8
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### CUDA 12.1
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### CPU Only
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Installation Options

### Option 1: Full Installation (All Regressors)

Install all dependencies for BVRC-X, BVRC-T, and BVRC-D:

```bash
pip install -r requirements.txt
```

### Option 2: Minimal Installation (BVRC-X Only)

If you only need XGBoost regressor:

```bash
pip install torch torchvision numpy pandas scikit-learn scipy Pillow tqdm xgboost
```

### Option 3: TabPFN Only (BVRC-T)

If you only need TabPFN regressor:

```bash
pip install torch torchvision numpy pandas scikit-learn scipy Pillow tqdm tabpfn
```

### Option 4: Training Only

If you only need the training script:

```bash
pip install torch torchvision numpy pandas scikit-learn Pillow tqdm transformers
```

## Verify Installation

Run the following Python code to verify your installation:

```python
import torch
import torchvision
import numpy as np
import pandas as pd
import xgboost
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from transformers import AutoModel

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

print("\nAll packages installed successfully!")
```

Save as `test_installation.py` and run:

```bash
python test_installation.py
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size or use CPU

```bash
python BVRC_main.py --device cpu ...
```

### Issue: TabPFN Installation Failed

**Solution**: Install from source

```bash
pip install git+https://github.com/automl/TabPFN.git
```

### Issue: Transformers Version Conflict

**Solution**: Update transformers

```bash
pip install --upgrade transformers
```

### Issue: XGBoost GPU Not Working

**Solution**: Ensure CUDA toolkit matches your PyTorch CUDA version

```bash
nvidia-smi  # Check your CUDA driver version
python -c "import torch; print(torch.version.cuda)"  # Check PyTorch CUDA version
```

## Platform-Specific Notes

### Linux
- Recommended for GPU training
- All features fully supported

### Windows
- Use WSL2 for best CUDA support
- Native Windows support available but may have performance limitations

### macOS
- CPU-only mode (Apple Silicon: MPS backend supported in PyTorch >=2.0)
- Recommended for small-scale experiments

## Updating Dependencies

To update all packages to the latest compatible versions:

```bash
pip install --upgrade -r requirements.txt
```

## Development Installation

For development with editable mode:

```bash
pip install -e .
```

(Requires setup.py - create one if needed)

## Docker Installation (Advanced)

A Dockerfile is provided for containerized deployment:

```bash
docker build -t bvrc:latest .
docker run --gpus all -v $(pwd):/workspace bvrc:latest
```

## Conda Environment Export

After successful installation, export your environment:

```bash
conda env export > environment.yml
```

To recreate the environment later:

```bash
conda env create -f environment.yml
```
