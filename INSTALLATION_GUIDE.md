# Installation Guide for Enhanced Criminal Archetypal Analysis

## Quick Setup

### 1. Automated Setup (Recommended)
```bash
# Make setup script executable
chmod +x setup_enhanced_env.sh

# Run setup
./setup_enhanced_env.sh
```

### 2. Manual Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Troubleshooting Common Issues

### 1. "ModuleNotFoundError" Errors

If you get import errors, try:

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall specific package
pip install --force-reinstall package_name
```

### 2. Specific Package Issues

#### **hmmlearn** Installation Failed
```bash
# Install build dependencies first
pip install numpy scipy scikit-learn
pip install hmmlearn
```

#### **ruptures** Installation Failed
```bash
# Install without optional dependencies
pip install ruptures --no-deps
pip install numpy scipy
```

#### **umap-learn** Installation Failed
```bash
# Install dependencies first
pip install numpy scipy scikit-learn numba
pip install umap-learn
```

#### **torch** Installation Issues
```bash
# CPU-only version (smaller)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or check PyTorch website for your system:
# https://pytorch.org/get-started/locally/
```

### 3. Minimal Installation

If you have persistent issues, use the minimal requirements:

```bash
pip install -r requirements_minimal.txt
```

This installs only essential packages. Some advanced features may be limited.

### 4. Platform-Specific Issues

#### **macOS**
```bash
# If you get compiler errors
brew install libomp

# For M1/M2 Macs with scipy issues
pip install --upgrade pip
pip install scipy --no-binary :all:
```

#### **Windows**
```bash
# Use Anaconda Python for easier installation
conda install -c conda-forge scikit-learn scipy numpy

# Then install remaining with pip
pip install -r requirements.txt
```

#### **Linux**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev build-essential

# For numerical libraries
sudo apt-get install libatlas-base-dev gfortran
```

### 5. Memory Issues

If you run out of memory during installation:

```bash
# Install packages one by one
pip install numpy
pip install pandas
pip install scikit-learn
# ... continue with other packages
```

### 6. Verify Installation

Check if everything is installed correctly:

```python
python -c "
import numpy
import pandas
import sklearn
import torch
import plotly
print('Core packages installed successfully!')
"
```

## Package Versions

If you need specific versions for compatibility:

```bash
# Create environment with specific Python version
python3.9 -m venv venv

# Install specific package versions
pip install numpy==1.21.6
pip install scikit-learn==1.0.2
pip install torch==1.13.1
```

## Optional Dependencies

These packages enhance functionality but aren't required:

- **prefixspan**: For advanced sequential pattern mining
- **hmmlearn**: For Hidden Markov Models
- **ruptures**: For advanced change point detection

Install them separately if needed:
```bash
pip install prefixspan hmmlearn ruptures
```

## Running Without Full Installation

If you can't install all packages, you can still run basic analysis:

```bash
# Use original analysis (without enhancements)
python analysis_integration.py

# Or use improved clustering only
python run_analysis_improved.py
```

## Getting Help

1. Check error messages carefully - they often indicate the specific missing package
2. Try installing the minimal requirements first
3. Use virtual environments to avoid conflicts
4. Consider using Anaconda for complex scientific packages

## Common Error Messages and Solutions

| Error | Solution |
|-------|----------|
| `No module named 'torch'` | `pip install torch` |
| `No module named 'plotly'` | `pip install plotly` |
| `No module named 'tqdm'` | `pip install tqdm` |
| `No module named 'statsmodels'` | `pip install statsmodels` |
| `ImportError: numba` | `pip install numba` |
| `Microsoft Visual C++ required` | Install Visual Studio Build Tools (Windows) |
| `error: Microsoft Visual C++ 14.0 or greater is required` | Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/ |

## Testing Your Installation

After installation, test with:

```bash
# Quick test of enhanced analysis
python run_enhanced_analysis.py --quick --n_permutations 10 --n_bootstrap 10
```

This runs a minimal version to verify everything works.