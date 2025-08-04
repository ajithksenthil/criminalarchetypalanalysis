#!/bin/bash
# setup_enhanced_env.sh
# Setup script for enhanced criminal archetypal analysis

echo "=========================================="
echo "Enhanced Criminal Analysis Setup"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Download NLTK data if needed
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create necessary directories
echo "Creating directories..."
mkdir -p enhanced_results
mkdir -p temp_embeddings

# Check installation
echo ""
echo "Checking installation..."
python -c "
import sys
print('Python:', sys.version)
print()

# Check each required module
modules = [
    'numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn',
    'plotly', 'networkx', 'torch', 'sentence_transformers',
    'openai', 'nltk', 'kneed', 'umap', 'tqdm', 'statsmodels',
    'hmmlearn', 'ruptures', 'prefixspan', 'numba'
]

missing = []
for module in modules:
    try:
        __import__(module)
        print(f'✓ {module}')
    except ImportError:
        print(f'✗ {module} - MISSING')
        missing.append(module)

if missing:
    print(f'\nMissing modules: {missing}')
    print('Please install missing modules manually')
    sys.exit(1)
else:
    print('\nAll modules installed successfully!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Setup complete! 🎉"
    echo "=========================================="
    echo ""
    echo "To run enhanced analysis:"
    echo "  source venv/bin/activate"
    echo "  python run_enhanced_analysis.py --quick"
    echo ""
    echo "For help:"
    echo "  python run_enhanced_analysis.py --help"
else
    echo ""
    echo "Setup encountered errors. Please check the output above."
    exit 1
fi