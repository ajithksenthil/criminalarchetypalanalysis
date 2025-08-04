#!/bin/bash
# run_improved_analysis_fixed.sh
# Run improved analysis with proper settings

# Activate virtual environment
source venv/bin/activate

# Set environment variable to avoid tokenizers warning
export TOKENIZERS_PARALLELISM=false

# Run the improved analysis with auto k selection
echo "Running improved criminal archetypal analysis..."
echo "=============================================="

python run_analysis_improved.py \
    --auto_k \
    --clustering_method kmeans \
    --reduce_dims \
    "$@"

echo ""
echo "Analysis complete!"