#!/bin/bash
# run_fast_improved.sh - Run improved analysis quickly

echo "Running FAST improved analysis (without slow LLM processing)..."
echo "============================================================"

# Load .env if exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✓ Loaded .env file"
fi

# Activate virtual environment
source venv/bin/activate

# Set options for fast processing
export TOKENIZERS_PARALLELISM=false

echo ""
echo "Using optimizations:"
echo "  - TF-IDF embeddings (fast)"
echo "  - Auto-select optimal k"
echo "  - Ensemble clustering"
echo "  - LLM labeling at the end (if API key available)"
echo ""

# Run with fast options
python run_analysis_improved.py \
    --auto_k \
    --clustering_method kmeans \
    --use_tfidf \
    --output_dir "results_improved_$(date +%Y%m%d_%H%M%S)" \
    "$@"

echo ""
echo "Analysis complete! Check the output directory for results."