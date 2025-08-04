#!/bin/bash
# run_with_5_clusters.sh - Run analysis with 5 clusters

echo "Running analysis with 5 clusters..."
echo "=================================="

# Load .env if exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✓ Loaded .env file"
fi

# Activate virtual environment
source venv/bin/activate

# Set environment variable to avoid warnings
export TOKENIZERS_PARALLELISM=false

# Run with 5 clusters
python run_analysis_improved.py \
    --n_clusters 5 \
    --clustering_method kmeans \
    --use_tfidf \
    --output_dir "results_5clusters_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "Analysis complete with 5 clusters!"