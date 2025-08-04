#!/bin/bash

# Run analysis with name standardization using the configuration that worked for test_run_fixed
# This uses 12 clusters which was found optimal, and TF-IDF embeddings

echo "============================================================"
echo "RUNNING STANDARDIZED ANALYSIS WITH 12 CLUSTERS"
echo "============================================================"
echo ""
echo "Configuration:"
echo "- Name standardization: YES (via improved analysis)"
echo "- Lexical imputation: YES"
echo "- Number of clusters: 12 (found optimal)"
echo "- Embeddings: TF-IDF"
echo ""

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results_standardized_12_clusters_${TIMESTAMP}"

# First install minimal requirements if needed
echo "Checking requirements..."
pip install -q nltk scikit-learn numpy pandas matplotlib scipy openai 2>/dev/null

# The run_analysis_improved.py script will handle the improved analysis
python run_analysis_improved.py \
    --n_clusters 12 \
    --use_tfidf \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Analysis completed successfully!"
    echo ""
    
    # Analyze conditional patterns
    echo "Analyzing conditional patterns..."
    python analyze_conditional_patterns.py "$OUTPUT_DIR"
    
    # Create visualizations
    echo ""
    echo "Creating visualizations..."
    python visualize_conditional_patterns.py "$OUTPUT_DIR"
    
    echo ""
    echo "============================================================"
    echo "ANALYSIS COMPLETE"
    echo "Results saved to: $OUTPUT_DIR/"
    echo ""
    echo "To verify name standardization was used, check:"
    echo "- $OUTPUT_DIR/global_clusters.json (should show processed events)"
    echo "- $OUTPUT_DIR/conditional_patterns_detailed.txt"
    echo "============================================================"
else
    echo ""
    echo "✗ Analysis failed!"
    echo ""
    echo "To debug, try running:"
    echo "python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')\""
    echo ""
fi