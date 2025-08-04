#!/bin/bash

# This script replicates the successful test_run_fixed analysis
# which used name standardization and found 12 optimal clusters

echo "============================================================"
echo "REPLICATING TEST_RUN_FIXED ANALYSIS"
echo "============================================================"
echo ""
echo "This will run the analysis with:"
echo "- Name standardization (via improved lexical imputation)"
echo "- 12 clusters (found optimal in test_run)"
echo "- TF-IDF embeddings"
echo ""

# Set timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results_like_test_run_${TIMESTAMP}"

# The test_run_fixed was created with these exact parameters
python run_analysis_improved.py \
    --n_clusters 12 \
    --use_tfidf \
    --output_dir "$OUTPUT_DIR"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Analysis completed!"
    echo ""
    
    # Run conditional patterns analysis
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
    echo "Expected results (based on test_run_fixed):"
    echo "- 12 clusters"
    echo "- ~125 significant conditional patterns (49.8%)"
    echo "- Strong patterns for demographics, education, criminal behavior"
    echo "============================================================"
fi