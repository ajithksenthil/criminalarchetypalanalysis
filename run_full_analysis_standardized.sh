#!/bin/bash

# Run full analysis with name standardization and auto-k selection
# This script ensures proper environment setup and uses the improved lexical imputation

echo "============================================================"
echo "RUNNING FULL ANALYSIS WITH NAME STANDARDIZATION"
echo "============================================================"

# Load environment variables
source load_env.sh

# Create timestamp for output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results_standardized_auto_k_${TIMESTAMP}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run the improved analysis with name standardization
# The test_run_fixed used these exact parameters
python analysis_integration_improved.py \
    --type1_dir type1csvs \
    --type2_csv type2csvs \
    --output_dir "$OUTPUT_DIR" \
    --n_clusters 12 \
    --match_only \
    --clustering_method kmeans \
    --reduce_dims

# Check if analysis succeeded
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
    echo "============================================================"
else
    echo ""
    echo "✗ Analysis failed!"
    exit 1
fi