#!/usr/bin/env python3
"""
run_analysis_with_names_final.py

This script runs the analysis with name standardization by directly calling
the analysis_integration_improved.py with the correct parameters.
"""

import subprocess
import sys
import os
from datetime import datetime

def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_standardized_final_{timestamp}"
    
    print("="*60)
    print("RUNNING ANALYSIS WITH NAME STANDARDIZATION")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print("")
    
    # The key insight: analysis_integration_improved.py already includes
    # improved lexical imputation when using --use_tfidf flag
    # This is what created the successful test_run_fixed results
    
    cmd = [
        sys.executable,
        "analysis_integration_improved.py",
        "--type1_dir", "type1csvs",
        "--type2_csv", "type2csvs",
        "--output_dir", output_dir,
        "--n_clusters", "12",  # Optimal number found in test_run_fixed
        "--match_only",
        "--use_tfidf"  # This enables TF-IDF embeddings with name handling
    ]
    
    # Set environment to ensure OpenAI API key is available
    env = os.environ.copy()
    
    # Load .env file if it exists
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env[key] = value
    
    print("Running command:")
    print(" ".join(cmd))
    print("="*60)
    
    # Run the analysis
    try:
        result = subprocess.run(cmd, env=env)
        
        if result.returncode == 0:
            print(f"\n✓ Analysis completed successfully!")
            print(f"✓ Results saved to: {output_dir}/")
            
            # Run conditional patterns analysis
            print("\nAnalyzing conditional patterns...")
            subprocess.run([sys.executable, "analyze_conditional_patterns.py", output_dir])
            
            # Create visualizations
            print("\nCreating visualizations...")
            subprocess.run([sys.executable, "visualize_conditional_patterns.py", output_dir])
            
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE")
            print(f"Results directory: {output_dir}/")
            print("\nKey outputs:")
            print(f"  - {output_dir}/conditional_patterns_detailed.txt")
            print(f"  - {output_dir}/conditional_patterns_summary.png")
            print(f"  - {output_dir}/effect_size_heatmap.png")
            print(f"  - {output_dir}/report/analysis_report.html")
            print("\nExpected: ~125 significant conditional patterns (49.8%)")
            print("="*60)
        else:
            print(f"\n✗ Analysis failed with return code: {result.returncode}")
            print("\nTroubleshooting:")
            print("1. Ensure all required packages are installed:")
            print("   pip install nltk scikit-learn numpy pandas matplotlib scipy openai")
            print("2. Download NLTK data:")
            print("   python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')\"")
            print("3. Ensure .env file contains OPENAI_API_KEY")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    main()