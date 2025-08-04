#!/usr/bin/env python3
"""
run_standardized_auto_k.py

Run the improved analysis with standardized names (using improved lexical imputation)
and automatic k selection for optimal clustering.
"""

import subprocess
import sys
import os
from datetime import datetime

def main():
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_standardized_auto_k_{timestamp}"
    
    # Build command
    cmd = [
        sys.executable,
        "run_analysis_improved.py",
        "--auto_k",
        "--output_dir", output_dir,
        "--use_tfidf"  # This enables better embeddings
    ]
    
    # Add any additional arguments passed to this script
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    print("="*60)
    print("RUNNING ANALYSIS WITH STANDARDIZED NAMES AND AUTO-K")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print("="*60)
    
    # First, ensure the improved lexical imputation is being used
    # We'll modify the analysis to use it by setting an environment variable
    env = os.environ.copy()
    env['USE_IMPROVED_LEXICAL_IMPUTATION'] = '1'
    
    # Run the analysis
    try:
        result = subprocess.run(cmd, env=env)
        
        if result.returncode == 0:
            print(f"\n✓ Analysis completed successfully!")
            print(f"✓ Results saved to: {output_dir}/")
            
            # Run the conditional patterns analysis
            print("\nAnalyzing conditional patterns...")
            analyze_cmd = [sys.executable, "analyze_conditional_patterns.py", output_dir]
            subprocess.run(analyze_cmd)
            
            # Create visualizations
            print("\nCreating visualizations...")
            viz_cmd = [sys.executable, "visualize_conditional_patterns.py", output_dir]
            subprocess.run(viz_cmd)
            
        else:
            print(f"\n✗ Analysis failed with return code: {result.returncode}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Error running analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()