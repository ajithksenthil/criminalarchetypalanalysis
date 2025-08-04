#!/usr/bin/env python3
"""
run_analysis_improved.py

Run the criminal archetypal analysis with improved clustering and LLM labeling enabled.
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Run improved criminal archetypal analysis")
    parser.add_argument("--type1_dir", default="type1csvs", help="Directory with Type1 CSV files")
    parser.add_argument("--type2_dir", default="type2csvs", help="Directory with Type2 CSV files") 
    parser.add_argument("--output_dir", help="Output directory (default: auto-generated)")
    parser.add_argument("--openai_key", help="OpenAI API key for LLM labeling")
    parser.add_argument("--clustering_method", default="kmeans", 
                       choices=["kmeans", "hierarchical", "dbscan", "spectral"],
                       help="Clustering method to use")
    parser.add_argument("--auto_k", action="store_true", help="Automatically select optimal k")
    parser.add_argument("--reduce_dims", action="store_true", default=True,
                       help="Reduce dimensions before clustering")
    parser.add_argument("--n_clusters", type=int, help="Number of clusters (overrides auto_k)")
    args = parser.parse_args()
    
    # Set up output directory
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"output_improved_{timestamp}"
    
    # Set environment variables
    env = os.environ.copy()
    
    # Set OpenAI API key if provided
    if args.openai_key:
        env["OPENAI_API_KEY"] = args.openai_key
        print("[INFO] OpenAI API key set for LLM labeling")
    elif "OPENAI_API_KEY" in env:
        print("[INFO] Using existing OPENAI_API_KEY from environment")
    else:
        print("[WARNING] No OpenAI API key found. LLM labeling will be disabled.")
        print("[INFO] To enable LLM labeling, either:")
        print("  1. Set OPENAI_API_KEY environment variable")
        print("  2. Use --openai_key argument")
    
    # Set clustering preferences
    env["USE_IMPROVED_CLUSTERING"] = "1"
    env["CLUSTERING_METHOD"] = args.clustering_method
    env["AUTO_SELECT_K"] = "1" if args.auto_k else "0"
    env["REDUCE_DIMENSIONS"] = "1" if args.reduce_dims else "0"
    
    print("\n" + "="*60)
    print("CRIMINAL ARCHETYPAL ANALYSIS - IMPROVED VERSION")
    print("="*60)
    
    print(f"[INFO] Type1 directory: {args.type1_dir}")
    print(f"[INFO] Type2 directory: {args.type2_dir}")
    print(f"[INFO] Output directory: {args.output_dir}")
    print(f"[INFO] Clustering method: {args.clustering_method}")
    print(f"[INFO] Auto-select k: {args.auto_k}")
    print(f"[INFO] Reduce dimensions: {args.reduce_dims}")
    if args.n_clusters:
        print(f"[INFO] Fixed clusters: {args.n_clusters}")
    
    # Build command
    cmd = [
        sys.executable,
        "analysis_integration_improved.py",
        "--type1_dir", args.type1_dir,
        "--type2_csv", args.type2_dir,
        "--output_dir", args.output_dir,
        "--match_only"
    ]
    
    # Add clustering options
    if args.n_clusters:
        cmd.extend(["--n_clusters", str(args.n_clusters)])
    elif args.auto_k:
        # Will be handled by the improved script
        pass
    
    # Don't add --no_llm flag if we have an API key
    if not args.openai_key and "OPENAI_API_KEY" not in env:
        cmd.append("--no_llm")
    
    print(f"\n[INFO] Running command: {' '.join(cmd)}")
    
    # Run the analysis
    try:
        result = subprocess.run(cmd, env=env, check=True)
        print("\n[SUCCESS] Analysis completed successfully!")
        print(f"[INFO] Results saved in: {args.output_dir}/")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Analysis failed with return code {e.returncode}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())