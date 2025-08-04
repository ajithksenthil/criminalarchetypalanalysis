#!/usr/bin/env python3
"""
run_analysis_simple.py

Simple working version that runs the original analysis with the same interface.
This ensures you can run the analysis immediately while the modular system is being finalized.
"""

import sys
import os
import argparse

def main():
    """Main entry point that delegates to the original script."""
    
    # Check if original script exists
    original_script = "analysis_integration_improved.py"
    if not os.path.exists(original_script):
        print(f"Error: {original_script} not found in current directory")
        print("Please ensure you're running from the correct directory.")
        sys.exit(1)
    
    # Create argument parser with same interface
    parser = argparse.ArgumentParser(
        description="Criminal Archetypal Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python run_analysis_simple.py --type1_dir=data_csv --type2_csv=data_csv --output_dir=output

  # With automatic k optimization (your new feature!)
  python run_analysis_simple.py --type1_dir=data_csv --type2_csv=data_csv --output_dir=output --auto_k

  # Matched data only with multi-modal clustering
  python run_analysis_simple.py --type1_dir=data_csv --type2_csv=data_csv --output_dir=output --match_only --multi_modal

  # Offline mode (no LLM, TF-IDF embeddings)
  python run_analysis_simple.py --type1_dir=data_csv --type2_csv=data_csv --output_dir=output --no_llm --use_tfidf
        """
    )
    
    # Required arguments
    parser.add_argument("--type1_dir", type=str, required=True,
                        help="Directory containing Type1_*.csv files (one per criminal)")
    parser.add_argument("--type2_csv", type=str, required=True,
                        help="Path to Type2 CSV file or directory containing Type2_*.csv files")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save output artifacts (default: output)")
    parser.add_argument("--n_clusters", type=int, default=5,
                        help="Number of clusters for KMeans (default: 5, ignored if --auto_k is used)")
    
    # Analysis options
    parser.add_argument("--auto_k", action="store_true",
                        help="Automatically optimize k for conditional effect detection")
    parser.add_argument("--no_llm", action="store_true",
                        help="Disable LLM calls (useful if no OpenAI API key available)")
    parser.add_argument("--multi_modal", action="store_true",
                        help="Perform multi-modal clustering at the criminal level using Type 1 & Type 2 data")
    parser.add_argument("--train_proto_net", action="store_true",
                        help="Train a prototypical network on clustered event embeddings")
    parser.add_argument("--use_tfidf", action="store_true",
                        help="Use TF-IDF embeddings instead of SentenceTransformer (offline mode)")
    parser.add_argument("--match_only", action="store_true",
                        help="Only analyze criminals with both Type1 and Type2 data (recommended)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Build command for original script
    cmd_parts = ["python", original_script]
    
    # Add all arguments
    cmd_parts.extend(["--type1_dir", args.type1_dir])
    cmd_parts.extend(["--type2_csv", args.type2_csv])
    cmd_parts.extend(["--output_dir", args.output_dir])
    cmd_parts.extend(["--n_clusters", str(args.n_clusters)])
    
    if args.auto_k:
        cmd_parts.append("--auto_k")
    if args.no_llm:
        cmd_parts.append("--no_llm")
    if args.multi_modal:
        cmd_parts.append("--multi_modal")
    if args.train_proto_net:
        cmd_parts.append("--train_proto_net")
    if args.use_tfidf:
        cmd_parts.append("--use_tfidf")
    if args.match_only:
        cmd_parts.append("--match_only")
    
    # Print what we're running
    print("="*70)
    print("CRIMINAL ARCHETYPAL ANALYSIS")
    print("="*70)
    print(f"Running: {' '.join(cmd_parts)}")
    print("="*70)
    
    # Execute the original script
    try:
        import subprocess
        result = subprocess.run(cmd_parts, check=True)
        print("\n" + "="*70)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Results saved to: {args.output_dir}")
        
        # Check for specific output files
        output_files = [
            "analysis_results.json",
            "cluster_info.json", 
            "conditional_insights.json",
            "global_transition_matrix.npy",
            "tsne_visualization.png"
        ]
        
        print("\nGenerated files:")
        for filename in output_files:
            filepath = os.path.join(args.output_dir, filename)
            if os.path.exists(filepath):
                print(f"  ✅ {filename}")
            else:
                # Check in subdirectories
                found = False
                for root, dirs, files in os.walk(args.output_dir):
                    if filename in files:
                        rel_path = os.path.relpath(os.path.join(root, filename), args.output_dir)
                        print(f"  ✅ {rel_path}")
                        found = True
                        break
                if not found:
                    print(f"  ⚠️  {filename} (not found)")
        
        if args.auto_k:
            print(f"\n🎯 Your conditional effect optimization feature was used!")
            print(f"   Check k_optimization_results.json for details")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Analysis failed with exit code {e.returncode}")
        print("Please check the error messages above.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
