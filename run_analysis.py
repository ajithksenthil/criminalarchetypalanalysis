#!/usr/bin/env python3
"""
run_analysis.py

Complete research-grade analysis pipeline for criminal archetypal analysis.
This script ensures reproducibility and generates comprehensive validation metrics.
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
import numpy as np


def setup_environment():
    """Set up environment variables and paths."""
    # Set random seeds for reproducibility
    os.environ['PYTHONHASHSEED'] = '42'
    np.random.seed(42)
    
    # Set number of threads for reproducibility
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    print("[INFO] Environment configured for reproducibility")


def run_main_analysis(args):
    """Run the main analysis pipeline."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{timestamp}"
    
    # Build command
    cmd = [
        "python", "analysis_integration.py",
        "--type1_dir", args.type1_dir,
        "--type2_csv", args.type2_dir,
        "--output_dir", output_dir,
        "--n_clusters", str(args.n_clusters),
        "--match_only"  # Always use matched data for research grade analysis
    ]
    
    # Add optional flags
    if args.use_tfidf:
        cmd.append("--use_tfidf")
    if args.no_llm:
        cmd.append("--no_llm")
    if args.train_proto_net:
        cmd.append("--train_proto_net")
    if args.multi_modal:
        cmd.append("--multi_modal")
    
    print(f"[INFO] Running main analysis with output directory: {output_dir}")
    print(f"[INFO] Command: {' '.join(cmd)}")
    
    # Run with logging
    log_file = f"logs/analysis_{timestamp}.log"
    os.makedirs("logs", exist_ok=True)
    
    with open(log_file, "w") as log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Print and log output in real-time
        for line in process.stdout:
            print(line, end='')
            log.write(line)
            log.flush()
        
        process.wait()
        
    if process.returncode != 0:
        print(f"[ERROR] Analysis failed with return code {process.returncode}")
        return None
        
    print(f"[INFO] Analysis completed. Log saved to {log_file}")
    return output_dir


def run_validation_analysis(output_dir):
    """Run comprehensive validation analysis."""
    print("\n[INFO] Running validation analysis...")
    
    # Import validation module
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from validation_analysis import generate_validation_report
    
    # Load analysis results
    try:
        # Load embeddings and labels
        embeddings_path = os.path.join(output_dir, "embeddings.npy")
        labels_path = os.path.join(output_dir, "labels.npy")
        
        if not os.path.exists(embeddings_path) or not os.path.exists(labels_path):
            print("[WARNING] Embeddings or labels not found. Skipping validation.")
            return
            
        embeddings = np.load(embeddings_path)
        labels = np.load(labels_path)
        
        # Load criminal sequences
        sequences_path = os.path.join(output_dir, "criminal_sequences.json")
        with open(sequences_path, 'r') as f:
            sequences = json.load(f)
            
        # Extract criminal IDs from sequences
        criminal_ids = []
        for cid, seq in sequences.items():
            criminal_ids.extend([cid] * len(seq))
            
        # Load transition matrix
        transition_matrix = np.load(os.path.join(output_dir, "global_transition_matrix.npy"))
        
        # Run validation
        validation_report = generate_validation_report(
            output_dir,
            embeddings,
            labels,
            sequences,
            criminal_ids,
            transition_matrix
        )
        
        print("[INFO] Validation analysis completed")
        
    except Exception as e:
        print(f"[ERROR] Validation analysis failed: {e}")
        import traceback
        traceback.print_exc()


def generate_reproducibility_info(output_dir):
    """Generate reproducibility information."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
        "environment_variables": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "not set"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "not set"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "not set")
        },
        "package_versions": {}
    }
    
    # Get package versions
    try:
        import pkg_resources
        packages = [
            'numpy', 'pandas', 'scikit-learn', 'scipy', 
            'matplotlib', 'seaborn', 'nltk', 'torch',
            'sentence-transformers', 'networkx'
        ]
        
        for package in packages:
            try:
                version = pkg_resources.get_distribution(package).version
                info["package_versions"][package] = version
            except:
                info["package_versions"][package] = "not installed"
                
    except ImportError:
        pass
    
    # Save reproducibility info
    repro_path = os.path.join(output_dir, "reproducibility_info.json")
    with open(repro_path, 'w') as f:
        json.dump(info, f, indent=4)
        
    print(f"[INFO] Reproducibility information saved to {repro_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive criminal archetypal analysis with validation"
    )
    parser.add_argument(
        "--type1_dir",
        default="type1csvs",
        help="Directory containing Type1 CSV files"
    )
    parser.add_argument(
        "--type2_dir", 
        default="type2csvs",
        help="Directory containing Type2 CSV files"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=5,
        help="Number of event clusters"
    )
    parser.add_argument(
        "--use_tfidf",
        action="store_true",
        help="Use TF-IDF embeddings (offline mode)"
    )
    parser.add_argument(
        "--no_llm",
        action="store_true",
        help="Disable LLM features"
    )
    parser.add_argument(
        "--train_proto_net",
        action="store_true",
        help="Train prototypical network"
    )
    parser.add_argument(
        "--multi_modal",
        action="store_true",
        help="Use multi-modal clustering"
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip validation analysis"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("CRIMINAL ARCHETYPAL ANALYSIS - RESEARCH GRADE")
    print("="*60)
    
    # Setup environment
    setup_environment()
    
    # First, we need to save embeddings and labels in the analysis
    # Let me check if they're being saved...
    
    # Run main analysis
    output_dir = run_main_analysis(args)
    
    if output_dir and not args.skip_validation:
        # Run validation
        run_validation_analysis(output_dir)
    
    if output_dir:
        # Generate reproducibility info
        generate_reproducibility_info(output_dir)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Results saved to: {output_dir}")
        print(f"To view report, open: {output_dir}/report/analysis_report.html")
    else:
        print("\n[ERROR] Analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()