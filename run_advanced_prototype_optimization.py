#!/usr/bin/env python3
"""
run_advanced_prototype_optimization.py

Command-line interface for two-layer prototype optimization.

This script runs advanced cluster optimization on existing analysis results,
providing a second layer of archetypal refinement with validation.
"""

import argparse
import sys
from pathlib import Path
import json
import time

def main():
    parser = argparse.ArgumentParser(
        description="Run advanced two-layer prototype optimization on existing analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on existing analysis results
  python run_advanced_prototype_optimization.py output_best_sota

  # Run with custom validation split
  python run_advanced_prototype_optimization.py output_best_sota --validation_split 0.4

  # Run on multiple output directories
  python run_advanced_prototype_optimization.py output_best_sota output_complete_prototype --compare

  # Run complete pipeline (Layer 1 + Layer 2)
  python run_advanced_prototype_optimization.py --run_complete_pipeline --output_dir output_advanced_complete
        """
    )
    
    parser.add_argument(
        "input_dirs",
        nargs="*",
        help="Input directories with existing analysis results"
    )
    
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.3,
        help="Fraction of cluster events to use for validation (default: 0.3)"
    )
    
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare optimization results across multiple input directories"
    )
    
    parser.add_argument(
        "--run_complete_pipeline",
        action="store_true",
        help="Run complete pipeline (Layer 1 + Layer 2) from scratch"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_advanced_complete",
        help="Output directory for complete pipeline (default: output_advanced_complete)"
    )
    
    # Layer 1 pipeline arguments (for complete pipeline)
    parser.add_argument(
        "--type1_dir",
        type=str,
        default="type1csvs",
        help="Directory containing Type 1 data (for complete pipeline)"
    )
    
    parser.add_argument(
        "--type2_csv",
        type=str,
        default="type2csvs",
        help="Type 2 CSV file (for complete pipeline)"
    )
    
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-mpnet-base-v2",
        help="Embedding model for Layer 1 (default: all-mpnet-base-v2)"
    )
    
    parser.add_argument(
        "--use_openai",
        action="store_true",
        help="Use OpenAI embeddings for Layer 1"
    )
    
    parser.add_argument(
        "--use_prototype",
        action="store_true",
        help="Use prototype embeddings for Layer 1"
    )
    
    parser.add_argument(
        "--use_lexical_bias_reduction",
        action="store_true",
        help="Use lexical bias reduction for Layer 1"
    )
    
    args = parser.parse_args()
    
    if args.run_complete_pipeline:
        run_complete_pipeline(args)
    elif args.input_dirs:
        if args.compare and len(args.input_dirs) > 1:
            run_comparison_analysis(args)
        else:
            run_optimization_on_existing(args)
    else:
        parser.print_help()
        sys.exit(1)

def run_complete_pipeline(args):
    """Run complete pipeline: Layer 1 + Layer 2."""
    print("🚀 RUNNING COMPLETE TWO-LAYER PROTOTYPE PIPELINE")
    print("=" * 60)
    print("Layer 1: Individual event prototypes + clustering")
    print("Layer 2: Cluster-level archetypal optimization")
    print()
    
    # Step 1: Run Layer 1 analysis
    print("[STEP 1] Running Layer 1 analysis...")
    layer1_success = run_layer1_analysis(args)
    
    if not layer1_success:
        print("❌ Layer 1 analysis failed")
        return False
    
    # Step 2: Run Layer 2 optimization
    print(f"\n[STEP 2] Running Layer 2 optimization on {args.output_dir}...")
    layer2_results = run_layer2_optimization(args.output_dir, args)
    
    # Step 3: Generate comprehensive report
    print(f"\n[STEP 3] Generating comprehensive report...")
    generate_complete_pipeline_report(args.output_dir, layer2_results)
    
    print(f"\n🎉 COMPLETE PIPELINE FINISHED!")
    print(f"📁 Results: {args.output_dir}/")
    print(f"📊 Advanced results: {args.output_dir}/advanced/")
    
    return True

def run_layer1_analysis(args):
    """Run Layer 1 analysis using existing modular system."""
    import subprocess
    
    # Build Layer 1 command
    cmd = [
        "python", "run_modular_analysis.py",
        f"--type1_dir={args.type1_dir}",
        f"--type2_csv={args.type2_csv}",
        f"--output_dir={args.output_dir}",
        f"--embedding_model={args.embedding_model}",
        "--auto_k",
        "--match_only",
        "--n_clusters=5"
    ]
    
    if args.use_openai:
        cmd.append("--use_openai")
    
    if args.use_prototype:
        cmd.append("--use_prototype")
    
    if args.use_lexical_bias_reduction:
        cmd.append("--use_lexical_bias_reduction")
    else:
        cmd.append("--no_llm")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
        print("✅ Layer 1 analysis completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Layer 1 analysis failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Layer 1 analysis timed out")
        return False

def run_layer2_optimization(output_dir: str, args):
    """Run Layer 2 optimization."""
    from modular_analysis.advanced.advanced_pipeline import run_advanced_pipeline
    
    try:
        results = run_advanced_pipeline(
            output_dir=output_dir,
            validation_split=args.validation_split,
            random_state=args.random_state
        )
        
        print("✅ Layer 2 optimization completed successfully")
        return results
        
    except Exception as e:
        print(f"❌ Layer 2 optimization failed: {e}")
        return None

def run_optimization_on_existing(args):
    """Run optimization on existing analysis results."""
    print("🔧 RUNNING ADVANCED OPTIMIZATION ON EXISTING RESULTS")
    print("=" * 60)
    
    results = {}
    
    for input_dir in args.input_dirs:
        print(f"\n📁 Processing: {input_dir}")
        
        if not Path(input_dir).exists():
            print(f"❌ Directory not found: {input_dir}")
            continue
        
        # Check if required files exist
        required_files = [
            "analysis_results.json",
            "clustering/cluster_info.json",
            "data/embeddings.npy",
            "data/labels.npy"
        ]
        
        missing_files = []
        for file in required_files:
            if not (Path(input_dir) / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            continue
        
        # Run optimization
        result = run_layer2_optimization(input_dir, args)
        if result:
            results[input_dir] = result
            
            # Print summary
            summary = result['summary']
            print(f"✅ Optimization complete")
            print(f"   Layer 1 silhouette: {summary['layer1_summary']['silhouette_score']:.4f}")
            print(f"   Layer 2 coherence: {summary['layer2_summary']['mean_coherence']:.4f}")
            print(f"   Optimization score: {summary['layer2_summary']['optimization_score']:.4f}")
            print(f"   Recommendation: {summary['recommendation']}")
    
    if not results:
        print("❌ No successful optimizations")
        return
    
    # Generate summary report
    generate_optimization_summary(results)

def run_comparison_analysis(args):
    """Run comparison analysis across multiple directories."""
    print("📊 RUNNING COMPARISON ANALYSIS")
    print("=" * 60)
    
    # First run optimization on all directories
    run_optimization_on_existing(args)
    
    # Then generate detailed comparison
    print(f"\n📈 Generating detailed comparison...")
    generate_detailed_comparison(args.input_dirs)

def generate_complete_pipeline_report(output_dir: str, layer2_results):
    """Generate comprehensive report for complete pipeline."""
    if not layer2_results:
        return
    
    report_path = Path(output_dir) / "advanced" / "complete_pipeline_report.txt"
    
    summary = layer2_results['summary']
    
    report = f"""
COMPLETE TWO-LAYER PROTOTYPE PIPELINE REPORT
============================================

METHODOLOGY
-----------
Layer 1: Individual Event Prototypes + Clustering
• Entity replacement (names → [PERSON], locations → [LOCATION])
• Prototype embeddings (lexical variation averaging)
• Initial clustering with k-means

Layer 2: Cluster-Level Archetypal Optimization
• Train/validation split ({summary['layer2_summary'].get('validation_split', 0.3):.1%} validation)
• Archetypal prototype creation (outlier removal + weighted centroid)
• Cross-validation for generalizability

RESULTS
-------
Layer 1 Results:
• Silhouette Score: {summary['layer1_summary']['silhouette_score']:.4f}
• Number of Clusters: {summary['layer1_summary']['n_clusters']}
• Total Events: {summary['layer1_summary']['n_events']}

Layer 2 Optimization:
• Mean Coherence: {summary['layer2_summary']['mean_coherence']:.4f}
• Validation Score: {summary['layer2_summary']['mean_validation_score']:.4f}
• Overall Optimization: {summary['layer2_summary']['optimization_score']:.4f}

KEY IMPROVEMENTS
----------------
• Archetypal Representation: {summary['key_improvements']['archetypal_representation']:.4f}
• Validation Generalizability: {summary['key_improvements']['validation_generalizability']:.4f}
• Overall Quality: {summary['key_improvements']['overall_quality']:.4f}

RECOMMENDATION
--------------
{summary['recommendation']}

NEXT STEPS
----------
"""
    
    for i, step in enumerate(summary['next_steps'], 1):
        report += f"{i}. {step}\n"
    
    report += f"""

FILES GENERATED
---------------
• Layer 1 Results: {output_dir}/
• Layer 2 Optimization: {output_dir}/advanced/
• Optimized Clusters: {output_dir}/advanced/optimized_cluster_info.json
• Comparison Analysis: {output_dir}/advanced/layer_comparison.json
• This Report: {report_path}
"""
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📄 Complete pipeline report saved: {report_path}")

def generate_optimization_summary(results: dict):
    """Generate summary of optimization results."""
    print(f"\n📊 OPTIMIZATION SUMMARY")
    print("=" * 40)
    
    # Sort by optimization score
    sorted_results = sorted(results.items(), 
                          key=lambda x: x[1]['summary']['layer2_summary']['optimization_score'], 
                          reverse=True)
    
    print(f"\nRanking by optimization score:")
    for i, (dir_name, result) in enumerate(sorted_results, 1):
        summary = result['summary']
        print(f"\n{i}. {dir_name}")
        print(f"   Optimization Score: {summary['layer2_summary']['optimization_score']:.4f}")
        print(f"   Layer 1 Silhouette: {summary['layer1_summary']['silhouette_score']:.4f}")
        print(f"   Layer 2 Coherence: {summary['layer2_summary']['mean_coherence']:.4f}")
        print(f"   Recommendation: {summary['recommendation']}")
    
    # Save comparison
    comparison_file = "advanced_optimization_comparison.json"
    with open(comparison_file, 'w') as f:
        # Make results JSON serializable
        serializable_results = {}
        for dir_name, result in results.items():
            serializable_results[dir_name] = result['summary']
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Comparison saved: {comparison_file}")

def generate_detailed_comparison(input_dirs: list):
    """Generate detailed comparison across directories."""
    print("📈 Detailed comparison analysis complete")
    print("Check individual advanced/ directories for detailed results")

if __name__ == "__main__":
    main()
