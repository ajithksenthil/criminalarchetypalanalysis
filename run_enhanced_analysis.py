#!/usr/bin/env python3
"""
run_enhanced_analysis.py

User-friendly script to run the enhanced criminal archetypal analysis.
"""

import argparse
import os
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description='Run Enhanced Criminal Archetypal Analysis with all improvements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python run_enhanced_analysis.py
  
  # Run with custom directories
  python run_enhanced_analysis.py --type1_dir my_type1_data --type2_dir my_type2_data
  
  # Run with specific configuration
  python run_enhanced_analysis.py --markov_order 4 --n_trajectories 5 --quick
  
  # Run full analysis with all validations
  python run_enhanced_analysis.py --full --n_permutations 5000 --n_bootstrap 5000
        """
    )
    
    # Data directories
    parser.add_argument('--type1_dir', default='type1csvs',
                       help='Directory containing Type 1 CSV files (default: type1csvs)')
    parser.add_argument('--type2_dir', default='type2csvs',
                       help='Directory containing Type 2 CSV files (default: type2csvs)')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory (default: enhanced_results_TIMESTAMP)')
    
    # Analysis configuration
    parser.add_argument('--markov_order', type=int, default=3,
                       help='Order for higher-order Markov chains (default: 3)')
    parser.add_argument('--n_clusters', type=int, default=None,
                       help='Number of clusters (default: auto-select)')
    parser.add_argument('--n_trajectories', type=int, default=4,
                       help='Number of trajectory types (default: 4)')
    parser.add_argument('--use_ensemble', action='store_true', default=True,
                       help='Use ensemble clustering (default: True)')
    parser.add_argument('--no_ensemble', dest='use_ensemble', action='store_false',
                       help='Disable ensemble clustering')
    
    # Statistical validation
    parser.add_argument('--n_permutations', type=int, default=1000,
                       help='Number of permutations for significance testing (default: 1000)')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                       help='Number of bootstrap samples (default: 1000)')
    
    # Quick vs full analysis
    parser.add_argument('--quick', action='store_true',
                       help='Quick analysis with reduced permutations/bootstrap (100 each)')
    parser.add_argument('--full', action='store_true',
                       help='Full analysis with extensive validation (5000 permutations/bootstrap)')
    
    # OpenAI configuration
    parser.add_argument('--openai_key', type=str, default=None,
                       help='OpenAI API key for LLM labeling')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'enhanced_results_{timestamp}'
    
    # Adjust for quick/full analysis
    if args.quick:
        args.n_permutations = 100
        args.n_bootstrap = 100
        print("[INFO] Running quick analysis with reduced validation samples")
    elif args.full:
        args.n_permutations = 5000
        args.n_bootstrap = 5000
        print("[INFO] Running full analysis with extensive validation")
    
    # Set OpenAI key if provided
    if args.openai_key:
        os.environ['OPENAI_API_KEY'] = args.openai_key
        print("[INFO] OpenAI API key set for LLM labeling")
    
    # Create configuration
    config = {
        'markov_order': args.markov_order,
        'n_clusters': args.n_clusters,
        'n_trajectories': args.n_trajectories,
        'use_ensemble': args.use_ensemble,
        'n_permutations': args.n_permutations,
        'n_bootstrap': args.n_bootstrap
    }
    
    print("\n" + "="*60)
    print("ENHANCED CRIMINAL ARCHETYPAL ANALYSIS")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Type 1 directory: {args.type1_dir}")
    print(f"  Type 2 directory: {args.type2_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Markov order: {config['markov_order']}")
    print(f"  Clusters: {'auto-select' if config['n_clusters'] is None else config['n_clusters']}")
    print(f"  Trajectories: {config['n_trajectories']}")
    print(f"  Ensemble clustering: {config['use_ensemble']}")
    print(f"  Permutations: {config['n_permutations']}")
    print(f"  Bootstrap samples: {config['n_bootstrap']}")
    print(f"  LLM labeling: {'enabled' if args.openai_key else 'disabled'}")
    
    # Check if directories exist
    if not os.path.exists(args.type1_dir):
        print(f"\n[ERROR] Type 1 directory not found: {args.type1_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.type2_dir):
        print(f"\n[ERROR] Type 2 directory not found: {args.type2_dir}")
        sys.exit(1)
    
    try:
        # Import and run enhanced analysis
        from enhanced_analysis_integration import run_enhanced_analysis
        
        print("\n[INFO] Starting enhanced analysis...")
        results = run_enhanced_analysis(
            type1_dir=args.type1_dir,
            type2_dir=args.type2_dir,
            output_dir=args.output_dir,
            config=config
        )
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"\nResults saved to: {args.output_dir}/")
        print(f"Report: {args.output_dir}/analysis_report.md")
        print(f"\nInteractive visualizations:")
        print(f"  - Sankey diagram: {args.output_dir}/sankey_diagram.html")
        print(f"  - 3D clusters: {args.output_dir}/clusters_3d.html")
        print(f"  - Dashboard: {args.output_dir}/dashboard.html")
        print(f"\nStatic visualizations:")
        print(f"  - Life phases: {args.output_dir}/life_phases.png")
        print(f"  - Trajectories: {args.output_dir}/trajectories.png")
        
        # Summary statistics
        print(f"\nKey findings:")
        print(f"  - Criminals analyzed: {results['data']['n_criminals']}")
        print(f"  - Criminal archetypes: {results['clustering']['n_clusters']}")
        print(f"  - Critical patterns: {len(results['markov']['critical_patterns'])}")
        print(f"  - Trajectory types: {len(results['trajectories']['trajectory_profiles'])}")
        
        # Risk summary
        risk_counts = {'Low Risk': 0, 'Moderate Risk': 0, 'High Risk': 0, 'Very High Risk': 0}
        for score in results['trajectories']['risk_scores']:
            risk_counts[score['risk_level']] += 1
        
        print(f"\nRisk distribution:")
        for level, count in risk_counts.items():
            print(f"  - {level}: {count}")
        
        print("\n✓ Enhanced analysis completed successfully!")
        
    except ImportError as e:
        print(f"\n[ERROR] Failed to import required modules: {e}")
        print("Please ensure all enhancement modules are installed:")
        print("  - markov_models.py")
        print("  - temporal_analysis.py")
        print("  - interactive_visualizations.py")
        print("  - ensemble_clustering.py")
        print("  - statistical_validation.py")
        print("  - trajectory_analysis.py")
        print("  - enhanced_analysis_integration.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()