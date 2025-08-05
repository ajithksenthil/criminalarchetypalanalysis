#!/usr/bin/env python3
"""
run_modular_analysis.py

Standalone script to run the modular criminal archetypal analysis system.
This script handles imports and provides the same interface as the original script.
"""

import sys
import os
import argparse

# Add modular_analysis to Python path
modular_path = os.path.join(os.path.dirname(__file__), 'modular_analysis')
sys.path.insert(0, modular_path)

def main():
    """Main entry point for the modular analysis system."""

    # Import here to avoid import issues
    try:
        from core.config import AnalysisConfig
        from integration.pipeline import CriminalArchetypalAnalysisPipeline
        print("[INFO] Successfully imported modular analysis components")
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Please ensure all required dependencies are installed:")
        print("pip install numpy pandas scikit-learn matplotlib seaborn networkx nltk sentence-transformers")
        print("\nIf the issue persists, try running the original script:")
        print("python analysis_integration_improved.py [same arguments]")
        sys.exit(1)
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Modular Criminal Archetypal Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python run_modular_analysis.py --type1_dir=data_csv --type2_csv=data_csv --output_dir=output

  # With automatic k optimization
  python run_modular_analysis.py --type1_dir=data_csv --type2_csv=data_csv --output_dir=output --auto_k

  # Matched data only with multi-modal clustering
  python run_modular_analysis.py --type1_dir=data_csv --type2_csv=data_csv --output_dir=output --match_only --multi_modal

  # Offline mode (no LLM, TF-IDF embeddings)
  python run_modular_analysis.py --type1_dir=data_csv --type2_csv=data_csv --output_dir=output --no_llm --use_tfidf
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--type1_dir", 
        type=str, 
        required=True,
        help="Directory containing Type1_*.csv files (one per criminal)"
    )
    
    parser.add_argument(
        "--type2_csv", 
        type=str, 
        required=True,
        help="Path to Type2 CSV file or directory containing Type2_*.csv files"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output",
        help="Directory to save output artifacts (default: output)"
    )
    
    parser.add_argument(
        "--n_clusters", 
        type=int, 
        default=5,
        help="Number of clusters for KMeans (default: 5, ignored if --auto_k is used)"
    )
    
    # Analysis options
    parser.add_argument(
        "--auto_k", 
        action="store_true",
        help="Automatically optimize k for conditional effect detection"
    )
    
    parser.add_argument(
        "--no_llm", 
        action="store_true",
        help="Disable LLM calls (useful if no OpenAI API key available)"
    )
    
    parser.add_argument(
        "--multi_modal", 
        action="store_true",
        help="Perform multi-modal clustering at the criminal level using Type 1 & Type 2 data"
    )
    
    parser.add_argument(
        "--train_proto_net", 
        action="store_true",
        help="Train a prototypical network on clustered event embeddings"
    )
    
    parser.add_argument(
        "--use_tfidf",
        action="store_true",
        help="Use TF-IDF embeddings instead of SentenceTransformer (offline mode)"
    )

    parser.add_argument(
        "--use_prototype",
        action="store_true",
        help="Use prototype embeddings to reduce lexical bias (requires OpenAI API key)"
    )

    parser.add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key for lexical variation generation (for prototype embeddings)"
    )

    parser.add_argument(
        "--use_statistical_validation",
        action="store_true",
        help="Use statistically valid k optimization with split-sample validation"
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence embedding model to use (default: all-MiniLM-L6-v2)"
    )

    parser.add_argument(
        "--use_openai",
        action="store_true",
        help="Use OpenAI embeddings (requires OPENAI_API_KEY in .env)"
    )

    parser.add_argument(
        "--openai_model",
        type=str,
        default="text-embedding-3-large",
        choices=["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"],
        help="OpenAI embedding model to use (default: text-embedding-3-large)"
    )

    parser.add_argument(
        "--use_lexical_bias_reduction",
        action="store_true",
        help="Apply lexical bias reduction (replace names, locations with generic labels)"
    )
    
    parser.add_argument(
        "--match_only", 
        action="store_true",
        help="Only analyze criminals with both Type1 and Type2 data (recommended)"
    )
    
    # Debugging and verbose options
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--dry_run", 
        action="store_true",
        help="Perform a dry run (validate inputs without running analysis)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create configuration
    config = AnalysisConfig.from_args(args)
    
    # Print configuration if verbose
    if args.verbose:
        print("\n" + "="*50)
        print("ANALYSIS CONFIGURATION")
        print("="*50)
        print(f"Type 1 directory:     {config.type1_dir}")
        print(f"Type 2 path:          {config.type2_csv}")
        print(f"Output directory:     {config.output_dir}")
        print(f"Number of clusters:   {config.n_clusters}")
        print(f"Auto k optimization:  {'Yes' if config.auto_k else 'No'}")
        print(f"LLM analysis:         {'No' if config.no_llm else 'Yes'}")
        print(f"Multi-modal:          {'Yes' if config.multi_modal else 'No'}")
        print(f"Embedding method:     {'TF-IDF' if config.use_tfidf else 'SentenceTransformer'}")
        print(f"Match only:           {'Yes' if config.match_only else 'No'}")
        print("="*50)
    
    # Validate inputs
    errors = []
    
    # Check Type 1 directory
    if not os.path.exists(config.type1_dir):
        errors.append(f"Type 1 directory does not exist: {config.type1_dir}")
    elif not os.path.isdir(config.type1_dir):
        errors.append(f"Type 1 path is not a directory: {config.type1_dir}")
    else:
        # Check for Type 1 files
        type1_files = [f for f in os.listdir(config.type1_dir) 
                      if f.startswith("Type1_") and f.endswith(".csv")]
        if not type1_files:
            errors.append(f"No Type1_*.csv files found in {config.type1_dir}")
    
    # Check Type 2 path
    if not os.path.exists(config.type2_csv):
        errors.append(f"Type 2 path does not exist: {config.type2_csv}")
    else:
        if os.path.isdir(config.type2_csv):
            # Check for Type 2 files in directory
            type2_files = [f for f in os.listdir(config.type2_csv) 
                          if f.startswith("Type2_") and f.endswith(".csv")]
            if not type2_files:
                errors.append(f"No Type2_*.csv files found in {config.type2_csv}")
        elif not config.type2_csv.endswith(".csv"):
            errors.append(f"Type 2 file must be a CSV file: {config.type2_csv}")
    
    # Check match_only constraint
    if config.match_only and not os.path.isdir(config.type2_csv):
        errors.append("When using --match_only, --type2_csv must be a directory")
    
    # Check clustering parameters
    if config.n_clusters < 2:
        errors.append("Number of clusters must be at least 2")
    
    # Print errors and exit if validation failed
    if errors:
        print("[ERROR] Input validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # Dry run mode
    if args.dry_run:
        print("\n[INFO] Dry run completed successfully. All inputs are valid.")
        print(f"[INFO] Analysis would be saved to: {config.output_dir}")
        return
    
    try:
        # Create and run pipeline
        print("\n[INFO] Starting modular criminal archetypal analysis...")
        pipeline = CriminalArchetypalAnalysisPipeline(config)
        results = pipeline.run_complete_analysis()
        
        # Print final summary
        print(f"\n[SUCCESS] Analysis completed successfully!")
        print(f"[INFO] Results saved to: {config.output_dir}")
        
        # Print key metrics
        if 'clustering' in results:
            silhouette = results['clustering'].get('silhouette', 'N/A')
            print(f"[INFO] Final clustering quality (silhouette): {silhouette}")
        
        if 'conditional_insights' in results:
            n_insights = len(results['conditional_insights'])
            print(f"[INFO] Conditional insights discovered: {n_insights}")
        
        if 'report_path' in results and results['report_path']:
            print(f"[INFO] HTML report available at: {results['report_path']}")
        
    except KeyboardInterrupt:
        print("\n[INFO] Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
