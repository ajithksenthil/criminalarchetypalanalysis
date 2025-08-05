#!/usr/bin/env python3
"""
run_with_best_embeddings.py

Run analysis with the best available embeddings:
1. Try OpenAI embeddings first (if quota available)
2. Fall back to state-of-the-art Sentence-BERT models
3. Use intelligent cluster labeling
"""

import subprocess
import os
import sys
from pathlib import Path

def check_openai_quota():
    """Check if OpenAI API is available and has quota."""
    try:
        import openai
        from dotenv import load_dotenv
        
        load_dotenv()
        
        if not os.getenv('OPENAI_API_KEY'):
            return False, "No OPENAI_API_KEY found"
        
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Test with a minimal request
        response = client.embeddings.create(
            input=["test"],
            model="text-embedding-3-small"  # Use smaller model for test
        )
        
        return True, "OpenAI API available"
        
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg:
            return False, "OpenAI quota exceeded"
        elif "401" in error_msg:
            return False, "OpenAI API key invalid"
        else:
            return False, f"OpenAI error: {error_msg}"

def run_analysis_with_best_embeddings():
    """Run analysis with the best available embedding method."""
    
    print("🚀 RUNNING ANALYSIS WITH BEST AVAILABLE EMBEDDINGS")
    print("=" * 60)
    
    # Check OpenAI availability
    print("\n[STEP 1] Checking OpenAI API availability...")
    openai_available, openai_status = check_openai_quota()
    print(f"OpenAI Status: {openai_status}")
    
    if openai_available:
        print("✅ Using OpenAI text-embedding-3-large (highest quality)")
        embedding_config = {
            "use_openai": True,
            "openai_model": "text-embedding-3-large",
            "embedding_model": "text-embedding-3-large"
        }
    else:
        print("⚠️  OpenAI not available, using state-of-the-art Sentence-BERT")
        print("📊 Using all-mpnet-base-v2 (768 dimensions, excellent performance)")
        embedding_config = {
            "use_openai": False,
            "embedding_model": "all-mpnet-base-v2"
        }
    
    # Prepare command
    output_dir = "output_best_embeddings"
    
    cmd = [
        "python", "run_modular_analysis.py",
        "--type1_dir=type1csvs",
        "--type2_csv=type2csvs",
        f"--output_dir={output_dir}",
        "--auto_k",
        "--match_only",
        "--no_llm",  # Disable LLM for now due to quota issues
        "--n_clusters=5",
        f"--embedding_model={embedding_config['embedding_model']}"
    ]
    
    if embedding_config.get("use_openai"):
        cmd.extend([
            "--use_openai",
            f"--openai_model={embedding_config['openai_model']}"
        ])
    
    print(f"\n[STEP 2] Running analysis...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Analysis completed successfully!")
        
        # Organize results
        print(f"\n[STEP 3] Organizing results...")
        organize_cmd = ["python", "organize_and_validate_results.py", output_dir]
        subprocess.run(organize_cmd, check=True)
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📊 Organized results in: {output_dir}/organized_results/")
        
        # Show key results
        show_key_results(output_dir)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    
    return True

def show_key_results(output_dir):
    """Show key results from the analysis."""
    try:
        import json
        
        # Load main results
        with open(f"{output_dir}/analysis_results.json", 'r') as f:
            results = json.load(f)
        
        print(f"\n📊 KEY RESULTS:")
        print(f"   • Dataset: {results['data_summary']['n_criminals']} criminals, {results['data_summary']['n_events']} events")
        print(f"   • Clustering: {results['optimization']['optimal_k']} clusters")
        print(f"   • Quality: {results['clustering']['silhouette']:.4f} silhouette score")
        
        # Load organized summary
        summary_file = f"{output_dir}/organized_results/executive_summary.txt"
        if Path(summary_file).exists():
            print(f"\n📋 EXECUTIVE SUMMARY:")
            with open(summary_file, 'r') as f:
                lines = f.readlines()
                for line in lines[10:20]:  # Show key findings section
                    if line.strip():
                        print(f"   {line.strip()}")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   • Review detailed results: {output_dir}/organized_results/")
        print(f"   • Check clustering: {output_dir}/organized_results/clustering_table.txt")
        print(f"   • Examine effects: {output_dir}/organized_results/effects_table.txt")
        print(f"   • Validation report: {output_dir}/organized_results/validation_report.txt")
        
    except Exception as e:
        print(f"Could not load results summary: {e}")

def compare_embedding_methods():
    """Compare different embedding methods."""
    print("\n🔬 EMBEDDING METHOD COMPARISON")
    print("=" * 40)
    
    methods = [
        ("all-MiniLM-L6-v2", "Current baseline (384 dim)"),
        ("all-mpnet-base-v2", "State-of-the-art (768 dim)"),
        ("all-MiniLM-L12-v2", "Improved baseline (384 dim)")
    ]
    
    results = {}
    
    for model, description in methods:
        print(f"\n🧪 Testing {model}...")
        print(f"   {description}")
        
        output_dir = f"output_compare_{model.replace('/', '_').replace('-', '_')}"
        
        cmd = [
            "python", "run_modular_analysis.py",
            "--type1_dir=type1csvs",
            "--type2_csv=type2csvs",
            f"--output_dir={output_dir}",
            "--auto_k",
            "--match_only",
            "--no_llm",
            "--n_clusters=5",
            f"--embedding_model={model}"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            
            # Load results
            with open(f"{output_dir}/analysis_results.json", 'r') as f:
                analysis_results = json.load(f)
            
            silhouette = analysis_results['clustering']['silhouette']
            results[model] = {
                'silhouette': silhouette,
                'description': description,
                'output_dir': output_dir
            }
            
            print(f"   ✅ Silhouette: {silhouette:.4f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[model] = {'error': str(e)}
    
    # Show comparison
    print(f"\n📊 COMPARISON RESULTS:")
    successful = {k: v for k, v in results.items() if 'silhouette' in v}
    
    if successful:
        sorted_results = sorted(successful.items(), key=lambda x: x[1]['silhouette'], reverse=True)
        
        print(f"   Ranking by clustering quality:")
        for i, (model, result) in enumerate(sorted_results, 1):
            print(f"   {i}. {model}: {result['silhouette']:.4f}")
        
        best_model, best_result = sorted_results[0]
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"   📊 Silhouette: {best_result['silhouette']:.4f}")
        print(f"   📁 Results: {best_result['output_dir']}")

def main():
    """Main function."""
    print("🎯 BEST EMBEDDINGS ANALYSIS SYSTEM")
    print("Automatically selects the best available embedding method")
    print()
    
    choice = input("Choose option:\n1. Run with best available embeddings\n2. Compare multiple embedding methods\n3. Both\nChoice (1/2/3): ")
    
    if choice == "1":
        run_analysis_with_best_embeddings()
    elif choice == "2":
        compare_embedding_methods()
    elif choice == "3":
        print("\n🚀 Running comprehensive analysis...")
        run_analysis_with_best_embeddings()
        print("\n" + "="*60)
        compare_embedding_methods()
    else:
        print("Invalid choice. Running with best available embeddings...")
        run_analysis_with_best_embeddings()

if __name__ == "__main__":
    main()
