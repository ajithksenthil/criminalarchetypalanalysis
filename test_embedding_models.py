#!/usr/bin/env python3
"""
test_embedding_models.py

Quick script to test different state-of-the-art embedding models on your criminal data.
"""

import subprocess
import time
import json
from pathlib import Path

def test_embedding_models():
    """Test different embedding models and compare results."""
    
    # State-of-the-art models to test
    models_to_test = [
        ("all-MiniLM-L6-v2", "Current baseline (384 dim)"),
        ("all-mpnet-base-v2", "Best general performance (768 dim)"),
        ("all-MiniLM-L12-v2", "Improved version (384 dim)"),
        ("paraphrase-mpnet-base-v2", "Good for similar events (768 dim)"),
        ("all-distilroberta-v1", "Fast and good (768 dim)")
    ]
    
    print("="*80)
    print("TESTING STATE-OF-THE-ART EMBEDDING MODELS")
    print("="*80)
    
    results = {}
    
    for model_name, description in models_to_test:
        print(f"\n🧪 TESTING: {model_name}")
        print(f"📝 Description: {description}")
        
        output_dir = f"output_embeddings_{model_name.replace('/', '_').replace('-', '_')}"
        
        # Run analysis with this embedding model
        cmd = [
            "python", "run_modular_analysis.py",
            "--type1_dir=type1csvs",
            "--type2_csv=type2csvs", 
            f"--output_dir={output_dir}",
            "--auto_k",
            "--match_only",
            "--no_llm",
            f"--embedding_model={model_name}",
            "--n_clusters=5"
        ]
        
        print(f"🚀 Running: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ Completed in {elapsed_time:.1f} seconds")
                
                # Load results
                try:
                    with open(f"{output_dir}/analysis_results.json", 'r') as f:
                        analysis_results = json.load(f)
                    
                    with open(f"{output_dir}/analysis/conditional_insights.json", 'r') as f:
                        insights = json.load(f)
                    
                    # Extract key metrics
                    silhouette = analysis_results['clustering']['silhouette']
                    n_insights = len(insights)
                    optimal_k = analysis_results['optimization']['optimal_k']
                    
                    # Calculate effect statistics
                    effect_sizes = [insight['l1_difference'] for insight in insights.values()]
                    p_values = [insight['statistics']['ks_pvalue'] for insight in insights.values()]
                    
                    significant_effects = sum(1 for p in p_values if p < 0.05)
                    unique_p_values = len(set(p_values))
                    mean_effect_size = sum(effect_sizes) / len(effect_sizes) if effect_sizes else 0
                    
                    results[model_name] = {
                        'description': description,
                        'success': True,
                        'time': elapsed_time,
                        'silhouette': silhouette,
                        'optimal_k': optimal_k,
                        'n_insights': n_insights,
                        'significant_effects': significant_effects,
                        'unique_p_values': unique_p_values,
                        'mean_effect_size': mean_effect_size,
                        'output_dir': output_dir
                    }
                    
                    print(f"📊 Silhouette: {silhouette:.4f}")
                    print(f"📊 Optimal k: {optimal_k}")
                    print(f"📊 Significant effects: {significant_effects}/{n_insights}")
                    print(f"📊 Unique p-values: {unique_p_values}")
                    
                except Exception as e:
                    print(f"❌ Error loading results: {e}")
                    results[model_name] = {
                        'description': description,
                        'success': False,
                        'error': f"Results loading error: {e}",
                        'time': elapsed_time
                    }
            else:
                print(f"❌ Failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
                results[model_name] = {
                    'description': description,
                    'success': False,
                    'error': result.stderr,
                    'time': elapsed_time
                }
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout after 10 minutes")
            results[model_name] = {
                'description': description,
                'success': False,
                'error': "Timeout",
                'time': 600
            }
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            results[model_name] = {
                'description': description,
                'success': False,
                'error': str(e),
                'time': 0
            }
    
    # Generate comparison report
    generate_comparison_report(results)
    
    return results

def generate_comparison_report(results):
    """Generate a comprehensive comparison report."""
    
    print("\n" + "="*80)
    print("EMBEDDING MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Filter successful results
    successful_results = {k: v for k, v in results.items() if v['success']}
    
    if not successful_results:
        print("❌ No models completed successfully!")
        return
    
    # Sort by silhouette score
    sorted_results = sorted(successful_results.items(), 
                          key=lambda x: x[1]['silhouette'], reverse=True)
    
    print("\n🏆 RANKING BY CLUSTERING QUALITY (Silhouette Score):")
    print("-" * 60)
    
    for i, (model_name, result) in enumerate(sorted_results, 1):
        print(f"\n{i}. {model_name}")
        print(f"   📝 {result['description']}")
        print(f"   📊 Silhouette: {result['silhouette']:.4f}")
        print(f"   🎯 Optimal k: {result['optimal_k']}")
        print(f"   📈 Significant effects: {result['significant_effects']}/{result['n_insights']}")
        print(f"   🔢 Unique p-values: {result['unique_p_values']}")
        print(f"   ⏱️  Time: {result['time']:.1f}s")
        print(f"   📁 Output: {result['output_dir']}")
    
    # Best model recommendation
    best_model, best_result = sorted_results[0]
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"Best model: {best_model}")
    print(f"Silhouette improvement: {best_result['silhouette']:.4f}")
    
    if best_result['silhouette'] > 0.1:
        print("✅ EXCELLENT: This model shows good clustering quality!")
    elif best_result['silhouette'] > 0.05:
        print("⚠️  MODERATE: This model shows some improvement")
    else:
        print("❌ LIMITED: Still low clustering quality - may need different approach")
    
    print(f"\n💡 TO USE THE BEST MODEL:")
    print(f"python run_modular_analysis.py \\")
    print(f"  --type1_dir=type1csvs \\")
    print(f"  --type2_csv=type2csvs \\")
    print(f"  --output_dir=output_best_model \\")
    print(f"  --embedding_model={best_model} \\")
    print(f"  --auto_k --match_only --no_llm")
    
    print(f"\n📊 TO ORGANIZE RESULTS:")
    print(f"python organize_and_validate_results.py {best_result['output_dir']}")
    
    # Save detailed comparison
    comparison_file = "embedding_model_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {comparison_file}")
    
    # Performance vs Quality analysis
    print(f"\n⚡ PERFORMANCE vs QUALITY ANALYSIS:")
    print("-" * 40)
    
    for model_name, result in sorted_results:
        if result['success']:
            quality_score = result['silhouette']
            speed_score = 1 / result['time'] if result['time'] > 0 else 0
            
            if quality_score > 0.1 and speed_score > 0.01:
                rating = "🌟 EXCELLENT"
            elif quality_score > 0.05:
                rating = "⭐ GOOD"
            else:
                rating = "💫 FAIR"
            
            print(f"{rating} {model_name}: Quality={quality_score:.4f}, Speed={result['time']:.1f}s")

def main():
    """Main function."""
    print("🧪 EMBEDDING MODEL TESTING SYSTEM")
    print("This will test multiple state-of-the-art embedding models on your criminal data")
    print("Each test takes 2-10 minutes depending on the model")
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    results = test_embedding_models()
    
    print(f"\n🎉 Testing complete!")
    print(f"Check the output directories for detailed results from each model.")

if __name__ == "__main__":
    main()
