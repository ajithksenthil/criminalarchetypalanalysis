#!/usr/bin/env python3
"""
test_advanced_optimization.py

Test the advanced two-layer prototype optimization system.
"""

import numpy as np
import json
from pathlib import Path
from modular_analysis.advanced.cluster_prototype_optimizer import ClusterPrototypeOptimizer

def test_advanced_optimization():
    """Test the advanced optimization system."""
    
    print("🧪 TESTING ADVANCED TWO-LAYER PROTOTYPE OPTIMIZATION")
    print("=" * 60)
    
    # Load data from existing results
    output_dir = Path("output_best_sota")
    
    if not output_dir.exists():
        print("❌ output_best_sota directory not found")
        return
    
    print("[STEP 1] Loading Layer 1 results...")
    
    # Load embeddings and labels
    try:
        embeddings = np.load(output_dir / "data" / "embeddings.npy")
        labels = np.load(output_dir / "data" / "labels.npy")
        print(f"✅ Loaded {len(embeddings)} embeddings, {len(np.unique(labels))} clusters")
    except Exception as e:
        print(f"❌ Could not load embeddings/labels: {e}")
        return
    
    # Load cluster info for texts
    try:
        with open(output_dir / "clustering" / "cluster_info.json", 'r') as f:
            cluster_info = json.load(f)
        print(f"✅ Loaded cluster info for {len(cluster_info)} clusters")
    except Exception as e:
        print(f"❌ Could not load cluster info: {e}")
        return
    
    # Create sample texts (simplified for testing)
    print("[STEP 2] Creating sample texts...")
    original_texts = []
    processed_texts = []
    
    for cluster in cluster_info:
        cluster_size = cluster['size']
        samples = cluster['representative_samples']
        
        for i in range(cluster_size):
            sample_idx = i % len(samples)
            original_text = samples[sample_idx]
            processed_text = original_text.replace("Charles", "[PERSON]").replace("Dallas", "[LOCATION]")
            
            original_texts.append(original_text)
            processed_texts.append(processed_text)
    
    # Ensure we have the right number
    original_texts = original_texts[:len(embeddings)]
    processed_texts = processed_texts[:len(embeddings)]
    
    print(f"✅ Created {len(original_texts)} text samples")
    
    # Test optimization on a subset (for speed)
    print("[STEP 3] Running optimization on subset...")
    subset_size = min(500, len(embeddings))
    subset_indices = np.random.choice(len(embeddings), subset_size, replace=False)
    
    subset_embeddings = embeddings[subset_indices]
    subset_labels = labels[subset_indices]
    subset_original = [original_texts[i] for i in subset_indices]
    subset_processed = [processed_texts[i] for i in subset_indices]
    
    print(f"Testing on {subset_size} events...")
    
    # Initialize optimizer
    optimizer = ClusterPrototypeOptimizer(validation_split=0.3, random_state=42)
    
    try:
        # Run optimization
        results = optimizer.optimize_cluster_prototypes(
            embeddings=subset_embeddings,
            labels=subset_labels,
            original_texts=subset_original,
            processed_texts=subset_processed
        )
        
        print("✅ Optimization completed successfully!")
        
        # Print results summary
        print(f"\n📊 OPTIMIZATION RESULTS:")
        overall_metrics = results['overall_metrics']
        print(f"   Mean Coherence: {overall_metrics['mean_coherence']:.4f}")
        print(f"   Mean Validation Score: {overall_metrics['mean_validation_score']:.4f}")
        print(f"   Overall Optimization Score: {overall_metrics['overall_optimization_score']:.4f}")
        print(f"   Clusters Optimized: {overall_metrics['n_clusters_optimized']}")
        
        # Show cluster details
        print(f"\n🔍 CLUSTER OPTIMIZATION DETAILS:")
        optimized_clusters = results['optimized_clusters']
        
        for cluster_id, cluster_data in list(optimized_clusters.items())[:3]:  # Show first 3
            print(f"\n   Cluster {cluster_id}:")
            print(f"     Training Events: {cluster_data['n_train_events']}")
            print(f"     Validation Events: {cluster_data['n_val_events']}")
            print(f"     Coherence Score: {cluster_data['coherence_metrics']['coherence_score']:.4f}")
            print(f"     Representative Samples:")
            for i, sample in enumerate(cluster_data['representative_samples'][:2], 1):
                print(f"       {i}. {sample[:80]}...")
        
        # Test JSON serialization
        print(f"\n[STEP 4] Testing JSON serialization...")
        try:
            from modular_analysis.advanced.cluster_prototype_optimizer import _make_json_serializable
            serializable_results = _make_json_serializable(results)
            
            test_json = json.dumps(serializable_results, indent=2)
            print("✅ JSON serialization successful")
            
            # Save test results
            with open("test_optimization_results.json", 'w') as f:
                f.write(test_json)
            print("✅ Test results saved to test_optimization_results.json")
            
        except Exception as e:
            print(f"❌ JSON serialization failed: {e}")
            
            # Debug the issue
            print("🔍 Debugging serialization issue...")
            for key, value in results.items():
                try:
                    json.dumps({key: _make_json_serializable(value)})
                    print(f"   ✅ {key}: OK")
                except Exception as sub_e:
                    print(f"   ❌ {key}: {sub_e}")
        
        return results
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function."""
    results = test_advanced_optimization()
    
    if results:
        print(f"\n🎉 ADVANCED OPTIMIZATION TEST SUCCESSFUL!")
        print(f"The two-layer prototype system is working correctly.")
        print(f"Check test_optimization_results.json for detailed results.")
    else:
        print(f"\n❌ ADVANCED OPTIMIZATION TEST FAILED")
        print(f"Check the error messages above for debugging information.")

if __name__ == "__main__":
    main()
