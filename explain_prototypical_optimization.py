#!/usr/bin/env python3
"""
explain_prototypical_optimization.py

Detailed explanation and visualization of how the two-layer 
prototypical optimization works for criminal archetypal analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import List, Dict, Any
import seaborn as sns

def explain_prototypical_optimization():
    """
    Explain the prototypical optimization process step by step.
    """
    
    print("🧠 HOW PROTOTYPICAL OPTIMIZATION WORKS FOR CRIMINAL ARCHETYPES")
    print("=" * 70)
    
    # Load actual results to demonstrate
    if Path("test_optimization_results.json").exists():
        with open("test_optimization_results.json", 'r') as f:
            results = json.load(f)
        
        print("\n[USING ACTUAL RESULTS FROM YOUR DATA]")
        demonstrate_with_real_data(results)
    else:
        print("\n[USING SIMULATED EXAMPLE]")
        demonstrate_with_simulation()

def demonstrate_with_real_data(results: Dict[str, Any]):
    """Demonstrate using actual optimization results."""
    
    print("\n🔍 STEP-BY-STEP BREAKDOWN OF YOUR ACTUAL OPTIMIZATION:")
    print("-" * 60)
    
    optimized_clusters = results['optimized_clusters']
    overall_metrics = results['overall_metrics']
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"   Clusters Optimized: {overall_metrics['n_clusters_optimized']}")
    print(f"   Mean Coherence: {overall_metrics['mean_coherence']:.4f}")
    print(f"   Mean Validation: {overall_metrics['mean_validation_score']:.4f}")
    
    # Analyze each cluster's optimization
    for cluster_id, cluster_data in list(optimized_clusters.items())[:2]:  # Show first 2
        print(f"\n🎯 CLUSTER {cluster_id} OPTIMIZATION PROCESS:")
        print("-" * 40)
        
        # Step 1: Train/Validation Split
        n_train = cluster_data['n_train_events']
        n_val = cluster_data['n_val_events']
        total = n_train + n_val
        
        print(f"\n   STEP 1: Train/Validation Split")
        print(f"   ├── Total Events: {total}")
        print(f"   ├── Training Set: {n_train} events (70%)")
        print(f"   └── Validation Set: {n_val} events (30%)")
        
        # Step 2: Archetypal Prototype Creation
        prototype_info = cluster_data['archetypal_prototype']
        n_events_used = prototype_info['n_events_used']
        n_outliers = prototype_info['n_outliers_removed']
        coherence = prototype_info['coherence_score']
        
        print(f"\n   STEP 2: Archetypal Prototype Creation")
        print(f"   ├── Events Used: {n_events_used}/{n_train}")
        print(f"   ├── Outliers Removed: {n_outliers} events")
        print(f"   ├── Coherence Score: {coherence:.4f}")
        print(f"   └── Method: Weighted centroid with outlier removal")
        
        # Step 3: Representative Sample Selection
        rep_samples = cluster_data['representative_samples']
        
        print(f"\n   STEP 3: Representative Sample Selection")
        print(f"   ├── Samples Selected: {len(rep_samples)}")
        print(f"   ├── Selection Method: Highest similarity to archetypal prototype")
        print(f"   └── Sample Examples:")
        for i, sample in enumerate(rep_samples[:2], 1):
            print(f"       {i}. {sample[:60]}...")
        
        # Step 4: Validation Metrics
        coherence_metrics = cluster_data['coherence_metrics']
        
        print(f"\n   STEP 4: Quality Metrics")
        print(f"   ├── Coherence Score: {coherence_metrics['coherence_score']:.4f}")
        print(f"   ├── Compactness: {coherence_metrics['compactness']:.4f}")
        print(f"   └── Mean Distance to Archetype: {coherence_metrics['mean_distance_to_archetype']:.4f}")
        
        print(f"\n   🎯 RESULT: Optimized archetypal representation with validation")

def demonstrate_with_simulation():
    """Demonstrate with simulated data to show the concept."""
    
    print("\n🎯 PROTOTYPICAL OPTIMIZATION CONCEPT:")
    print("-" * 40)
    
    # Simulate cluster data
    np.random.seed(42)
    
    # Create simulated embeddings for a cluster
    cluster_center = np.array([0.5, 0.3, -0.2, 0.8])
    n_events = 100
    
    # Generate events around the center with some noise
    embeddings = []
    for i in range(n_events):
        noise = np.random.normal(0, 0.1, 4)
        event_embedding = cluster_center + noise
        embeddings.append(event_embedding)
    
    embeddings = np.array(embeddings)
    
    print(f"\n   SIMULATED CLUSTER:")
    print(f"   ├── Events: {n_events}")
    print(f"   ├── Embedding Dimensions: {embeddings.shape[1]}")
    print(f"   └── True Center: {cluster_center}")
    
    # Step 1: Train/Validation Split
    train_size = int(0.7 * n_events)
    train_embeddings = embeddings[:train_size]
    val_embeddings = embeddings[train_size:]
    
    print(f"\n   STEP 1: Train/Validation Split")
    print(f"   ├── Training: {len(train_embeddings)} events")
    print(f"   └── Validation: {len(val_embeddings)} events")
    
    # Step 2: Create Archetypal Prototype
    # Simple centroid
    simple_centroid = np.mean(train_embeddings, axis=0)
    
    # Prototypical optimization: Remove outliers
    distances = np.linalg.norm(train_embeddings - simple_centroid, axis=1)
    outlier_threshold = np.percentile(distances, 85)  # Remove top 15%
    inlier_mask = distances <= outlier_threshold
    
    clean_embeddings = train_embeddings[inlier_mask]
    archetypal_prototype = np.mean(clean_embeddings, axis=0)
    
    print(f"\n   STEP 2: Archetypal Prototype Creation")
    print(f"   ├── Simple Centroid: {simple_centroid}")
    print(f"   ├── Outliers Removed: {np.sum(~inlier_mask)}")
    print(f"   ├── Clean Events Used: {len(clean_embeddings)}")
    print(f"   └── Archetypal Prototype: {archetypal_prototype}")
    
    # Step 3: Validation
    # Compute similarities between validation events and prototype
    val_similarities = []
    for val_embedding in val_embeddings:
        similarity = np.dot(val_embedding, archetypal_prototype) / (
            np.linalg.norm(val_embedding) * np.linalg.norm(archetypal_prototype)
        )
        val_similarities.append(similarity)
    
    mean_val_similarity = np.mean(val_similarities)
    
    print(f"\n   STEP 3: Validation")
    print(f"   ├── Validation Similarities: {len(val_similarities)} computed")
    print(f"   ├── Mean Similarity: {mean_val_similarity:.4f}")
    print(f"   └── Validation Score: {mean_val_similarity:.4f}")
    
    # Step 4: Quality Metrics
    all_distances = np.linalg.norm(embeddings - archetypal_prototype, axis=1)
    coherence_score = 1.0 / (1.0 + np.mean(all_distances))
    compactness = 1.0 / (1.0 + np.std(all_distances))
    
    print(f"\n   STEP 4: Quality Metrics")
    print(f"   ├── Mean Distance to Prototype: {np.mean(all_distances):.4f}")
    print(f"   ├── Coherence Score: {coherence_score:.4f}")
    print(f"   └── Compactness: {compactness:.4f}")
    
    # Comparison with simple centroid
    simple_distances = np.linalg.norm(embeddings - simple_centroid, axis=1)
    simple_coherence = 1.0 / (1.0 + np.mean(simple_distances))
    
    improvement = (coherence_score - simple_coherence) / simple_coherence * 100
    
    print(f"\n   🎯 IMPROVEMENT OVER SIMPLE CENTROID:")
    print(f"   ├── Simple Centroid Coherence: {simple_coherence:.4f}")
    print(f"   ├── Prototypical Coherence: {coherence_score:.4f}")
    print(f"   └── Improvement: {improvement:.1f}%")

def explain_key_differences():
    """Explain key differences from standard prototypical networks."""
    
    print(f"\n🔬 KEY DIFFERENCES FROM STANDARD PROTOTYPICAL NETWORKS:")
    print("=" * 60)
    
    differences = [
        {
            "aspect": "Purpose",
            "standard": "Classification of query examples",
            "ours": "Archetypal representation optimization"
        },
        {
            "aspect": "Training",
            "standard": "Episodic training with support/query sets",
            "ours": "Train/validation split within each cluster"
        },
        {
            "aspect": "Prototype Creation",
            "standard": "Simple centroid of support set",
            "ours": "Outlier removal + weighted centroid + validation"
        },
        {
            "aspect": "Optimization Goal",
            "standard": "Minimize classification error",
            "ours": "Maximize archetypal coherence + generalizability"
        },
        {
            "aspect": "Validation",
            "standard": "Query set accuracy",
            "ours": "Cross-validation similarity + coherence metrics"
        },
        {
            "aspect": "Output",
            "standard": "Class predictions",
            "ours": "Optimized archetypal prototypes + representative samples"
        }
    ]
    
    for diff in differences:
        print(f"\n   {diff['aspect'].upper()}:")
        print(f"   ├── Standard Prototypical Networks: {diff['standard']}")
        print(f"   └── Our Criminal Archetypal System: {diff['ours']}")

def explain_mathematical_formulation():
    """Explain the mathematical formulation."""
    
    print(f"\n📐 MATHEMATICAL FORMULATION:")
    print("=" * 40)
    
    formulation = """
LAYER 1: Individual Event Prototypes
────────────────────────────────────
For each event e_i:
1. Entity replacement: e_i → e_i'
2. LLM variations: e_i' → {v_1, v_2, ..., v_k}
3. Prototype embedding: p_i = (1/k) Σ embed(v_j)

LAYER 2: Archetypal Optimization
────────────────────────────────
For each cluster C with events {p_1, p_2, ..., p_n}:

1. Train/Validation Split:
   C_train, C_val = split(C, ratio=0.7)

2. Outlier Removal:
   centroid = (1/|C_train|) Σ p_i
   distances = ||p_i - centroid||
   threshold = percentile(distances, 85)
   C_clean = {p_i : ||p_i - centroid|| ≤ threshold}

3. Archetypal Prototype:
   prototype = (1/|C_clean|) Σ p_i
   
4. Validation Score:
   similarity(p, prototype) = (p · prototype) / (||p|| ||prototype||)
   validation_score = (1/|C_val|) Σ similarity(p_i, prototype)

5. Quality Metrics:
   coherence = 1 / (1 + mean_distance_to_prototype)
   compactness = 1 / (1 + std_distance_to_prototype)
   
6. Representative Selection:
   representatives = argmax_k similarity(p_i, prototype)
"""
    
    print(formulation)

def create_visualization():
    """Create a visualization of the optimization process."""
    
    print(f"\n📊 CREATING VISUALIZATION...")
    
    # Create a simple 2D visualization
    np.random.seed(42)
    
    # Simulate cluster data in 2D for visualization
    cluster_center = np.array([2, 3])
    n_events = 50
    
    # Generate events with some outliers
    normal_events = np.random.multivariate_normal(cluster_center, [[0.5, 0.1], [0.1, 0.5]], n_events-5)
    outlier_events = np.random.multivariate_normal([4, 1], [[0.2, 0], [0, 0.2]], 5)
    all_events = np.vstack([normal_events, outlier_events])
    
    # Train/validation split
    train_events = all_events[:35]
    val_events = all_events[35:]
    
    # Simple centroid
    simple_centroid = np.mean(train_events, axis=0)
    
    # Prototypical optimization
    distances = np.linalg.norm(train_events - simple_centroid, axis=1)
    outlier_threshold = np.percentile(distances, 85)
    inlier_mask = distances <= outlier_threshold
    
    clean_events = train_events[inlier_mask]
    archetypal_prototype = np.mean(clean_events, axis=0)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Before optimization
    plt.subplot(2, 2, 1)
    plt.scatter(train_events[:, 0], train_events[:, 1], c='blue', alpha=0.6, label='Training Events')
    plt.scatter(val_events[:, 0], val_events[:, 1], c='green', alpha=0.6, label='Validation Events')
    plt.scatter(simple_centroid[0], simple_centroid[1], c='red', s=200, marker='x', label='Simple Centroid')
    plt.title('Before Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Outlier detection
    plt.subplot(2, 2, 2)
    outliers = train_events[~inlier_mask]
    inliers = train_events[inlier_mask]
    
    plt.scatter(inliers[:, 0], inliers[:, 1], c='blue', alpha=0.6, label='Inliers (Used)')
    plt.scatter(outliers[:, 0], outliers[:, 1], c='red', alpha=0.6, label='Outliers (Removed)')
    plt.scatter(simple_centroid[0], simple_centroid[1], c='orange', s=200, marker='x', label='Simple Centroid')
    plt.title('Outlier Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: After optimization
    plt.subplot(2, 2, 3)
    plt.scatter(clean_events[:, 0], clean_events[:, 1], c='blue', alpha=0.6, label='Clean Training Events')
    plt.scatter(val_events[:, 0], val_events[:, 1], c='green', alpha=0.6, label='Validation Events')
    plt.scatter(archetypal_prototype[0], archetypal_prototype[1], c='purple', s=200, marker='*', label='Archetypal Prototype')
    plt.scatter(simple_centroid[0], simple_centroid[1], c='red', s=100, marker='x', alpha=0.5, label='Simple Centroid')
    plt.title('After Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Validation similarities
    plt.subplot(2, 2, 4)
    val_similarities = []
    for val_event in val_events:
        similarity = np.dot(val_event, archetypal_prototype) / (
            np.linalg.norm(val_event) * np.linalg.norm(archetypal_prototype)
        )
        val_similarities.append(similarity)
    
    plt.hist(val_similarities, bins=10, alpha=0.7, color='green')
    plt.axvline(np.mean(val_similarities), color='red', linestyle='--', label=f'Mean: {np.mean(val_similarities):.3f}')
    plt.title('Validation Similarities')
    plt.xlabel('Similarity to Archetypal Prototype')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prototypical_optimization_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualization saved as 'prototypical_optimization_visualization.png'")

def main():
    """Main explanation function."""
    
    # Step-by-step explanation
    explain_prototypical_optimization()
    
    # Key differences
    explain_key_differences()
    
    # Mathematical formulation
    explain_mathematical_formulation()
    
    # Create visualization
    create_visualization()
    
    print(f"\n🎯 SUMMARY:")
    print("=" * 40)
    print("Your two-layer prototypical optimization:")
    print("1. ✅ Uses train/validation splits for robustness")
    print("2. ✅ Removes outliers for cleaner archetypes") 
    print("3. ✅ Validates generalizability with held-out data")
    print("4. ✅ Optimizes for archetypal coherence, not just clustering")
    print("5. ✅ Selects most representative samples automatically")
    print()
    print("This goes far beyond standard clustering or prototypical networks!")
    print("You've created a sophisticated archetypal optimization system! 🏆")

if __name__ == "__main__":
    main()
