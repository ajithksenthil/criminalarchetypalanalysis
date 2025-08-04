#!/usr/bin/env python3
"""
example_conditional_k_optimization.py

Example script demonstrating the conditional effect-optimized clustering functionality.
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.cluster import KMeans

# Import the new functions
from analysis_integration_improved import (
    get_condition_map,
    find_optimal_k_for_conditional_analysis,
    multi_objective_k_selection,
    build_conditional_markov,
    compute_stationary_distribution
)

def create_realistic_test_data():
    """Create more realistic test data that mimics actual criminal data patterns."""
    np.random.seed(42)
    
    # Create embeddings with some structure
    n_events = 200
    n_dims = 50
    
    # Create clusters with different characteristics
    cluster_centers = [
        np.array([2, 1, 0, -1, 0.5] + [0] * (n_dims - 5)),  # Violence cluster
        np.array([-1, 2, 1, 0, -0.5] + [0] * (n_dims - 5)),  # Substance abuse cluster
        np.array([0, -1, 2, 1, 0] + [0] * (n_dims - 5)),     # Property crime cluster
        np.array([1, 0, -1, 2, 0.5] + [0] * (n_dims - 5)),   # Mental health cluster
    ]
    
    embeddings = []
    true_labels = []
    
    for i in range(n_events):
        cluster_id = np.random.choice(len(cluster_centers))
        center = cluster_centers[cluster_id]
        noise = np.random.normal(0, 0.5, n_dims)
        embedding = center + noise
        embeddings.append(embedding)
        true_labels.append(cluster_id)
    
    embeddings = np.array(embeddings)
    
    # Create criminal sequences with patterns
    criminal_sequences = {}
    event_idx = 0
    
    for crim_id in range(1, 21):  # 20 criminals
        criminal_sequences[str(crim_id)] = []
        
        # Different criminals have different patterns
        if crim_id <= 5:  # Violence-prone criminals
            preferred_clusters = [0, 3]  # Violence and mental health
        elif crim_id <= 10:  # Substance abuse criminals
            preferred_clusters = [1, 0]  # Substance abuse and violence
        elif crim_id <= 15:  # Property criminals
            preferred_clusters = [2, 1]  # Property and substance abuse
        else:  # Mixed patterns
            preferred_clusters = [0, 1, 2, 3]
        
        num_events = np.random.randint(8, 15)
        for _ in range(num_events):
            if event_idx < len(true_labels):
                # Bias towards preferred clusters for this criminal type
                if np.random.random() < 0.7:
                    cluster = np.random.choice(preferred_clusters)
                    # Find an event from this cluster
                    cluster_events = [i for i, label in enumerate(true_labels) if label == cluster]
                    if cluster_events:
                        event_idx = np.random.choice(cluster_events)
                
                criminal_sequences[str(crim_id)].append(event_idx)
                event_idx = (event_idx + 1) % len(true_labels)
    
    # Create Type 2 data with correlations to criminal patterns
    type2_data = []
    
    for crim_id in range(1, 21):
        # Create correlated characteristics
        if crim_id <= 5:  # Violence-prone
            sex = "Male"
            abused = "Yes" if np.random.random() < 0.8 else "No"
            victims = str(np.random.randint(3, 8))
            education = np.random.choice(["None", "High School"], p=[0.6, 0.4])
        elif crim_id <= 10:  # Substance abuse
            sex = "Male" if np.random.random() < 0.7 else "Female"
            abused = "Yes" if np.random.random() < 0.6 else "No"
            victims = str(np.random.randint(1, 4))
            education = np.random.choice(["High School", "None"], p=[0.7, 0.3])
        elif crim_id <= 15:  # Property criminals
            sex = "Male" if np.random.random() < 0.6 else "Female"
            abused = "No" if np.random.random() < 0.7 else "Yes"
            victims = str(np.random.randint(1, 3))
            education = np.random.choice(["High School", "College"], p=[0.8, 0.2])
        else:  # Mixed
            sex = np.random.choice(["Male", "Female"])
            abused = np.random.choice(["Yes", "No"])
            victims = str(np.random.randint(1, 5))
            education = np.random.choice(["None", "High School", "College"])
        
        # Add data for each heading
        type2_data.extend([
            {"CriminalID": str(crim_id), "Heading": "Sex", "Value": sex},
            {"CriminalID": str(crim_id), "Heading": "Physically abused?", "Value": abused},
            {"CriminalID": str(crim_id), "Heading": "Number of victims", "Value": victims},
            {"CriminalID": str(crim_id), "Heading": "Education level", "Value": education}
        ])
    
    type2_df = pd.DataFrame(type2_data)
    
    return embeddings, criminal_sequences, type2_df, true_labels

def demonstrate_k_optimization():
    """Demonstrate the k optimization functionality."""
    print("="*70)
    print("CONDITIONAL EFFECT-OPTIMIZED CLUSTERING DEMONSTRATION")
    print("="*70)
    
    # Create test data
    print("\n1. Creating realistic test data...")
    embeddings, criminal_sequences, type2_df, true_labels = create_realistic_test_data()
    print(f"   - Created {len(embeddings)} event embeddings")
    print(f"   - Created sequences for {len(criminal_sequences)} criminals")
    print(f"   - Created Type 2 data with {len(type2_df)} records")
    print(f"   - True number of clusters: {len(set(true_labels))}")
    
    # Test get_condition_map
    print("\n2. Testing condition mapping...")
    sex_map = get_condition_map(type2_df, "Sex")
    abuse_map = get_condition_map(type2_df, "Physically abused?")
    print(f"   - Sex distribution: {dict(pd.Series(list(sex_map.values())).value_counts())}")
    print(f"   - Abuse distribution: {dict(pd.Series(list(abuse_map.values())).value_counts())}")
    
    # Test conditional k optimization
    print("\n3. Running conditional k optimization...")
    print("   This may take a moment...")
    
    optimal_k, results = find_optimal_k_for_conditional_analysis(
        embeddings, criminal_sequences, type2_df, k_range=range(2, 8)
    )
    
    print(f"\n   Optimal k selected: {optimal_k}")
    print(f"   True k was: {len(set(true_labels))}")
    
    print("\n   Detailed results:")
    for k, metrics in results.items():
        print(f"   k={k}: {metrics['significant_effects']}/{metrics['total_effects']} "
              f"significant ({metrics['significance_rate']:.1%}), "
              f"mean effect={metrics['mean_effect_size']:.3f}, "
              f"score={metrics['score']:.3f}")
    
    # Test multi-objective optimization
    print("\n4. Running multi-objective k optimization...")
    multi_k, multi_results = multi_objective_k_selection(
        embeddings, criminal_sequences, type2_df, k_range=range(2, 8)
    )
    
    print(f"\n   Multi-objective optimal k: {multi_k}")
    print("\n   Multi-objective results:")
    for k, metrics in multi_results.items():
        print(f"   k={k}: silhouette={metrics['silhouette']:.3f}, "
              f"effect={metrics['effect_score']:.3f}, "
              f"combined={metrics['combined_score']:.3f}")
    
    # Compare with standard clustering
    print("\n5. Comparing with standard k-means...")
    from sklearn.metrics import silhouette_score
    
    best_silhouette_k = None
    best_silhouette_score = -1
    
    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        
        if score > best_silhouette_score:
            best_silhouette_score = score
            best_silhouette_k = k
    
    print(f"   Best k by silhouette score: {best_silhouette_k} (score: {best_silhouette_score:.3f})")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"True number of clusters:              {len(set(true_labels))}")
    print(f"Conditional effect optimization:      {optimal_k}")
    print(f"Multi-objective optimization:         {multi_k}")
    print(f"Standard silhouette optimization:     {best_silhouette_k}")
    print("\nThe conditional effect optimization should be closer to the true number")
    print("of clusters when the data has meaningful conditional patterns.")
    
    # Save results
    output_data = {
        "true_k": len(set(true_labels)),
        "conditional_optimal_k": optimal_k,
        "multi_objective_k": multi_k,
        "silhouette_optimal_k": best_silhouette_k,
        "conditional_results": results,
        "multi_objective_results": multi_results
    }
    
    with open("k_optimization_demo_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: k_optimization_demo_results.json")

if __name__ == "__main__":
    demonstrate_k_optimization()
