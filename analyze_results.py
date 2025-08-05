#!/usr/bin/env python3
"""
analyze_results.py

Analyze the conditional matrix analysis results from the modular system.
"""

import numpy as np
import json

def analyze_conditional_results():
    """Analyze the conditional matrix analysis results."""
    
    print("="*70)
    print("CONDITIONAL MATRIX ANALYSIS RESULTS")
    print("="*70)
    
    # Load the transition matrix and stationary distribution
    try:
        transition_matrix = np.load('output/data/global_transition_matrix.npy')
        stationary = np.load('output/data/global_stationary_distribution.npy')
        
        print("\n1. GLOBAL TRANSITION MATRIX:")
        print("-" * 40)
        print("This shows the probability of transitioning from one cluster to another:")
        print(f"Shape: {transition_matrix.shape}")
        print("\nMatrix (rows=from cluster, columns=to cluster):")
        for i in range(len(transition_matrix)):
            row_str = f"Cluster {i}: "
            for j in range(len(transition_matrix[i])):
                row_str += f"{transition_matrix[i][j]:.3f} "
            print(row_str)
        
        print("\n2. STATIONARY DISTRIBUTION:")
        print("-" * 40)
        print("Long-term probability of being in each cluster:")
        for i, prob in enumerate(stationary):
            print(f"  Cluster {i}: {prob:.3f} ({prob*100:.1f}%)")
        
        # Find dominant clusters
        dominant_cluster = np.argmax(stationary)
        print(f"\nDominant cluster: {dominant_cluster} ({stationary[dominant_cluster]*100:.1f}%)")
        
    except Exception as e:
        print(f"Error loading transition data: {e}")
        return
    
    # Load criminal sequence
    try:
        with open('output/criminal_sequences.json', 'r') as f:
            sequences = json.load(f)
        
        print("\n3. CRIMINAL SEQUENCE ANALYSIS:")
        print("-" * 40)
        
        for criminal_id, sequence in sequences.items():
            print(f"Criminal: {criminal_id}")
            print(f"Sequence: {sequence}")
            print(f"Length: {len(sequence)} events")
            
            # Analyze transitions
            transitions = []
            for i in range(len(sequence)-1):
                from_state = sequence[i]
                to_state = sequence[i+1]
                transitions.append((from_state, to_state))
            
            print(f"\nObserved Transitions ({len(transitions)} total):")
            for i, (from_state, to_state) in enumerate(transitions):
                prob = transition_matrix[from_state, to_state] if transition_matrix[from_state, to_state] > 0 else 0
                print(f"  Step {i+1}: Cluster {from_state} → Cluster {to_state} (prob: {prob:.3f})")
            
            # Count cluster frequencies
            cluster_counts = {}
            for cluster in sequence:
                cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
            
            print(f"\nCluster Frequencies:")
            for cluster in sorted(cluster_counts.keys()):
                count = cluster_counts[cluster]
                percentage = (count / len(sequence)) * 100
                print(f"  Cluster {cluster}: {count} events ({percentage:.1f}%)")
    
    except Exception as e:
        print(f"Error loading sequence data: {e}")
        return
    
    # Load cluster information
    try:
        with open('output/analysis_results.json', 'r') as f:
            results = json.load(f)
        
        print("\n4. CLUSTER THEMES:")
        print("-" * 40)
        
        clusters = results.get('clusters', [])
        for cluster in clusters:
            cluster_id = cluster['cluster_id']
            size = cluster['size']
            theme = cluster.get('archetypal_theme', 'Unknown')
            samples = cluster.get('representative_samples', [])
            
            print(f"\nCluster {cluster_id} ({size} events): {theme}")
            if samples:
                print("  Representative events:")
                for i, sample in enumerate(samples[:2]):  # Show first 2 samples
                    truncated = sample[:100] + "..." if len(sample) > 100 else sample
                    print(f"    {i+1}. {truncated}")
    
    except Exception as e:
        print(f"Error loading cluster data: {e}")
    
    # Calculate transition entropy
    try:
        def transition_entropy(matrix):
            entropy = 0.0
            for i in range(len(matrix)):
                row_sum = np.sum(matrix[i])
                if row_sum > 0:
                    for j in range(len(matrix[i])):
                        if matrix[i][j] > 0:
                            p = matrix[i][j] / row_sum
                            entropy -= p * np.log2(p)
            return entropy / len(matrix)
        
        entropy = transition_entropy(transition_matrix)
        
        print("\n5. TRANSITION ANALYSIS:")
        print("-" * 40)
        print(f"Transition Entropy: {entropy:.3f}")
        print("  (0.0 = completely predictable, higher = more random)")
        
        if entropy < 0.5:
            print("  → HIGHLY PREDICTABLE transition patterns")
        elif entropy < 1.0:
            print("  → MODERATELY PREDICTABLE transition patterns")
        elif entropy < 1.5:
            print("  → SOMEWHAT RANDOM transition patterns")
        else:
            print("  → HIGHLY RANDOM transition patterns")
        
        # Find most common transitions
        print("\nMost Probable Transitions:")
        max_transitions = []
        for i in range(len(transition_matrix)):
            for j in range(len(transition_matrix[i])):
                if transition_matrix[i][j] > 0:
                    max_transitions.append((i, j, transition_matrix[i][j]))
        
        max_transitions.sort(key=lambda x: x[2], reverse=True)
        for i, (from_cluster, to_cluster, prob) in enumerate(max_transitions[:5]):
            print(f"  {i+1}. Cluster {from_cluster} → Cluster {to_cluster}: {prob:.3f}")
    
    except Exception as e:
        print(f"Error calculating entropy: {e}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Check for conditional insights
    try:
        with open('output/analysis/conditional_insights.json', 'r') as f:
            insights = json.load(f)
        
        if insights:
            print("Conditional insights were found!")
            for heading, data in insights.items():
                print(f"  {heading}: {len(data)} insights")
        else:
            print("No conditional insights were found.")
            print("This could be because:")
            print("  - Only 1 criminal in the dataset (need multiple for comparison)")
            print("  - No significant differences in transition patterns")
            print("  - Insufficient Type 2 data variation")
    
    except Exception as e:
        print(f"Could not load conditional insights: {e}")

if __name__ == "__main__":
    analyze_conditional_results()
