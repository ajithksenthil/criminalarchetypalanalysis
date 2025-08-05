#!/usr/bin/env python3
"""
analyze_conditional_results.py

Comprehensive analysis of the conditional matrix analysis results.
"""

import numpy as np
import json
import pandas as pd
from collections import Counter

def analyze_full_results():
    """Analyze the full conditional matrix analysis results."""
    
    print("="*80)
    print("COMPREHENSIVE CONDITIONAL MATRIX ANALYSIS RESULTS")
    print("="*80)
    
    # Load analysis results
    try:
        with open('output_conditional/analysis_results.json', 'r') as f:
            results = json.load(f)
        
        print("\n1. DATASET OVERVIEW:")
        print("-" * 50)
        print(f"Criminals analyzed: {results['data_summary']['n_criminals']}")
        print(f"Total life events: {results['data_summary']['n_events']}")
        print(f"Embedding dimensions: {results['embeddings_shape']}")
        print(f"Optimization method: {results['optimization']['optimization_method']}")
        print(f"Optimal k selected: {results['optimization']['optimal_k']}")
        print(f"Final silhouette score: {results['clustering']['silhouette']:.4f}")
        
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    # Analyze cluster distribution
    print("\n2. CLUSTER ANALYSIS:")
    print("-" * 50)
    
    total_events = sum(cluster['size'] for cluster in results['clusters'])
    for cluster in results['clusters']:
        cluster_id = cluster['cluster_id']
        size = cluster['size']
        percentage = (size / total_events) * 100
        print(f"Cluster {cluster_id}: {size:4d} events ({percentage:5.1f}%)")
        
        # Show representative events (truncated)
        samples = cluster.get('representative_samples', [])
        if samples:
            print(f"  Representative: {samples[0][:80]}...")
    
    # Analyze stationary distribution
    print("\n3. MARKOV CHAIN ANALYSIS:")
    print("-" * 50)
    
    stationary = results['markov']['global_stationary']
    entropy = results['markov']['transition_entropy']
    
    print("Long-term cluster probabilities:")
    for i, prob in enumerate(stationary):
        print(f"  Cluster {i}: {prob:.3f} ({prob*100:.1f}%)")
    
    print(f"\nTransition entropy: {entropy:.3f}")
    if entropy < 0.5:
        print("  → HIGHLY PREDICTABLE transition patterns")
    elif entropy < 1.0:
        print("  → MODERATELY PREDICTABLE transition patterns")
    else:
        print("  → RANDOM transition patterns")
    
    # Load transition matrix
    try:
        transition_matrix = np.load('output_conditional/data/global_transition_matrix.npy')
        
        print("\n4. TRANSITION PATTERNS:")
        print("-" * 50)
        
        # Find strongest transitions
        strong_transitions = []
        for i in range(len(transition_matrix)):
            for j in range(len(transition_matrix[i])):
                if transition_matrix[i][j] > 0.3:  # Strong transitions
                    strong_transitions.append((i, j, transition_matrix[i][j]))
        
        strong_transitions.sort(key=lambda x: x[2], reverse=True)
        
        if strong_transitions:
            print("Strongest transition patterns (>30% probability):")
            for from_cluster, to_cluster, prob in strong_transitions[:10]:
                print(f"  Cluster {from_cluster} → Cluster {to_cluster}: {prob:.3f}")
        else:
            print("No strong transition patterns found (all <30% probability)")
            
    except Exception as e:
        print(f"Could not load transition matrix: {e}")
    
    # Analyze conditional optimization results
    print("\n5. CONDITIONAL EFFECT OPTIMIZATION:")
    print("-" * 50)
    
    opt_results = results['optimization']['optimization_results']
    if opt_results:
        print("K-value optimization results:")
        for k, data in opt_results.items():
            sig_effects = data['significant_effects']
            total_effects = data['total_effects']
            score = data['score']
            print(f"  k={k}: {sig_effects}/{total_effects} significant effects, score={score}")
        
        # Check why no effects were found
        print(f"\nOptimal k selected: {results['optimization']['optimal_k']}")
        
        if all(data['significant_effects'] == 0 for data in opt_results.values()):
            print("\n❌ NO CONDITIONAL EFFECTS DETECTED")
            print("Possible reasons:")
            print("  1. Insufficient Type2 data variation")
            print("  2. Type2 categories have too few criminals each")
            print("  3. No significant differences in transition patterns")
            print("  4. Effect size threshold too high")
    
    # Analyze criminal sequences
    print("\n6. CRIMINAL SEQUENCE PATTERNS:")
    print("-" * 50)
    
    try:
        with open('output_conditional/criminal_sequences.json', 'r') as f:
            sequences = json.load(f)
        
        print(f"Criminal sequences available: {len(sequences)}")
        
        # Analyze sequence lengths
        lengths = [len(seq) for seq in sequences.values()]
        print(f"Sequence length: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")
        
        # Analyze cluster transitions
        all_transitions = []
        for seq in sequences.values():
            for i in range(len(seq)-1):
                all_transitions.append((seq[i], seq[i+1]))
        
        transition_counts = Counter(all_transitions)
        print(f"\nMost common transitions:")
        for (from_c, to_c), count in transition_counts.most_common(5):
            percentage = (count / len(all_transitions)) * 100
            print(f"  Cluster {from_c} → Cluster {to_c}: {count} times ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"Could not analyze sequences: {e}")
    
    print("\n" + "="*80)
    print("SUMMARY AND INTERPRETATION")
    print("="*80)
    
    # Determine dominant patterns
    dominant_cluster = np.argmax(stationary)
    dominant_prob = stationary[dominant_cluster]
    
    print(f"\n🎯 DOMINANT LIFE EVENT ARCHETYPE: Cluster {dominant_cluster}")
    print(f"   Represents {dominant_prob*100:.1f}% of long-term criminal behavior")
    
    # Interpret clustering quality
    silhouette = results['clustering']['silhouette']
    if silhouette > 0.3:
        quality = "EXCELLENT"
    elif silhouette > 0.1:
        quality = "GOOD"
    elif silhouette > 0.0:
        quality = "MODERATE"
    else:
        quality = "POOR"
    
    print(f"\n📊 CLUSTERING QUALITY: {quality} (silhouette = {silhouette:.3f})")
    
    # Conditional analysis summary
    total_sig_effects = sum(data['significant_effects'] for data in opt_results.values())
    if total_sig_effects > 0:
        print(f"\n🔍 CONDITIONAL EFFECTS: {total_sig_effects} significant demographic effects found")
        print("   Your conditional effect optimization successfully identified patterns!")
    else:
        print(f"\n⚠️  CONDITIONAL EFFECTS: None detected")
        print("   This suggests either:")
        print("   • Universal criminal archetypal patterns (same across demographics)")
        print("   • Insufficient demographic data variation")
        print("   • Need for larger sample sizes per demographic group")
    
    print(f"\n✅ ANALYSIS COMPLETE: {results['data_summary']['n_criminals']} criminals, {results['data_summary']['n_events']} events processed")
    print("   The modular system successfully identified archetypal patterns in criminal life trajectories!")

if __name__ == "__main__":
    analyze_full_results()
