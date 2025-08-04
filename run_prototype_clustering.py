#!/usr/bin/env python3
"""
run_prototype_clustering.py

Runs clustering using lexically imputed prototype representations of events.
Maintains original event descriptions for interpretability while clustering
on the semantic prototypes.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import required modules
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from improved_lexical_imputation import ImprovedLexicalImputation
from data_loading import load_matched_criminal_data

def create_prototype_embeddings(events, imputer, use_tfidf=True, num_variants=5):
    """
    Create prototype embeddings for events using lexical imputation.
    
    Returns:
        prototype_embeddings: Array of prototype embeddings
        event_mapping: Dict mapping index to original event text
        all_variations: List of all variations for debugging
    """
    print(f"\n[INFO] Creating prototype representations for {len(events)} events...")
    
    prototype_embeddings = []
    event_mapping = {}
    all_variations = []
    
    # If using TF-IDF, we need to collect all text first
    if use_tfidf:
        all_texts = []
        variation_groups = []
        
        for idx, event in enumerate(events):
            if idx % 100 == 0:
                print(f"  Processing event {idx}/{len(events)}...")
            
            # Generate variations with name standardization
            variations = imputer.generate_improved_variations(event, num_variants=num_variants)
            
            # Include the standardized original
            standardized_original, _ = imputer.standardize_names(event)
            all_versions = variations + [standardized_original]
            
            variation_groups.append(all_versions)
            all_texts.extend(all_versions)
            
            # Store original event for interpretability
            event_mapping[idx] = event
            all_variations.append({
                'original': event,
                'standardized': standardized_original,
                'variations': variations
            })
        
        # Create TF-IDF vectorizer on all text
        print("\n[INFO] Creating TF-IDF embeddings for all variations...")
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        all_embeddings = vectorizer.fit_transform(all_texts).toarray()
        
        # Create prototypes by averaging embeddings for each event's variations
        print("\n[INFO] Creating prototype embeddings...")
        start_idx = 0
        for group in variation_groups:
            end_idx = start_idx + len(group)
            group_embeddings = all_embeddings[start_idx:end_idx]
            prototype = np.mean(group_embeddings, axis=0)
            prototype_embeddings.append(prototype)
            start_idx = end_idx
    
    else:
        # Use sentence embeddings (requires sentence-transformers)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        for idx, event in enumerate(events):
            if idx % 100 == 0:
                print(f"  Processing event {idx}/{len(events)}...")
            
            # Generate variations
            variations = imputer.generate_improved_variations(event, num_variants=num_variants)
            standardized_original, _ = imputer.standardize_names(event)
            all_versions = variations + [standardized_original]
            
            # Get embeddings and create prototype
            embeddings = model.encode(all_versions)
            prototype = np.mean(embeddings, axis=0)
            prototype_embeddings.append(prototype)
            
            event_mapping[idx] = event
            all_variations.append({
                'original': event,
                'standardized': standardized_original,
                'variations': variations
            })
    
    prototype_embeddings = np.array(prototype_embeddings)
    print(f"\n[INFO] Created {len(prototype_embeddings)} prototype embeddings")
    print(f"[INFO] Embedding shape: {prototype_embeddings.shape}")
    
    return prototype_embeddings, event_mapping, all_variations

def cluster_prototypes(prototype_embeddings, n_clusters=None, auto_k=False):
    """Cluster the prototype embeddings."""
    
    if auto_k and n_clusters is None:
        print("\n[INFO] Finding optimal number of clusters...")
        silhouette_scores = []
        k_range = range(2, min(21, len(prototype_embeddings)//10))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(prototype_embeddings)
            score = silhouette_score(prototype_embeddings, labels)
            silhouette_scores.append(score)
            print(f"  k={k}: silhouette score = {score:.4f}")
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"\n[INFO] Optimal k = {optimal_k}")
        n_clusters = optimal_k
    elif n_clusters is None:
        n_clusters = 12  # Default based on previous analysis
    
    print(f"\n[INFO] Clustering with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(prototype_embeddings)
    
    # Calculate metrics
    sil_score = silhouette_score(prototype_embeddings, cluster_labels)
    
    print(f"[INFO] Clustering complete!")
    print(f"[INFO] Silhouette score: {sil_score:.4f}")
    
    return cluster_labels, kmeans, n_clusters

def save_results(output_dir, cluster_labels, event_mapping, all_variations, 
                 prototype_embeddings, n_clusters, criminal_events):
    """Save clustering results with original event descriptions."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create cluster info with original events
    cluster_info = {i: [] for i in range(n_clusters)}
    
    for idx, label in enumerate(cluster_labels):
        cluster_info[label].append({
            'index': idx,
            'original_event': event_mapping[idx],
            'standardized': all_variations[idx]['standardized'],
            'variations': all_variations[idx]['variations']
        })
    
    # Save detailed cluster information
    cluster_report = {
        'n_clusters': n_clusters,
        'n_events': len(cluster_labels),
        'silhouette_score': float(silhouette_score(prototype_embeddings, cluster_labels)),
        'cluster_info': {}
    }
    
    for cluster_id, events in cluster_info.items():
        cluster_report['cluster_info'][str(cluster_id)] = {
            'size': len(events),
            'sample_events': [
                {
                    'original': e['original_event'],
                    'standardized': e['standardized']
                } for e in events[:5]  # Show first 5 examples
            ]
        }
    
    # Save cluster report
    with open(os.path.join(output_dir, 'prototype_cluster_report.json'), 'w') as f:
        json.dump(cluster_report, f, indent=2)
    
    # Save full mapping for analysis
    full_mapping = {
        'event_to_cluster': {event_mapping[i]: int(cluster_labels[i]) 
                           for i in range(len(cluster_labels))},
        'event_to_variations': {v['original']: v['variations'] 
                              for v in all_variations}
    }
    
    with open(os.path.join(output_dir, 'event_cluster_mapping.json'), 'w') as f:
        json.dump(full_mapping, f, indent=2)
    
    # Create criminal sequences using cluster labels
    criminal_sequences = {}
    event_idx = 0
    for criminal_id, events in criminal_events.items():
        sequence = []
        for _ in events:
            sequence.append(int(cluster_labels[event_idx]))
            event_idx += 1
        criminal_sequences[criminal_id] = sequence
    
    with open(os.path.join(output_dir, 'criminal_sequences.json'), 'w') as f:
        json.dump(criminal_sequences, f, indent=2)
    
    # Save embeddings and labels
    np.save(os.path.join(output_dir, 'prototype_embeddings.npy'), prototype_embeddings)
    np.save(os.path.join(output_dir, 'cluster_labels.npy'), cluster_labels)
    
    print(f"\n[INFO] Results saved to {output_dir}/")
    print(f"[INFO] Key files:")
    print(f"  - prototype_cluster_report.json: Cluster summary with example events")
    print(f"  - event_cluster_mapping.json: Full mapping of events to clusters")
    print(f"  - criminal_sequences.json: Cluster sequences for each criminal")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run clustering on prototype representations')
    parser.add_argument('--type1_dir', default='type1csvs', help='Directory with Type1 CSV files')
    parser.add_argument('--type2_dir', default='type2csvs', help='Directory with Type2 CSV files')
    parser.add_argument('--output_dir', help='Output directory')
    parser.add_argument('--n_clusters', type=int, help='Number of clusters')
    parser.add_argument('--auto_k', action='store_true', help='Automatically select optimal k')
    parser.add_argument('--use_tfidf', action='store_true', default=True, 
                      help='Use TF-IDF embeddings (default: True)')
    parser.add_argument('--num_variants', type=int, default=5, 
                      help='Number of lexical variations to generate')
    
    args = parser.parse_args()
    
    # Set output directory
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results_prototype_clustering_{timestamp}"
    
    print("="*60)
    print("PROTOTYPE-BASED CLUSTERING ANALYSIS")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Number of variations: {args.num_variants}")
    print(f"Using TF-IDF: {args.use_tfidf}")
    print(f"Auto-select k: {args.auto_k}")
    
    # Load data
    print("\n[INFO] Loading criminal event data...")
    try:
        # Try with match_only parameter
        type1_data, type2_data = load_matched_criminal_data(
            args.type1_dir, args.type2_dir, match_only=True
        )
    except TypeError:
        # Fallback without match_only
        type1_data, type2_data = load_matched_criminal_data(
            args.type1_dir, args.type2_dir
        )
    
    # Flatten events while keeping track of criminals
    all_events = []
    criminal_events = {}
    
    # Type1_data contains dicts with 'events' key for each criminal
    for criminal_id, criminal_data in type1_data.items():
        if 'events' in criminal_data:
            events = criminal_data['events']
            criminal_events[criminal_id] = events
            all_events.extend(events)
    
    print(f"[INFO] Loaded {len(all_events)} events from {len(type1_data)} criminals")
    
    # Initialize imputer
    imputer = ImprovedLexicalImputation()
    
    # Create prototype embeddings
    prototype_embeddings, event_mapping, all_variations = create_prototype_embeddings(
        all_events, imputer, use_tfidf=args.use_tfidf, num_variants=args.num_variants
    )
    
    # Cluster prototypes
    cluster_labels, kmeans, n_clusters = cluster_prototypes(
        prototype_embeddings, n_clusters=args.n_clusters, auto_k=args.auto_k
    )
    
    # Save results
    save_results(
        args.output_dir, cluster_labels, event_mapping, all_variations,
        prototype_embeddings, n_clusters, criminal_events
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review prototype_cluster_report.json to see cluster contents")
    print("2. Run conditional analysis on the criminal sequences")
    print("3. Create visualizations of the improved clustering")

if __name__ == "__main__":
    main()