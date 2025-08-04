#!/usr/bin/env python3
"""
demo_improved_clustering.py

Demonstrate the improved clustering without running the full analysis.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from improved_clustering import improved_clustering, plot_clustering_results
from data_cleaning import clean_type2_data
from data_matching import match_criminal_data

# Ensure we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_and_process_data():
    """Load and process data for clustering demo."""
    print("[INFO] Loading data...")
    
    # Load matched data
    matches = match_criminal_data('type1csvs', 'type2csvs')
    matched_files = [(m['type1_file'], m['type2_file']) for m in matches['matches']]
    
    print(f"[INFO] Found {len(matched_files)} matched criminals")
    
    # Load and clean Type2 data
    all_events = []
    
    for type1_file, type2_file in matched_files[:30]:  # Use first 30 for demo
        type2_path = os.path.join('type2csvs', type2_file)
        
        try:
            df = pd.read_csv(type2_path)
            # Clean the data
            df_clean = clean_type2_data(df)
            
            # Extract events
            if 'Value' in df_clean.columns:
                events = df_clean['Value'].dropna().tolist()
                # Convert None to Unknown
                events = ['Unknown' if e is None else str(e) for e in events]
                all_events.extend(events)
                
        except Exception as e:
            print(f"[WARNING] Error processing {type2_file}: {e}")
            continue
    
    print(f"[INFO] Collected {len(all_events)} total events")
    return all_events

def create_embeddings(events):
    """Create TF-IDF embeddings for events."""
    print("[INFO] Creating TF-IDF embeddings...")
    
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    embeddings = vectorizer.fit_transform(events)
    
    print(f"[INFO] Created embeddings with shape: {embeddings.shape}")
    return embeddings

def main():
    """Run the clustering demo."""
    print("="*60)
    print("IMPROVED CLUSTERING DEMO")
    print("="*60)
    
    # Load data
    events = load_and_process_data()
    
    if not events:
        print("[ERROR] No events loaded")
        return
    
    # Create embeddings
    embeddings = create_embeddings(events)
    
    # Original clustering (k=5, no improvements)
    print("\n" + "="*60)
    print("ORIGINAL CLUSTERING (k=5, no improvements)")
    print("="*60)
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    kmeans_orig = KMeans(n_clusters=5, random_state=42)
    labels_orig = kmeans_orig.fit_predict(embeddings)
    sil_orig = silhouette_score(embeddings, labels_orig)
    
    print(f"[ORIGINAL] Number of clusters: 5")
    print(f"[ORIGINAL] Silhouette score: {sil_orig:.3f}")
    
    # Improved clustering
    print("\n" + "="*60)
    print("IMPROVED CLUSTERING (auto-k, dimensionality reduction)")
    print("="*60)
    
    labels_improved, clusterer, metrics = improved_clustering(
        embeddings,
        n_clusters=None,  # Auto-select
        method='kmeans',
        reduce_dims=True,
        dim_reduction='truncated_svd',
        n_components=50,
        auto_select_k=True
    )
    
    # Create visualization
    print("\n[INFO] Creating visualization...")
    plot_clustering_results(embeddings.toarray(), labels_improved, 'clustering_demo.png')
    
    # Save results
    results = {
        'original': {
            'n_clusters': 5,
            'silhouette_score': float(sil_orig)
        },
        'improved': metrics,
        'improvement': {
            'silhouette_increase': metrics['silhouette_score'] - sil_orig,
            'silhouette_increase_pct': ((metrics['silhouette_score'] - sil_orig) / abs(sil_orig)) * 100
        }
    }
    
    with open('clustering_demo_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Original clustering: k=5, silhouette={sil_orig:.3f}")
    print(f"Improved clustering: k={metrics['n_clusters']}, silhouette={metrics['silhouette_score']:.3f}")
    print(f"Improvement: {results['improvement']['silhouette_increase']:.3f} ({results['improvement']['silhouette_increase_pct']:.1f}%)")
    print("\nResults saved to:")
    print("  - clustering_demo_results.json")
    print("  - clustering_demo.png")

if __name__ == "__main__":
    main()