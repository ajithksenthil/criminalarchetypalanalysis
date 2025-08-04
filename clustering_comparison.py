#!/usr/bin/env python3
"""
clustering_comparison.py

Quick comparison of original vs improved clustering.
"""

import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from improved_clustering import improved_clustering
import matplotlib.pyplot as plt

def generate_demo_data():
    """Generate some demo criminal event data."""
    events = [
        # Childhood trauma cluster
        "physically abused by father",
        "mother died when young", 
        "parents divorced early",
        "abandoned by parents",
        "raised by grandparents",
        "witnessed domestic violence",
        
        # School problems cluster
        "dropped out of high school",
        "expelled from school",
        "poor academic performance", 
        "truancy issues",
        "learning disabilities",
        "held back a grade",
        
        # Substance abuse cluster
        "began drinking at 12",
        "marijuana use in teens",
        "cocaine addiction",
        "alcohol dependency", 
        "drug dealing arrests",
        "DUI convictions",
        
        # Violence escalation cluster
        "tortured animals as child",
        "first assault at 15",
        "armed robbery conviction",
        "domestic violence arrests",
        "bar fight incidents", 
        "weapon possession charges",
        
        # Mental health cluster
        "diagnosed with schizophrenia",
        "suicide attempts", 
        "psychiatric hospitalization",
        "prescribed antipsychotics",
        "paranoid delusions",
        "hearing voices",
    ] * 20  # Repeat to create more data
    
    # Add some noise/variations
    import random
    random.shuffle(events)
    
    return events

def main():
    print("="*60)
    print("CLUSTERING COMPARISON: Original vs Improved")
    print("="*60)
    
    # Generate demo data
    events = generate_demo_data()
    print(f"\n[INFO] Generated {len(events)} demo events")
    
    # Create TF-IDF embeddings
    print("[INFO] Creating TF-IDF embeddings...")
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    embeddings = vectorizer.fit_transform(events)
    print(f"[INFO] Embedding shape: {embeddings.shape}")
    
    # Original clustering
    print("\n" + "-"*60)
    print("ORIGINAL CLUSTERING (k=5, no preprocessing)")
    print("-"*60)
    
    kmeans_orig = KMeans(n_clusters=5, random_state=42)
    labels_orig = kmeans_orig.fit_predict(embeddings)
    sil_orig = silhouette_score(embeddings, labels_orig)
    
    print(f"✗ Fixed k=5 (no optimization)")
    print(f"✗ No dimensionality reduction")
    print(f"✗ Basic k-means only")
    print(f"\nResults:")
    print(f"  - Silhouette score: {sil_orig:.3f}")
    
    # Improved clustering
    print("\n" + "-"*60)
    print("IMPROVED CLUSTERING (auto-k, dimensionality reduction)")
    print("-"*60)
    
    print("✓ Automatic k selection using multiple metrics")
    print("✓ Dimensionality reduction with SVD")
    print("✓ Optimized parameters")
    
    labels_improved, clusterer, metrics = improved_clustering(
        embeddings,
        n_clusters=None,  # Auto-select
        method='kmeans',
        reduce_dims=True,
        dim_reduction='truncated_svd',
        n_components=20,
        auto_select_k=True
    )
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    improvement = metrics['silhouette_score'] - sil_orig
    pct_improvement = (improvement / abs(sil_orig)) * 100
    
    print(f"\nOriginal method:")
    print(f"  - k = 5 (fixed)")
    print(f"  - Silhouette = {sil_orig:.3f}")
    
    print(f"\nImproved method:")
    print(f"  - k = {metrics['n_clusters']} (auto-selected)")
    print(f"  - Silhouette = {metrics['silhouette_score']:.3f}")
    
    print(f"\nImprovement: +{improvement:.3f} ({pct_improvement:+.1f}%)")
    
    if improvement > 0:
        print("\n✓ The improved clustering provides better separation!")
    
    # Save results
    results = {
        'original': {
            'method': 'Basic K-means',
            'k': 5,
            'silhouette': float(sil_orig),
            'preprocessing': 'None'
        },
        'improved': {
            'method': 'Optimized K-means',
            'k': metrics['n_clusters'],
            'silhouette': metrics['silhouette_score'],
            'preprocessing': 'SVD dimensionality reduction',
            'k_selection': 'Automatic (consensus of 4 metrics)'
        },
        'improvement': {
            'absolute': float(improvement),
            'percentage': float(pct_improvement)
        }
    }
    
    with open('clustering_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: clustering_comparison_results.json")
    
    # Create simple bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    methods = ['Original\n(k=5)', f'Improved\n(k={metrics["n_clusters"]})']
    scores = [sil_orig, metrics['silhouette_score']]
    colors = ['#ff6b6b', '#4ecdc4']
    
    bars = ax.bar(methods, scores, color=colors, alpha=0.8)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Clustering Quality Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(scores) * 1.2)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}',
                ha='center', va='bottom', fontsize=11)
    
    # Add improvement annotation
    if improvement > 0:
        ax.annotate(f'+{pct_improvement:.1f}%', 
                    xy=(1, metrics['silhouette_score']), 
                    xytext=(1.2, metrics['silhouette_score']),
                    fontsize=12, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('clustering_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: clustering_comparison.png")

if __name__ == "__main__":
    main()