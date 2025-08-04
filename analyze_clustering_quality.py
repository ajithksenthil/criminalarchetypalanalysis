#!/usr/bin/env python3
"""
analyze_clustering_quality.py

Analyze why clustering quality is poor and suggest improvements.
"""

import json
import numpy as np
import os
from collections import Counter

def analyze_results(results_dir):
    """Analyze clustering results and identify issues."""
    
    print(f"\nAnalyzing: {results_dir}")
    print("="*60)
    
    # Load cluster metrics
    metrics_file = os.path.join(results_dir, 'cluster_metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        print(f"Clustering Metrics:")
        sil_score = metrics.get('silhouette_score', None)
        db_score = metrics.get('davies_bouldin_score', None)
        
        if sil_score is not None:
            print(f"  - Silhouette Score: {sil_score:.4f}")
        else:
            print(f"  - Silhouette Score: N/A")
            
        if db_score is not None:
            print(f"  - Davies-Bouldin Score: {db_score:.4f}")
        else:
            print(f"  - Davies-Bouldin Score: N/A")
        print(f"  - Number of Clusters: {metrics.get('n_clusters', 'N/A')}")
        
        # Check cluster balance
        sizes = metrics.get('cluster_sizes', {})
        if sizes:
            values = list(sizes.values())
            print(f"\nCluster Balance:")
            print(f"  - Largest cluster: {max(values)} events")
            print(f"  - Smallest cluster: {min(values)} events")
            print(f"  - Ratio: {max(values)/min(values):.2f}x")
    
    # Load conditional insights
    insights_file = os.path.join(results_dir, 'conditional_insights.json')
    if os.path.exists(insights_file):
        with open(insights_file, 'r') as f:
            insights = json.load(f)
        
        # Analyze p-values and sample sizes
        sig_count = 0
        small_n_sig = 0
        p_values = []
        n_criminals = []
        
        for key, value in insights.items():
            if value.get('significant', False):
                sig_count += 1
                n = value['n_criminals']
                p_val = value['statistics']['ks_pvalue']
                
                p_values.append(p_val)
                n_criminals.append(n)
                
                if n < 10:
                    small_n_sig += 1
        
        print(f"\nConditional Analysis Issues:")
        print(f"  - Total significant patterns: {sig_count}")
        print(f"  - Patterns with n<10: {small_n_sig} ({small_n_sig/sig_count*100:.1f}%)")
        print(f"  - Median sample size: {np.median(n_criminals):.0f}")
        print(f"  - Min p-value: {min(p_values):.2e}")
        
        # Check for multiple testing
        bonferroni_threshold = 0.05 / len(insights)
        print(f"\nMultiple Testing:")
        print(f"  - Tests performed: {len(insights)}")
        print(f"  - Bonferroni threshold: {bonferroni_threshold:.2e}")
        print(f"  - Would survive correction: {sum(1 for p in p_values if p < bonferroni_threshold)}")

def main():
    """Analyze all available results."""
    
    print("CLUSTERING QUALITY ANALYSIS")
    print("="*60)
    
    # Find all results directories
    dirs = [d for d in os.listdir('.') if d.startswith(('results_', 'test_run', 'output_'))]
    
    for d in sorted(dirs):
        if os.path.isdir(d):
            analyze_results(d)
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    print("1. Increase minimum sample size to n≥10 for conditional analysis")
    print("2. Apply Bonferroni or FDR correction for multiple testing")
    print("3. Use better embeddings (e.g., sentence-transformers with semantic similarity)")
    print("4. Consider hierarchical clustering to capture nested patterns")
    print("5. Aggregate similar events before clustering (e.g., group all abuse events)")
    print("6. Remove or impute 'Unknown' values more intelligently")
    print("7. Use cross-validation to assess clustering stability")

if __name__ == "__main__":
    main()