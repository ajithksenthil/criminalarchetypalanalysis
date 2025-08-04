#!/usr/bin/env python3
"""
validation_analysis.py

Research-grade validation and evaluation metrics for criminal archetypal analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from scipy.stats import spearmanr, pearsonr, chi2_contingency
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


def clustering_stability_analysis(embeddings, n_clusters, n_iterations=10, subsample_ratio=0.8):
    """
    Assess clustering stability through subsampling.
    
    Returns:
        dict: Stability metrics including ARI scores and cluster assignment consistency
    """
    from sklearn.metrics import adjusted_rand_score
    
    n_samples = len(embeddings)
    subsample_size = int(n_samples * subsample_ratio)
    
    # Store cluster assignments from each iteration
    all_assignments = []
    ari_scores = []
    
    # Reference clustering on full data
    km_ref = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_ref = km_ref.fit_predict(embeddings)
    
    for i in range(n_iterations):
        # Random subsample
        indices = np.random.choice(n_samples, subsample_size, replace=False)
        subsample = embeddings[indices]
        
        # Cluster the subsample
        km = KMeans(n_clusters=n_clusters, random_state=i, n_init=10)
        labels_subsample = km.fit_predict(subsample)
        
        # Project back to full dataset
        full_labels = np.zeros(n_samples, dtype=int) - 1
        full_labels[indices] = labels_subsample
        
        # Compare with reference (only for samples in subsample)
        ari = adjusted_rand_score(labels_ref[indices], labels_subsample)
        ari_scores.append(ari)
        
        all_assignments.append(full_labels)
    
    # Calculate consistency metrics
    mean_ari = np.mean(ari_scores)
    std_ari = np.std(ari_scores)
    
    return {
        'mean_ari': float(mean_ari),
        'std_ari': float(std_ari),
        'stability_score': float(mean_ari - std_ari),  # Higher is better
        'ari_scores': ari_scores
    }


def optimal_cluster_analysis(embeddings, max_clusters=15):
    """
    Determine optimal number of clusters using multiple metrics.
    """
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []
    inertias = []
    
    cluster_range = range(2, min(max_clusters + 1, len(embeddings) // 2))
    
    for n_clusters in cluster_range:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        
        silhouette_scores.append(silhouette_score(embeddings, labels))
        calinski_scores.append(calinski_harabasz_score(embeddings, labels))
        davies_bouldin_scores.append(davies_bouldin_score(embeddings, labels))
        inertias.append(km.inertia_)
    
    # Plot the metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(cluster_range, silhouette_scores, 'bo-')
    axes[0, 0].set_xlabel('Number of Clusters')
    axes[0, 0].set_ylabel('Silhouette Score')
    axes[0, 0].set_title('Silhouette Score vs Number of Clusters')
    
    axes[0, 1].plot(cluster_range, calinski_scores, 'go-')
    axes[0, 1].set_xlabel('Number of Clusters')
    axes[0, 1].set_ylabel('Calinski-Harabasz Score')
    axes[0, 1].set_title('Calinski-Harabasz Score vs Number of Clusters')
    
    axes[1, 0].plot(cluster_range, davies_bouldin_scores, 'ro-')
    axes[1, 0].set_xlabel('Number of Clusters')
    axes[1, 0].set_ylabel('Davies-Bouldin Score')
    axes[1, 0].set_title('Davies-Bouldin Score vs Number of Clusters (lower is better)')
    
    axes[1, 1].plot(cluster_range, inertias, 'mo-')
    axes[1, 1].set_xlabel('Number of Clusters')
    axes[1, 1].set_ylabel('Inertia')
    axes[1, 1].set_title('Elbow Method - Inertia vs Number of Clusters')
    
    plt.tight_layout()
    
    return {
        'cluster_range': list(cluster_range),
        'silhouette_scores': silhouette_scores,
        'calinski_scores': calinski_scores,
        'davies_bouldin_scores': davies_bouldin_scores,
        'inertias': inertias,
        'optimal_silhouette': int(cluster_range[np.argmax(silhouette_scores)]),
        'optimal_calinski': int(cluster_range[np.argmax(calinski_scores)]),
        'optimal_davies_bouldin': int(cluster_range[np.argmin(davies_bouldin_scores)])
    }, fig


def markov_chain_validation(transition_matrix, sequences, n_bootstrap=100):
    """
    Validate Markov chain properties and assess uncertainty.
    """
    n_states = transition_matrix.shape[0]
    
    # Check if matrix is valid stochastic matrix
    row_sums = transition_matrix.sum(axis=1)
    is_stochastic = np.allclose(row_sums[row_sums > 0], 1.0)
    
    # Check irreducibility (all states reachable)
    reachability = np.linalg.matrix_power(transition_matrix > 0, n_states)
    is_irreducible = np.all(reachability > 0)
    
    # Bootstrap confidence intervals for transition probabilities
    bootstrap_matrices = []
    for _ in range(n_bootstrap):
        # Resample sequences with replacement
        sample_indices = np.random.choice(len(sequences), len(sequences), replace=True)
        sample_sequences = [sequences[i] for i in sample_indices]
        
        # Build transition matrix from bootstrap sample
        boot_matrix = np.zeros((n_states, n_states))
        for seq in sample_sequences:
            if len(seq) < 2:
                continue
            for s1, s2 in zip(seq[:-1], seq[1:]):
                if s1 < n_states and s2 < n_states:
                    boot_matrix[s1, s2] += 1
        
        # Normalize
        row_sums = boot_matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            boot_matrix = np.divide(boot_matrix, row_sums, where=row_sums != 0)
            for i in range(n_states):
                if row_sums[i] == 0:
                    boot_matrix[i] = 1.0 / n_states
        
        bootstrap_matrices.append(boot_matrix)
    
    # Calculate confidence intervals
    bootstrap_matrices = np.array(bootstrap_matrices)
    ci_lower = np.percentile(bootstrap_matrices, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_matrices, 97.5, axis=0)
    ci_width = ci_upper - ci_lower
    
    return {
        'is_valid_stochastic': bool(is_stochastic),
        'is_irreducible': bool(is_irreducible),
        'mean_ci_width': float(np.mean(ci_width[transition_matrix > 0])),
        'max_ci_width': float(np.max(ci_width)),
        'transition_matrix_lower': ci_lower.tolist(),
        'transition_matrix_upper': ci_upper.tolist()
    }


def cross_validation_analysis(embeddings, labels, criminal_ids, n_folds=5):
    """
    Perform cross-validation to assess generalization of clustering.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    unique_criminals = list(set(criminal_ids))
    n_criminals = len(unique_criminals)
    
    if n_criminals < n_folds:
        print(f"[WARNING] Not enough criminals ({n_criminals}) for {n_folds}-fold CV")
        return None
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    ari_scores = []
    nmi_scores = []
    
    for train_idx, test_idx in kf.split(unique_criminals):
        # Get train and test criminals
        train_criminals = [unique_criminals[i] for i in train_idx]
        test_criminals = [unique_criminals[i] for i in test_idx]
        
        # Get indices for train and test events
        train_event_idx = [i for i, cid in enumerate(criminal_ids) if cid in train_criminals]
        test_event_idx = [i for i, cid in enumerate(criminal_ids) if cid in test_criminals]
        
        if len(train_event_idx) < 10 or len(test_event_idx) < 10:
            continue
        
        # Cluster on training data
        train_embeddings = embeddings[train_event_idx]
        km = KMeans(n_clusters=len(np.unique(labels)), random_state=42, n_init=10)
        km.fit(train_embeddings)
        
        # Predict on test data
        test_embeddings = embeddings[test_event_idx]
        test_pred = km.predict(test_embeddings)
        test_true = labels[test_event_idx]
        
        # Calculate metrics
        ari = adjusted_rand_score(test_true, test_pred)
        nmi = normalized_mutual_info_score(test_true, test_pred)
        
        ari_scores.append(ari)
        nmi_scores.append(nmi)
    
    return {
        'mean_ari': float(np.mean(ari_scores)),
        'std_ari': float(np.std(ari_scores)),
        'mean_nmi': float(np.mean(nmi_scores)),
        'std_nmi': float(np.std(nmi_scores)),
        'n_folds_completed': len(ari_scores)
    }


def statistical_significance_tests(conditional_matrices, global_matrix, criminal_groups):
    """
    Perform rigorous statistical tests for conditional analysis.
    """
    from scipy.stats import anderson_ksamp, kruskal
    
    results = {}
    
    # Anderson-Darling k-sample test for distribution differences
    # Flatten matrices and compare distributions
    matrix_samples = []
    group_labels = []
    
    for group_name, matrix in conditional_matrices.items():
        flat = matrix.flatten()
        matrix_samples.extend(flat)
        group_labels.extend([group_name] * len(flat))
    
    # Add global matrix as baseline
    flat_global = global_matrix.flatten()
    matrix_samples.extend(flat_global)
    group_labels.extend(['global'] * len(flat_global))
    
    # Group samples by label
    unique_groups = list(set(group_labels))
    grouped_samples = []
    for group in unique_groups:
        group_data = [matrix_samples[i] for i, g in enumerate(group_labels) if g == group]
        grouped_samples.append(group_data)
    
    # Perform Anderson-Darling test
    if len(grouped_samples) >= 2:
        ad_result = anderson_ksamp(grouped_samples)
        results['anderson_darling'] = {
            'statistic': float(ad_result.statistic),
            'critical_values': ad_result.critical_values.tolist(),
            'significance_level': float(ad_result.significance_level)
        }
    
    # Kruskal-Wallis test
    if len(grouped_samples) >= 2:
        kw_result = kruskal(*grouped_samples)
        results['kruskal_wallis'] = {
            'statistic': float(kw_result.statistic),
            'pvalue': float(kw_result.pvalue)
        }
    
    # Pairwise comparisons with Bonferroni correction
    n_comparisons = len(conditional_matrices)
    bonferroni_alpha = 0.05 / n_comparisons if n_comparisons > 0 else 0.05
    
    results['bonferroni_alpha'] = bonferroni_alpha
    results['pairwise_tests'] = {}
    
    for group_name, matrix in conditional_matrices.items():
        # Compare with global
        flat_group = matrix.flatten()
        _, pvalue = kruskal(flat_group, flat_global)
        
        results['pairwise_tests'][f'{group_name}_vs_global'] = {
            'pvalue': float(pvalue),
            'significant_bonferroni': pvalue < bonferroni_alpha,
            'effect_size': float(np.mean(np.abs(matrix - global_matrix)))  # Simple effect size
        }
    
    return results


def generate_validation_report(output_dir, embeddings, labels, sequences, criminal_ids, 
                             transition_matrix, conditional_results=None):
    """
    Generate comprehensive validation report with all metrics.
    """
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'data_summary': {
            'n_events': len(embeddings),
            'n_criminals': len(set(criminal_ids)),
            'n_clusters': len(np.unique(labels)),
            'events_per_criminal': len(embeddings) / len(set(criminal_ids))
        }
    }
    
    # Clustering validation
    print("[INFO] Running clustering stability analysis...")
    stability = clustering_stability_analysis(embeddings, len(np.unique(labels)))
    report['clustering_stability'] = stability
    
    # Optimal clusters analysis
    print("[INFO] Analyzing optimal number of clusters...")
    optimal_clusters, fig = optimal_cluster_analysis(embeddings, max_clusters=15)
    report['optimal_clusters'] = optimal_clusters
    fig.savefig(os.path.join(output_dir, 'optimal_clusters_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Cross-validation
    print("[INFO] Running cross-validation analysis...")
    cv_results = cross_validation_analysis(embeddings, labels, criminal_ids)
    if cv_results:
        report['cross_validation'] = cv_results
    
    # Markov chain validation
    print("[INFO] Validating Markov chain properties...")
    sequences_list = list(sequences.values())
    markov_validation = markov_chain_validation(transition_matrix, sequences_list)
    report['markov_validation'] = markov_validation
    
    # Statistical tests for conditional analysis
    if conditional_results:
        print("[INFO] Running statistical significance tests...")
        conditional_matrices = {}
        for condition_name, data in conditional_results.items():
            if 'transition_matrix' in data:
                conditional_matrices[condition_name] = data['transition_matrix']
        
        if conditional_matrices:
            criminal_groups = {}  # Would need to extract from data
            significance_tests = statistical_significance_tests(
                conditional_matrices, transition_matrix, criminal_groups
            )
            report['statistical_significance'] = significance_tests
    
    # Save report
    report_path = os.path.join(output_dir, 'validation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"[INFO] Validation report saved to {report_path}")
    
    # Generate summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Data: {report['data_summary']['n_events']} events from {report['data_summary']['n_criminals']} criminals")
    print(f"Clustering Stability (ARI): {stability['mean_ari']:.3f} ± {stability['std_ari']:.3f}")
    print(f"Optimal clusters: {optimal_clusters['optimal_silhouette']} (by Silhouette)")
    if cv_results:
        print(f"Cross-validation ARI: {cv_results['mean_ari']:.3f} ± {cv_results['std_ari']:.3f}")
    print(f"Markov chain valid: {markov_validation['is_valid_stochastic']}")
    print(f"Markov chain irreducible: {markov_validation['is_irreducible']}")
    print("="*60)
    
    return report


if __name__ == "__main__":
    print("[INFO] Validation analysis module loaded. Use generate_validation_report() to run full validation.")