#!/usr/bin/env python3
"""
improved_clustering.py

Improved clustering methods for criminal archetypal analysis.
Includes:
- Multiple clustering algorithms
- Automatic optimal k selection
- Dimensionality reduction
- Better evaluation metrics
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
import umap
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from kneed import KneeLocator
import warnings
warnings.filterwarnings('ignore')


def reduce_dimensions(embeddings, method='pca', n_components=50, random_state=42):
    """
    Reduce dimensionality of embeddings before clustering.
    
    Args:
        embeddings: Input embeddings
        method: 'pca', 'umap', or 'truncated_svd'
        n_components: Number of components to keep
        random_state: Random seed
        
    Returns:
        Reduced embeddings
    """
    print(f"[INFO] Reducing dimensions using {method.upper()} to {n_components} components...")
    
    if method == 'pca':
        # Standardize before PCA
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced = reducer.fit_transform(embeddings_scaled)
        print(f"[INFO] PCA explained variance ratio: {reducer.explained_variance_ratio_.sum():.3f}")
        
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        reduced = reducer.fit_transform(embeddings)
        
    elif method == 'truncated_svd':
        # Good for sparse matrices (TF-IDF)
        reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
        reduced = reducer.fit_transform(embeddings)
        print(f"[INFO] SVD explained variance ratio: {reducer.explained_variance_ratio_.sum():.3f}")
        
    else:
        raise ValueError(f"Unknown method: {method}")
        
    return reduced


def find_optimal_k(embeddings, k_range=None, method='kmeans'):
    """
    Find optimal number of clusters using multiple metrics.
    
    Args:
        embeddings: Input embeddings
        k_range: Range of k values to test (default: 2 to 15)
        method: Clustering method to use
        
    Returns:
        Dictionary with optimal k values for each metric
    """
    if k_range is None:
        k_range = range(2, min(16, len(embeddings) // 10))
    
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    inertias = []
    
    print(f"[INFO] Finding optimal k for {method} clustering...")
    
    for k in k_range:
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = clusterer.fit_predict(embeddings)
            inertias.append(clusterer.inertia_)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=k)
            labels = clusterer.fit_predict(embeddings)
            # Hierarchical doesn't have inertia
            inertias.append(0)
        elif method == 'spectral':
            clusterer = SpectralClustering(n_clusters=k, random_state=42)
            labels = clusterer.fit_predict(embeddings)
            inertias.append(0)
        else:
            # Default to kmeans if unknown method
            clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = clusterer.fit_predict(embeddings)
            inertias.append(clusterer.inertia_)
        
        # Calculate metrics
        sil_score = silhouette_score(embeddings, labels)
        cal_score = calinski_harabasz_score(embeddings, labels)
        dav_score = davies_bouldin_score(embeddings, labels)
        
        silhouette_scores.append(sil_score)
        calinski_scores.append(cal_score)
        davies_scores.append(dav_score)
        
        print(f"  k={k}: Silhouette={sil_score:.3f}, Calinski={cal_score:.1f}, Davies-Bouldin={dav_score:.3f}")
    
    # Find optimal k for each metric
    optimal_k_silhouette = list(k_range)[np.argmax(silhouette_scores)]
    optimal_k_calinski = list(k_range)[np.argmax(calinski_scores)]
    optimal_k_davies = list(k_range)[np.argmin(davies_scores)]  # Lower is better
    
    # Use elbow method for k-means
    optimal_k_elbow = None
    if method == 'kmeans' and len(inertias) > 2:
        try:
            kneedle = KneeLocator(list(k_range), inertias, S=1.0, curve="convex", direction="decreasing")
            optimal_k_elbow = kneedle.elbow
        except:
            optimal_k_elbow = None
    
    results = {
        'silhouette': optimal_k_silhouette,
        'calinski': optimal_k_calinski,
        'davies_bouldin': optimal_k_davies,
        'elbow': optimal_k_elbow,
        'scores': {
            'k_values': list(k_range),
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'davies_scores': davies_scores,
            'inertias': inertias
        }
    }
    
    # Consensus optimal k (majority vote)
    k_votes = [optimal_k_silhouette, optimal_k_calinski, optimal_k_davies]
    if optimal_k_elbow:
        k_votes.append(optimal_k_elbow)
    
    from collections import Counter
    k_counts = Counter(k_votes)
    consensus_k = k_counts.most_common(1)[0][0]
    results['consensus'] = consensus_k
    
    print(f"\n[INFO] Optimal k recommendations:")
    print(f"  - Silhouette: {optimal_k_silhouette}")
    print(f"  - Calinski-Harabasz: {optimal_k_calinski}")
    print(f"  - Davies-Bouldin: {optimal_k_davies}")
    if optimal_k_elbow:
        print(f"  - Elbow method: {optimal_k_elbow}")
    print(f"  - CONSENSUS: {consensus_k}")
    
    return results


def improved_clustering(embeddings, n_clusters=None, method='kmeans', 
                       reduce_dims=True, dim_reduction='pca', n_components=50,
                       auto_select_k=True, random_state=42):
    """
    Improved clustering with multiple algorithms and automatic k selection.
    
    Args:
        embeddings: Input embeddings
        n_clusters: Number of clusters (if None, auto-select)
        method: 'kmeans', 'hierarchical', 'dbscan', 'spectral'
        reduce_dims: Whether to reduce dimensions first
        dim_reduction: Method for dimensionality reduction
        n_components: Number of components for dim reduction
        auto_select_k: Whether to automatically select optimal k
        random_state: Random seed
        
    Returns:
        labels, clusterer, metrics
    """
    # Optionally reduce dimensions
    if reduce_dims:
        embeddings_reduced = reduce_dimensions(embeddings, method=dim_reduction, 
                                              n_components=n_components, 
                                              random_state=random_state)
    else:
        embeddings_reduced = embeddings
    
    # Auto-select k if requested
    if auto_select_k and n_clusters is None and method != 'dbscan':
        optimal_k_results = find_optimal_k(embeddings_reduced, method=method)
        n_clusters = optimal_k_results['consensus']
        print(f"\n[INFO] Auto-selected k={n_clusters}")
    
    # Apply clustering
    print(f"\n[INFO] Applying {method} clustering with k={n_clusters}...")
    
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
        labels = clusterer.fit_predict(embeddings_reduced)
        
    elif method == 'hierarchical':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = clusterer.fit_predict(embeddings_reduced)
        
    elif method == 'dbscan':
        # DBSCAN doesn't need n_clusters, it finds them automatically
        # We need to tune eps parameter
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors_fit = neighbors.fit(embeddings_reduced)
        distances, indices = neighbors_fit.kneighbors(embeddings_reduced)
        distances = np.sort(distances[:, -1])
        
        # Find knee point for eps
        try:
            kneedle = KneeLocator(range(len(distances)), distances, S=1.0, curve="convex", direction="increasing")
            eps = distances[kneedle.elbow] if kneedle.elbow else np.percentile(distances, 90)
        except:
            eps = np.percentile(distances, 90)
            
        clusterer = DBSCAN(eps=eps, min_samples=5)
        labels = clusterer.fit_predict(embeddings_reduced)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"[INFO] DBSCAN found {n_clusters} clusters (eps={eps:.3f})")
        
    elif method == 'spectral':
        clusterer = SpectralClustering(n_clusters=n_clusters, random_state=random_state,
                                      affinity='nearest_neighbors', n_neighbors=10)
        labels = clusterer.fit_predict(embeddings_reduced)
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Calculate metrics
    if len(set(labels)) > 1:  # Need at least 2 clusters for metrics
        sil_score = silhouette_score(embeddings_reduced, labels)
        cal_score = calinski_harabasz_score(embeddings_reduced, labels)
        dav_score = davies_bouldin_score(embeddings_reduced, labels)
    else:
        sil_score = cal_score = dav_score = 0
    
    # Convert numpy types to Python types for JSON serialization
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(label): int(count) for label, count in zip(unique_labels, counts)}
    
    metrics = {
        'silhouette_score': float(sil_score),
        'calinski_harabasz_score': float(cal_score),
        'davies_bouldin_score': float(dav_score),
        'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
        'cluster_sizes': cluster_sizes
    }
    
    print(f"\n[INFO] Clustering results:")
    print(f"  - Number of clusters: {metrics['n_clusters']}")
    print(f"  - Silhouette score: {sil_score:.3f}")
    print(f"  - Calinski-Harabasz score: {cal_score:.1f}")
    print(f"  - Davies-Bouldin score: {dav_score:.3f}")
    print(f"  - Cluster sizes: {metrics['cluster_sizes']}")
    
    return labels, clusterer, metrics


def plot_clustering_results(embeddings, labels, output_path='clustering_results.png'):
    """
    Visualize clustering results using t-SNE.
    """
    print("[INFO] Creating t-SNE visualization...")
    
    # Use t-SNE for 2D visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.title('t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Visualization saved to {output_path}")


def hierarchical_clustering_dendrogram(embeddings, output_path='dendrogram.png', max_display=50):
    """
    Create a dendrogram for hierarchical clustering analysis.
    """
    print("[INFO] Creating hierarchical clustering dendrogram...")
    
    # Sample if too many points
    if len(embeddings) > max_display:
        sample_idx = np.random.choice(len(embeddings), max_display, replace=False)
        embeddings_sample = embeddings[sample_idx]
    else:
        embeddings_sample = embeddings
    
    # Create linkage matrix
    Z = linkage(embeddings_sample, 'ward')
    
    plt.figure(figsize=(15, 8))
    dendrogram(Z, no_labels=True)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Dendrogram saved to {output_path}")


if __name__ == "__main__":
    # Test the improved clustering
    print("Improved clustering module loaded successfully!")