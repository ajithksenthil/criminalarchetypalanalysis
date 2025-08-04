#!/usr/bin/env python3
"""
improved_clustering.py

Advanced clustering methods with automatic k selection and dimensionality reduction.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from ..core.config import RANDOM_SEED

class ImprovedClusterer:
    """Advanced clustering with multiple algorithms and automatic optimization."""
    
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.scaler = StandardScaler()
    
    def improved_clustering(self, embeddings: np.ndarray, n_clusters: Optional[int] = None,
                          method: str = 'kmeans', reduce_dims: bool = True,
                          dim_reduction: str = 'pca', n_components: int = 50,
                          auto_select_k: bool = False) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
        """
        Perform improved clustering with optional dimensionality reduction and auto k selection.
        
        Args:
            embeddings: Input embeddings
            n_clusters: Number of clusters (ignored if auto_select_k=True)
            method: Clustering method ('kmeans', 'hierarchical', 'dbscan')
            reduce_dims: Whether to apply dimensionality reduction
            dim_reduction: Dimensionality reduction method ('pca', 'truncated_svd', 'tsne')
            n_components: Number of components for dimensionality reduction
            auto_select_k: Whether to automatically select optimal k
            
        Returns:
            Tuple of (labels, clusterer, metrics)
        """
        print(f"[INFO] Starting improved clustering with method: {method}")
        
        # Prepare data
        X = embeddings.copy()
        
        # Standardize features
        X = self.scaler.fit_transform(X)
        
        # Dimensionality reduction
        if reduce_dims and X.shape[1] > n_components:
            X = self._apply_dimensionality_reduction(X, dim_reduction, n_components)
            print(f"[INFO] Applied {dim_reduction} dimensionality reduction: {embeddings.shape} -> {X.shape}")
        
        # Automatic k selection
        if auto_select_k:
            optimal_k = self._find_optimal_k(X, method)
            n_clusters = optimal_k
            print(f"[INFO] Auto-selected k = {optimal_k}")
        
        # Apply clustering
        labels, clusterer = self._apply_clustering(X, method, n_clusters)
        
        # Evaluate clustering
        metrics = self._evaluate_clustering(X, labels, n_clusters)
        metrics['method'] = method
        metrics['dimensionality_reduction'] = dim_reduction if reduce_dims else None
        metrics['auto_selected_k'] = auto_select_k
        
        print(f"[INFO] Clustering complete. Silhouette score: {metrics.get('silhouette', 'N/A'):.3f}")
        
        return labels, clusterer, metrics
    
    def _apply_dimensionality_reduction(self, X: np.ndarray, method: str, 
                                      n_components: int) -> np.ndarray:
        """Apply dimensionality reduction."""
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=self.random_state)
        elif method == 'truncated_svd':
            reducer = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        elif method == 'tsne':
            # t-SNE is typically used for visualization, not preprocessing
            reducer = TSNE(n_components=min(n_components, 3), random_state=self.random_state)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        return reducer.fit_transform(X)
    
    def _apply_clustering(self, X: np.ndarray, method: str, 
                         n_clusters: int) -> Tuple[np.ndarray, Any]:
        """Apply the specified clustering method."""
        if method == 'kmeans':
            clusterer = KMeans(
                n_clusters=n_clusters, 
                random_state=self.random_state, 
                n_init=10
            )
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
        elif method == 'dbscan':
            # For DBSCAN, we don't specify n_clusters
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        labels = clusterer.fit_predict(X)
        return labels, clusterer
    
    def _find_optimal_k(self, X: np.ndarray, method: str, 
                       k_range: Optional[range] = None) -> int:
        """Find optimal number of clusters."""
        if k_range is None:
            k_range = range(2, min(16, len(X) // 10))
        
        if method == 'dbscan':
            # DBSCAN doesn't use k, return a reasonable default
            return -1
        
        best_k = 2
        best_score = -1
        
        for k in k_range:
            try:
                if method == 'kmeans':
                    clusterer = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                elif method == 'hierarchical':
                    clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')
                else:
                    continue
                
                labels = clusterer.fit_predict(X)
                
                if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            except:
                continue
        
        return best_k
    
    def _evaluate_clustering(self, X: np.ndarray, labels: np.ndarray, 
                           n_clusters: int) -> Dict[str, Any]:
        """Evaluate clustering quality."""
        unique_labels = set(labels)
        n_clusters_actual = len(unique_labels)
        
        metrics = {
            'n_clusters': n_clusters,
            'n_clusters_actual': n_clusters_actual,
            'n_noise_points': np.sum(labels == -1) if -1 in labels else 0
        }
        
        if n_clusters_actual > 1:
            try:
                metrics['silhouette'] = float(silhouette_score(X, labels))
            except:
                metrics['silhouette'] = 0.0
            
            try:
                metrics['calinski_harabasz'] = float(calinski_harabasz_score(X, labels))
            except:
                metrics['calinski_harabasz'] = 0.0
            
            try:
                metrics['davies_bouldin'] = float(davies_bouldin_score(X, labels))
            except:
                metrics['davies_bouldin'] = float('inf')
        else:
            metrics['silhouette'] = 0.0
            metrics['calinski_harabasz'] = 0.0
            metrics['davies_bouldin'] = float('inf')
        
        return metrics

class ClusteringVisualizer:
    """Visualization for improved clustering results."""
    
    @staticmethod
    def plot_clustering_results(embeddings: np.ndarray, labels: np.ndarray, 
                              output_path: str) -> None:
        """Plot comprehensive clustering results."""
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        # Subsample if too many points
        if len(embeddings) > 1000:
            indices = np.random.choice(len(embeddings), 1000, replace=False)
            embeddings_sub = embeddings[indices]
            labels_sub = labels[indices]
        else:
            embeddings_sub = embeddings
            labels_sub = labels
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # t-SNE plot
        print("[INFO] Computing t-SNE for visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_sub)-1))
        embeddings_tsne = tsne.fit_transform(embeddings_sub)
        
        scatter = axes[0, 0].scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                                   c=labels_sub, cmap='tab10', alpha=0.7, s=30)
        axes[0, 0].set_title('t-SNE Visualization')
        axes[0, 0].set_xlabel('t-SNE 1')
        axes[0, 0].set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # PCA plot
        if embeddings_sub.shape[1] > 2:
            pca = PCA(n_components=2)
            embeddings_pca = pca.fit_transform(embeddings_sub)
            
            scatter = axes[0, 1].scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                                       c=labels_sub, cmap='tab10', alpha=0.7, s=30)
            axes[0, 1].set_title(f'PCA Visualization (explained var: {pca.explained_variance_ratio_.sum():.2%})')
            axes[0, 1].set_xlabel('PC1')
            axes[0, 1].set_ylabel('PC2')
            plt.colorbar(scatter, ax=axes[0, 1])
        
        # Cluster size distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        axes[1, 0].bar(unique_labels, counts, alpha=0.7)
        axes[1, 0].set_title('Cluster Size Distribution')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('Number of Points')
        
        # Cluster statistics
        try:
            sil_score = silhouette_score(embeddings_sub, labels_sub)
            cal_score = calinski_harabasz_score(embeddings_sub, labels_sub)
            
            stats_text = f'Silhouette Score: {sil_score:.3f}\nCalinski-Harabasz: {cal_score:.1f}\nNumber of Clusters: {len(unique_labels)}'
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
            axes[1, 1].set_title('Clustering Statistics')
            axes[1, 1].axis('off')
        except:
            axes[1, 1].text(0.1, 0.5, 'Statistics unavailable', fontsize=12, verticalalignment='center')
            axes[1, 1].set_title('Clustering Statistics')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Clustering visualization saved to {output_path}")
    
    @staticmethod
    def hierarchical_clustering_dendrogram(embeddings: np.ndarray, 
                                         output_path: str) -> None:
        """Create hierarchical clustering dendrogram."""
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import pdist
        
        # Subsample if too many points
        if len(embeddings) > 100:
            indices = np.random.choice(len(embeddings), 100, replace=False)
            embeddings_sub = embeddings[indices]
            labels = [f"Point {i}" for i in indices]
        else:
            embeddings_sub = embeddings
            labels = [f"Point {i}" for i in range(len(embeddings))]
        
        print("[INFO] Computing hierarchical clustering dendrogram...")
        
        # Compute distance matrix and linkage
        dist_matrix = pdist(embeddings_sub, metric='euclidean')
        linkage_matrix = linkage(dist_matrix, method='ward')
        
        # Plot dendrogram
        plt.figure(figsize=(15, 8))
        dendrogram(linkage_matrix, labels=labels, orientation='top', leaf_rotation=90)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Data Points')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Dendrogram saved to {output_path}")
