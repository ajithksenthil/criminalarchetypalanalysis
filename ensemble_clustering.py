#!/usr/bin/env python3
"""
ensemble_clustering.py

Ensemble clustering methods for robust archetype identification.
Combines multiple clustering algorithms to find stable patterns.
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')


class EnsembleClustering:
    """
    Ensemble clustering using consensus matrix approach.
    Combines multiple clustering algorithms for robust results.
    """
    
    def __init__(self, n_clusters=5, methods=None, n_iterations=10):
        """
        Initialize ensemble clustering.
        
        Args:
            n_clusters: Number of clusters to find
            methods: List of clustering methods to use
            n_iterations: Number of iterations per method for stability
        """
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        
        if methods is None:
            self.methods = ['kmeans', 'gmm', 'spectral', 'hierarchical']
        else:
            self.methods = methods
            
        self.consensus_matrix = None
        self.final_labels = None
        self.method_results = {}
        
    def fit(self, X):
        """
        Fit ensemble clustering model.
        
        Args:
            X: Data matrix (n_samples, n_features)
            
        Returns:
            Final cluster labels
        """
        n_samples = X.shape[0]
        
        # Initialize consensus matrix
        self.consensus_matrix = np.zeros((n_samples, n_samples))
        total_runs = 0
        
        # Apply each clustering method multiple times
        all_labelings = []
        
        for method in self.methods:
            print(f"[Ensemble] Running {method} clustering...")
            method_labelings = []
            
            for iteration in range(self.n_iterations):
                # Add small random perturbation for diversity
                if iteration > 0:
                    noise = np.random.normal(0, 0.01, X.shape)
                    X_perturbed = X + noise
                else:
                    X_perturbed = X
                
                # Apply clustering method
                labels = self._apply_clustering(X_perturbed, method)
                
                if labels is not None:
                    method_labelings.append(labels)
                    all_labelings.append(labels)
                    
                    # Update consensus matrix
                    self._update_consensus_matrix(labels)
                    total_runs += 1
            
            # Store method results
            if method_labelings:
                self.method_results[method] = {
                    'labelings': method_labelings,
                    'stability': self._compute_stability(method_labelings)
                }
        
        # Normalize consensus matrix
        if total_runs > 0:
            self.consensus_matrix /= total_runs
        
        # Final clustering on consensus matrix
        print("[Ensemble] Computing final consensus clustering...")
        self.final_labels = self._consensus_clustering(self.consensus_matrix)
        
        # Compute ensemble metrics
        self._compute_ensemble_metrics(X, all_labelings)
        
        return self.final_labels
    
    def _apply_clustering(self, X, method):
        """Apply a specific clustering method."""
        try:
            if method == 'kmeans':
                clusterer = KMeans(n_clusters=self.n_clusters, random_state=None, n_init=10)
                labels = clusterer.fit_predict(X)
                
            elif method == 'gmm':
                clusterer = GaussianMixture(n_components=self.n_clusters, random_state=None)
                labels = clusterer.fit_predict(X)
                
            elif method == 'spectral':
                clusterer = SpectralClustering(n_clusters=self.n_clusters, random_state=None,
                                              affinity='nearest_neighbors', n_neighbors=10)
                labels = clusterer.fit_predict(X)
                
            elif method == 'hierarchical':
                clusterer = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='ward')
                labels = clusterer.fit_predict(X)
                
            elif method == 'affinity':
                from sklearn.cluster import AffinityPropagation
                clusterer = AffinityPropagation(random_state=None, damping=0.9)
                labels = clusterer.fit_predict(X)
                
                # Adjust number of clusters if needed
                n_found = len(np.unique(labels))
                if n_found != self.n_clusters:
                    # Re-cluster to desired number
                    cluster_centers = []
                    for i in range(n_found):
                        cluster_centers.append(X[labels == i].mean(axis=0))
                    
                    kmeans = KMeans(n_clusters=self.n_clusters, init=np.array(cluster_centers)[:self.n_clusters])
                    labels = kmeans.fit_predict(X)
                    
            else:
                return None
                
            return labels
            
        except Exception as e:
            print(f"[Warning] {method} clustering failed: {e}")
            return None
    
    def _update_consensus_matrix(self, labels):
        """Update consensus matrix based on clustering labels."""
        n_samples = len(labels)
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if labels[i] == labels[j]:
                    self.consensus_matrix[i, j] += 1
                    self.consensus_matrix[j, i] += 1
    
    def _consensus_clustering(self, consensus_matrix):
        """Perform final clustering on consensus matrix."""
        # Convert consensus matrix to distance matrix
        distance_matrix = 1 - consensus_matrix
        
        # Use hierarchical clustering on consensus
        # Note: metric='precomputed' instead of affinity for newer sklearn
        from sklearn import __version__ as sklearn_version
        if sklearn_version >= '1.2':
            clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                metric='precomputed',
                linkage='average'
            )
        else:
            clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                affinity='precomputed',
                linkage='average'
            )
        
        return clusterer.fit_predict(distance_matrix)
    
    def _compute_stability(self, labelings):
        """Compute stability of clustering across iterations."""
        if len(labelings) < 2:
            return 0.0
        
        # Compute average pairwise ARI
        ari_scores = []
        for i in range(len(labelings)):
            for j in range(i + 1, len(labelings)):
                ari = adjusted_rand_score(labelings[i], labelings[j])
                ari_scores.append(ari)
        
        return np.mean(ari_scores) if ari_scores else 0.0
    
    def _compute_ensemble_metrics(self, X, all_labelings):
        """Compute metrics for ensemble clustering."""
        # Individual method performances
        self.ensemble_metrics = {
            'final_silhouette': silhouette_score(X, self.final_labels),
            'consensus_strength': np.mean(self.consensus_matrix[self.consensus_matrix > 0]),
            'method_agreement': self._compute_method_agreement(all_labelings),
            'robustness': self._compute_robustness(X)
        }
        
        # Method-specific silhouette scores
        for method, results in self.method_results.items():
            scores = []
            for labels in results['labelings']:
                try:
                    score = silhouette_score(X, labels)
                    scores.append(score)
                except:
                    pass
            
            if scores:
                results['avg_silhouette'] = np.mean(scores)
                results['std_silhouette'] = np.std(scores)
    
    def _compute_method_agreement(self, all_labelings):
        """Compute agreement between different methods."""
        if len(all_labelings) < 2:
            return 0.0
        
        agreements = []
        for i in range(len(all_labelings)):
            for j in range(i + 1, len(all_labelings)):
                ari = adjusted_rand_score(all_labelings[i], all_labelings[j])
                agreements.append(ari)
        
        return np.mean(agreements)
    
    def _compute_robustness(self, X):
        """Compute robustness of final clustering to perturbations."""
        robustness_scores = []
        
        for _ in range(5):
            # Add noise to data
            noise = np.random.normal(0, 0.05 * np.std(X), X.shape)
            X_noisy = X + noise
            
            # Re-cluster
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            noisy_labels = kmeans.fit_predict(X_noisy)
            
            # Compare to final labels
            ari = adjusted_rand_score(self.final_labels, noisy_labels)
            robustness_scores.append(ari)
        
        return np.mean(robustness_scores)
    
    def get_confidence_scores(self):
        """
        Get confidence scores for each sample's cluster assignment.
        
        Returns:
            Array of confidence scores (0-1) for each sample
        """
        if self.consensus_matrix is None or self.final_labels is None:
            return None
        
        n_samples = len(self.final_labels)
        confidence_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Get samples in same cluster
            same_cluster = self.final_labels == self.final_labels[i]
            
            # Average consensus with cluster members
            if np.sum(same_cluster) > 1:
                consensus_values = self.consensus_matrix[i, same_cluster]
                confidence_scores[i] = np.mean(consensus_values)
        
        return confidence_scores
    
    def get_summary(self):
        """Get summary of ensemble clustering results."""
        summary = {
            'n_clusters': self.n_clusters,
            'methods_used': self.methods,
            'n_iterations': self.n_iterations,
            'ensemble_metrics': self.ensemble_metrics,
            'method_performances': {}
        }
        
        for method, results in self.method_results.items():
            summary['method_performances'][method] = {
                'stability': results['stability'],
                'avg_silhouette': results.get('avg_silhouette', 0),
                'std_silhouette': results.get('std_silhouette', 0)
            }
        
        return summary


class MultiViewClustering:
    """
    Multi-view clustering for different representations of the same data.
    Useful when you have multiple feature sets (e.g., TF-IDF and embeddings).
    """
    
    def __init__(self, n_clusters=5, fusion_method='late'):
        """
        Initialize multi-view clustering.
        
        Args:
            n_clusters: Number of clusters
            fusion_method: 'early' (concatenate features) or 'late' (ensemble)
        """
        self.n_clusters = n_clusters
        self.fusion_method = fusion_method
        self.view_results = {}
        self.final_labels = None
        
    def fit(self, views, view_names=None):
        """
        Fit multi-view clustering.
        
        Args:
            views: List of data matrices (different views)
            view_names: Optional names for views
            
        Returns:
            Final cluster labels
        """
        if view_names is None:
            view_names = [f"View_{i}" for i in range(len(views))]
        
        if self.fusion_method == 'early':
            # Concatenate all views
            print("[MultiView] Using early fusion (concatenation)...")
            X_concat = np.hstack([StandardScaler().fit_transform(v) for v in views])
            
            # Cluster concatenated features
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            self.final_labels = kmeans.fit_predict(X_concat)
            
        else:  # late fusion
            print("[MultiView] Using late fusion (ensemble)...")
            all_labels = []
            
            # Cluster each view separately
            for view, name in zip(views, view_names):
                print(f"  Clustering {name}...")
                
                # Standardize view
                view_scaled = StandardScaler().fit_transform(view)
                
                # Apply multiple algorithms
                view_labels = []
                
                # K-means
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
                labels_kmeans = kmeans.fit_predict(view_scaled)
                view_labels.append(labels_kmeans)
                
                # GMM
                gmm = GaussianMixture(n_components=self.n_clusters, random_state=42)
                labels_gmm = gmm.fit_predict(view_scaled)
                view_labels.append(labels_gmm)
                
                all_labels.extend(view_labels)
                
                # Store view results
                self.view_results[name] = {
                    'labels': view_labels,
                    'silhouette_kmeans': silhouette_score(view_scaled, labels_kmeans),
                    'silhouette_gmm': silhouette_score(view_scaled, labels_gmm)
                }
            
            # Ensemble all labelings
            ensemble = EnsembleClustering(n_clusters=self.n_clusters, n_iterations=1)
            ensemble.consensus_matrix = np.zeros((len(views[0]), len(views[0])))
            
            for labels in all_labels:
                ensemble._update_consensus_matrix(labels)
            
            ensemble.consensus_matrix /= len(all_labels)
            self.final_labels = ensemble._consensus_clustering(ensemble.consensus_matrix)
        
        return self.final_labels
    
    def get_view_importance(self, X_combined):
        """
        Compute importance of each view for final clustering.
        
        Args:
            X_combined: Combined data matrix for evaluation
            
        Returns:
            Dictionary of view importance scores
        """
        if not self.view_results:
            return {}
        
        final_silhouette = silhouette_score(X_combined, self.final_labels)
        
        importance = {}
        for view_name, results in self.view_results.items():
            # Average agreement with final clustering
            agreements = []
            for labels in results['labels']:
                ari = adjusted_rand_score(self.final_labels, labels)
                agreements.append(ari)
            
            importance[view_name] = {
                'agreement_with_final': np.mean(agreements),
                'internal_quality': np.mean([results['silhouette_kmeans'], 
                                            results['silhouette_gmm']]),
                'contribution': np.mean(agreements) * final_silhouette
            }
        
        return importance


def create_robust_clusters(X, n_clusters_range=(3, 10), methods=None):
    """
    Create robust clusters using ensemble approach with automatic k selection.
    
    Args:
        X: Data matrix
        n_clusters_range: Range of cluster numbers to try
        methods: Clustering methods to use
        
    Returns:
        Best clustering results
    """
    print("Finding robust clusters using ensemble approach...")
    
    best_score = -1
    best_k = None
    best_ensemble = None
    
    # Try different numbers of clusters
    for k in range(n_clusters_range[0], n_clusters_range[1] + 1):
        print(f"\nTrying k={k}...")
        
        # Create ensemble
        ensemble = EnsembleClustering(n_clusters=k, methods=methods)
        labels = ensemble.fit(X)
        
        # Evaluate
        score = ensemble.ensemble_metrics['final_silhouette']
        
        print(f"  Silhouette: {score:.3f}")
        print(f"  Consensus strength: {ensemble.ensemble_metrics['consensus_strength']:.3f}")
        print(f"  Method agreement: {ensemble.ensemble_metrics['method_agreement']:.3f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            best_ensemble = ensemble
    
    print(f"\n[Best] k={best_k} with silhouette={best_score:.3f}")
    
    return best_ensemble


if __name__ == "__main__":
    # Test ensemble clustering
    print("Testing Ensemble Clustering...")
    
    # Generate sample data
    from sklearn.datasets import make_blobs
    X, true_labels = make_blobs(n_samples=300, n_features=10, 
                               centers=4, random_state=42)
    
    # Test basic ensemble
    print("\n1. Basic Ensemble Clustering")
    ensemble = EnsembleClustering(n_clusters=4)
    labels = ensemble.fit(X)
    
    summary = ensemble.get_summary()
    print(f"\nEnsemble Summary:")
    print(f"  Final silhouette: {summary['ensemble_metrics']['final_silhouette']:.3f}")
    print(f"  Consensus strength: {summary['ensemble_metrics']['consensus_strength']:.3f}")
    print(f"  Method agreement: {summary['ensemble_metrics']['method_agreement']:.3f}")
    
    # Test confidence scores
    confidence = ensemble.get_confidence_scores()
    print(f"\nSample confidence scores: {confidence[:10]}")
    
    # Test multi-view clustering
    print("\n2. Multi-View Clustering")
    
    # Create two views of the data
    view1 = X[:, :5]  # First 5 features
    view2 = X[:, 5:]  # Last 5 features
    
    mvc = MultiViewClustering(n_clusters=4, fusion_method='late')
    mvc_labels = mvc.fit([view1, view2], ['Features_1-5', 'Features_6-10'])
    
    # Get view importance
    importance = mvc.get_view_importance(X)
    print("\nView Importance:")
    for view, scores in importance.items():
        print(f"  {view}: agreement={scores['agreement_with_final']:.3f}")
    
    print("\nEnsemble clustering tests complete!")