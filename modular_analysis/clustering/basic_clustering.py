#!/usr/bin/env python3
"""
basic_clustering.py

Basic clustering functionality for criminal archetypal analysis.
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Constants to avoid import issues
DEFAULT_KMEANS_INIT = 10
RANDOM_SEED = 42

class BasicClusterer:
    """Basic clustering functionality."""
    
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
    
    def kmeans_cluster(self, embeddings: np.ndarray, n_clusters: int = 5) -> Tuple[np.ndarray, KMeans]:
        """
        Run KMeans clustering on embeddings.
        
        Args:
            embeddings: Input embeddings
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (labels, fitted_model)
        """
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=self.random_state, 
            n_init=DEFAULT_KMEANS_INIT
        )
        labels = kmeans.fit_predict(embeddings)
        return labels, kmeans
    
    def evaluate_clustering(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering quality using multiple metrics.
        
        Args:
            embeddings: Input embeddings
            labels: Cluster labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        n_clusters = len(set(labels))
        
        if n_clusters < 2:
            return {
                'n_clusters': n_clusters,
                'silhouette': 0.0,
                'calinski_harabasz': 0.0,
                'davies_bouldin': float('inf')
            }
        
        try:
            silhouette = silhouette_score(embeddings, labels)
        except:
            silhouette = 0.0
        
        try:
            calinski = calinski_harabasz_score(embeddings, labels)
        except:
            calinski = 0.0
        
        try:
            davies = davies_bouldin_score(embeddings, labels)
        except:
            davies = float('inf')
        
        return {
            'n_clusters': n_clusters,
            'silhouette': float(silhouette),
            'calinski_harabasz': float(calinski),
            'davies_bouldin': float(davies)
        }

class KOptimizer:
    """Optimize the number of clusters."""
    
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.clusterer = BasicClusterer(random_state)
    
    def find_optimal_k_standard(self, embeddings: np.ndarray, k_range: Optional[range] = None) -> Dict[str, Any]:
        """
        Find optimal k using standard clustering metrics.
        
        Args:
            embeddings: Input embeddings
            k_range: Range of k values to test
            
        Returns:
            Dictionary with optimal k values and metrics
        """
        if k_range is None:
            k_range = range(2, min(16, len(embeddings) // 10))
        
        results = {}
        silhouette_scores = []
        calinski_scores = []
        davies_scores = []
        
        print(f"[INFO] Finding optimal k using standard metrics...")
        
        for k in k_range:
            labels, _ = self.clusterer.kmeans_cluster(embeddings, n_clusters=k)
            metrics = self.clusterer.evaluate_clustering(embeddings, labels)
            
            silhouette_scores.append(metrics['silhouette'])
            calinski_scores.append(metrics['calinski_harabasz'])
            davies_scores.append(metrics['davies_bouldin'])
            
            results[k] = metrics
            print(f"  k={k}: silhouette={metrics['silhouette']:.3f}, "
                  f"calinski={metrics['calinski_harabasz']:.1f}, "
                  f"davies={metrics['davies_bouldin']:.3f}")
        
        # Find optimal k for each metric
        optimal_silhouette = list(k_range)[np.argmax(silhouette_scores)]
        optimal_calinski = list(k_range)[np.argmax(calinski_scores)]
        optimal_davies = list(k_range)[np.argmin(davies_scores)]
        
        # Consensus: most common optimal k
        votes = [optimal_silhouette, optimal_calinski, optimal_davies]
        consensus = max(set(votes), key=votes.count)
        
        return {
            'results': results,
            'optimal_silhouette': optimal_silhouette,
            'optimal_calinski': optimal_calinski,
            'optimal_davies': optimal_davies,
            'consensus': consensus,
            'k_range': list(k_range)
        }

class MultiModalClusterer:
    """Multi-modal clustering combining Type 1 and Type 2 data."""
    
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.clusterer = BasicClusterer(random_state)
    
    def create_multimodal_vectors(self, criminal_sequences: Dict[str, List[int]], 
                                 type2_df, n_clusters: int) -> Tuple[List[str], np.ndarray]:
        """
        Create multi-modal feature vectors combining Type 1 and Type 2 data.
        
        Args:
            criminal_sequences: Criminal cluster sequences
            type2_df: Type 2 DataFrame
            n_clusters: Number of event clusters
            
        Returns:
            Tuple of (criminal_ids, feature_vectors)
        """
        from ..data.loaders import Type2DataProcessor
        
        modal_criminal_ids = []
        multi_modal_vectors = []
        
        for crim_id, seq in criminal_sequences.items():
            # Type 1 features: cluster distribution
            cluster_counts = np.zeros(n_clusters)
            for cluster_id in seq:
                cluster_counts[cluster_id] += 1
            
            # Normalize to get distribution
            if len(seq) > 0:
                cluster_dist = cluster_counts / len(seq)
            else:
                cluster_dist = cluster_counts
            
            # Type 2 features
            type2_features = Type2DataProcessor.extract_feature_vector(
                crim_id, type2_df, ["Physically abused?", "Sex", "Number of victims"]
            )
            
            if type2_features is not None:
                # Combine Type 1 and Type 2 features
                combined_vector = np.concatenate([cluster_dist, type2_features])
                multi_modal_vectors.append(combined_vector)
                modal_criminal_ids.append(crim_id)
        
        return modal_criminal_ids, np.array(multi_modal_vectors) if multi_modal_vectors else np.array([])
    
    def cluster_multimodal(self, criminal_sequences: Dict[str, List[int]], 
                          type2_df, n_event_clusters: int, n_criminal_clusters: int = 3) -> Dict[str, Any]:
        """
        Perform multi-modal clustering at the criminal level.
        
        Args:
            criminal_sequences: Criminal cluster sequences
            type2_df: Type 2 DataFrame
            n_event_clusters: Number of event clusters
            n_criminal_clusters: Number of criminal clusters
            
        Returns:
            Clustering results
        """
        criminal_ids, vectors = self.create_multimodal_vectors(
            criminal_sequences, type2_df, n_event_clusters
        )
        
        if len(vectors) == 0:
            print("[WARNING] No multi-modal vectors created")
            return {}
        
        labels, model = self.clusterer.kmeans_cluster(vectors, n_clusters=n_criminal_clusters)
        metrics = self.clusterer.evaluate_clustering(vectors, labels)
        
        # Create results
        results = {
            'criminal_ids': criminal_ids,
            'labels': labels.tolist(),
            'metrics': metrics,
            'cluster_assignments': {cid: int(label) for cid, label in zip(criminal_ids, labels)}
        }
        
        print("[INFO] Multi-modal clustering complete. Cluster assignments:")
        for cid, label in zip(criminal_ids, labels):
            print(f"  Criminal {cid}: Cluster {label}")
        
        return results

class ClusterAnalyzer:
    """Analyze clustering results."""
    
    @staticmethod
    def analyze_cluster_characteristics(embeddings: np.ndarray, labels: np.ndarray, 
                                      sentences: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Analyze characteristics of each cluster.
        
        Args:
            embeddings: Input embeddings
            labels: Cluster labels
            sentences: Original sentences
            
        Returns:
            Dictionary mapping cluster IDs to their characteristics
        """
        cluster_analysis = {}
        unique_labels = sorted(set(labels))
        
        for cluster_id in unique_labels:
            mask = labels == cluster_id
            cluster_embeddings = embeddings[mask]
            cluster_sentences = [sentences[i] for i in range(len(sentences)) if mask[i]]
            
            # Basic statistics
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Intra-cluster distances
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(cluster_embeddings, [centroid]).flatten()
            
            cluster_analysis[cluster_id] = {
                'size': int(np.sum(mask)),
                'percentage': float(np.sum(mask) / len(labels) * 100),
                'centroid': centroid.tolist(),
                'mean_distance_to_centroid': float(np.mean(distances)),
                'std_distance_to_centroid': float(np.std(distances)),
                'sample_sentences': cluster_sentences[:5]  # First 5 as samples
            }
        
        return cluster_analysis
    
    @staticmethod
    def compute_cluster_stability(embeddings: np.ndarray, n_clusters: int, 
                                n_iterations: int = 10) -> Dict[str, float]:
        """
        Compute clustering stability across multiple runs.
        
        Args:
            embeddings: Input embeddings
            n_clusters: Number of clusters
            n_iterations: Number of iterations to test
            
        Returns:
            Stability metrics
        """
        from sklearn.metrics import adjusted_rand_score
        
        clusterer = BasicClusterer()
        all_labels = []
        
        for i in range(n_iterations):
            labels, _ = clusterer.kmeans_cluster(embeddings, n_clusters)
            all_labels.append(labels)
        
        # Compute pairwise ARI scores
        ari_scores = []
        for i in range(n_iterations):
            for j in range(i + 1, n_iterations):
                ari = adjusted_rand_score(all_labels[i], all_labels[j])
                ari_scores.append(ari)
        
        return {
            'mean_ari': float(np.mean(ari_scores)),
            'std_ari': float(np.std(ari_scores)),
            'min_ari': float(np.min(ari_scores)),
            'max_ari': float(np.max(ari_scores))
        }

# Backward compatibility aliases
def kmeans_cluster(embeddings: np.ndarray, n_clusters: int = 5) -> Tuple[np.ndarray, Any]:
    """Backward compatibility alias for BasicClusterer.kmeans_cluster()."""
    clusterer = BasicClusterer()
    return clusterer.kmeans_cluster(embeddings, n_clusters)

def evaluate_clustering(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Backward compatibility alias for BasicClusterer.evaluate_clustering()."""
    clusterer = BasicClusterer()
    return clusterer.evaluate_clustering(embeddings, labels)
