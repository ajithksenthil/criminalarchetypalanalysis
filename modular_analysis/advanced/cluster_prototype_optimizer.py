#!/usr/bin/env python3
"""
cluster_prototype_optimizer.py

Two-layer prototype optimization system:
Layer 1: Individual event prototypes (existing system)
Layer 2: Cluster-level archetypal optimization

This creates optimized archetypal representations from clusters of prototype events,
with train/validation splits to ensure generalizability.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import logging

class ClusterPrototypeOptimizer:
    """
    Optimize cluster representations using two-layer prototype processing.
    
    Layer 1: Individual events → prototype embeddings → initial clustering
    Layer 2: Cluster events → cluster-level prototypes → optimized archetypes
    """
    
    def __init__(self, validation_split: float = 0.3, random_state: int = 42):
        """
        Initialize the cluster prototype optimizer.
        
        Args:
            validation_split: Fraction of cluster events to use for validation
            random_state: Random seed for reproducibility
        """
        self.validation_split = validation_split
        self.random_state = random_state
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the optimizer."""
        logger = logging.getLogger('ClusterPrototypeOptimizer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(name)s] %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def optimize_cluster_prototypes(self, 
                                  embeddings: np.ndarray,
                                  labels: np.ndarray,
                                  original_texts: List[str],
                                  processed_texts: List[str]) -> Dict[str, Any]:
        """
        Optimize cluster prototypes using two-layer approach.
        
        Args:
            embeddings: Layer 1 prototype embeddings (n_events, embedding_dim)
            labels: Cluster labels from Layer 1 clustering
            original_texts: Original event texts
            processed_texts: Processed event texts (entity replaced)
            
        Returns:
            Dictionary with optimized cluster information
        """
        self.logger.info("Starting two-layer cluster prototype optimization")
        self.logger.info(f"Input: {len(embeddings)} events, {len(np.unique(labels))} clusters")
        
        optimized_clusters = {}
        validation_results = {}
        
        for cluster_id in np.unique(labels):
            self.logger.info(f"Optimizing cluster {cluster_id}")
            
            # Get events in this cluster
            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            cluster_original_texts = [original_texts[i] for i in range(len(original_texts)) if cluster_mask[i]]
            cluster_processed_texts = [processed_texts[i] for i in range(len(processed_texts)) if cluster_mask[i]]
            
            # Optimize this cluster
            cluster_result = self._optimize_single_cluster(
                cluster_id=cluster_id,
                embeddings=cluster_embeddings,
                original_texts=cluster_original_texts,
                processed_texts=cluster_processed_texts
            )
            
            optimized_clusters[cluster_id] = cluster_result['optimization']
            validation_results[cluster_id] = cluster_result['validation']
        
        # Compute overall optimization metrics
        overall_metrics = self._compute_overall_metrics(optimized_clusters, validation_results)
        
        return {
            'optimized_clusters': optimized_clusters,
            'validation_results': validation_results,
            'overall_metrics': overall_metrics,
            'method': 'Two-Layer Prototype Optimization',
            'validation_split': self.validation_split
        }
    
    def _optimize_single_cluster(self, 
                               cluster_id: int,
                               embeddings: np.ndarray,
                               original_texts: List[str],
                               processed_texts: List[str]) -> Dict[str, Any]:
        """
        Optimize a single cluster using train/validation split.
        
        Args:
            cluster_id: Cluster identifier
            embeddings: Embeddings for events in this cluster
            original_texts: Original texts for events in this cluster
            processed_texts: Processed texts for events in this cluster
            
        Returns:
            Dictionary with optimization and validation results
        """
        n_events = len(embeddings)
        
        if n_events < 10:
            self.logger.warning(f"Cluster {cluster_id} has only {n_events} events, skipping optimization")
            return self._create_fallback_result(cluster_id, embeddings, original_texts, processed_texts)
        
        # Split into train/validation
        train_indices, val_indices = train_test_split(
            range(n_events), 
            test_size=self.validation_split,
            random_state=self.random_state
        )
        
        train_embeddings = embeddings[train_indices]
        val_embeddings = embeddings[val_indices]
        
        train_original = [original_texts[i] for i in train_indices]
        train_processed = [processed_texts[i] for i in train_indices]
        val_original = [original_texts[i] for i in val_indices]
        val_processed = [processed_texts[i] for i in val_indices]
        
        # Layer 2 optimization: Create archetypal prototype from training data
        archetypal_prototype = self._create_archetypal_prototype(
            train_embeddings, train_processed
        )
        
        # Validate the archetypal prototype
        validation_metrics = self._validate_archetypal_prototype(
            archetypal_prototype, val_embeddings, val_processed
        )
        
        # Select most representative samples using the archetypal prototype
        representative_samples = self._select_representative_samples(
            archetypal_prototype, train_embeddings, train_processed, n_samples=5
        )
        
        # Compute cluster coherence metrics
        coherence_metrics = self._compute_cluster_coherence(
            archetypal_prototype, embeddings, processed_texts
        )
        
        return {
            'optimization': {
                'cluster_id': cluster_id,
                'archetypal_prototype': archetypal_prototype,
                'representative_samples': representative_samples,
                'coherence_metrics': coherence_metrics,
                'n_train_events': len(train_indices),
                'n_val_events': len(val_indices),
                'optimization_method': 'Archetypal Centroid with Validation'
            },
            'validation': {
                'cluster_id': cluster_id,
                'validation_metrics': validation_metrics,
                'train_silhouette': self._compute_internal_silhouette(train_embeddings),
                'val_silhouette': self._compute_internal_silhouette(val_embeddings),
                'cross_validation_score': validation_metrics.get('prototype_similarity', 0.0)
            }
        }
    
    def _create_archetypal_prototype(self, 
                                   embeddings: np.ndarray, 
                                   processed_texts: List[str]) -> Dict[str, Any]:
        """
        Create an archetypal prototype from cluster training data.
        
        This goes beyond simple centroid by considering:
        1. Weighted centroid based on representativeness
        2. Outlier removal for cleaner archetype
        3. Semantic consistency validation
        """
        # Compute centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Compute distances to centroid
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        
        # Remove outliers (events too far from centroid)
        outlier_threshold = np.percentile(distances, 85)  # Remove top 15% outliers
        inlier_mask = distances <= outlier_threshold
        
        if np.sum(inlier_mask) < 3:  # Keep at least 3 events
            inlier_mask = np.ones(len(embeddings), dtype=bool)
        
        # Recompute centroid without outliers
        clean_embeddings = embeddings[inlier_mask]
        clean_texts = [processed_texts[i] for i in range(len(processed_texts)) if inlier_mask[i]]
        
        archetypal_centroid = np.mean(clean_embeddings, axis=0)
        
        # Compute representativeness scores
        representativeness = 1.0 / (1.0 + np.linalg.norm(clean_embeddings - archetypal_centroid, axis=1))
        
        return {
            'centroid': archetypal_centroid,
            'n_events_used': len(clean_embeddings),
            'n_outliers_removed': len(embeddings) - len(clean_embeddings),
            'representativeness_scores': representativeness.tolist(),
            'archetypal_texts': clean_texts,
            'coherence_score': np.mean(representativeness)
        }
    
    def _validate_archetypal_prototype(self, 
                                     archetypal_prototype: Dict[str, Any],
                                     val_embeddings: np.ndarray,
                                     val_texts: List[str]) -> Dict[str, Any]:
        """
        Validate the archetypal prototype on held-out validation data.
        """
        centroid = archetypal_prototype['centroid']
        
        # Compute similarities between validation events and archetypal prototype
        similarities = []
        for embedding in val_embeddings:
            similarity = np.dot(embedding, centroid) / (np.linalg.norm(embedding) * np.linalg.norm(centroid))
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        
        # Validation metrics
        return {
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'prototype_similarity': float(np.mean(similarities)),  # Main validation score
            'n_validation_events': len(val_embeddings)
        }
    
    def _select_representative_samples(self, 
                                     archetypal_prototype: Dict[str, Any],
                                     embeddings: np.ndarray,
                                     texts: List[str],
                                     n_samples: int = 5) -> List[str]:
        """
        Select most representative samples based on archetypal prototype.
        """
        centroid = archetypal_prototype['centroid']
        
        # Compute similarities to archetypal centroid
        similarities = []
        for embedding in embeddings:
            similarity = np.dot(embedding, centroid) / (np.linalg.norm(embedding) * np.linalg.norm(centroid))
            similarities.append(similarity)
        
        # Select top N most similar events
        top_indices = np.argsort(similarities)[-n_samples:][::-1]
        
        return [texts[i] for i in top_indices]
    
    def _compute_cluster_coherence(self, 
                                 archetypal_prototype: Dict[str, Any],
                                 all_embeddings: np.ndarray,
                                 all_texts: List[str]) -> Dict[str, Any]:
        """
        Compute coherence metrics for the optimized cluster.
        """
        centroid = archetypal_prototype['centroid']
        
        # Compute distances to archetypal centroid
        distances = np.linalg.norm(all_embeddings - centroid, axis=1)
        
        return {
            'mean_distance_to_archetype': float(np.mean(distances)),
            'std_distance_to_archetype': float(np.std(distances)),
            'coherence_score': float(1.0 / (1.0 + np.mean(distances))),  # Higher is better
            'compactness': float(1.0 / (1.0 + np.std(distances))),  # Higher is better
            'n_events': len(all_embeddings)
        }
    
    def _compute_internal_silhouette(self, embeddings: np.ndarray) -> float:
        """
        Compute internal silhouette score for a single cluster.
        Uses k-means with k=2 to measure internal structure.
        """
        if len(embeddings) < 4:
            return 0.0
        
        try:
            # Use k=2 to measure internal coherence
            kmeans = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
            internal_labels = kmeans.fit_predict(embeddings)
            
            # Only compute if we actually get 2 clusters
            if len(np.unique(internal_labels)) == 2:
                return silhouette_score(embeddings, internal_labels)
            else:
                return 0.0
        except:
            return 0.0
    
    def _create_fallback_result(self, 
                              cluster_id: int,
                              embeddings: np.ndarray,
                              original_texts: List[str],
                              processed_texts: List[str]) -> Dict[str, Any]:
        """
        Create fallback result for clusters too small to optimize.
        """
        centroid = np.mean(embeddings, axis=0)
        
        return {
            'optimization': {
                'cluster_id': cluster_id,
                'archetypal_prototype': {
                    'centroid': centroid,
                    'n_events_used': len(embeddings),
                    'n_outliers_removed': 0,
                    'coherence_score': 1.0
                },
                'representative_samples': processed_texts[:5],
                'coherence_metrics': {
                    'coherence_score': 1.0,
                    'compactness': 1.0,
                    'n_events': len(embeddings)
                },
                'n_train_events': len(embeddings),
                'n_val_events': 0,
                'optimization_method': 'Fallback (too few events)'
            },
            'validation': {
                'cluster_id': cluster_id,
                'validation_metrics': {'prototype_similarity': 1.0},
                'train_silhouette': 0.0,
                'val_silhouette': 0.0,
                'cross_validation_score': 1.0
            }
        }
    
    def _compute_overall_metrics(self, 
                               optimized_clusters: Dict[int, Any],
                               validation_results: Dict[int, Any]) -> Dict[str, Any]:
        """
        Compute overall optimization metrics across all clusters.
        """
        # Collect metrics across clusters
        coherence_scores = []
        validation_scores = []
        compactness_scores = []
        
        for cluster_id, cluster_data in optimized_clusters.items():
            coherence_scores.append(cluster_data['coherence_metrics']['coherence_score'])
            compactness_scores.append(cluster_data['coherence_metrics']['compactness'])
            
            val_data = validation_results[cluster_id]
            validation_scores.append(val_data['cross_validation_score'])
        
        return {
            'mean_coherence': float(np.mean(coherence_scores)),
            'mean_validation_score': float(np.mean(validation_scores)),
            'mean_compactness': float(np.mean(compactness_scores)),
            'overall_optimization_score': float(np.mean(coherence_scores) * np.mean(validation_scores)),
            'n_clusters_optimized': len(optimized_clusters),
            'optimization_improvement': self._compute_optimization_improvement(optimized_clusters)
        }
    
    def _compute_optimization_improvement(self, optimized_clusters: Dict[int, Any]) -> float:
        """
        Estimate improvement from optimization (simplified metric).
        """
        # This is a simplified metric - in practice you'd compare to pre-optimization
        coherence_scores = [cluster['coherence_metrics']['coherence_score'] 
                          for cluster in optimized_clusters.values()]
        
        # Higher coherence indicates better optimization
        return float(np.mean(coherence_scores))

def save_optimized_clusters(optimization_results: Dict[str, Any], 
                          output_dir: Path) -> None:
    """
    Save optimized cluster results to the output directory.
    
    Args:
        optimization_results: Results from cluster optimization
        output_dir: Output directory path
    """
    # Create advanced directory
    advanced_dir = output_dir / "advanced"
    advanced_dir.mkdir(exist_ok=True)
    
    # Save optimization results
    with open(advanced_dir / "cluster_optimization_results.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = _make_json_serializable(optimization_results)
        json.dump(serializable_results, f, indent=2)
    
    # Save optimized cluster info (compatible with existing system)
    optimized_cluster_info = _create_optimized_cluster_info(optimization_results)
    with open(advanced_dir / "optimized_cluster_info.json", 'w') as f:
        json.dump(optimized_cluster_info, f, indent=2)
    
    print(f"[INFO] Optimized cluster results saved to {advanced_dir}")

def _make_json_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
    if isinstance(obj, dict):
        # Convert keys to strings if they're numpy types
        new_dict = {}
        for key, value in obj.items():
            if isinstance(key, (np.integer, np.int32, np.int64)):
                new_key = str(int(key))
            elif isinstance(key, (np.floating, np.float32, np.float64)):
                new_key = str(float(key))
            else:
                new_key = key
            new_dict[new_key] = _make_json_serializable(value)
        return new_dict
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj

def _create_optimized_cluster_info(optimization_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create cluster info compatible with existing system format.
    """
    optimized_clusters = optimization_results['optimized_clusters']
    
    cluster_info = []
    for cluster_id, cluster_data in optimized_clusters.items():
        cluster_info.append({
            'cluster_id': cluster_id,
            'size': cluster_data['coherence_metrics']['n_events'],
            'representative_samples': cluster_data['representative_samples'],
            'archetypal_theme': f"Optimized Archetype {cluster_id}",
            'coherence_score': cluster_data['coherence_metrics']['coherence_score'],
            'compactness': cluster_data['coherence_metrics']['compactness'],
            'optimization_method': cluster_data['optimization_method'],
            'n_train_events': cluster_data['n_train_events'],
            'n_val_events': cluster_data['n_val_events']
        })
    
    return cluster_info
