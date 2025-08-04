#!/usr/bin/env python3
"""
conditional_optimization.py

Conditional effect-optimized clustering for maximizing significant conditional effects.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    from core.config import RANDOM_SEED, DEFAULT_KMEANS_INIT, MIN_CRIMINALS_FOR_ANALYSIS
    from data.loaders import Type2DataProcessor
    from clustering.basic_clustering import BasicClusterer
except ImportError:
    try:
        from ..core.config import RANDOM_SEED, DEFAULT_KMEANS_INIT, MIN_CRIMINALS_FOR_ANALYSIS
        from ..data.loaders import Type2DataProcessor
        from .basic_clustering import BasicClusterer
    except ImportError:
        RANDOM_SEED = 42
        DEFAULT_KMEANS_INIT = 10
        MIN_CRIMINALS_FOR_ANALYSIS = 5
        # Will need to import these classes later

class ConditionalEffectOptimizer:
    """Optimize clustering for conditional effect detection."""
    
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.clusterer = BasicClusterer(random_state)
    
    def find_optimal_k_for_conditional_analysis(self, embeddings: np.ndarray, 
                                              criminal_sequences: Dict[str, List[int]], 
                                              type2_df, k_range: Optional[range] = None, 
                                              min_effect_size: float = 0.1) -> Tuple[int, Dict[int, Dict[str, Any]]]:
        """
        Find optimal k by maximizing significant conditional effects.
        
        Args:
            embeddings: Event embeddings
            criminal_sequences: Criminal ID to sequence mapping
            type2_df: Type 2 data for conditional analysis
            k_range: Range of k values to test
            min_effect_size: Minimum L1 difference to consider significant
        
        Returns:
            Tuple of (optimal_k, effect_metrics)
        """
        if k_range is None:
            k_range = range(3, min(20, len(embeddings) // 50))
        
        best_k = None
        best_score = 0
        results = {}
        
        print(f"[INFO] Optimizing k for conditional effect detection...")
        
        for k in k_range:
            print(f"  Testing k={k} for conditional effects...")
            
            # Cluster with this k
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=DEFAULT_KMEANS_INIT)
            labels = kmeans.fit_predict(embeddings)
            
            # Build sequences with new clustering
            test_sequences = self._build_test_sequences(criminal_sequences, labels)
            
            # Compute global transition matrix and stationary distribution
            from ..markov.transition_analysis import TransitionMatrixBuilder
            builder = TransitionMatrixBuilder()
            global_matrix = builder.build_conditional_markov(list(test_sequences.keys()), test_sequences, k)
            global_stationary = builder.compute_stationary_distribution(global_matrix)
            
            # Test conditional effects
            effect_results = self._test_conditional_effects(
                type2_df, test_sequences, k, global_stationary, min_effect_size
            )
            
            # Score this k value
            score = self._compute_k_score(effect_results)
            results[k] = effect_results
            results[k]['score'] = score
            
            print(f"    k={k}: {effect_results['significant_effects']}/{effect_results['total_effects']} "
                  f"significant ({effect_results['significance_rate']:.2%}), "
                  f"mean effect={effect_results['mean_effect_size']:.3f}, score={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        
        return best_k, results
    
    def _build_test_sequences(self, criminal_sequences: Dict[str, List[int]], 
                            labels: np.ndarray) -> Dict[str, List[int]]:
        """Build test sequences with new cluster labels."""
        test_sequences = {}
        event_idx = 0
        
        for crim_id, events in criminal_sequences.items():
            test_sequences[crim_id] = []
            for _ in events:
                if event_idx < len(labels):
                    test_sequences[crim_id].append(labels[event_idx])
                    event_idx += 1
        
        return test_sequences
    
    def _test_conditional_effects(self, type2_df, test_sequences: Dict[str, List[int]], 
                                k: int, global_stationary: np.ndarray, 
                                min_effect_size: float) -> Dict[str, Any]:
        """Test conditional effects for a given k value."""
        from ..markov.transition_analysis import TransitionMatrixBuilder
        
        builder = TransitionMatrixBuilder()
        significant_effects = 0
        total_effects = 0
        effect_sizes = []
        
        # Test each heading in Type 2 data
        for heading in type2_df["Heading"].unique():
            condition_map = Type2DataProcessor.get_condition_map(type2_df, heading)
            unique_values = set(condition_map.values())
            
            for val in unique_values:
                selected_ids = [cid for cid, v in condition_map.items() if v == val]
                if len(selected_ids) < MIN_CRIMINALS_FOR_ANALYSIS:
                    continue
                
                total_effects += 1
                
                # Build conditional matrix
                cond_matrix = builder.build_conditional_markov(selected_ids, test_sequences, k)
                cond_stationary = builder.compute_stationary_distribution(cond_matrix)
                
                # Compute effect size (L1 distance)
                l1_diff = np.sum(np.abs(cond_stationary - global_stationary))
                effect_sizes.append(l1_diff)
                
                if l1_diff >= min_effect_size:
                    significant_effects += 1
        
        # Compute summary statistics
        significance_rate = significant_effects / total_effects if total_effects > 0 else 0
        mean_effect_size = np.mean(effect_sizes) if effect_sizes else 0
        
        return {
            'significant_effects': significant_effects,
            'total_effects': total_effects,
            'significance_rate': significance_rate,
            'mean_effect_size': mean_effect_size,
            'effect_sizes': effect_sizes
        }
    
    def _compute_k_score(self, effect_results: Dict[str, Any]) -> float:
        """Compute score for a k value based on conditional effects."""
        significance_rate = effect_results['significance_rate']
        mean_effect_size = effect_results['mean_effect_size']
        
        # Combined score: rate of significant effects * mean effect size
        return significance_rate * mean_effect_size
    
    def multi_objective_k_selection(self, embeddings: np.ndarray, 
                                   criminal_sequences: Dict[str, List[int]], 
                                   type2_df, k_range: Optional[range] = None) -> Tuple[int, Dict[int, Dict[str, float]]]:
        """
        Select k using both clustering quality and conditional effect strength.
        
        Args:
            embeddings: Event embeddings
            criminal_sequences: Criminal ID to sequence mapping
            type2_df: Type 2 data for conditional analysis
            k_range: Range of k values to test
            
        Returns:
            Tuple of (optimal_k, results)
        """
        if k_range is None:
            k_range = range(3, 16)
        
        results = {}
        
        print(f"[INFO] Multi-objective k optimization...")
        
        for k in k_range:
            # Clustering quality metrics
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=DEFAULT_KMEANS_INIT)
            labels = kmeans.fit_predict(embeddings)
            
            try:
                silhouette = silhouette_score(embeddings, labels)
            except:
                silhouette = 0.0
            
            # Conditional effect metrics
            _, effect_results = self.find_optimal_k_for_conditional_analysis(
                embeddings, criminal_sequences, type2_df, k_range=[k]
            )
            
            effect_score = effect_results[k]['score'] if k in effect_results else 0
            
            # Combined score (weighted)
            combined_score = 0.3 * silhouette + 0.7 * effect_score
            
            results[k] = {
                'silhouette': silhouette,
                'effect_score': effect_score,
                'combined_score': combined_score
            }
            
            print(f"  k={k}: silhouette={silhouette:.3f}, effect={effect_score:.3f}, "
                  f"combined={combined_score:.3f}")
        
        best_k = max(results.keys(), key=lambda k: results[k]['combined_score'])
        return best_k, results

class ConditionalClusteringPipeline:
    """Complete pipeline for conditional effect-optimized clustering."""
    
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.optimizer = ConditionalEffectOptimizer(random_state)
        self.clusterer = BasicClusterer(random_state)
    
    def run_conditional_optimization(self, embeddings: np.ndarray, 
                                   criminal_sequences: Dict[str, List[int]], 
                                   type2_df, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete conditional optimization pipeline.
        
        Args:
            embeddings: Event embeddings
            criminal_sequences: Criminal sequences (temporary for k optimization)
            type2_df: Type 2 DataFrame
            config: Configuration dictionary
            
        Returns:
            Optimization results
        """
        results = {}
        
        if config.get('auto_k', False) and type2_df is not None:
            print("\n[INFO] Running conditional effect optimization...")
            
            # Choose optimization method
            if config.get('multi_objective', False):
                optimal_k, optimization_results = self.optimizer.multi_objective_k_selection(
                    embeddings, criminal_sequences, type2_df
                )
                results['optimization_method'] = 'multi_objective'
            else:
                optimal_k, optimization_results = self.optimizer.find_optimal_k_for_conditional_analysis(
                    embeddings, criminal_sequences, type2_df
                )
                results['optimization_method'] = 'conditional_effects'
            
            results['optimal_k'] = optimal_k
            results['optimization_results'] = optimization_results
            
            print(f"[INFO] Selected k={optimal_k} for maximum conditional effects")
            
        elif config.get('auto_k', False):
            print("[WARNING] --auto_k specified but no Type 2 data available for conditional optimization")
            results['optimal_k'] = config.get('n_clusters', 5)
            results['optimization_method'] = 'fallback'
        else:
            results['optimal_k'] = config.get('n_clusters', 5)
            results['optimization_method'] = 'manual'
        
        return results
    
    def apply_optimized_clustering(self, embeddings: np.ndarray, 
                                 optimal_k: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply clustering with the optimized k value.
        
        Args:
            embeddings: Event embeddings
            optimal_k: Optimal number of clusters
            
        Returns:
            Tuple of (labels, metrics)
        """
        labels, model = self.clusterer.kmeans_cluster(embeddings, n_clusters=optimal_k)
        metrics = self.clusterer.evaluate_clustering(embeddings, labels)
        metrics['n_clusters'] = optimal_k
        
        return labels, metrics
