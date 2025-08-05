#!/usr/bin/env python3
"""
conditional_optimization.py

Conditional effect-optimized clustering for maximizing significant conditional effects.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Constants to avoid import issues
RANDOM_SEED = 42
DEFAULT_KMEANS_INIT = 10
MIN_CRIMINALS_FOR_ANALYSIS = 5

# Import classes - will be handled at runtime
try:
    from data.loaders import Type2DataProcessor
except ImportError:
    Type2DataProcessor = None

try:
    from clustering.basic_clustering import BasicClusterer
except ImportError:
    BasicClusterer = None

class ConditionalEffectOptimizer:
    """Optimize clustering for conditional effect detection."""
    
    def __init__(self, random_state: int = RANDOM_SEED, use_statistical_validation: bool = True):
        self.random_state = random_state
        self.use_statistical_validation = use_statistical_validation
        # Import BasicClusterer at runtime if not available
        if BasicClusterer is None:
            try:
                from clustering.basic_clustering import BasicClusterer as BC
                self.clusterer = BC(random_state)
            except ImportError:
                from sklearn.cluster import KMeans
                self.clusterer = None  # Will create KMeans directly
        else:
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
            # Use more conservative k range to avoid over-clustering
            n_samples = len(embeddings)
            k_max = min(15, int(np.sqrt(n_samples / 2)))  # Rule of thumb: k <= sqrt(n/2)
            k_range = range(3, k_max + 1)
            print(f"[INFO] Auto-selected k range: 3 to {k_max} (n_samples={n_samples})")
        
        best_k = None
        best_score = 0
        results = {}
        
        print(f"[INFO] Optimizing k for conditional effect detection...")
        
        for k in k_range:
            print(f"  Testing k={k} for conditional effects...")
            
            # Cluster with this k
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=DEFAULT_KMEANS_INIT)
            labels = kmeans.fit_predict(embeddings)

            # Evaluate clustering quality
            from sklearn.metrics import silhouette_score
            try:
                silhouette = silhouette_score(embeddings, labels)
            except:
                silhouette = 0.0

            # Skip if clustering quality is too poor
            if silhouette < 0.1:
                print(f"    k={k}: Poor clustering quality (silhouette={silhouette:.3f}), skipping")
                results[k] = {
                    'score': 0.0,
                    'significant_effects': 0,
                    'total_effects': 0,
                    'silhouette': silhouette,
                    'skipped': True
                }
                continue

            # Build sequences with new clustering
            test_sequences = self._build_test_sequences(criminal_sequences, labels)
            
            # Compute global transition matrix and stationary distribution
            try:
                from markov.transition_analysis import TransitionMatrixBuilder
            except ImportError:
                # Fallback for relative imports
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
                from markov.transition_analysis import TransitionMatrixBuilder
            builder = TransitionMatrixBuilder()
            global_matrix = builder.build_conditional_markov(list(test_sequences.keys()), test_sequences, k)
            global_stationary = builder.compute_stationary_distribution(global_matrix)
            
            # Test conditional effects
            effect_results = self._test_conditional_effects(
                type2_df, test_sequences, k, global_stationary, min_effect_size
            )
            
            # Score this k value (combine clustering quality + conditional effects)
            effect_score = self._compute_k_score(effect_results)
            combined_score = 0.3 * silhouette + 0.7 * effect_score  # Weight conditional effects more

            results[k] = effect_results
            results[k]['score'] = combined_score
            results[k]['effect_score'] = effect_score
            results[k]['silhouette'] = silhouette

            print(f"    k={k}: {effect_results['significant_effects']}/{effect_results['total_effects']} "
                  f"significant ({effect_results['significance_rate']:.2%}), "
                  f"silhouette={silhouette:.3f}, combined_score={combined_score:.3f}")

            if combined_score > best_score:
                best_score = combined_score
                best_k = k
        
        return best_k, results

    def find_optimal_k_statistically_valid(self, embeddings: np.ndarray,
                                          criminal_sequences: Dict[str, List[int]],
                                          type2_df, k_range: Optional[range] = None,
                                          validation_split: float = 0.5,
                                          n_null_simulations: int = 50) -> Tuple[int, Dict[str, Any]]:
        """
        Statistically valid k optimization using split-sample validation.

        This method addresses multiple testing and validation concerns by:
        1. Splitting data into optimization and validation sets
        2. Optimizing k on optimization set only
        3. Validating on independent validation set
        4. Generating null distribution for comparison
        5. Applying multiple testing correction

        Args:
            embeddings: Event embeddings
            criminal_sequences: Criminal sequences
            type2_df: Type 2 demographic data
            k_range: Range of k values to test
            validation_split: Fraction of data for validation
            n_null_simulations: Number of null simulations

        Returns:
            Tuple of (optimal_k, comprehensive_results)
        """
        print("[INFO] Using statistically valid k optimization")
        print("[INFO] This addresses multiple testing and validation concerns")

        # Import the statistically valid optimizer
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        try:
            from statistically_valid_conditional_optimization import StatisticallyValidConditionalOptimizer

            optimizer = StatisticallyValidConditionalOptimizer(
                random_state=self.random_state,
                validation_split=validation_split
            )

            if k_range is None:
                n_samples = len(embeddings)
                k_max = min(10, int(np.sqrt(n_samples / 2)))
                k_range = list(range(3, k_max + 1))
            else:
                k_range = list(k_range)

            # Run statistically valid analysis
            results = optimizer.run_valid_exploratory_analysis(
                embeddings, criminal_sequences, type2_df,
                k_range=k_range, n_null_simulations=n_null_simulations
            )

            optimal_k = results['selected_k']

            print(f"[INFO] Statistically valid k optimization complete")
            print(f"[INFO] Selected k: {optimal_k}")
            print(f"[INFO] Validation results: {results['validation_results']['raw_significant_effects']} raw effects")
            print(f"[INFO] After correction: {results['validation_results']['bonferroni_significant']} Bonferroni, {results['validation_results']['fdr_significant']} FDR")

            return optimal_k, results

        except ImportError as e:
            print(f"[WARNING] Could not import statistical validation: {e}")
            print("[INFO] Falling back to standard optimization")
            return self.find_optimal_k_for_conditional_analysis(
                embeddings, criminal_sequences, type2_df, k_range
            )
    
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
        try:
            from markov.transition_analysis import TransitionMatrixBuilder
        except ImportError:
            # Fallback for relative imports
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from markov.transition_analysis import TransitionMatrixBuilder
        
        builder = TransitionMatrixBuilder()

        # Build global transition matrix (k×k) for comparison
        global_matrix = builder.build_conditional_markov(list(test_sequences.keys()), test_sequences, k)

        significant_effects = 0
        total_effects = 0
        effect_sizes = []
        detailed_effects = []

        # Test each heading in Type 2 data
        for heading in type2_df["Heading"].unique():
            # Get Type2DataProcessor at runtime if not available
            if Type2DataProcessor is None:
                try:
                    from data.loaders import Type2DataProcessor as T2DP
                    condition_map = T2DP.get_condition_map(type2_df, heading)
                except ImportError:
                    # Fallback implementation
                    condition_map = {}
                    for _, row in type2_df.iterrows():
                        if row["Heading"].strip().lower() == heading.strip().lower():
                            crim_id = str(row["CriminalID"])
                            val = str(row["Value"]).strip() if row["Value"] else "Unknown"
                            condition_map[crim_id] = val
            else:
                condition_map = Type2DataProcessor.get_condition_map(type2_df, heading)
            unique_values = set(condition_map.values())

            for val in unique_values:
                selected_ids = [cid for cid, v in condition_map.items() if v == val]
                if len(selected_ids) < MIN_CRIMINALS_FOR_ANALYSIS:
                    continue

                total_effects += 1

                # Build conditional transition matrix for this demographic group
                cond_matrix = builder.build_conditional_markov(selected_ids, test_sequences, k)

                # Compare TRANSITION MATRICES (not just stationary distributions)
                matrix_diff = np.linalg.norm(global_matrix - cond_matrix, 'fro')  # Frobenius norm

                # Statistical test for matrix differences
                p_value = self._test_matrix_difference(global_matrix, cond_matrix)

                effect_sizes.append(matrix_diff)

                # Record detailed effect
                detailed_effects.append({
                    'heading': heading,
                    'value': val,
                    'n_criminals': len(selected_ids),
                    'matrix_difference': float(matrix_diff),
                    'p_value': float(p_value),
                    'significant': matrix_diff >= min_effect_size and p_value < 0.05
                })

                if matrix_diff >= min_effect_size and p_value < 0.05:
                    significant_effects += 1
        
        # Compute summary statistics
        significance_rate = significant_effects / total_effects if total_effects > 0 else 0
        mean_effect_size = np.mean(effect_sizes) if effect_sizes else 0
        
        return {
            'significant_effects': significant_effects,
            'total_effects': total_effects,
            'significance_rate': significance_rate,
            'mean_effect_size': mean_effect_size,
            'effect_sizes': effect_sizes,
            'detailed_effects': detailed_effects
        }

    def _test_matrix_difference(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """
        Statistical test for difference between transition matrices.
        Uses chi-square test on the transition counts.
        """
        try:
            from scipy.stats import chi2_contingency

            # Convert probabilities back to counts for chi-square test
            counts1 = (matrix1 * 100).astype(int)
            counts2 = (matrix2 * 100).astype(int)

            # Flatten and combine
            flat1 = counts1.flatten()
            flat2 = counts2.flatten()

            # Remove zero entries
            non_zero_mask = (flat1 > 0) | (flat2 > 0)
            if np.sum(non_zero_mask) < 2:
                return 1.0

            # Chi-square test
            contingency = np.array([flat1[non_zero_mask], flat2[non_zero_mask]])
            _, p_value, _, _ = chi2_contingency(contingency)

            return p_value
        except:
            return 1.0
    
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

            try:
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

                # Ensure optimal_k is valid
                if optimal_k is None or optimal_k < 2:
                    print(f"[WARNING] Invalid optimal k ({optimal_k}), using default")
                    optimal_k = config.get('n_clusters', 5)

                results['optimal_k'] = optimal_k
                results['optimization_results'] = optimization_results

                print(f"[INFO] Selected k={optimal_k} for maximum conditional effects")

            except Exception as e:
                print(f"[WARNING] Conditional optimization failed: {e}")
                print("[INFO] Falling back to default k")
                results['optimal_k'] = config.get('n_clusters', 5)
                results['optimization_method'] = 'fallback_error'

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
        # Ensure optimal_k is valid
        if optimal_k is None or optimal_k < 2:
            print(f"[WARNING] Invalid optimal_k ({optimal_k}), using default k=5")
            optimal_k = 5

        # Ensure we don't have more clusters than data points
        max_k = min(optimal_k, len(embeddings) // 2)
        if max_k < 2:
            max_k = 2

        if max_k != optimal_k:
            print(f"[WARNING] Adjusted k from {optimal_k} to {max_k} due to data size")
            optimal_k = max_k

        try:
            if self.clusterer:
                labels, model = self.clusterer.kmeans_cluster(embeddings, n_clusters=optimal_k)
                metrics = self.clusterer.evaluate_clustering(embeddings, labels)
            else:
                # Fallback to direct sklearn
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                metrics = {'n_clusters': optimal_k, 'silhouette': 0.0}

            metrics['n_clusters'] = optimal_k
            return labels, metrics

        except Exception as e:
            print(f"[ERROR] Clustering failed: {e}")
            # Return dummy results
            labels = np.zeros(len(embeddings), dtype=int)
            metrics = {'n_clusters': 1, 'silhouette': 0.0, 'error': str(e)}
            return labels, metrics
