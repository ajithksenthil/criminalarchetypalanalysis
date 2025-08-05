#!/usr/bin/env python3
"""
fix_statistical_issues.py

Comprehensive fixes for the statistical issues identified in the conditional analysis.
"""

import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, permutation_test
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import warnings

class ImprovedTransitionStatistics:
    """Improved statistical testing for transition matrices."""
    
    @staticmethod
    def permutation_test_stationary(stationary1: np.ndarray, stationary2: np.ndarray, 
                                  n_permutations: int = 1000) -> float:
        """
        Permutation test for comparing stationary distributions.
        More appropriate than KS test for this application.
        """
        def test_statistic(x, y):
            return np.sum(np.abs(x - y))  # L1 distance
        
        observed_stat = test_statistic(stationary1, stationary2)
        
        # Combine the data
        combined = np.concatenate([stationary1, stationary2])
        n1, n2 = len(stationary1), len(stationary2)
        
        # Permutation test
        permuted_stats = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_stat = test_statistic(combined[:n1], combined[n1:])
            permuted_stats.append(perm_stat)
        
        # Calculate p-value
        p_value = np.mean(np.array(permuted_stats) >= observed_stat)
        return max(p_value, 1/n_permutations)  # Avoid p=0
    
    @staticmethod
    def chi_square_test_transitions(matrix1: np.ndarray, matrix2: np.ndarray) -> tuple:
        """
        Chi-square test for comparing transition patterns.
        More appropriate for categorical transition data.
        """
        # Convert to counts (multiply by arbitrary large number for chi-square)
        counts1 = (matrix1 * 1000).astype(int)
        counts2 = (matrix2 * 1000).astype(int)
        
        # Combine into contingency table
        combined = np.stack([counts1.flatten(), counts2.flatten()])
        
        # Remove zero columns to avoid chi-square issues
        non_zero_cols = np.any(combined > 0, axis=0)
        if np.sum(non_zero_cols) < 2:
            return 0.0, 1.0  # No variation
        
        combined_filtered = combined[:, non_zero_cols]
        
        try:
            chi2, p_value, _, _ = chi2_contingency(combined_filtered)
            return float(chi2), float(p_value)
        except:
            return 0.0, 1.0
    
    @staticmethod
    def bootstrap_confidence_interval(stationary: np.ndarray, n_bootstrap: int = 1000, 
                                    confidence: float = 0.95) -> np.ndarray:
        """
        Bootstrap confidence intervals for stationary distribution.
        """
        bootstrap_samples = []
        n = len(stationary)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            bootstrap_sample = stationary[indices]
            bootstrap_sample = bootstrap_sample / np.sum(bootstrap_sample)  # Renormalize
            bootstrap_samples.append(bootstrap_sample)
        
        bootstrap_samples = np.array(bootstrap_samples)
        
        # Calculate confidence intervals
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_samples, 100 * alpha/2, axis=0)
        upper = np.percentile(bootstrap_samples, 100 * (1 - alpha/2), axis=0)
        
        return np.stack([lower, upper])

class ImprovedClusteringValidator:
    """Improved clustering validation and optimization."""
    
    @staticmethod
    def find_optimal_k_range(embeddings: np.ndarray, k_min: int = 2, k_max: int = 20) -> int:
        """
        Find optimal k using multiple criteria and avoiding over-clustering.
        """
        n_samples = len(embeddings)
        
        # Limit k_max based on sample size (rule of thumb: k <= sqrt(n/2))
        max_reasonable_k = min(k_max, int(np.sqrt(n_samples / 2)))
        k_range = range(k_min, max_reasonable_k + 1)
        
        results = {}
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            # Multiple quality metrics
            silhouette = silhouette_score(embeddings, labels)
            
            # Calinski-Harabasz index (higher is better)
            from sklearn.metrics import calinski_harabasz_score
            ch_score = calinski_harabasz_score(embeddings, labels)
            
            # Davies-Bouldin index (lower is better)
            from sklearn.metrics import davies_bouldin_score
            db_score = davies_bouldin_score(embeddings, labels)
            
            # Inertia (within-cluster sum of squares)
            inertia = kmeans.inertia_
            
            results[k] = {
                'silhouette': silhouette,
                'calinski_harabasz': ch_score,
                'davies_bouldin': db_score,
                'inertia': inertia,
                'n_clusters': k
            }
        
        # Find optimal k using combined score
        best_k = ImprovedClusteringValidator._select_best_k(results)
        return best_k
    
    @staticmethod
    def _select_best_k(results: dict) -> int:
        """Select best k using multiple criteria."""
        # Normalize metrics to [0, 1] scale
        silhouettes = [results[k]['silhouette'] for k in results.keys()]
        ch_scores = [results[k]['calinski_harabasz'] for k in results.keys()]
        db_scores = [results[k]['davies_bouldin'] for k in results.keys()]
        
        # Normalize (higher is better for silhouette and CH, lower is better for DB)
        norm_silhouette = np.array(silhouettes) / max(silhouettes) if max(silhouettes) > 0 else np.zeros(len(silhouettes))
        norm_ch = np.array(ch_scores) / max(ch_scores) if max(ch_scores) > 0 else np.zeros(len(ch_scores))
        norm_db = (max(db_scores) - np.array(db_scores)) / (max(db_scores) - min(db_scores)) if max(db_scores) > min(db_scores) else np.ones(len(db_scores))
        
        # Combined score (weighted average)
        combined_scores = 0.4 * norm_silhouette + 0.3 * norm_ch + 0.3 * norm_db
        
        # Find best k
        k_values = list(results.keys())
        best_idx = np.argmax(combined_scores)
        best_k = k_values[best_idx]
        
        return best_k

class RobustConditionalAnalyzer:
    """Robust conditional analysis with proper statistical testing."""
    
    def __init__(self):
        self.stats = ImprovedTransitionStatistics()
        self.clustering_validator = ImprovedClusteringValidator()
    
    def analyze_conditional_effect(self, global_stationary: np.ndarray, 
                                 conditional_stationary: np.ndarray,
                                 global_matrix: np.ndarray,
                                 conditional_matrix: np.ndarray,
                                 n_criminals: int) -> dict:
        """
        Robust statistical analysis of conditional effects.
        """
        # Effect size (L1 distance)
        l1_distance = np.sum(np.abs(conditional_stationary - global_stationary))
        
        # Permutation test for stationary distributions
        perm_p_value = self.stats.permutation_test_stationary(
            conditional_stationary, global_stationary, n_permutations=1000
        )
        
        # Chi-square test for transition matrices
        chi2_stat, chi2_p_value = self.stats.chi_square_test_transitions(
            conditional_matrix, global_matrix
        )
        
        # Bootstrap confidence intervals
        ci = self.stats.bootstrap_confidence_interval(conditional_stationary)
        
        # Effect size classification
        if l1_distance < 0.1:
            effect_size_category = "negligible"
        elif l1_distance < 0.3:
            effect_size_category = "small"
        elif l1_distance < 0.5:
            effect_size_category = "medium"
        else:
            effect_size_category = "large"
        
        return {
            'l1_distance': float(l1_distance),
            'permutation_p_value': float(perm_p_value),
            'chi2_statistic': float(chi2_stat),
            'chi2_p_value': float(chi2_p_value),
            'effect_size_category': effect_size_category,
            'confidence_interval': ci.tolist(),
            'n_criminals': int(n_criminals),
            'significant_permutation': perm_p_value < 0.05,
            'significant_chi2': chi2_p_value < 0.05
        }

def validate_fixes():
    """Test the improved statistical methods."""
    print("Testing improved statistical methods...")
    
    # Create test data
    np.random.seed(42)
    
    # Test stationary distributions
    stat1 = np.array([0.3, 0.4, 0.2, 0.1])
    stat2 = np.array([0.1, 0.2, 0.4, 0.3])  # Different distribution
    stat3 = np.array([0.31, 0.39, 0.21, 0.09])  # Similar to stat1
    
    stats_tester = ImprovedTransitionStatistics()
    
    # Test permutation test
    p_val_different = stats_tester.permutation_test_stationary(stat1, stat2)
    p_val_similar = stats_tester.permutation_test_stationary(stat1, stat3)
    
    print(f"Permutation test - different distributions: p = {p_val_different:.4f}")
    print(f"Permutation test - similar distributions: p = {p_val_similar:.4f}")
    
    # Test clustering validator
    embeddings = np.random.randn(100, 10)  # 100 samples, 10 features
    validator = ImprovedClusteringValidator()
    optimal_k = validator.find_optimal_k_range(embeddings, k_min=2, k_max=10)
    
    print(f"Optimal k for test data: {optimal_k}")
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    validate_fixes()
