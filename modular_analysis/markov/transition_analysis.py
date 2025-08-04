#!/usr/bin/env python3
"""
transition_analysis.py

Markov chain transition analysis for criminal behavior patterns.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy.stats import wasserstein_distance, ks_2samp

from ..core.config import MIN_CRIMINALS_FOR_ANALYSIS
from ..utils.helpers import safe_filename

class TransitionMatrixBuilder:
    """Build and analyze transition matrices."""
    
    def build_global_transition_matrix(self, criminal_sequences: Dict[str, List[int]], 
                                     n_clusters: int) -> np.ndarray:
        """
        Build global transition matrix from all criminal sequences.
        
        Args:
            criminal_sequences: Dictionary mapping criminal IDs to cluster sequences
            n_clusters: Number of clusters
            
        Returns:
            Row-normalized transition matrix
        """
        matrix = np.zeros((n_clusters, n_clusters))
        
        for seq in criminal_sequences.values():
            if len(seq) < 2:
                continue
            for s1, s2 in zip(seq[:-1], seq[1:]):
                matrix[s1, s2] += 1
        
        return self._normalize_transition_matrix(matrix, n_clusters)
    
    def build_conditional_markov(self, selected_criminal_ids: List[str], 
                               criminal_sequences: Dict[str, List[int]], 
                               n_clusters: int) -> np.ndarray:
        """
        Build conditional Markov transition matrix for a subset of criminals.
        
        Args:
            selected_criminal_ids: Criminal IDs for the subgroup
            criminal_sequences: Mapping from CriminalID to cluster sequences
            n_clusters: Number of clusters
            
        Returns:
            Row-normalized transition matrix
        """
        matrix = np.zeros((n_clusters, n_clusters))
        
        for cid in selected_criminal_ids:
            seq = criminal_sequences.get(cid, [])
            if len(seq) < 2:
                continue
            for s1, s2 in zip(seq[:-1], seq[1:]):
                matrix[s1, s2] += 1
        
        return self._normalize_transition_matrix(matrix, n_clusters)
    
    def _normalize_transition_matrix(self, matrix: np.ndarray, n_clusters: int) -> np.ndarray:
        """Normalize transition matrix rows to probabilities."""
        row_sums = matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            matrix = np.divide(matrix, row_sums, where=row_sums != 0)
            # Handle zero rows by setting uniform distribution
            for i in range(n_clusters):
                if row_sums[i] == 0:
                    matrix[i] = 1.0 / n_clusters
        return matrix
    
    def compute_stationary_distribution(self, transition_matrix: np.ndarray) -> np.ndarray:
        """
        Compute stationary distribution by finding eigenvector with eigenvalue=1.
        
        Args:
            transition_matrix: Transition matrix
            
        Returns:
            Stationary distribution
        """
        eigvals, eigvecs = np.linalg.eig(transition_matrix.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        stat_dist = eigvecs[:, idx].real
        stat_dist = np.abs(stat_dist)  # Ensure non-negative
        stat_dist /= stat_dist.sum()   # Normalize
        return stat_dist

class TransitionStatistics:
    """Compute statistical measures for transition matrices."""
    
    @staticmethod
    def compute_transition_statistics(matrix1: np.ndarray, matrix2: np.ndarray, 
                                    stationary1: np.ndarray, stationary2: np.ndarray) -> Dict[str, float]:
        """
        Compute statistical measures comparing two transition matrices.
        
        Args:
            matrix1: First transition matrix
            matrix2: Second transition matrix
            stationary1: First stationary distribution
            stationary2: Second stationary distribution
            
        Returns:
            Dictionary of statistical test results
        """
        # Flatten matrices for comparison
        flat1 = matrix1.flatten()
        flat2 = matrix2.flatten()
        
        # Wasserstein distance between stationary distributions
        wasserstein = wasserstein_distance(stationary1, stationary2)
        
        # KS test on the flattened transition probabilities
        ks_stat, ks_pvalue = ks_2samp(flat1, flat2)
        
        # Frobenius norm of the difference
        frobenius = np.linalg.norm(matrix1 - matrix2, 'fro')
        
        # Total variation distance between stationary distributions
        tv_distance = 0.5 * np.sum(np.abs(stationary1 - stationary2))
        
        # L1 distance between stationary distributions
        l1_distance = np.sum(np.abs(stationary1 - stationary2))
        
        return {
            'wasserstein_distance': float(wasserstein),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'frobenius_norm': float(frobenius),
            'tv_distance': float(tv_distance),
            'l1_distance': float(l1_distance)
        }
    
    @staticmethod
    def transition_entropy(transition_matrix: np.ndarray) -> float:
        """
        Compute entropy of transition matrix.
        
        Args:
            transition_matrix: Transition matrix
            
        Returns:
            Entropy value
        """
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        matrix_safe = transition_matrix + epsilon
        
        # Compute entropy for each row and take average
        row_entropies = -np.sum(matrix_safe * np.log2(matrix_safe), axis=1)
        return float(np.mean(row_entropies))

class ConditionalAnalyzer:
    """Analyze conditional patterns in transition matrices."""
    
    def __init__(self):
        self.builder = TransitionMatrixBuilder()
        self.stats = TransitionStatistics()
    
    def analyze_all_conditional_insights(self, type2_df, criminal_sequences: Dict[str, List[int]], 
                                       n_clusters: int, global_stationary: np.ndarray, 
                                       global_transition_matrix: np.ndarray, 
                                       diff_threshold: float = 0.1) -> Dict[str, Dict[str, Any]]:
        """
        Analyze conditional insights for all headings in Type 2 data.
        
        Args:
            type2_df: Type 2 DataFrame
            criminal_sequences: Criminal sequences
            n_clusters: Number of clusters
            global_stationary: Global stationary distribution
            global_transition_matrix: Global transition matrix
            diff_threshold: Threshold for significant differences
            
        Returns:
            Dictionary of insights
        """
        from ..data.loaders import Type2DataProcessor
        
        insights = {}
        headings = type2_df["Heading"].unique()
        
        for heading in headings:
            print(f"[INFO] Processing conditional analysis for heading: {heading}")
            
            # Get condition mapping
            condition_map = Type2DataProcessor.get_condition_map(type2_df, heading)
            unique_values = set(condition_map.values())
            
            for val in unique_values:
                selected_ids = [cid for cid, v in condition_map.items() if v == val]
                
                if len(selected_ids) < MIN_CRIMINALS_FOR_ANALYSIS:
                    print(f"[INFO] Skipping {heading} = {val} due to insufficient criminals (n={len(selected_ids)})")
                    continue
                
                # Build conditional matrix
                matrix = self.builder.build_conditional_markov(selected_ids, criminal_sequences, n_clusters)
                stationary_cond = self.builder.compute_stationary_distribution(matrix)
                
                # Compute differences
                l1_diff = np.sum(np.abs(stationary_cond - global_stationary))
                stats = self.stats.compute_transition_statistics(
                    matrix, global_transition_matrix, stationary_cond, global_stationary
                )
                
                # Record insight if significant
                if l1_diff > diff_threshold or stats['ks_pvalue'] < 0.05:
                    safe_heading = safe_filename(heading)
                    safe_val = safe_filename(str(val))
                    key = f"{safe_heading}={safe_val}"
                    
                    insights[key] = {
                        "heading": heading,
                        "value": val,
                        "n_criminals": len(selected_ids),
                        "stationary_cond": stationary_cond.tolist(),
                        "global_stationary": global_stationary.tolist(),
                        "l1_difference": l1_diff,
                        "statistics": stats,
                        "significant": stats['ks_pvalue'] < 0.05
                    }
                    
                    significance_str = " (SIGNIFICANT)" if stats['ks_pvalue'] < 0.05 else ""
                    print(f"[INSIGHT] {heading} = {val}, L1 diff={l1_diff:.3f}, "
                          f"KS p-value={stats['ks_pvalue']:.4f}{significance_str} (n={len(selected_ids)})")
        
        return insights
    
    def run_conditional_markov_analysis(self, type2_df, criminal_sequences: Dict[str, List[int]], 
                                      n_clusters: int, output_dir: str) -> None:
        """
        Run conditional Markov analysis and save transition diagrams.
        
        Args:
            type2_df: Type 2 DataFrame
            criminal_sequences: Criminal sequences
            n_clusters: Number of clusters
            output_dir: Output directory for diagrams
        """
        from ..data.loaders import Type2DataProcessor
        from ..visualization.diagrams import TransitionDiagramGenerator
        
        diagram_generator = TransitionDiagramGenerator()
        headings = type2_df["Heading"].unique()
        
        for heading in headings:
            print(f"[INFO] Processing conditional analysis for heading: {heading}")
            
            condition_map = Type2DataProcessor.get_condition_map(type2_df, heading)
            unique_values = set(condition_map.values())
            
            for val in unique_values:
                selected_ids = [cid for cid, v in condition_map.items() if v == val]
                
                if len(selected_ids) < MIN_CRIMINALS_FOR_ANALYSIS:
                    print(f"[INFO] Skipping {heading} = {val} due to insufficient criminals (n={len(selected_ids)})")
                    continue
                
                # Build conditional matrix
                matrix = self.builder.build_conditional_markov(selected_ids, criminal_sequences, n_clusters)
                
                # Generate diagram
                safe_heading = safe_filename(heading)
                safe_val = safe_filename(str(val))
                filename = f"state_transition_{safe_heading}_{safe_val}.png"
                output_path = f"{output_dir}/{filename}"
                
                try:
                    diagram_generator.plot_state_transition_diagram(matrix, output_path)
                    print(f"[INFO] Conditional Markov chain for {heading} = {val} saved to {output_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to save diagram for {heading} = {val}: {e}")

class CriminalTransitionAnalyzer:
    """Analyze transition patterns at the criminal level."""
    
    def __init__(self):
        self.builder = TransitionMatrixBuilder()
    
    def cluster_criminals_by_transition_patterns(self, criminal_sequences: Dict[str, List[int]], 
                                               n_clusters: int, n_criminal_clusters: int = 3) -> Dict[str, Any]:
        """
        Cluster criminals based on their individual transition matrices.
        
        Args:
            criminal_sequences: Criminal sequences
            n_clusters: Number of event clusters
            n_criminal_clusters: Number of criminal clusters
            
        Returns:
            Clustering results
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # Build individual transition matrices
        criminal_matrices = {}
        criminal_ids = []
        
        for crim_id, seq in criminal_sequences.items():
            if len(seq) < 2:
                continue
            
            matrix = self.builder.build_conditional_markov([crim_id], {crim_id: seq}, n_clusters)
            criminal_matrices[crim_id] = matrix
            criminal_ids.append(crim_id)
        
        if len(criminal_matrices) < 2:
            print("[WARNING] Not enough criminals with sufficient data for clustering")
            return {}
        
        # Flatten matrices for clustering
        matrix_vectors = np.array([criminal_matrices[cid].flatten() for cid in criminal_ids])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_criminal_clusters, random_state=42, n_init=10)
        criminal_clusters = kmeans.fit_predict(matrix_vectors)
        
        # Compute metrics
        silhouette = silhouette_score(matrix_vectors, criminal_clusters)
        
        # Create results
        results = {
            'criminal_clusters': {criminal_ids[i]: int(criminal_clusters[i]) 
                                 for i in range(len(criminal_ids))},
            'cluster_sizes': {i: int(np.sum(criminal_clusters == i)) 
                             for i in range(n_criminal_clusters)},
            'silhouette_score': float(silhouette),
            'n_criminals': len(criminal_ids)
        }
        
        print(f"[INFO] Criminal clustering complete. Silhouette score: {silhouette:.3f}")
        
        return results

# Backward compatibility aliases
def build_conditional_markov(selected_criminal_ids: List[str],
                           criminal_sequences: Dict[str, List[int]],
                           n_clusters: int) -> np.ndarray:
    """Backward compatibility alias."""
    builder = TransitionMatrixBuilder()
    return builder.build_conditional_markov(selected_criminal_ids, criminal_sequences, n_clusters)

def compute_stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    """Backward compatibility alias."""
    builder = TransitionMatrixBuilder()
    return builder.compute_stationary_distribution(transition_matrix)

def transition_entropy(transition_matrix: np.ndarray) -> float:
    """Backward compatibility alias."""
    stats = TransitionStatistics()
    return stats.transition_entropy(transition_matrix)

def analyze_all_conditional_insights(type2_df, criminal_sequences: Dict[str, List[int]],
                                   n_clusters: int, output_dir: str, global_stationary: np.ndarray,
                                   global_transition_matrix: np.ndarray,
                                   diff_threshold: float = 0.1) -> Dict[str, Dict[str, Any]]:
    """Backward compatibility alias."""
    analyzer = ConditionalAnalyzer()
    return analyzer.analyze_all_conditional_insights(
        type2_df, criminal_sequences, n_clusters, global_stationary,
        global_transition_matrix, diff_threshold
    )

def run_all_conditional_markov_analysis(type2_df, criminal_sequences: Dict[str, List[int]],
                                      n_clusters: int, output_dir: str) -> None:
    """Backward compatibility alias."""
    analyzer = ConditionalAnalyzer()
    analyzer.run_conditional_markov_analysis(type2_df, criminal_sequences, n_clusters, output_dir)

def cluster_criminals_by_transition_patterns(criminal_sequences: Dict[str, List[int]],
                                           n_clusters: int, output_dir: str,
                                           n_criminal_clusters: int = 3) -> Dict[str, Any]:
    """Backward compatibility alias."""
    analyzer = CriminalTransitionAnalyzer()
    return analyzer.cluster_criminals_by_transition_patterns(
        criminal_sequences, n_clusters, n_criminal_clusters
    )
