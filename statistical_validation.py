#!/usr/bin/env python3
"""
statistical_validation.py

Statistical validation methods for criminal archetypal analysis.
Includes permutation tests, bootstrap confidence intervals, and hypothesis testing.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.utils import resample
try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("[Warning] statsmodels not installed. Multiple testing correction will be unavailable.")
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, desc=None, total=None):
        return iterable
import warnings
warnings.filterwarnings('ignore')


class PermutationTest:
    """
    Permutation tests for clustering and transition matrix significance.
    """
    
    def __init__(self, n_permutations=1000, random_state=42):
        """
        Initialize permutation test.
        
        Args:
            n_permutations: Number of permutations
            random_state: Random seed for reproducibility
        """
        self.n_permutations = n_permutations
        self.random_state = random_state
        np.random.seed(random_state)
        
    def test_clustering_significance(self, X, labels):
        """
        Test if clustering structure is significant vs random.
        
        Args:
            X: Data matrix
            labels: Cluster labels
            
        Returns:
            Dictionary with test results
        """
        print(f"Running permutation test for clustering ({self.n_permutations} permutations)...")
        
        # Observed silhouette score
        observed_score = silhouette_score(X, labels)
        
        # Permutation scores
        permuted_scores = []
        
        for i in tqdm(range(self.n_permutations), desc="Permutations"):
            # Randomly shuffle labels
            permuted_labels = np.random.permutation(labels)
            
            # Calculate silhouette score
            perm_score = silhouette_score(X, permuted_labels)
            permuted_scores.append(perm_score)
        
        permuted_scores = np.array(permuted_scores)
        
        # Calculate p-value
        p_value = np.mean(permuted_scores >= observed_score)
        
        # Effect size (standardized difference)
        effect_size = (observed_score - np.mean(permuted_scores)) / np.std(permuted_scores)
        
        results = {
            'observed_score': observed_score,
            'permuted_mean': np.mean(permuted_scores),
            'permuted_std': np.std(permuted_scores),
            'p_value': p_value,
            'effect_size': effect_size,
            'permuted_scores': permuted_scores,
            'significant': p_value < 0.05
        }
        
        return results
    
    def test_transition_differences(self, matrix1, matrix2, labels1=None, labels2=None):
        """
        Test if two transition matrices are significantly different.
        
        Args:
            matrix1, matrix2: Transition matrices to compare
            labels1, labels2: Optional group labels
            
        Returns:
            Dictionary with test results
        """
        print("Testing transition matrix differences...")
        
        # Flatten matrices
        flat1 = matrix1.flatten()
        flat2 = matrix2.flatten()
        
        # Observed difference
        observed_diff = np.linalg.norm(flat1 - flat2)
        
        # Combine data
        combined = np.concatenate([flat1, flat2])
        n1 = len(flat1)
        
        # Permutation test
        permuted_diffs = []
        
        for _ in tqdm(range(self.n_permutations), desc="Permutations"):
            # Shuffle combined data
            shuffled = np.random.permutation(combined)
            
            # Split back
            perm1 = shuffled[:n1].reshape(matrix1.shape)
            perm2 = shuffled[n1:].reshape(matrix2.shape)
            
            # Calculate difference
            perm_diff = np.linalg.norm(perm1.flatten() - perm2.flatten())
            permuted_diffs.append(perm_diff)
        
        permuted_diffs = np.array(permuted_diffs)
        
        # P-value
        p_value = np.mean(permuted_diffs >= observed_diff)
        
        results = {
            'observed_difference': observed_diff,
            'permuted_mean': np.mean(permuted_diffs),
            'permuted_std': np.std(permuted_diffs),
            'p_value': p_value,
            'effect_size': (observed_diff - np.mean(permuted_diffs)) / np.std(permuted_diffs),
            'significant': p_value < 0.05
        }
        
        return results
    
    def test_pattern_significance(self, sequences, pattern, background_freq=None):
        """
        Test if a sequential pattern appears more often than expected.
        
        Args:
            sequences: List of sequences
            pattern: Pattern to test (list of events)
            background_freq: Expected frequency (if None, uses uniform)
            
        Returns:
            Dictionary with test results
        """
        # Count pattern occurrences
        observed_count = 0
        for seq in sequences:
            if self._contains_pattern(seq, pattern):
                observed_count += 1
        
        observed_freq = observed_count / len(sequences)
        
        # If no background frequency, assume uniform based on pattern length
        if background_freq is None:
            n_unique_events = len(set(event for seq in sequences for event in seq))
            background_freq = (1 / n_unique_events) ** len(pattern)
        
        # Binomial test - use binomtest for newer scipy
        try:
            # New API (scipy >= 1.7)
            result = stats.binomtest(observed_count, len(sequences), 
                                   background_freq, alternative='greater')
            p_value = result.pvalue
        except AttributeError:
            # Old API
            p_value = stats.binom_test(observed_count, len(sequences), 
                                     background_freq, alternative='greater')
        
        # Permutation test for more robust p-value
        permuted_counts = []
        
        for _ in range(self.n_permutations):
            # Shuffle events within each sequence
            perm_count = 0
            for seq in sequences:
                perm_seq = np.random.permutation(seq)
                if self._contains_pattern(perm_seq, pattern):
                    perm_count += 1
            permuted_counts.append(perm_count)
        
        permuted_counts = np.array(permuted_counts)
        perm_p_value = np.mean(permuted_counts >= observed_count)
        
        results = {
            'pattern': pattern,
            'observed_count': observed_count,
            'observed_frequency': observed_freq,
            'expected_frequency': background_freq,
            'binomial_p_value': p_value,
            'permutation_p_value': perm_p_value,
            'fold_enrichment': observed_freq / background_freq if background_freq > 0 else np.inf,
            'significant': perm_p_value < 0.05
        }
        
        return results
    
    def _contains_pattern(self, sequence, pattern):
        """Check if sequence contains pattern as subsequence."""
        pattern_idx = 0
        for event in sequence:
            if pattern_idx < len(pattern) and event == pattern[pattern_idx]:
                pattern_idx += 1
                if pattern_idx == len(pattern):
                    return True
        return False


class BootstrapValidation:
    """
    Bootstrap methods for computing confidence intervals.
    """
    
    def __init__(self, n_bootstrap=1000, confidence_level=0.95, random_state=42):
        """
        Initialize bootstrap validation.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            random_state: Random seed
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.random_state = random_state
        np.random.seed(random_state)
    
    def bootstrap_transition_matrix(self, sequences):
        """
        Compute bootstrap confidence intervals for transition probabilities.
        
        Args:
            sequences: List of sequences
            
        Returns:
            Dictionary with mean matrix and confidence intervals
        """
        print(f"Computing bootstrap confidence intervals ({self.n_bootstrap} samples)...")
        
        # Get unique states
        all_states = sorted(set(state for seq in sequences for state in seq))
        n_states = len(all_states)
        state_to_idx = {state: i for i, state in enumerate(all_states)}
        
        # Bootstrap samples
        bootstrap_matrices = []
        
        for _ in tqdm(range(self.n_bootstrap), desc="Bootstrap samples"):
            # Resample sequences with replacement
            boot_sequences = resample(sequences, n_samples=len(sequences))
            
            # Build transition matrix
            matrix = np.zeros((n_states, n_states))
            
            for seq in boot_sequences:
                for i in range(len(seq) - 1):
                    from_idx = state_to_idx[seq[i]]
                    to_idx = state_to_idx[seq[i + 1]]
                    matrix[from_idx, to_idx] += 1
            
            # Normalize rows
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            matrix = matrix / row_sums
            
            bootstrap_matrices.append(matrix)
        
        bootstrap_matrices = np.array(bootstrap_matrices)
        
        # Compute statistics
        mean_matrix = np.mean(bootstrap_matrices, axis=0)
        
        # Confidence intervals
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_matrices, lower_percentile, axis=0)
        ci_upper = np.percentile(bootstrap_matrices, upper_percentile, axis=0)
        
        # Standard errors
        se_matrix = np.std(bootstrap_matrices, axis=0)
        
        results = {
            'mean_matrix': mean_matrix,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'standard_errors': se_matrix,
            'states': all_states,
            'n_bootstrap': self.n_bootstrap,
            'confidence_level': self.confidence_level
        }
        
        return results
    
    def bootstrap_clustering_stability(self, X, clustering_func, **clustering_kwargs):
        """
        Assess clustering stability using bootstrap.
        
        Args:
            X: Data matrix
            clustering_func: Clustering function that returns labels
            **clustering_kwargs: Arguments for clustering function
            
        Returns:
            Dictionary with stability metrics
        """
        print("Assessing clustering stability via bootstrap...")
        
        n_samples = X.shape[0]
        
        # Original clustering
        original_labels = clustering_func(X, **clustering_kwargs)
        n_clusters = len(np.unique(original_labels))
        
        # Bootstrap
        agreement_scores = []
        silhouette_scores = []
        cluster_assignments = np.zeros((n_samples, self.n_bootstrap))
        
        for b in tqdm(range(self.n_bootstrap), desc="Bootstrap samples"):
            # Resample data
            indices = resample(range(n_samples), n_samples=n_samples)
            X_boot = X[indices]
            
            # Cluster bootstrap sample
            boot_labels = clustering_func(X_boot, **clustering_kwargs)
            
            # Map back to original indices
            full_labels = np.full(n_samples, -1)
            for i, idx in enumerate(indices):
                full_labels[idx] = boot_labels[i]
            
            # Store assignments
            cluster_assignments[:, b] = full_labels
            
            # Compute agreement with original (where both assigned)
            mask = full_labels != -1
            if np.sum(mask) > 0:
                ari = adjusted_rand_score(original_labels[mask], full_labels[mask])
                agreement_scores.append(ari)
            
            # Silhouette score
            try:
                sil_score = silhouette_score(X_boot, boot_labels)
                silhouette_scores.append(sil_score)
            except:
                pass
        
        # Compute stability for each sample
        sample_stability = []
        for i in range(n_samples):
            assignments = cluster_assignments[i, :]
            valid_assignments = assignments[assignments != -1]
            
            if len(valid_assignments) > 0:
                # Frequency of most common assignment
                mode_freq = np.max(np.bincount(valid_assignments.astype(int))) / len(valid_assignments)
                sample_stability.append(mode_freq)
            else:
                sample_stability.append(0)
        
        results = {
            'mean_agreement': np.mean(agreement_scores),
            'ci_agreement': np.percentile(agreement_scores, [2.5, 97.5]),
            'mean_silhouette': np.mean(silhouette_scores),
            'ci_silhouette': np.percentile(silhouette_scores, [2.5, 97.5]),
            'sample_stability': np.array(sample_stability),
            'overall_stability': np.mean(sample_stability),
            'unstable_samples': np.where(np.array(sample_stability) < 0.5)[0]
        }
        
        return results
    
    def bootstrap_metric(self, data, metric_func, **kwargs):
        """
        Generic bootstrap for any metric.
        
        Args:
            data: Input data
            metric_func: Function to compute metric
            **kwargs: Additional arguments for metric_func
            
        Returns:
            Dictionary with bootstrap results
        """
        bootstrap_values = []
        
        for _ in range(self.n_bootstrap):
            # Resample
            boot_data = resample(data, n_samples=len(data))
            
            # Compute metric
            value = metric_func(boot_data, **kwargs)
            bootstrap_values.append(value)
        
        bootstrap_values = np.array(bootstrap_values)
        
        results = {
            'mean': np.mean(bootstrap_values),
            'std': np.std(bootstrap_values),
            'ci_lower': np.percentile(bootstrap_values, (self.alpha/2) * 100),
            'ci_upper': np.percentile(bootstrap_values, (1-self.alpha/2) * 100),
            'values': bootstrap_values
        }
        
        return results


class MultipleTestingCorrection:
    """
    Handle multiple testing corrections for large-scale analyses.
    """
    
    @staticmethod
    def correct_pvalues(p_values, method='fdr_bh', alpha=0.05):
        """
        Apply multiple testing correction.
        
        Args:
            p_values: Array of p-values
            method: Correction method ('bonferroni', 'fdr_bh', 'fdr_by', etc.)
            alpha: Significance level
            
        Returns:
            Dictionary with corrected results
        """
        if not STATSMODELS_AVAILABLE:
            # Simple Bonferroni correction as fallback
            p_corrected = np.minimum(np.array(p_values) * len(p_values), 1.0)
            rejected = p_corrected < alpha
            
            return {
                'original_pvalues': p_values,
                'corrected_pvalues': p_corrected,
                'rejected': rejected,
                'n_significant_original': np.sum(p_values < alpha),
                'n_significant_corrected': np.sum(rejected),
                'correction_method': 'bonferroni (fallback)',
                'alpha': alpha
            }
        
        # Apply correction
        rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
            p_values, alpha=alpha, method=method
        )
        
        results = {
            'original_pvalues': p_values,
            'corrected_pvalues': p_corrected,
            'rejected': rejected,
            'n_significant_original': np.sum(p_values < alpha),
            'n_significant_corrected': np.sum(rejected),
            'correction_method': method,
            'alpha': alpha
        }
        
        return results
    
    @staticmethod
    def summarize_discoveries(results_dict, p_value_key='p_value', 
                            effect_key='effect_size', name_key='name'):
        """
        Summarize discoveries from multiple tests.
        
        Args:
            results_dict: Dictionary of test results
            p_value_key: Key for p-values in results
            effect_key: Key for effect sizes
            name_key: Key for test names
            
        Returns:
            Summary DataFrame
        """
        import pandas as pd
        
        # Extract p-values and effects
        tests = []
        for test_name, result in results_dict.items():
            tests.append({
                'test': test_name,
                'p_value': result.get(p_value_key, 1.0),
                'effect_size': result.get(effect_key, 0.0),
                'significant_uncorrected': result.get(p_value_key, 1.0) < 0.05
            })
        
        df = pd.DataFrame(tests)
        
        # Apply FDR correction
        if len(df) > 0:
            correction = MultipleTestingCorrection.correct_pvalues(
                df['p_value'].values, method='fdr_bh'
            )
            
            df['p_value_corrected'] = correction['corrected_pvalues']
            df['significant_corrected'] = correction['rejected']
        
        return df.sort_values('p_value')


def plot_validation_results(validation_results, save_path='validation_plots.png'):
    """
    Create visualization of validation results.
    
    Args:
        validation_results: Dictionary with validation results
        save_path: Path to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Permutation test histogram
    if 'permutation_test' in validation_results:
        ax = axes[0, 0]
        perm_results = validation_results['permutation_test']
        
        ax.hist(perm_results['permuted_scores'], bins=30, alpha=0.7, 
                label='Permuted', color='gray')
        ax.axvline(perm_results['observed_score'], color='red', 
                  linestyle='--', linewidth=2, label='Observed')
        ax.set_xlabel('Silhouette Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Permutation Test (p={perm_results["p_value"]:.3f})')
        ax.legend()
    
    # 2. Bootstrap confidence intervals
    if 'bootstrap_stability' in validation_results:
        ax = axes[0, 1]
        boot_results = validation_results['bootstrap_stability']
        
        # Plot sample stability distribution
        ax.hist(boot_results['sample_stability'], bins=20, alpha=0.7, color='blue')
        ax.axvline(0.5, color='red', linestyle='--', label='Stability threshold')
        ax.set_xlabel('Sample Stability')
        ax.set_ylabel('Number of Samples')
        ax.set_title(f'Clustering Stability (mean={boot_results["overall_stability"]:.3f})')
        ax.legend()
    
    # 3. Transition matrix confidence
    if 'transition_confidence' in validation_results:
        ax = axes[1, 0]
        trans_results = validation_results['transition_confidence']
        
        # Plot mean transition matrix
        im = ax.imshow(trans_results['mean_matrix'], cmap='Blues', aspect='auto')
        ax.set_title('Mean Transition Probabilities')
        ax.set_xlabel('To State')
        ax.set_ylabel('From State')
        plt.colorbar(im, ax=ax)
    
    # 4. Multiple testing summary
    if 'multiple_testing' in validation_results:
        ax = axes[1, 1]
        mt_results = validation_results['multiple_testing']
        
        # Bar plot of significant tests
        categories = ['Original', 'Corrected']
        counts = [mt_results['n_significant_original'], 
                 mt_results['n_significant_corrected']]
        
        ax.bar(categories, counts, color=['lightblue', 'darkblue'])
        ax.set_ylabel('Number of Significant Tests')
        ax.set_title(f'Multiple Testing Correction ({mt_results["correction_method"]})')
        
        # Add text
        for i, count in enumerate(counts):
            ax.text(i, count + 0.5, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Validation plots saved to {save_path}")


if __name__ == "__main__":
    # Test statistical validation methods
    print("Testing Statistical Validation Methods...")
    
    # Generate sample data
    from sklearn.datasets import make_blobs
    X, true_labels = make_blobs(n_samples=200, n_features=10, 
                               centers=3, random_state=42)
    
    # 1. Permutation test for clustering
    print("\n1. Permutation Test for Clustering")
    perm_test = PermutationTest(n_permutations=100)
    perm_results = perm_test.test_clustering_significance(X, true_labels)
    
    print(f"Observed silhouette: {perm_results['observed_score']:.3f}")
    print(f"Permuted mean: {perm_results['permuted_mean']:.3f}")
    print(f"P-value: {perm_results['p_value']:.4f}")
    print(f"Significant: {perm_results['significant']}")
    
    # 2. Bootstrap validation
    print("\n2. Bootstrap Validation")
    
    # Create dummy sequences
    sequences = [
        ['A', 'B', 'C', 'D'],
        ['A', 'C', 'B', 'D'],
        ['B', 'A', 'C', 'D'],
        ['A', 'B', 'D', 'C']
    ] * 10
    
    bootstrap = BootstrapValidation(n_bootstrap=100)
    trans_results = bootstrap.bootstrap_transition_matrix(sequences)
    
    print(f"Transition matrix shape: {trans_results['mean_matrix'].shape}")
    print(f"States: {trans_results['states']}")
    
    # 3. Multiple testing correction
    print("\n3. Multiple Testing Correction")
    
    # Simulate multiple p-values
    p_values = np.random.uniform(0, 1, 20)
    p_values[:5] = np.random.uniform(0, 0.03, 5)  # Some significant
    
    mt_results = MultipleTestingCorrection.correct_pvalues(p_values)
    print(f"Significant before correction: {mt_results['n_significant_original']}")
    print(f"Significant after correction: {mt_results['n_significant_corrected']}")
    
    print("\nStatistical validation tests complete!")