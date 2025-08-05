#!/usr/bin/env python3
"""
statistically_valid_conditional_optimization.py

Statistically valid implementation of conditional k optimization for exploratory analysis.
Addresses multiple testing, validation, and proper statistical reporting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency
import json
import warnings

class StatisticallyValidConditionalOptimizer:
    """
    Statistically valid conditional k optimization for exploratory analysis.
    
    Key features:
    1. Split-sample validation (k selection vs. hypothesis testing)
    2. Proper multiple testing correction
    3. Null distribution estimation
    4. Transparent reporting of exploratory nature
    """
    
    def __init__(self, random_state: int = 42, validation_split: float = 0.5):
        self.random_state = random_state
        self.validation_split = validation_split
        self.optimization_results = {}
        self.validation_results = {}
        self.null_distribution = {}
        
        np.random.seed(random_state)
    
    def run_valid_exploratory_analysis(self, embeddings: np.ndarray, 
                                     criminal_sequences: Dict[str, List[int]], 
                                     type2_df, k_range: List[int] = None,
                                     n_null_simulations: int = 100) -> Dict[str, Any]:
        """
        Run statistically valid exploratory conditional analysis.
        
        Steps:
        1. Split data into optimization and validation sets
        2. Optimize k on optimization set only
        3. Validate selected k on validation set
        4. Generate null distribution for comparison
        5. Apply proper multiple testing correction
        6. Report with appropriate caveats
        """
        print("="*80)
        print("STATISTICALLY VALID EXPLORATORY CONDITIONAL ANALYSIS")
        print("="*80)
        
        # Step 1: Split data for unbiased validation
        print(f"\n[STEP 1] Splitting data (validation_split={self.validation_split})")
        opt_data, val_data = self._split_data_for_validation(
            embeddings, criminal_sequences, type2_df
        )
        
        print(f"Optimization set: {len(opt_data['embeddings'])} events, {len(opt_data['sequences'])} criminals")
        print(f"Validation set: {len(val_data['embeddings'])} events, {len(val_data['sequences'])} criminals")
        
        # Step 2: Optimize k using only optimization data
        print(f"\n[STEP 2] K optimization on optimization set only")
        optimal_k, optimization_results = self._optimize_k_on_subset(
            opt_data['embeddings'], opt_data['sequences'], opt_data['type2_df'], k_range
        )
        
        self.optimization_results = optimization_results
        print(f"Selected k: {optimal_k} (based on optimization set only)")
        
        # Step 3: Validate on independent validation set
        print(f"\n[STEP 3] Validation on independent validation set")
        validation_results = self._validate_k_on_independent_data(
            val_data['embeddings'], val_data['sequences'], val_data['type2_df'], optimal_k
        )
        
        self.validation_results = validation_results
        
        # Step 4: Generate null distribution
        print(f"\n[STEP 4] Generating null distribution ({n_null_simulations} simulations)")
        null_results = self._generate_null_distribution(
            val_data['embeddings'], val_data['sequences'], val_data['type2_df'], 
            optimal_k, n_null_simulations
        )
        
        self.null_distribution = null_results
        
        # Step 5: Apply multiple testing correction
        print(f"\n[STEP 5] Applying multiple testing correction")
        corrected_results = self._apply_multiple_testing_correction(validation_results)
        
        # Step 6: Generate comprehensive report
        print(f"\n[STEP 6] Generating statistical validity report")
        final_report = self._generate_validity_report(
            optimal_k, optimization_results, validation_results, 
            corrected_results, null_results
        )
        
        return final_report
    
    def _split_data_for_validation(self, embeddings: np.ndarray, 
                                 criminal_sequences: Dict[str, List[int]], 
                                 type2_df) -> Tuple[Dict, Dict]:
        """Split data into optimization and validation sets."""
        
        # Split criminals (not events) to maintain sequence integrity
        criminal_ids = list(criminal_sequences.keys())
        opt_criminals, val_criminals = train_test_split(
            criminal_ids, test_size=self.validation_split, 
            random_state=self.random_state
        )
        
        # Split embeddings based on criminal assignment
        opt_indices = []
        val_indices = []
        event_idx = 0
        
        for crim_id, events in criminal_sequences.items():
            if crim_id in opt_criminals:
                opt_indices.extend(range(event_idx, event_idx + len(events)))
            else:
                val_indices.extend(range(event_idx, event_idx + len(events)))
            event_idx += len(events)
        
        # Create optimization data
        opt_data = {
            'embeddings': embeddings[opt_indices],
            'sequences': {cid: seq for cid, seq in criminal_sequences.items() if cid in opt_criminals},
            'type2_df': type2_df[type2_df['CriminalID'].isin(opt_criminals)]
        }
        
        # Create validation data
        val_data = {
            'embeddings': embeddings[val_indices],
            'sequences': {cid: seq for cid, seq in criminal_sequences.items() if cid in val_criminals},
            'type2_df': type2_df[type2_df['CriminalID'].isin(val_criminals)]
        }
        
        return opt_data, val_data
    
    def _optimize_k_on_subset(self, embeddings: np.ndarray, 
                            criminal_sequences: Dict[str, List[int]], 
                            type2_df, k_range: List[int] = None) -> Tuple[int, Dict]:
        """Optimize k using only the optimization subset."""
        
        if k_range is None:
            n_samples = len(embeddings)
            k_max = min(10, int(np.sqrt(n_samples / 2)))  # Conservative range
            k_range = list(range(3, k_max + 1))
        
        print(f"Testing k values: {k_range}")
        
        results = {}
        best_k = k_range[0]
        best_score = -1
        
        for k in k_range:
            print(f"  Optimizing k={k}...")
            
            # Cluster with this k
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            # Evaluate clustering quality
            try:
                silhouette = silhouette_score(embeddings, labels)
            except:
                silhouette = 0.0
            
            # Skip poor clustering
            if silhouette < 0.1:
                print(f"    Skipping k={k} due to poor clustering (silhouette={silhouette:.3f})")
                results[k] = {'score': 0.0, 'silhouette': silhouette, 'skipped': True}
                continue
            
            # Build test sequences
            test_sequences = self._build_test_sequences(criminal_sequences, labels)
            
            # Evaluate conditional effects (on optimization set only)
            effect_score = self._evaluate_conditional_effects_fast(
                test_sequences, type2_df, k
            )
            
            # Combined score
            combined_score = 0.3 * silhouette + 0.7 * effect_score
            
            results[k] = {
                'score': combined_score,
                'silhouette': silhouette,
                'effect_score': effect_score,
                'n_tests': len(type2_df["Heading"].unique())
            }
            
            print(f"    k={k}: combined_score={combined_score:.3f}, silhouette={silhouette:.3f}")
            
            if combined_score > best_score:
                best_score = combined_score
                best_k = k
        
        print(f"Optimization complete. Selected k={best_k} (score={best_score:.3f})")
        return best_k, results
    
    def _validate_k_on_independent_data(self, embeddings: np.ndarray, 
                                      criminal_sequences: Dict[str, List[int]], 
                                      type2_df, k: int) -> Dict[str, Any]:
        """Validate selected k on completely independent validation data."""
        
        print(f"Validating k={k} on independent validation set...")
        
        # Cluster validation data with selected k
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Build sequences
        test_sequences = self._build_test_sequences(criminal_sequences, labels)
        
        # Comprehensive conditional analysis on validation data
        validation_results = self._comprehensive_conditional_analysis(
            test_sequences, type2_df, k
        )
        
        print(f"Validation complete: {validation_results['significant_effects']} significant effects found")
        
        return validation_results
    
    def _generate_null_distribution(self, embeddings: np.ndarray, 
                                  criminal_sequences: Dict[str, List[int]], 
                                  type2_df, k: int, n_simulations: int = 100) -> Dict[str, Any]:
        """Generate null distribution by permuting demographic labels."""
        
        print(f"Generating null distribution with {n_simulations} permutations...")
        
        null_significant_counts = []
        null_effect_sizes = []
        
        # Get original demographic data
        original_type2 = type2_df.copy()
        
        for sim in range(n_simulations):
            if sim % 20 == 0:
                print(f"  Simulation {sim+1}/{n_simulations}")
            
            # Permute demographic labels while preserving structure
            permuted_type2 = self._permute_demographics(original_type2)
            
            # Cluster with selected k
            kmeans = KMeans(n_clusters=k, random_state=self.random_state + sim, n_init=5)
            labels = kmeans.fit_predict(embeddings)
            
            # Build sequences
            test_sequences = self._build_test_sequences(criminal_sequences, labels)
            
            # Test conditional effects on permuted data
            null_results = self._comprehensive_conditional_analysis(
                test_sequences, permuted_type2, k
            )
            
            null_significant_counts.append(null_results['significant_effects'])
            null_effect_sizes.extend(null_results['effect_sizes'])
        
        # Compute null distribution statistics
        null_stats = {
            'mean_significant_effects': np.mean(null_significant_counts),
            'std_significant_effects': np.std(null_significant_counts),
            'max_significant_effects': np.max(null_significant_counts),
            'percentile_95': np.percentile(null_significant_counts, 95),
            'percentile_99': np.percentile(null_significant_counts, 99),
            'mean_effect_size': np.mean(null_effect_sizes),
            'percentile_95_effect_size': np.percentile(null_effect_sizes, 95),
            'all_significant_counts': null_significant_counts
        }
        
        print(f"Null distribution: mean={null_stats['mean_significant_effects']:.1f}, "
              f"95th percentile={null_stats['percentile_95']:.1f}")
        
        return null_stats
    
    def _apply_multiple_testing_correction(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply multiple testing correction to validation results."""
        
        n_tests = validation_results['total_effects']
        
        # Bonferroni correction
        bonferroni_alpha = 0.05 / n_tests
        
        # Benjamini-Hochberg (FDR) correction
        p_values = [effect['p_value'] for effect in validation_results['detailed_effects']]
        fdr_significant = self._benjamini_hochberg_correction(p_values, 0.05)
        
        corrected_results = validation_results.copy()
        corrected_results['bonferroni_alpha'] = bonferroni_alpha
        corrected_results['bonferroni_significant'] = sum(
            1 for effect in validation_results['detailed_effects'] 
            if effect['p_value'] < bonferroni_alpha
        )
        corrected_results['fdr_significant'] = fdr_significant
        
        print(f"Multiple testing correction:")
        print(f"  Original significant: {validation_results['significant_effects']}")
        print(f"  Bonferroni significant: {corrected_results['bonferroni_significant']}")
        print(f"  FDR significant: {corrected_results['fdr_significant']}")
        
        return corrected_results

    def _build_test_sequences(self, criminal_sequences: Dict[str, List[int]],
                            labels: np.ndarray) -> Dict[str, List[int]]:
        """Build test sequences with new cluster labels."""
        test_sequences = {}
        event_idx = 0

        for crim_id, events in criminal_sequences.items():
            test_sequences[crim_id] = []
            for _ in events:
                if event_idx < len(labels):
                    test_sequences[crim_id].append(int(labels[event_idx]))
                    event_idx += 1

        return test_sequences

    def _evaluate_conditional_effects_fast(self, test_sequences: Dict[str, List[int]],
                                         type2_df, k: int) -> float:
        """Fast evaluation of conditional effects for k optimization."""
        significant_effects = 0
        total_effects = 0

        # Sample subset of demographics for speed during optimization
        headings = type2_df["Heading"].unique()
        sample_size = min(20, len(headings))  # Limit for optimization speed
        sampled_headings = np.random.choice(headings, size=sample_size, replace=False)

        for heading in sampled_headings:
            condition_map = self._get_condition_map(type2_df, heading)
            unique_values = list(set(condition_map.values()))[:3]  # Limit values

            for val in unique_values:
                selected_ids = [cid for cid, v in condition_map.items() if v == val]
                if len(selected_ids) < 5:
                    continue

                total_effects += 1

                # Quick significance check
                if self._quick_significance_test(selected_ids, test_sequences, k):
                    significant_effects += 1

        return significant_effects / total_effects if total_effects > 0 else 0.0

    def _comprehensive_conditional_analysis(self, test_sequences: Dict[str, List[int]],
                                          type2_df, k: int) -> Dict[str, Any]:
        """Comprehensive conditional analysis for validation."""
        from modular_analysis.markov.transition_analysis import TransitionMatrixBuilder

        builder = TransitionMatrixBuilder()
        global_matrix = builder.build_conditional_markov(list(test_sequences.keys()), test_sequences, k)

        significant_effects = 0
        total_effects = 0
        effect_sizes = []
        detailed_effects = []

        for heading in type2_df["Heading"].unique():
            condition_map = self._get_condition_map(type2_df, heading)
            unique_values = set(condition_map.values())

            for val in unique_values:
                selected_ids = [cid for cid, v in condition_map.items() if v == val]
                if len(selected_ids) < 5:
                    continue

                total_effects += 1

                # Build conditional matrix
                cond_matrix = builder.build_conditional_markov(selected_ids, test_sequences, k)

                # Compute effect size
                matrix_diff = np.linalg.norm(global_matrix - cond_matrix, 'fro')
                effect_sizes.append(matrix_diff)

                # Statistical test
                p_value = self._matrix_difference_test(global_matrix, cond_matrix)

                detailed_effects.append({
                    'heading': heading,
                    'value': val,
                    'n_criminals': len(selected_ids),
                    'effect_size': matrix_diff,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })

                if p_value < 0.05:
                    significant_effects += 1

        return {
            'significant_effects': significant_effects,
            'total_effects': total_effects,
            'significance_rate': significant_effects / total_effects if total_effects > 0 else 0,
            'mean_effect_size': np.mean(effect_sizes) if effect_sizes else 0,
            'effect_sizes': effect_sizes,
            'detailed_effects': detailed_effects
        }

    def _get_condition_map(self, type2_df, heading: str) -> Dict[str, str]:
        """Get condition map for a heading."""
        condition_map = {}
        for _, row in type2_df.iterrows():
            if row["Heading"].strip().lower() == heading.strip().lower():
                crim_id = str(row["CriminalID"])
                val = str(row["Value"]).strip() if row["Value"] else "Unknown"
                condition_map[crim_id] = val
        return condition_map

    def _quick_significance_test(self, selected_ids: List[str],
                               test_sequences: Dict[str, List[int]], k: int) -> bool:
        """Quick heuristic significance test for optimization."""
        total_criminals = len(test_sequences)
        group_size = len(selected_ids)

        # Quick filters
        if group_size < 5 or group_size > total_criminals * 0.8:
            return False

        # Check sequence diversity
        group_sequences = [test_sequences.get(cid, []) for cid in selected_ids]
        group_sequences = [seq for seq in group_sequences if len(seq) > 0]

        if len(group_sequences) < 3:
            return False

        all_states = [state for seq in group_sequences for state in seq]
        unique_states = len(set(all_states))

        return unique_states >= k * 0.5  # Group uses at least half the states

    def _matrix_difference_test(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """Statistical test for matrix differences."""
        try:
            # Convert to counts for chi-square test
            counts1 = (matrix1 * 100).astype(int)
            counts2 = (matrix2 * 100).astype(int)

            flat1 = counts1.flatten()
            flat2 = counts2.flatten()

            non_zero_mask = (flat1 > 0) | (flat2 > 0)
            if np.sum(non_zero_mask) < 2:
                return 1.0

            contingency = np.array([flat1[non_zero_mask], flat2[non_zero_mask]])
            _, p_value, _, _ = chi2_contingency(contingency)

            return p_value
        except:
            return 1.0

    def _permute_demographics(self, type2_df) -> pd.DataFrame:
        """Permute demographic labels while preserving structure."""
        permuted_df = type2_df.copy()

        # Permute values within each heading
        for heading in permuted_df["Heading"].unique():
            mask = permuted_df["Heading"] == heading
            values = permuted_df.loc[mask, "Value"].values
            np.random.shuffle(values)
            permuted_df.loc[mask, "Value"] = values

        return permuted_df

    def _benjamini_hochberg_correction(self, p_values: List[float], alpha: float = 0.05) -> int:
        """Benjamini-Hochberg FDR correction."""
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]

        significant_count = 0
        for i, p in enumerate(sorted_p):
            threshold = (i + 1) / len(sorted_p) * alpha
            if p <= threshold:
                significant_count = i + 1
            else:
                break

        return significant_count

    def _generate_validity_report(self, optimal_k: int, optimization_results: Dict,
                                validation_results: Dict, corrected_results: Dict,
                                null_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive statistical validity report."""

        # Compare validation results to null distribution
        observed_effects = validation_results['significant_effects']
        null_mean = null_results['mean_significant_effects']
        null_95th = null_results['percentile_95']

        # Statistical significance of overall pattern
        empirical_p = np.mean(np.array(null_results['all_significant_counts']) >= observed_effects)

        report = {
            'analysis_type': 'EXPLORATORY',
            'statistical_validity': {
                'data_splitting': True,
                'independent_validation': True,
                'null_distribution': True,
                'multiple_testing_correction': True
            },
            'selected_k': optimal_k,
            'optimization_results': optimization_results,
            'validation_results': {
                'raw_significant_effects': validation_results['significant_effects'],
                'total_tests': validation_results['total_effects'],
                'bonferroni_significant': corrected_results['bonferroni_significant'],
                'fdr_significant': corrected_results['fdr_significant'],
                'mean_effect_size': validation_results['mean_effect_size']
            },
            'null_comparison': {
                'observed_effects': observed_effects,
                'null_mean': null_mean,
                'null_95th_percentile': null_95th,
                'empirical_p_value': empirical_p,
                'exceeds_null_95th': observed_effects > null_95th
            },
            'statistical_interpretation': self._generate_interpretation(
                observed_effects, null_mean, null_95th, empirical_p,
                corrected_results['bonferroni_significant'], corrected_results['fdr_significant']
            ),
            'limitations_and_caveats': [
                "This is an EXPLORATORY analysis for hypothesis generation",
                "Results require independent replication for confirmation",
                "Multiple testing burden is substantial",
                "Effect sizes should be interpreted cautiously",
                "Clustering quality affects all downstream results"
            ]
        }

        return report

    def _generate_interpretation(self, observed: int, null_mean: float, null_95th: float,
                               empirical_p: float, bonferroni_sig: int, fdr_sig: int) -> str:
        """Generate statistical interpretation."""

        interpretation = []

        # Overall pattern significance
        if empirical_p < 0.05:
            interpretation.append(f"The overall pattern of {observed} significant effects is unlikely under the null hypothesis (empirical p = {empirical_p:.3f})")
        else:
            interpretation.append(f"The overall pattern of {observed} significant effects could plausibly occur by chance (empirical p = {empirical_p:.3f})")

        # Multiple testing results
        if bonferroni_sig > 0:
            interpretation.append(f"{bonferroni_sig} effects survive conservative Bonferroni correction")
        else:
            interpretation.append("No effects survive conservative Bonferroni correction")

        if fdr_sig > 0:
            interpretation.append(f"{fdr_sig} effects survive FDR correction (less conservative)")

        # Comparison to null
        if observed > null_95th:
            interpretation.append(f"Observed effects ({observed}) exceed 95th percentile of null distribution ({null_95th:.1f})")

        return "; ".join(interpretation)

def run_valid_exploratory_analysis_example():
    """Example of running statistically valid exploratory analysis."""

    # Create test data
    np.random.seed(42)
    embeddings = np.random.randn(200, 10)
    criminal_sequences = {f"criminal_{i}": [0, 1, 2, 1, 0] for i in range(40)}

    type2_data = []
    for i in range(40):
        type2_data.extend([
            {"CriminalID": f"criminal_{i}", "Heading": "Sex", "Value": "Male" if i % 2 == 0 else "Female"},
            {"CriminalID": f"criminal_{i}", "Heading": "Age", "Value": "Young" if i < 20 else "Old"}
        ])
    type2_df = pd.DataFrame(type2_data)

    # Run analysis
    optimizer = StatisticallyValidConditionalOptimizer(validation_split=0.5)
    results = optimizer.run_valid_exploratory_analysis(
        embeddings, criminal_sequences, type2_df,
        k_range=[3, 4, 5], n_null_simulations=20
    )

    print("\n" + "="*80)
    print("FINAL STATISTICAL VALIDITY REPORT")
    print("="*80)
    print(json.dumps(results['statistical_interpretation'], indent=2))

if __name__ == "__main__":
    run_valid_exploratory_analysis_example()
