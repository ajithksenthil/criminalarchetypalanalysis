#!/usr/bin/env python3
"""
fix_conditional_markov_properly.py

Proper implementation of conditional Markov analysis and lexical bias fixes.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency
import json

class ProperConditionalMarkovAnalyzer:
    """Correct implementation of conditional Markov analysis."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def find_optimal_k_for_conditional_analysis(self, embeddings: np.ndarray, 
                                              criminal_sequences: Dict[str, List[int]], 
                                              type2_df, k_range: List[int] = None, 
                                              min_effect_size: float = 0.1) -> Tuple[int, Dict[int, Any]]:
        """
        Find optimal k by maximizing significant conditional effects on TRANSITION MATRICES.
        
        This is the correct implementation that compares transition matrices between
        demographic groups, not just stationary distributions.
        """
        if k_range is None:
            n_samples = len(embeddings)
            k_max = min(15, int(np.sqrt(n_samples / 2)))
            k_range = list(range(3, k_max + 1))
        
        print(f"[INFO] Testing k values for conditional TRANSITION MATRIX effects: {k_range}")
        
        best_k = None
        best_score = 0
        results = {}
        
        for k in k_range:
            print(f"  Testing k={k} for conditional transition effects...")
            
            # Cluster with this k
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            # Build sequences with new clustering
            test_sequences = self._build_test_sequences(criminal_sequences, labels)
            
            # Compute global transition matrix (k×k)
            global_matrix = self._build_conditional_markov(list(test_sequences.keys()), test_sequences, k)
            
            # Test conditional effects on TRANSITION MATRICES
            effect_results = self._test_conditional_transition_effects(
                type2_df, test_sequences, k, global_matrix, min_effect_size
            )
            
            # Score this k value
            score = self._compute_transition_score(effect_results)
            results[k] = effect_results
            results[k]['score'] = score
            
            print(f"    k={k}: {effect_results['significant_effects']}/{effect_results['total_effects']} "
                  f"significant transition effects, score={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        
        print(f"[INFO] Optimal k selected: {best_k} (transition score: {best_score:.3f})")
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
                    test_sequences[crim_id].append(int(labels[event_idx]))
                    event_idx += 1
        
        return test_sequences
    
    def _build_conditional_markov(self, selected_criminal_ids: List[str], 
                                criminal_sequences: Dict[str, List[int]], 
                                n_clusters: int) -> np.ndarray:
        """
        Build a k×k transition matrix for a subset of criminals.
        This is the core of conditional Markov analysis.
        """
        matrix = np.zeros((n_clusters, n_clusters))
        
        for cid in selected_criminal_ids:
            seq = criminal_sequences.get(cid, [])
            if len(seq) < 2:
                continue
            
            # Count transitions
            for s1, s2 in zip(seq[:-1], seq[1:]):
                if 0 <= s1 < n_clusters and 0 <= s2 < n_clusters:
                    matrix[s1, s2] += 1
        
        # Normalize rows to probabilities
        row_sums = matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            matrix = np.divide(matrix, row_sums, where=row_sums != 0)
            # Handle empty rows
            for i in range(n_clusters):
                if row_sums[i] == 0:
                    matrix[i] = 1.0 / n_clusters
        
        return matrix
    
    def _test_conditional_transition_effects(self, type2_df, criminal_sequences: Dict[str, List[int]], 
                                           k: int, global_matrix: np.ndarray, 
                                           min_effect_size: float) -> Dict[str, Any]:
        """
        Test conditional effects on TRANSITION MATRICES (not just stationary distributions).
        This is the correct approach for conditional Markov analysis.
        """
        significant_effects = 0
        total_effects = 0
        effect_sizes = []
        detailed_effects = []
        
        # Test each demographic heading
        for heading in type2_df["Heading"].unique():
            condition_map = self._get_condition_map(type2_df, heading)
            unique_values = set(condition_map.values())
            
            # Test each value within this heading
            for val in unique_values:
                selected_ids = [cid for cid, v in condition_map.items() if v == val]
                if len(selected_ids) < 5:  # Skip small groups
                    continue
                
                total_effects += 1
                
                # Build conditional transition matrix for this demographic group
                cond_matrix = self._build_conditional_markov(selected_ids, criminal_sequences, k)
                
                # Compare TRANSITION MATRICES (not just stationary distributions)
                matrix_diff = self._compute_matrix_difference(global_matrix, cond_matrix)
                
                # Statistical test for transition matrix differences
                p_value = self._test_matrix_difference(global_matrix, cond_matrix)
                
                effect_sizes.append(matrix_diff)
                
                # Record detailed effect
                detailed_effects.append({
                    'heading': heading,
                    'value': val,
                    'n_criminals': len(selected_ids),
                    'matrix_difference': float(matrix_diff),
                    'p_value': float(p_value),
                    'global_matrix': global_matrix.tolist(),
                    'conditional_matrix': cond_matrix.tolist()
                })
                
                if matrix_diff >= min_effect_size and p_value < 0.05:
                    significant_effects += 1
        
        return {
            'significant_effects': significant_effects,
            'total_effects': total_effects,
            'significance_rate': significant_effects / total_effects if total_effects > 0 else 0,
            'mean_effect_size': np.mean(effect_sizes) if effect_sizes else 0,
            'detailed_effects': detailed_effects
        }
    
    def _get_condition_map(self, type2_df, heading: str) -> Dict[str, str]:
        """Get mapping from criminal ID to value for a specific heading."""
        condition_map = {}
        for _, row in type2_df.iterrows():
            if row["Heading"].strip().lower() == heading.strip().lower():
                crim_id = str(row["CriminalID"])
                val = str(row["Value"]).strip() if row["Value"] else "Unknown"
                condition_map[crim_id] = val
        return condition_map
    
    def _compute_matrix_difference(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """
        Compute difference between two transition matrices.
        Uses Frobenius norm of the difference.
        """
        return np.linalg.norm(matrix1 - matrix2, 'fro')
    
    def _test_matrix_difference(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """
        Statistical test for difference between transition matrices.
        Uses chi-square test on the transition counts.
        """
        try:
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
    
    def _compute_transition_score(self, effect_results: Dict[str, Any]) -> float:
        """
        Compute score for k based on transition matrix effects.
        """
        if effect_results['total_effects'] == 0:
            return 0.0
        
        significance_rate = effect_results['significance_rate']
        mean_effect_size = effect_results['mean_effect_size']
        
        # Combined score: rate of significant effects * mean effect size
        return significance_rate * mean_effect_size

class LexicalBiasFixer:
    """Fix lexical bias in embeddings using prototype-based approach."""
    
    def __init__(self, client=None):
        self.client = client
    
    def generate_lexical_variations(self, text: str, num_variants: int = 5) -> List[str]:
        """
        Generate lexical variations of text to capture semantic meaning.
        """
        if not self.client:
            return [text]  # Fallback to original text
        
        prompt = (
            f"Generate {num_variants} alternative versions of the following sentence, "
            f"using synonyms and varied phrasing, while preserving the meaning:\n\n"
            f"{text}\n\nAlternative versions:"
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            reply = response.choices[0].message.content.strip()
            variations = [line.strip() for line in reply.split("\n") if line.strip()]
            
            if not variations:
                return [text]
            
            return variations
        except Exception as e:
            print(f"[WARNING] Could not generate lexical variations: {e}")
            return [text]
    
    def get_prototype_embedding(self, event_text: str, model, num_variants: int = 5) -> np.ndarray:
        """
        Get prototype embedding by averaging embeddings of lexical variations.
        This reduces bias from specific word choices.
        """
        # Generate lexical variations
        variations = self.generate_lexical_variations(event_text, num_variants)
        
        # Include original text
        all_versions = variations + [event_text]
        
        # Compute embeddings for all versions
        embeddings = model.encode(all_versions)
        
        # Return centroid (prototype) embedding
        return np.mean(embeddings, axis=0)

def test_proper_implementation():
    """Test the proper conditional Markov implementation."""
    print("Testing proper conditional Markov implementation...")
    
    # Create test data
    np.random.seed(42)
    
    # Test embeddings
    embeddings = np.random.randn(100, 10)
    
    # Test criminal sequences
    criminal_sequences = {
        f"criminal_{i}": [0, 1, 2, 1, 0] for i in range(20)
    }
    
    # Test type2 data
    import pandas as pd
    type2_data = []
    for i in range(20):
        type2_data.extend([
            {"CriminalID": f"criminal_{i}", "Heading": "Sex", "Value": "Male" if i % 2 == 0 else "Female"},
            {"CriminalID": f"criminal_{i}", "Heading": "Age", "Value": "Young" if i < 10 else "Old"}
        ])
    
    type2_df = pd.DataFrame(type2_data)
    
    # Test analyzer
    analyzer = ProperConditionalMarkovAnalyzer()
    optimal_k, results = analyzer.find_optimal_k_for_conditional_analysis(
        embeddings, criminal_sequences, type2_df, k_range=[3, 4, 5]
    )
    
    print(f"✅ Test completed. Optimal k: {optimal_k}")
    print(f"✅ Results structure: {list(results.keys())}")

if __name__ == "__main__":
    test_proper_implementation()
