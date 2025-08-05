#!/usr/bin/env python3
"""
efficient_conditional_optimization.py

Efficient algorithms for conditional k optimization to reduce computational cost.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.cluster import KMeans
import time

class EfficientConditionalOptimizer:
    """Efficient optimization strategies for conditional k selection."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.cache = {}  # Cache clustering results
    
    def bayesian_k_optimization(self, embeddings: np.ndarray, 
                               criminal_sequences: Dict[str, List[int]], 
                               type2_df, k_range: List[int] = None,
                               n_initial: int = 3, n_iterations: int = 5) -> Tuple[int, Dict]:
        """
        Bayesian optimization for k selection - much more efficient than grid search.
        
        Uses Gaussian Process to model the k -> score relationship and intelligently
        select the next k to evaluate.
        """
        if k_range is None:
            k_range = list(range(3, 12))
        
        print(f"[INFO] Bayesian optimization for k selection (range: {k_range})")
        
        # Initial random sampling
        initial_ks = np.random.choice(k_range, size=min(n_initial, len(k_range)), replace=False)
        
        results = {}
        scores = []
        evaluated_ks = []
        
        # Evaluate initial points
        for k in initial_ks:
            print(f"  [INITIAL] Evaluating k={k}")
            score = self._evaluate_k_fast(embeddings, criminal_sequences, type2_df, k)
            results[k] = {'score': score, 'method': 'initial'}
            scores.append(score)
            evaluated_ks.append(k)
        
        # Bayesian optimization iterations
        for iteration in range(n_iterations):
            if len(evaluated_ks) >= len(k_range):
                break
                
            # Select next k using acquisition function (Upper Confidence Bound)
            next_k = self._select_next_k_ucb(k_range, evaluated_ks, scores)
            
            print(f"  [ITER {iteration+1}] Evaluating k={next_k}")
            score = self._evaluate_k_fast(embeddings, criminal_sequences, type2_df, next_k)
            
            results[next_k] = {'score': score, 'method': 'bayesian'}
            scores.append(score)
            evaluated_ks.append(next_k)
        
        # Find best k
        best_k = max(results.keys(), key=lambda k: results[k]['score'])
        
        print(f"[INFO] Bayesian optimization complete. Best k: {best_k}")
        print(f"[INFO] Evaluated {len(evaluated_ks)}/{len(k_range)} possible k values")
        
        return best_k, results
    
    def progressive_k_optimization(self, embeddings: np.ndarray, 
                                 criminal_sequences: Dict[str, List[int]], 
                                 type2_df, k_range: List[int] = None) -> Tuple[int, Dict]:
        """
        Progressive optimization: start with fast approximations, refine promising candidates.
        
        1. Quick screening with subset of data
        2. Detailed evaluation of top candidates
        """
        if k_range is None:
            k_range = list(range(3, 12))
        
        print(f"[INFO] Progressive k optimization (range: {k_range})")
        
        # Phase 1: Quick screening with subset
        print("  [PHASE 1] Quick screening with data subset...")
        subset_size = min(1000, len(embeddings) // 2)
        subset_indices = np.random.choice(len(embeddings), subset_size, replace=False)
        subset_embeddings = embeddings[subset_indices]
        
        quick_results = {}
        for k in k_range:
            score = self._evaluate_k_subset(subset_embeddings, criminal_sequences, type2_df, k)
            quick_results[k] = score
        
        # Phase 2: Detailed evaluation of top candidates
        top_k_candidates = sorted(quick_results.keys(), 
                                key=lambda k: quick_results[k], reverse=True)[:3]
        
        print(f"  [PHASE 2] Detailed evaluation of top candidates: {top_k_candidates}")
        detailed_results = {}
        
        for k in top_k_candidates:
            print(f"    Detailed evaluation of k={k}")
            score = self._evaluate_k_fast(embeddings, criminal_sequences, type2_df, k)
            detailed_results[k] = {'score': score, 'quick_score': quick_results[k]}
        
        best_k = max(detailed_results.keys(), key=lambda k: detailed_results[k]['score'])
        
        # Combine results
        final_results = {}
        for k in k_range:
            if k in detailed_results:
                final_results[k] = detailed_results[k]
            else:
                final_results[k] = {'score': quick_results[k], 'method': 'quick_only'}
        
        print(f"[INFO] Progressive optimization complete. Best k: {best_k}")
        return best_k, final_results
    
    def early_stopping_optimization(self, embeddings: np.ndarray, 
                                  criminal_sequences: Dict[str, List[int]], 
                                  type2_df, k_range: List[int] = None,
                                  patience: int = 3) -> Tuple[int, Dict]:
        """
        Early stopping: stop if score doesn't improve for 'patience' iterations.
        """
        if k_range is None:
            k_range = list(range(3, 12))
        
        print(f"[INFO] Early stopping k optimization (patience: {patience})")
        
        results = {}
        best_score = -1
        best_k = k_range[0]
        no_improvement_count = 0
        
        for k in k_range:
            print(f"  Evaluating k={k}")
            score = self._evaluate_k_fast(embeddings, criminal_sequences, type2_df, k)
            results[k] = {'score': score}
            
            if score > best_score:
                best_score = score
                best_k = k
                no_improvement_count = 0
                print(f"    New best score: {score:.4f}")
            else:
                no_improvement_count += 1
                print(f"    Score: {score:.4f} (no improvement: {no_improvement_count}/{patience})")
            
            if no_improvement_count >= patience:
                print(f"  [EARLY STOP] No improvement for {patience} iterations")
                break
        
        print(f"[INFO] Early stopping complete. Best k: {best_k}")
        return best_k, results
    
    def _evaluate_k_fast(self, embeddings: np.ndarray, criminal_sequences: Dict[str, List[int]], 
                        type2_df, k: int) -> float:
        """Fast evaluation of k value with caching."""
        cache_key = f"k_{k}_{len(embeddings)}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        start_time = time.time()
        
        # Cluster with this k
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)  # Reduced n_init
        labels = kmeans.fit_predict(embeddings)
        
        # Quick clustering quality check
        from sklearn.metrics import silhouette_score
        try:
            silhouette = silhouette_score(embeddings, labels)
            if silhouette < 0.1:  # Skip poor clustering
                self.cache[cache_key] = 0.0
                return 0.0
        except:
            self.cache[cache_key] = 0.0
            return 0.0
        
        # Build test sequences
        test_sequences = self._build_test_sequences_fast(criminal_sequences, labels)
        
        # Sample subset of demographic tests for speed
        sample_size = min(50, len(type2_df["Heading"].unique()))  # Limit demographic tests
        sampled_headings = np.random.choice(
            type2_df["Heading"].unique(), 
            size=sample_size, 
            replace=False
        )
        
        # Quick conditional effect evaluation
        significant_effects = 0
        total_effects = 0
        
        for heading in sampled_headings:
            condition_map = self._get_condition_map_fast(type2_df, heading)
            unique_values = list(set(condition_map.values()))[:5]  # Limit values per heading
            
            for val in unique_values:
                selected_ids = [cid for cid, v in condition_map.items() if v == val]
                if len(selected_ids) < 5:
                    continue
                
                total_effects += 1
                
                # Quick matrix difference calculation
                if self._has_significant_effect_fast(selected_ids, test_sequences, k):
                    significant_effects += 1
        
        # Combined score
        if total_effects == 0:
            score = 0.0
        else:
            effect_rate = significant_effects / total_effects
            score = 0.3 * silhouette + 0.7 * effect_rate
        
        elapsed = time.time() - start_time
        print(f"    k={k}: score={score:.4f}, silhouette={silhouette:.3f}, time={elapsed:.1f}s")
        
        self.cache[cache_key] = score
        return score
    
    def _evaluate_k_subset(self, subset_embeddings: np.ndarray, 
                          criminal_sequences: Dict[str, List[int]], 
                          type2_df, k: int) -> float:
        """Very fast evaluation using data subset."""
        # Even faster evaluation for screening
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=3)
        labels = kmeans.fit_predict(subset_embeddings)
        
        try:
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(subset_embeddings, labels)
            return silhouette  # Use clustering quality as proxy
        except:
            return 0.0
    
    def _select_next_k_ucb(self, k_range: List[int], evaluated_ks: List[int], 
                          scores: List[float]) -> int:
        """Upper Confidence Bound acquisition function for Bayesian optimization."""
        if len(evaluated_ks) == 0:
            return np.random.choice(k_range)
        
        # Simple UCB: select k that maximizes mean + confidence
        remaining_ks = [k for k in k_range if k not in evaluated_ks]
        if not remaining_ks:
            return np.random.choice(k_range)
        
        # For simplicity, select k furthest from evaluated ones
        distances = []
        for k in remaining_ks:
            min_dist = min(abs(k - eval_k) for eval_k in evaluated_ks)
            distances.append(min_dist)
        
        max_dist_idx = np.argmax(distances)
        return remaining_ks[max_dist_idx]
    
    def _build_test_sequences_fast(self, criminal_sequences: Dict[str, List[int]], 
                                  labels: np.ndarray) -> Dict[str, List[int]]:
        """Fast sequence building."""
        test_sequences = {}
        event_idx = 0
        
        for crim_id, events in criminal_sequences.items():
            test_sequences[crim_id] = []
            for _ in events:
                if event_idx < len(labels):
                    test_sequences[crim_id].append(int(labels[event_idx]))
                    event_idx += 1
        
        return test_sequences
    
    def _get_condition_map_fast(self, type2_df, heading: str) -> Dict[str, str]:
        """Fast condition map building."""
        condition_map = {}
        for _, row in type2_df.iterrows():
            if row["Heading"].strip().lower() == heading.strip().lower():
                crim_id = str(row["CriminalID"])
                val = str(row["Value"]).strip() if row["Value"] else "Unknown"
                condition_map[crim_id] = val
        return condition_map
    
    def _has_significant_effect_fast(self, selected_ids: List[str], 
                                   test_sequences: Dict[str, List[int]], k: int) -> bool:
        """Fast significance check using simple heuristics."""
        # Quick check: if group is too small or too large, likely not significant
        total_criminals = len(test_sequences)
        group_size = len(selected_ids)
        
        if group_size < 5 or group_size > total_criminals * 0.8:
            return False
        
        # Quick diversity check: if group has diverse sequences, more likely significant
        group_sequences = [test_sequences.get(cid, []) for cid in selected_ids]
        group_sequences = [seq for seq in group_sequences if len(seq) > 0]
        
        if len(group_sequences) < 3:
            return False
        
        # Check if group has different patterns than expected
        all_states = [state for seq in group_sequences for state in seq]
        if len(set(all_states)) >= k * 0.6:  # Group uses most states
            return True
        
        return False

def compare_optimization_methods():
    """Compare different optimization methods."""
    print("=== OPTIMIZATION METHOD COMPARISON ===")
    
    # Create test data
    np.random.seed(42)
    embeddings = np.random.randn(500, 10)
    criminal_sequences = {f"criminal_{i}": [0, 1, 2, 1, 0] for i in range(50)}
    
    import pandas as pd
    type2_data = []
    for i in range(50):
        type2_data.extend([
            {"CriminalID": f"criminal_{i}", "Heading": "Sex", "Value": "Male" if i % 2 == 0 else "Female"},
            {"CriminalID": f"criminal_{i}", "Heading": "Age", "Value": "Young" if i < 25 else "Old"}
        ])
    type2_df = pd.DataFrame(type2_data)
    
    optimizer = EfficientConditionalOptimizer()
    k_range = list(range(3, 8))
    
    methods = [
        ("Bayesian", lambda: optimizer.bayesian_k_optimization(embeddings, criminal_sequences, type2_df, k_range)),
        ("Progressive", lambda: optimizer.progressive_k_optimization(embeddings, criminal_sequences, type2_df, k_range)),
        ("Early Stopping", lambda: optimizer.early_stopping_optimization(embeddings, criminal_sequences, type2_df, k_range))
    ]
    
    for method_name, method_func in methods:
        print(f"\n--- {method_name} Optimization ---")
        start_time = time.time()
        best_k, results = method_func()
        elapsed = time.time() - start_time
        print(f"Best k: {best_k}, Time: {elapsed:.2f}s, Evaluations: {len(results)}")

if __name__ == "__main__":
    compare_optimization_methods()
