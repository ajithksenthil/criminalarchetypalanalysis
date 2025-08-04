#!/usr/bin/env python3
"""
markov_models.py

Advanced Markov chain models for criminal archetypal analysis.
Includes higher-order chains, HMMs, and time-varying models.
"""

import numpy as np
from collections import defaultdict, Counter
from itertools import groupby
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("[Warning] hmmlearn not installed. Hidden Markov Models will be unavailable.")


class HigherOrderMarkov:
    """
    Higher-order Markov chain model that captures multi-step dependencies.
    
    For example, a 2nd order model considers the last 2 states to predict the next:
    P(X_t | X_{t-1}, X_{t-2})
    """
    
    def __init__(self, order=2):
        """
        Initialize higher-order Markov model.
        
        Args:
            order: The order of the Markov chain (number of previous states to consider)
        """
        self.order = order
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.state_encoder = LabelEncoder()
        self.is_fitted = False
        
    def fit(self, sequences):
        """
        Fit the higher-order Markov model on sequences.
        
        Args:
            sequences: List of state sequences (each sequence is a list of states)
        """
        # Flatten all states to fit encoder
        all_states = []
        for seq in sequences:
            all_states.extend(seq)
        
        # Fit state encoder
        self.state_encoder.fit(all_states)
        
        # Count transitions
        for seq in sequences:
            if len(seq) <= self.order:
                continue
                
            # Encode sequence
            encoded_seq = self.state_encoder.transform(seq)
            
            # Extract all n-grams of length order+1
            for i in range(len(encoded_seq) - self.order):
                # Previous states tuple
                prev_states = tuple(encoded_seq[i:i+self.order])
                # Next state
                next_state = encoded_seq[i+self.order]
                
                self.transition_counts[prev_states][next_state] += 1
        
        self.is_fitted = True
        self._compute_probabilities()
        
    def _compute_probabilities(self):
        """Compute transition probabilities from counts."""
        self.transition_probs = {}
        
        for prev_states, next_counts in self.transition_counts.items():
            total = sum(next_counts.values())
            self.transition_probs[prev_states] = {
                state: count/total for state, count in next_counts.items()
            }
    
    def predict_next_state(self, current_states, top_k=3):
        """
        Predict next state(s) given current state history.
        
        Args:
            current_states: List of current states (length should be >= order)
            top_k: Return top k most probable next states
            
        Returns:
            List of (state, probability) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Take last 'order' states
        if len(current_states) < self.order:
            raise ValueError(f"Need at least {self.order} states for prediction")
            
        recent_states = current_states[-self.order:]
        
        # Encode states
        try:
            encoded_states = tuple(self.state_encoder.transform(recent_states))
        except:
            # Handle unseen states
            return []
        
        # Get probabilities
        if encoded_states in self.transition_probs:
            probs = self.transition_probs[encoded_states]
            # Sort by probability
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            
            # Decode states and return top k
            results = []
            for state_idx, prob in sorted_probs[:top_k]:
                state = self.state_encoder.inverse_transform([state_idx])[0]
                results.append((state, prob))
            
            return results
        else:
            return []
    
    def get_pattern_frequency(self, pattern):
        """
        Get frequency of a specific pattern in the training data.
        
        Args:
            pattern: List of states representing the pattern
            
        Returns:
            Frequency count
        """
        if len(pattern) != self.order + 1:
            raise ValueError(f"Pattern length should be {self.order + 1}")
            
        try:
            encoded_pattern = self.state_encoder.transform(pattern)
            prev_states = tuple(encoded_pattern[:-1])
            next_state = encoded_pattern[-1]
            
            return self.transition_counts[prev_states][next_state]
        except:
            return 0
    
    def find_common_patterns(self, min_frequency=5, top_n=10):
        """
        Find the most common multi-step patterns.
        
        Args:
            min_frequency: Minimum frequency to consider
            top_n: Return top n patterns
            
        Returns:
            List of (pattern, frequency) tuples
        """
        patterns = []
        
        for prev_states, next_counts in self.transition_counts.items():
            for next_state, count in next_counts.items():
                if count >= min_frequency:
                    # Decode pattern
                    full_pattern = list(prev_states) + [next_state]
                    decoded_pattern = self.state_encoder.inverse_transform(full_pattern)
                    patterns.append((list(decoded_pattern), count))
        
        # Sort by frequency
        patterns.sort(key=lambda x: x[1], reverse=True)
        
        return patterns[:top_n]


class TimeVaryingMarkov:
    """
    Markov model where transition probabilities vary over time/age.
    Useful for modeling how criminal patterns change across life stages.
    """
    
    def __init__(self, time_windows=None):
        """
        Initialize time-varying Markov model.
        
        Args:
            time_windows: List of tuples defining time windows [(start1, end1), (start2, end2), ...]
                         If None, will use automatic segmentation
        """
        self.time_windows = time_windows
        self.models = {}
        self.state_encoder = LabelEncoder()
        
    def fit(self, sequences_with_times):
        """
        Fit time-varying model.
        
        Args:
            sequences_with_times: List of (sequence, times) tuples where times are timestamps/ages
        """
        # Extract all states for encoding
        all_states = []
        for seq, _ in sequences_with_times:
            all_states.extend(seq)
        self.state_encoder.fit(all_states)
        
        # If no time windows specified, create automatic ones
        if self.time_windows is None:
            all_times = []
            for _, times in sequences_with_times:
                all_times.extend(times)
            self.time_windows = self._create_time_windows(all_times)
        
        # Fit separate model for each time window
        for window in self.time_windows:
            window_sequences = self._extract_window_sequences(sequences_with_times, window)
            
            if window_sequences:
                model = MarkovChain()
                model.state_encoder = self.state_encoder  # Share encoder
                model.fit_from_sequences(window_sequences)
                self.models[window] = model
    
    def _create_time_windows(self, times, n_windows=4):
        """Automatically create time windows based on data distribution."""
        times = sorted(times)
        percentiles = np.linspace(0, 100, n_windows + 1)
        boundaries = np.percentile(times, percentiles)
        
        windows = []
        for i in range(len(boundaries) - 1):
            windows.append((boundaries[i], boundaries[i+1]))
        
        return windows
    
    def _extract_window_sequences(self, sequences_with_times, window):
        """Extract transitions that occur within a time window."""
        window_sequences = []
        start, end = window
        
        for seq, times in sequences_with_times:
            window_seq = []
            for i, (state, time) in enumerate(zip(seq, times)):
                if start <= time <= end:
                    window_seq.append(state)
            
            if len(window_seq) > 1:
                window_sequences.append(window_seq)
        
        return window_sequences
    
    def get_transition_matrix(self, time):
        """Get transition matrix for a specific time."""
        # Find which window this time belongs to
        for window in self.time_windows:
            if window[0] <= time <= window[1]:
                if window in self.models:
                    return self.models[window].transition_matrix
        
        return None
    
    def compare_time_periods(self):
        """Compare transition patterns across time periods."""
        comparisons = {}
        
        windows = list(self.models.keys())
        for i in range(len(windows) - 1):
            window1, window2 = windows[i], windows[i+1]
            
            if window1 in self.models and window2 in self.models:
                matrix1 = self.models[window1].transition_matrix
                matrix2 = self.models[window2].transition_matrix
                
                # Compute difference
                if matrix1 is not None and matrix2 is not None:
                    # Ensure same dimensions
                    n_states = max(matrix1.shape[0], matrix2.shape[0])
                    m1 = np.zeros((n_states, n_states))
                    m2 = np.zeros((n_states, n_states))
                    
                    m1[:matrix1.shape[0], :matrix1.shape[1]] = matrix1
                    m2[:matrix2.shape[0], :matrix2.shape[1]] = matrix2
                    
                    # Compute metrics
                    frobenius_diff = np.linalg.norm(m1 - m2, 'fro')
                    
                    comparisons[f"{window1}_vs_{window2}"] = {
                        'frobenius_distance': frobenius_diff,
                        'max_difference': np.max(np.abs(m1 - m2)),
                        'changed_transitions': np.sum(np.abs(m1 - m2) > 0.1)
                    }
        
        return comparisons


class MarkovChain:
    """Basic Markov chain utilities (enhanced version)."""
    
    def __init__(self):
        self.transition_matrix = None
        self.state_encoder = LabelEncoder()
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        
    def fit_from_sequences(self, sequences):
        """Fit Markov chain from sequences."""
        # Count transitions
        for seq in sequences:
            for i in range(len(seq) - 1):
                self.transition_counts[seq[i]][seq[i+1]] += 1
        
        # Get all unique states
        all_states = set()
        for seq in sequences:
            all_states.update(seq)
        
        # Fit encoder
        self.state_encoder.fit(list(all_states))
        n_states = len(all_states)
        
        # Build transition matrix
        self.transition_matrix = np.zeros((n_states, n_states))
        
        for state_from, transitions in self.transition_counts.items():
            i = self.state_encoder.transform([state_from])[0]
            total = sum(transitions.values())
            
            for state_to, count in transitions.items():
                j = self.state_encoder.transform([state_to])[0]
                self.transition_matrix[i, j] = count / total
        
        return self
    
    def get_steady_state(self):
        """Compute steady-state distribution."""
        if self.transition_matrix is None:
            return None
            
        # Find eigenvector for eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        
        # Find index of eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1))
        
        # Get corresponding eigenvector
        steady_state = np.real(eigenvectors[:, idx])
        
        # Normalize to sum to 1
        steady_state = steady_state / steady_state.sum()
        
        return steady_state
    
    def simulate_sequence(self, start_state, length, seed=None):
        """Simulate a sequence from the Markov chain."""
        if seed is not None:
            np.random.seed(seed)
            
        # Encode start state
        current_idx = self.state_encoder.transform([start_state])[0]
        
        sequence = [start_state]
        
        for _ in range(length - 1):
            # Get transition probabilities
            probs = self.transition_matrix[current_idx]
            
            # Sample next state
            next_idx = np.random.choice(len(probs), p=probs)
            next_state = self.state_encoder.inverse_transform([next_idx])[0]
            
            sequence.append(next_state)
            current_idx = next_idx
        
        return sequence


def identify_critical_patterns(sequences, order=3, min_support=0.05):
    """
    Identify critical multi-step patterns that appear frequently.
    
    Args:
        sequences: List of state sequences
        order: Length of patterns to find
        min_support: Minimum frequency (as fraction of total sequences)
        
    Returns:
        List of (pattern, frequency, sequences_containing) tuples
    """
    pattern_counts = defaultdict(lambda: {'count': 0, 'sequences': set()})
    
    for seq_idx, seq in enumerate(sequences):
        # Extract all patterns of specified order
        for i in range(len(seq) - order + 1):
            pattern = tuple(seq[i:i+order])
            pattern_counts[pattern]['count'] += 1
            pattern_counts[pattern]['sequences'].add(seq_idx)
    
    # Filter by minimum support
    min_count = int(min_support * len(sequences))
    critical_patterns = []
    
    for pattern, info in pattern_counts.items():
        if info['count'] >= min_count:
            critical_patterns.append({
                'pattern': list(pattern),
                'total_occurrences': info['count'],
                'unique_sequences': len(info['sequences']),
                'support': len(info['sequences']) / len(sequences)
            })
    
    # Sort by support
    critical_patterns.sort(key=lambda x: x['support'], reverse=True)
    
    return critical_patterns


if __name__ == "__main__":
    # Test higher-order Markov
    print("Testing Higher-Order Markov Models...")
    
    # Example sequences
    sequences = [
        ["childhood_trauma", "substance_abuse", "violence", "arrest"],
        ["childhood_trauma", "school_problems", "substance_abuse", "violence"],
        ["normal_childhood", "school_success", "stable_job", "normal_life"],
        ["childhood_trauma", "substance_abuse", "mental_illness", "violence"]
    ]
    
    # Test 2nd order model
    model = HigherOrderMarkov(order=2)
    model.fit(sequences)
    
    # Find common patterns
    patterns = model.find_common_patterns(min_frequency=1)
    print("\nCommon 3-step patterns:")
    for pattern, freq in patterns[:5]:
        print(f"  {' → '.join(pattern)}: {freq} times")
    
    # Test prediction
    current = ["childhood_trauma", "substance_abuse"]
    predictions = model.predict_next_state(current)
    print(f"\nGiven sequence: {' → '.join(current)}")
    print("Predicted next states:")
    for state, prob in predictions:
        print(f"  {state}: {prob:.2f}")