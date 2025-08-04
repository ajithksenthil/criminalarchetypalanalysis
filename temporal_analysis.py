#!/usr/bin/env python3
"""
temporal_analysis.py

Advanced temporal analysis for criminal life events.
Includes change point detection, life phase segmentation, and pattern mining.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    print("[Warning] ruptures not installed. Advanced change point detection will be limited.")

try:
    from prefixspan import PrefixSpan
    PREFIXSPAN_AVAILABLE = True
except ImportError:
    PREFIXSPAN_AVAILABLE = False
    print("[Warning] prefixspan not installed. Advanced pattern mining will be limited.")


class ChangePointDetector:
    """
    Detect significant change points in criminal life trajectories.
    These represent critical transitions like escalation to violence.
    """
    
    def __init__(self, method='bayesian', min_segment_length=3):
        """
        Initialize change point detector.
        
        Args:
            method: Detection method ('bayesian', 'pelt', 'binary_segmentation')
            min_segment_length: Minimum events between change points
        """
        self.method = method
        self.min_segment_length = min_segment_length
        self.change_points = {}
        
    def detect_change_points(self, sequence, embeddings=None, max_changes=5):
        """
        Detect change points in a sequence.
        
        Args:
            sequence: List of events
            embeddings: Optional embeddings for each event
            max_changes: Maximum number of change points to detect
            
        Returns:
            List of change point indices
        """
        if len(sequence) < 2 * self.min_segment_length:
            return []
        
        if self.method == 'bayesian':
            return self._bayesian_change_points(sequence, embeddings, max_changes)
        elif self.method == 'binary_segmentation':
            return self._binary_segmentation(sequence, embeddings, max_changes)
        else:
            return self._statistical_change_points(sequence, embeddings)
    
    def _bayesian_change_points(self, sequence, embeddings, max_changes):
        """Bayesian change point detection."""
        n = len(sequence)
        
        # Convert to numerical representation
        if embeddings is not None:
            data = embeddings
        else:
            # Use simple encoding
            unique_events = list(set(sequence))
            event_to_idx = {event: idx for idx, event in enumerate(unique_events)}
            data = np.array([event_to_idx[event] for event in sequence])
        
        # Compute change probability at each point
        change_probs = []
        
        for t in range(self.min_segment_length, n - self.min_segment_length):
            # Compare distributions before and after point t
            before = data[:t]
            after = data[t:]
            
            # Compute divergence between segments
            if len(before.shape) > 1:  # Embeddings
                before_mean = np.mean(before, axis=0)
                after_mean = np.mean(after, axis=0)
                divergence = np.linalg.norm(before_mean - after_mean)
            else:  # Simple encoding
                before_dist = np.bincount(before.astype(int), minlength=len(unique_events))
                after_dist = np.bincount(after.astype(int), minlength=len(unique_events))
                before_dist = before_dist / before_dist.sum()
                after_dist = after_dist / after_dist.sum()
                divergence = stats.entropy(before_dist, after_dist)
            
            change_probs.append((t, divergence))
        
        # Sort by divergence and return top change points
        change_probs.sort(key=lambda x: x[1], reverse=True)
        change_points = sorted([cp[0] for cp in change_probs[:max_changes]])
        
        return change_points
    
    def _binary_segmentation(self, sequence, embeddings, max_changes):
        """Binary segmentation for change point detection."""
        if embeddings is not None:
            data = embeddings
        else:
            # Simple encoding
            unique_events = list(set(sequence))
            event_to_idx = {event: idx for idx, event in enumerate(unique_events)}
            data = np.array([[event_to_idx[event]] for event in sequence])
        
        change_points = []
        segments = [(0, len(data))]
        
        for _ in range(max_changes):
            best_gain = -np.inf
            best_point = None
            best_segment_idx = None
            
            # Check each segment for best split
            for seg_idx, (start, end) in enumerate(segments):
                if end - start < 2 * self.min_segment_length:
                    continue
                
                segment_data = data[start:end]
                
                # Try each possible split point
                for t in range(self.min_segment_length, 
                             len(segment_data) - self.min_segment_length):
                    # Calculate cost reduction
                    cost_full = self._segment_cost(segment_data)
                    cost_left = self._segment_cost(segment_data[:t])
                    cost_right = self._segment_cost(segment_data[t:])
                    
                    gain = cost_full - (cost_left + cost_right)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = start + t
                        best_segment_idx = seg_idx
            
            if best_point is None:
                break
            
            # Add change point and update segments
            change_points.append(best_point)
            old_segment = segments.pop(best_segment_idx)
            segments.extend([(old_segment[0], best_point), 
                           (best_point, old_segment[1])])
            segments.sort()
        
        return sorted(change_points)
    
    def _segment_cost(self, segment_data):
        """Calculate cost (variance) of a segment."""
        if len(segment_data) == 0:
            return 0
        
        if len(segment_data.shape) > 1:  # Multidimensional
            return np.sum(np.var(segment_data, axis=0))
        else:
            return np.var(segment_data)
    
    def _statistical_change_points(self, sequence, embeddings):
        """Statistical test-based change point detection."""
        n = len(sequence)
        change_scores = []
        
        # Slide window and test for distribution change
        window_size = max(self.min_segment_length, n // 10)
        
        for i in range(window_size, n - window_size):
            before = sequence[i-window_size:i]
            after = sequence[i:i+window_size]
            
            # Chi-square test for independence
            all_events = list(set(before + after))
            contingency = np.zeros((2, len(all_events)))
            
            for j, event in enumerate(all_events):
                contingency[0, j] = before.count(event)
                contingency[1, j] = after.count(event)
            
            # Remove empty columns
            contingency = contingency[:, contingency.sum(axis=0) > 0]
            
            if contingency.shape[1] > 1:
                chi2, p_value = stats.chi2_contingency(contingency)[:2]
                change_scores.append((i, chi2))
        
        # Find peaks in change scores
        change_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return significant change points
        threshold = np.percentile([s[1] for s in change_scores], 90)
        change_points = [cp[0] for cp in change_scores if cp[1] > threshold]
        
        return sorted(change_points)[:5]  # Return top 5
    
    def analyze_change_points(self, sequences, labels=None):
        """
        Analyze change points across multiple sequences.
        
        Args:
            sequences: List of event sequences
            labels: Optional labels for sequences (e.g., criminal types)
            
        Returns:
            Dictionary with analysis results
        """
        all_change_points = []
        change_events = defaultdict(list)
        
        for seq_idx, seq in enumerate(sequences):
            cps = self.detect_change_points(seq)
            self.change_points[seq_idx] = cps
            
            # Record events around change points
            for cp in cps:
                if cp > 0 and cp < len(seq):
                    before_event = seq[cp-1] if cp > 0 else None
                    after_event = seq[cp] if cp < len(seq) else None
                    
                    change_events['before'].append(before_event)
                    change_events['after'].append(after_event)
                    
                    if labels is not None:
                        change_events['label'].append(labels[seq_idx])
                    
                    all_change_points.append(cp / len(seq))  # Normalized position
        
        # Analyze patterns
        results = {
            'total_change_points': len(all_change_points),
            'avg_change_points_per_sequence': len(all_change_points) / len(sequences),
            'change_point_positions': np.array(all_change_points),
            'common_before_events': Counter(change_events['before']).most_common(10),
            'common_after_events': Counter(change_events['after']).most_common(10)
        }
        
        # Identify critical transitions
        transitions = []
        for before, after in zip(change_events['before'], change_events['after']):
            if before and after:
                transitions.append(f"{before} → {after}")
        
        results['common_transitions'] = Counter(transitions).most_common(10)
        
        return results


class LifePhaseSegmenter:
    """
    Automatically segment criminal lives into developmental phases.
    """
    
    def __init__(self, n_phases=5):
        """
        Initialize life phase segmenter.
        
        Args:
            n_phases: Number of life phases to identify
        """
        self.n_phases = n_phases
        self.phase_model = None
        self.phase_characteristics = {}
        
    def segment_life_phases(self, sequences_with_ages):
        """
        Segment sequences into life phases based on age and event patterns.
        
        Args:
            sequences_with_ages: List of (sequence, ages) tuples
            
        Returns:
            Phase assignments for each event
        """
        # Extract features for each event
        features = []
        event_info = []
        
        for seq, ages in sequences_with_ages:
            for event, age in zip(seq, ages):
                # Feature vector: [age, event_type_encoding, sequence_position]
                seq_position = len(event_info) / len(seq)  # Normalized position
                
                features.append([
                    age,
                    seq_position,
                    len(event),  # Event complexity (string length as proxy)
                ])
                
                event_info.append({
                    'event': event,
                    'age': age,
                    'sequence_idx': len(event_info)
                })
        
        features = np.array(features)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Cluster to find phases
        self.phase_model = KMeans(n_clusters=self.n_phases, random_state=42)
        phase_labels = self.phase_model.fit_predict(features_scaled)
        
        # Analyze phase characteristics
        for phase in range(self.n_phases):
            phase_mask = phase_labels == phase
            phase_events = [event_info[i] for i in range(len(event_info)) if phase_mask[i]]
            
            if phase_events:
                ages = [e['age'] for e in phase_events]
                events = [e['event'] for e in phase_events]
                
                self.phase_characteristics[phase] = {
                    'age_range': (min(ages), max(ages)),
                    'mean_age': np.mean(ages),
                    'common_events': Counter(events).most_common(5),
                    'n_events': len(phase_events)
                }
        
        # Sort phases by mean age
        sorted_phases = sorted(self.phase_characteristics.items(), 
                             key=lambda x: x[1]['mean_age'])
        
        # Create mapping from old to new phase labels
        phase_mapping = {old_phase: new_phase 
                        for new_phase, (old_phase, _) in enumerate(sorted_phases)}
        
        # Remap labels
        remapped_labels = [phase_mapping[label] for label in phase_labels]
        
        # Update characteristics with new labels
        new_characteristics = {}
        for new_phase, (old_phase, chars) in enumerate(sorted_phases):
            new_characteristics[new_phase] = chars
            new_characteristics[new_phase]['phase_name'] = self._name_phase(new_phase, chars)
        
        self.phase_characteristics = new_characteristics
        
        return remapped_labels, event_info
    
    def _name_phase(self, phase_idx, characteristics):
        """Generate descriptive name for a life phase."""
        age_range = characteristics['age_range']
        mean_age = characteristics['mean_age']
        
        if mean_age < 12:
            return "Early Childhood"
        elif mean_age < 18:
            return "Adolescence"
        elif mean_age < 25:
            return "Early Adulthood"
        elif mean_age < 40:
            return "Middle Adulthood"
        else:
            return "Late Adulthood"
    
    def plot_life_phases(self, save_path='life_phases.png'):
        """Visualize the identified life phases."""
        if not self.phase_characteristics:
            print("No phases identified yet. Run segment_life_phases first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Age ranges for each phase
        phases = sorted(self.phase_characteristics.keys())
        for phase in phases:
            chars = self.phase_characteristics[phase]
            age_range = chars['age_range']
            
            ax1.barh(phase, age_range[1] - age_range[0], 
                    left=age_range[0], height=0.6,
                    label=chars['phase_name'])
            
            # Add mean age marker
            ax1.plot(chars['mean_age'], phase, 'ko', markersize=8)
        
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Life Phase')
        ax1.set_title('Criminal Life Phases by Age Range')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Event distribution across phases
        phase_names = [self.phase_characteristics[p]['phase_name'] for p in phases]
        event_counts = [self.phase_characteristics[p]['n_events'] for p in phases]
        
        ax2.bar(phase_names, event_counts, color='skyblue', edgecolor='navy')
        ax2.set_xlabel('Life Phase')
        ax2.set_ylabel('Number of Events')
        ax2.set_title('Event Distribution Across Life Phases')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Life phase visualization saved to {save_path}")


class SequentialPatternMiner:
    """
    Mine frequent sequential patterns from criminal life events.
    """
    
    def __init__(self, min_support=0.1, max_pattern_length=5):
        """
        Initialize pattern miner.
        
        Args:
            min_support: Minimum fraction of sequences containing pattern
            max_pattern_length: Maximum length of patterns to find
        """
        self.min_support = min_support
        self.max_pattern_length = max_pattern_length
        self.frequent_patterns = []
        
    def find_patterns(self, sequences):
        """
        Find frequent sequential patterns using a simplified PrefixSpan approach.
        
        Args:
            sequences: List of event sequences
            
        Returns:
            List of (pattern, support, occurrences) tuples
        """
        n_sequences = len(sequences)
        min_count = int(self.min_support * n_sequences)
        
        # Start with 1-item patterns
        patterns_by_length = defaultdict(list)
        
        # Find frequent 1-items
        item_counts = Counter()
        for seq in sequences:
            seen_items = set(seq)  # Count each item once per sequence
            for item in seen_items:
                item_counts[item] += 1
        
        frequent_items = {item for item, count in item_counts.items() 
                         if count >= min_count}
        
        for item in frequent_items:
            support = item_counts[item] / n_sequences
            patterns_by_length[1].append(([item], support))
        
        # Extend patterns iteratively
        for length in range(2, self.max_pattern_length + 1):
            if not patterns_by_length[length - 1]:
                break
            
            # Generate candidate patterns
            candidates = set()
            prev_patterns = patterns_by_length[length - 1]
            
            for pattern1, _ in prev_patterns:
                for pattern2, _ in prev_patterns:
                    # Try combining patterns
                    if pattern1[:-1] == pattern2[:-1]:
                        candidate = pattern1 + [pattern2[-1]]
                        candidates.add(tuple(candidate))
                    elif pattern1[1:] == pattern2[:-1]:
                        candidate = pattern1 + [pattern2[-1]]
                        candidates.add(tuple(candidate))
            
            # Count support for candidates
            for candidate in candidates:
                count = 0
                containing_sequences = []
                
                for seq_idx, seq in enumerate(sequences):
                    if self._contains_pattern(seq, candidate):
                        count += 1
                        containing_sequences.append(seq_idx)
                
                if count >= min_count:
                    support = count / n_sequences
                    patterns_by_length[length].append((list(candidate), support))
                    
                    self.frequent_patterns.append({
                        'pattern': list(candidate),
                        'length': length,
                        'support': support,
                        'count': count,
                        'sequences': containing_sequences
                    })
        
        # Sort by support
        self.frequent_patterns.sort(key=lambda x: x['support'], reverse=True)
        
        return self.frequent_patterns
    
    def _contains_pattern(self, sequence, pattern):
        """Check if sequence contains pattern as a subsequence."""
        pattern = list(pattern)
        pattern_idx = 0
        
        for item in sequence:
            if pattern_idx < len(pattern) and item == pattern[pattern_idx]:
                pattern_idx += 1
                if pattern_idx == len(pattern):
                    return True
        
        return False
    
    def find_association_rules(self, min_confidence=0.5):
        """
        Find association rules from frequent patterns.
        
        Args:
            min_confidence: Minimum confidence for rules
            
        Returns:
            List of (antecedent, consequent, confidence, support) tuples
        """
        rules = []
        
        for pattern_info in self.frequent_patterns:
            pattern = pattern_info['pattern']
            if len(pattern) < 2:
                continue
            
            # Try all possible splits
            for i in range(1, len(pattern)):
                antecedent = pattern[:i]
                consequent = pattern[i:]
                
                # Find support of antecedent
                antecedent_count = 0
                for seq in pattern_info['sequences']:
                    if self._contains_pattern(seq, antecedent):
                        antecedent_count += 1
                
                if antecedent_count > 0:
                    confidence = pattern_info['count'] / antecedent_count
                    
                    if confidence >= min_confidence:
                        rules.append({
                            'antecedent': antecedent,
                            'consequent': consequent,
                            'confidence': confidence,
                            'support': pattern_info['support'],
                            'lift': confidence / (pattern_info['count'] / len(pattern_info['sequences']))
                        })
        
        # Sort by confidence
        rules.sort(key=lambda x: x['confidence'], reverse=True)
        
        return rules


def analyze_temporal_patterns(sequences, sequences_with_ages=None):
    """
    Comprehensive temporal analysis of criminal sequences.
    
    Args:
        sequences: List of event sequences
        sequences_with_ages: Optional list of (sequence, ages) tuples
        
    Returns:
        Dictionary with all temporal analysis results
    """
    results = {}
    
    # 1. Change point detection
    print("Detecting change points...")
    detector = ChangePointDetector(method='bayesian')
    change_analysis = detector.analyze_change_points(sequences)
    results['change_points'] = change_analysis
    
    print(f"Found {change_analysis['total_change_points']} total change points")
    print(f"Average {change_analysis['avg_change_points_per_sequence']:.2f} per criminal")
    print("\nMost common transitions at change points:")
    for transition, count in change_analysis['common_transitions'][:5]:
        print(f"  {transition}: {count} times")
    
    # 2. Sequential pattern mining
    print("\nMining sequential patterns...")
    miner = SequentialPatternMiner(min_support=0.1)
    patterns = miner.find_patterns(sequences)
    results['frequent_patterns'] = patterns
    
    print(f"Found {len(patterns)} frequent patterns")
    print("\nTop 5 patterns:")
    for pattern_info in patterns[:5]:
        pattern = ' → '.join(pattern_info['pattern'])
        print(f"  {pattern}: {pattern_info['support']:.2%} support")
    
    # 3. Life phase segmentation (if ages provided)
    if sequences_with_ages:
        print("\nSegmenting life phases...")
        segmenter = LifePhaseSegmenter(n_phases=5)
        phase_labels, event_info = segmenter.segment_life_phases(sequences_with_ages)
        results['life_phases'] = segmenter.phase_characteristics
        
        print(f"Identified {len(segmenter.phase_characteristics)} life phases:")
        for phase, chars in segmenter.phase_characteristics.items():
            print(f"  {chars['phase_name']}: ages {chars['age_range'][0]:.0f}-{chars['age_range'][1]:.0f}")
    
    return results


if __name__ == "__main__":
    # Test the temporal analysis tools
    print("Testing Temporal Analysis Tools...")
    
    # Example sequences
    sequences = [
        ["normal_childhood", "school_success", "minor_delinquency", 
         "substance_abuse", "violent_crime", "arrest"],
        ["abusive_childhood", "school_problems", "gang_involvement",
         "drug_dealing", "murder", "arrest"],
        ["neglect", "foster_care", "runaway", "theft", "assault", "imprisonment"],
        ["normal_childhood", "mental_illness", "isolation", 
         "stalking", "kidnapping", "arrest"]
    ]
    
    # Run analysis
    results = analyze_temporal_patterns(sequences)
    
    print("\n" + "="*50)
    print("Temporal Analysis Complete!")
    print("="*50)