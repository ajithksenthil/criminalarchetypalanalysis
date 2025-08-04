#!/usr/bin/env python3
"""
trajectory_analysis.py

Analyze and classify criminal development trajectories.
Includes trajectory clustering, prediction, and risk assessment.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from scipy.interpolate import UnivariateSpline
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')


class TrajectoryAnalyzer:
    """
    Analyze criminal life trajectories to identify development patterns.
    """
    
    def __init__(self, n_trajectories=4):
        """
        Initialize trajectory analyzer.
        
        Args:
            n_trajectories: Number of trajectory types to identify
        """
        self.n_trajectories = n_trajectories
        self.trajectory_model = None
        self.trajectory_profiles = {}
        self.feature_scaler = StandardScaler()
        
    def extract_trajectory_features(self, sequences, ages=None, embeddings=None):
        """
        Extract features that characterize criminal trajectories.
        
        Args:
            sequences: List of event sequences
            ages: Optional ages for each event
            embeddings: Optional embeddings for each event
            
        Returns:
            Feature matrix for trajectories
        """
        features = []
        
        for seq_idx, seq in enumerate(sequences):
            seq_features = {}
            
            # 1. Sequence length and complexity
            seq_features['length'] = len(seq)
            seq_features['unique_events'] = len(set(seq))
            seq_features['diversity'] = seq_features['unique_events'] / seq_features['length']
            
            # 2. Temporal features (if ages provided)
            if ages and seq_idx < len(ages):
                seq_ages = ages[seq_idx]
                seq_features['start_age'] = seq_ages[0] if seq_ages else 0
                seq_features['end_age'] = seq_ages[-1] if seq_ages else 0
                seq_features['duration'] = seq_features['end_age'] - seq_features['start_age']
                
                # Rate of events
                if len(seq_ages) > 1:
                    time_diffs = np.diff(seq_ages)
                    seq_features['event_rate_mean'] = 1 / np.mean(time_diffs) if np.mean(time_diffs) > 0 else 0
                    seq_features['event_rate_std'] = np.std(1 / time_diffs[time_diffs > 0]) if any(time_diffs > 0) else 0
                else:
                    seq_features['event_rate_mean'] = 0
                    seq_features['event_rate_std'] = 0
            
            # 3. Transition patterns
            transitions = defaultdict(int)
            for i in range(len(seq) - 1):
                transition = f"{seq[i]}_to_{seq[i+1]}"
                transitions[transition] += 1
            
            seq_features['n_unique_transitions'] = len(transitions)
            seq_features['transition_entropy'] = entropy(list(transitions.values())) if transitions else 0
            
            # 4. Severity progression (simplified - would need domain knowledge)
            # For now, use position in sequence as proxy
            severity_progression = []
            for i, event in enumerate(seq):
                # Simple heuristic: violent crimes have "violen", "murder", "assault" etc
                severity = 1.0
                if any(term in str(event).lower() for term in ['violen', 'murder', 'assault', 'rape']):
                    severity = 3.0
                elif any(term in str(event).lower() for term in ['drug', 'theft', 'robbery']):
                    severity = 2.0
                severity_progression.append(severity * (i + 1) / len(seq))  # Weight by position
            
            seq_features['severity_slope'] = np.polyfit(range(len(severity_progression)), 
                                                       severity_progression, 1)[0] if len(seq) > 1 else 0
            seq_features['max_severity'] = max(severity_progression) if severity_progression else 0
            
            # 5. Embedding-based features (if provided)
            if embeddings is not None and seq_idx < len(embeddings):
                seq_embeddings = embeddings[seq_idx]
                if len(seq_embeddings) > 0:
                    # Trajectory in embedding space
                    if len(seq_embeddings) > 1:
                        # Compute distances between consecutive events
                        distances = []
                        for i in range(len(seq_embeddings) - 1):
                            dist = np.linalg.norm(seq_embeddings[i] - seq_embeddings[i+1])
                            distances.append(dist)
                        
                        seq_features['embedding_distance_mean'] = np.mean(distances)
                        seq_features['embedding_distance_std'] = np.std(distances)
                        seq_features['embedding_total_distance'] = sum(distances)
                    else:
                        seq_features['embedding_distance_mean'] = 0
                        seq_features['embedding_distance_std'] = 0
                        seq_features['embedding_total_distance'] = 0
            
            # Convert to list
            feature_list = [seq_features.get(key, 0) for key in sorted(seq_features.keys())]
            features.append(feature_list)
        
        return np.array(features)
    
    def identify_trajectories(self, sequences, ages=None, embeddings=None):
        """
        Identify distinct criminal trajectory types.
        
        Args:
            sequences: List of event sequences
            ages: Optional ages for events
            embeddings: Optional embeddings
            
        Returns:
            Trajectory labels for each sequence
        """
        print(f"Identifying {self.n_trajectories} criminal trajectory types...")
        
        # Extract features
        features = self.extract_trajectory_features(sequences, ages, embeddings)
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Fit Gaussian Mixture Model
        self.trajectory_model = GaussianMixture(
            n_components=self.n_trajectories,
            covariance_type='full',
            random_state=42,
            n_init=10
        )
        
        trajectory_labels = self.trajectory_model.fit_predict(features_scaled)
        
        # Compute trajectory profiles
        self._compute_trajectory_profiles(sequences, trajectory_labels, features, ages)
        
        # Sort trajectories by severity/risk
        self._sort_trajectories_by_risk()
        
        return trajectory_labels
    
    def _compute_trajectory_profiles(self, sequences, labels, features, ages):
        """Compute profiles for each trajectory type."""
        
        for traj_type in range(self.n_trajectories):
            mask = labels == traj_type
            traj_sequences = [seq for seq, m in zip(sequences, mask) if m]
            traj_features = features[mask]
            
            if len(traj_sequences) == 0:
                continue
            
            # Compute profile statistics
            profile = {
                'n_criminals': len(traj_sequences),
                'avg_sequence_length': np.mean([len(seq) for seq in traj_sequences]),
                'feature_means': np.mean(traj_features, axis=0),
                'feature_stds': np.std(traj_features, axis=0)
            }
            
            # Common events
            event_counts = Counter()
            for seq in traj_sequences:
                event_counts.update(seq)
            profile['common_events'] = event_counts.most_common(10)
            
            # Age statistics (if available)
            if ages is not None:
                traj_ages = [ages[i] for i, m in enumerate(mask) if m and i < len(ages)]
                if traj_ages:
                    all_ages = [age for age_list in traj_ages for age in age_list if age_list]
                    if all_ages:
                        profile['age_range'] = (min(all_ages), max(all_ages))
                        profile['mean_age'] = np.mean(all_ages)
            
            # Trajectory characteristics
            profile['name'] = self._name_trajectory(profile)
            
            self.trajectory_profiles[traj_type] = profile
    
    def _name_trajectory(self, profile):
        """Generate descriptive name for trajectory based on profile."""
        # Simple heuristic naming based on features
        avg_length = profile['avg_sequence_length']
        n_criminals = profile['n_criminals']
        
        # Look at common events
        common_events = profile['common_events']
        violent_count = sum(count for event, count in common_events 
                          if any(term in str(event).lower() 
                                for term in ['violen', 'murder', 'assault']))
        
        # Name based on characteristics
        if avg_length < 5:
            length_desc = "Brief"
        elif avg_length < 10:
            length_desc = "Moderate"
        else:
            length_desc = "Extended"
        
        if violent_count > len(common_events) * 0.3:
            severity_desc = "High-Violence"
        elif violent_count > 0:
            severity_desc = "Mixed-Severity"
        else:
            severity_desc = "Non-Violent"
        
        return f"{length_desc} {severity_desc} Trajectory"
    
    def _sort_trajectories_by_risk(self):
        """Sort trajectory types by estimated risk level."""
        risk_scores = []
        
        for traj_type, profile in self.trajectory_profiles.items():
            # Simple risk scoring
            risk = 0
            
            # Length indicates chronic behavior
            risk += profile['avg_sequence_length'] * 0.3
            
            # Violent events indicate severity
            violent_events = sum(1 for event, _ in profile['common_events']
                               if any(term in str(event).lower() 
                                     for term in ['violen', 'murder', 'assault']))
            risk += violent_events * 2
            
            # Early start age indicates early onset
            if 'mean_age' in profile:
                risk += max(0, 25 - profile['mean_age']) * 0.1
            
            risk_scores.append((traj_type, risk))
        
        # Sort by risk
        risk_scores.sort(key=lambda x: x[1])
        
        # Create mapping
        self.risk_ranking = {old_idx: new_idx 
                           for new_idx, (old_idx, _) in enumerate(risk_scores)}
    
    def predict_trajectory(self, sequence, age=None, embedding=None):
        """
        Predict trajectory type for a new sequence.
        
        Args:
            sequence: Event sequence
            age: Optional ages
            embedding: Optional embeddings
            
        Returns:
            Predicted trajectory type and probability
        """
        if self.trajectory_model is None:
            raise ValueError("Model not fitted yet")
        
        # Extract features
        features = self.extract_trajectory_features([sequence], 
                                                   [age] if age else None,
                                                   [embedding] if embedding else None)
        
        # Scale
        features_scaled = self.feature_scaler.transform(features)
        
        # Predict
        probs = self.trajectory_model.predict_proba(features_scaled)[0]
        predicted_type = np.argmax(probs)
        
        return predicted_type, probs[predicted_type]
    
    def analyze_trajectory_transitions(self, sequences, labels):
        """
        Analyze how criminals transition between trajectory types.
        
        Args:
            sequences: Event sequences
            labels: Trajectory labels
            
        Returns:
            Transition analysis results
        """
        # Split sequences into phases and check if trajectory type changes
        trajectory_changes = []
        
        for seq, label in zip(sequences, labels):
            if len(seq) < 6:  # Need enough events to split
                continue
            
            # Split into early and late phase
            mid_point = len(seq) // 2
            early_seq = seq[:mid_point]
            late_seq = seq[mid_point:]
            
            # Predict trajectory for each phase
            early_traj, _ = self.predict_trajectory(early_seq)
            late_traj, _ = self.predict_trajectory(late_seq)
            
            trajectory_changes.append({
                'original_label': label,
                'early_trajectory': early_traj,
                'late_trajectory': late_traj,
                'changed': early_traj != late_traj
            })
        
        # Analyze changes
        n_changed = sum(1 for tc in trajectory_changes if tc['changed'])
        change_rate = n_changed / len(trajectory_changes) if trajectory_changes else 0
        
        # Transition matrix
        n_types = self.n_trajectories
        transition_matrix = np.zeros((n_types, n_types))
        
        for tc in trajectory_changes:
            transition_matrix[tc['early_trajectory'], tc['late_trajectory']] += 1
        
        # Normalize
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_matrix / row_sums
        
        results = {
            'change_rate': change_rate,
            'n_changed': n_changed,
            'transition_matrix': transition_matrix,
            'trajectory_changes': trajectory_changes
        }
        
        return results


class RiskAssessment:
    """
    Assess risk levels based on trajectory analysis.
    """
    
    def __init__(self):
        self.risk_model = None
        self.risk_factors = {}
        
    def compute_risk_score(self, sequence, trajectory_type, age=None):
        """
        Compute risk score for a criminal.
        
        Args:
            sequence: Event sequence
            trajectory_type: Identified trajectory type
            age: Current age
            
        Returns:
            Risk score and components
        """
        risk_components = {}
        
        # 1. Trajectory-based risk
        trajectory_risk = {
            'Low-Risk': 0.2,
            'Moderate-Risk': 0.5,
            'High-Risk': 0.8,
            'Chronic-Offender': 0.9
        }
        risk_components['trajectory'] = trajectory_risk.get(trajectory_type, 0.5)
        
        # 2. Recent activity
        if len(sequence) >= 3:
            recent_events = sequence[-3:]
            violent_recent = sum(1 for event in recent_events
                               if any(term in str(event).lower() 
                                     for term in ['violen', 'murder', 'assault']))
            risk_components['recent_violence'] = violent_recent / 3
        else:
            risk_components['recent_violence'] = 0
        
        # 3. Escalation pattern
        escalation = self._detect_escalation(sequence)
        risk_components['escalation'] = escalation
        
        # 4. Age factor (younger violent offenders often higher risk)
        if age and age < 25:
            risk_components['age_factor'] = 0.3
        else:
            risk_components['age_factor'] = 0.1
        
        # 5. Criminal versatility (diverse crime types)
        crime_types = set()
        for event in sequence:
            if 'violen' in str(event).lower():
                crime_types.add('violent')
            elif 'drug' in str(event).lower():
                crime_types.add('drug')
            elif 'theft' in str(event).lower() or 'robbery' in str(event).lower():
                crime_types.add('property')
            elif 'sex' in str(event).lower() or 'rape' in str(event).lower():
                crime_types.add('sexual')
        
        risk_components['versatility'] = len(crime_types) / 4
        
        # Compute weighted risk score
        weights = {
            'trajectory': 0.3,
            'recent_violence': 0.25,
            'escalation': 0.2,
            'age_factor': 0.1,
            'versatility': 0.15
        }
        
        total_risk = sum(risk_components[key] * weights[key] 
                        for key in risk_components)
        
        return total_risk, risk_components
    
    def _detect_escalation(self, sequence):
        """Detect if criminal behavior is escalating."""
        if len(sequence) < 3:
            return 0
        
        # Simple severity scoring
        severities = []
        for event in sequence:
            if any(term in str(event).lower() for term in ['murder', 'kill']):
                severities.append(10)
            elif any(term in str(event).lower() for term in ['violen', 'assault', 'rape']):
                severities.append(7)
            elif any(term in str(event).lower() for term in ['armed', 'weapon']):
                severities.append(5)
            elif any(term in str(event).lower() for term in ['drug', 'theft']):
                severities.append(3)
            else:
                severities.append(1)
        
        # Check if severity is increasing
        if len(severities) > 1:
            # Fit linear trend
            x = np.arange(len(severities))
            slope, _ = np.polyfit(x, severities, 1)
            
            # Normalize slope
            escalation = min(1.0, max(0.0, slope / 2))
        else:
            escalation = 0
        
        return escalation
    
    def classify_risk_level(self, risk_score):
        """
        Classify risk into categories.
        
        Args:
            risk_score: Computed risk score (0-1)
            
        Returns:
            Risk level and recommended action
        """
        if risk_score < 0.3:
            level = "Low Risk"
            action = "Standard supervision"
        elif risk_score < 0.5:
            level = "Moderate Risk"
            action = "Enhanced monitoring"
        elif risk_score < 0.7:
            level = "High Risk"
            action = "Intensive supervision"
        else:
            level = "Very High Risk"
            action = "Maximum supervision and intervention"
        
        return level, action


def visualize_trajectories(trajectory_analyzer, sequences, labels, save_path='trajectories.png'):
    """
    Visualize criminal trajectories.
    
    Args:
        trajectory_analyzer: Fitted TrajectoryAnalyzer
        sequences: Event sequences
        labels: Trajectory labels
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Trajectory distribution
    ax = axes[0, 0]
    trajectory_counts = Counter(labels)
    
    names = [trajectory_analyzer.trajectory_profiles.get(i, {}).get('name', f'Type {i}')
             for i in sorted(trajectory_counts.keys())]
    counts = [trajectory_counts[i] for i in sorted(trajectory_counts.keys())]
    
    ax.bar(names, counts, color='skyblue', edgecolor='navy')
    ax.set_xlabel('Trajectory Type')
    ax.set_ylabel('Number of Criminals')
    ax.set_title('Distribution of Criminal Trajectories')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Trajectory characteristics
    ax = axes[0, 1]
    
    # Create radar chart for trajectory profiles
    categories = ['Avg Length', 'Violence', 'Versatility', 'Early Onset']
    
    for traj_type, profile in trajectory_analyzer.trajectory_profiles.items():
        values = [
            profile['avg_sequence_length'] / 20,  # Normalize
            sum(1 for e, _ in profile['common_events'][:5] 
                if 'violen' in str(e).lower()) / 5,
            profile.get('n_criminals', 0) / 100,  # Normalize
            1 - (profile.get('mean_age', 30) / 50) if 'mean_age' in profile else 0.5
        ]
        
        ax.plot(categories + [categories[0]], values + [values[0]], 
               'o-', label=profile['name'])
    
    ax.set_title('Trajectory Characteristics')
    ax.legend(bbox_to_anchor=(1.1, 1))
    
    # 3. Sequence length by trajectory
    ax = axes[1, 0]
    
    lengths_by_traj = defaultdict(list)
    for seq, label in zip(sequences, labels):
        lengths_by_traj[label].append(len(seq))
    
    box_data = [lengths_by_traj[i] for i in sorted(lengths_by_traj.keys())]
    box_labels = [trajectory_analyzer.trajectory_profiles.get(i, {}).get('name', f'Type {i}')
                  for i in sorted(lengths_by_traj.keys())]
    
    ax.boxplot(box_data, labels=box_labels)
    ax.set_xlabel('Trajectory Type')
    ax.set_ylabel('Sequence Length')
    ax.set_title('Criminal Career Length by Trajectory')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Transition patterns
    ax = axes[1, 1]
    
    # Analyze transitions
    trans_results = trajectory_analyzer.analyze_trajectory_transitions(sequences, labels)
    
    im = ax.imshow(trans_results['transition_matrix'], cmap='Blues', aspect='auto')
    ax.set_title(f'Trajectory Transitions (Change Rate: {trans_results["change_rate"]:.1%})')
    ax.set_xlabel('Late Phase Trajectory')
    ax.set_ylabel('Early Phase Trajectory')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Trajectory visualization saved to {save_path}")


if __name__ == "__main__":
    # Test trajectory analysis
    print("Testing Trajectory Analysis...")
    
    # Generate sample criminal sequences
    sequences = []
    ages = []
    
    # Type 1: Early onset, escalating
    for _ in range(30):
        seq = ['childhood_trauma', 'school_problems', 'petty_theft', 
               'drug_use', 'assault', 'armed_robbery', 'murder']
        age = [8, 12, 15, 16, 19, 22, 25]
        sequences.append(seq)
        ages.append(age)
    
    # Type 2: Late onset, property crimes
    for _ in range(25):
        seq = ['financial_problems', 'theft', 'fraud', 'theft', 'fraud']
        age = [35, 36, 37, 39, 40]
        sequences.append(seq)
        ages.append(age)
    
    # Type 3: Drug-related trajectory
    for _ in range(20):
        seq = ['peer_pressure', 'drug_use', 'drug_dealing', 
               'possession_arrest', 'drug_dealing', 'overdose']
        age = [16, 17, 19, 20, 22, 24]
        sequences.append(seq)
        ages.append(age)
    
    # Analyze trajectories
    analyzer = TrajectoryAnalyzer(n_trajectories=3)
    labels = analyzer.identify_trajectories(sequences, ages)
    
    print("\nIdentified Trajectory Types:")
    for traj_type, profile in analyzer.trajectory_profiles.items():
        print(f"\n{profile['name']}:")
        print(f"  - {profile['n_criminals']} criminals")
        print(f"  - Avg length: {profile['avg_sequence_length']:.1f} events")
        print(f"  - Common events: {profile['common_events'][:3]}")
    
    # Test risk assessment
    risk_assessor = RiskAssessment()
    
    test_sequence = ['assault', 'armed_robbery', 'murder']
    risk_score, components = risk_assessor.compute_risk_score(test_sequence, 'High-Violence', age=22)
    risk_level, action = risk_assessor.classify_risk_level(risk_score)
    
    print(f"\nRisk Assessment Example:")
    print(f"  Risk Score: {risk_score:.2f}")
    print(f"  Risk Level: {risk_level}")
    print(f"  Recommended: {action}")
    
    # Visualize
    visualize_trajectories(analyzer, sequences, labels)
    
    print("\nTrajectory analysis complete!")