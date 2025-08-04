#!/usr/bin/env python3
"""
enhanced_analysis_integration.py

Integration module for all enhancements to the criminal archetypal analysis.
Brings together advanced Markov models, temporal analysis, enhanced clustering,
statistical validation, and trajectory analysis.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Import all enhancement modules
from markov_models import HigherOrderMarkov, TimeVaryingMarkov, identify_critical_patterns
from temporal_analysis import ChangePointDetector, LifePhaseSegmenter, SequentialPatternMiner, analyze_temporal_patterns
from interactive_visualizations import create_sankey_diagram, create_3d_cluster_visualization, create_comprehensive_dashboard
from ensemble_clustering import EnsembleClustering, create_robust_clusters
from statistical_validation import PermutationTest, BootstrapValidation, MultipleTestingCorrection
from trajectory_analysis import TrajectoryAnalyzer, RiskAssessment, visualize_trajectories

# Import base modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis_integration import load_all_criminals_type1, load_all_type2_data
from data_cleaning import clean_type2_data
from data_matching import match_criminals_by_name
from improved_clustering import improved_clustering


class EnhancedCriminalAnalysis:
    """
    Main class integrating all enhancements for comprehensive criminal analysis.
    """
    
    def __init__(self, output_dir='enhanced_results'):
        """
        Initialize enhanced analysis.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.higher_order_markov = None
        self.time_varying_markov = None
        self.change_detector = ChangePointDetector()
        self.pattern_miner = SequentialPatternMiner()
        self.trajectory_analyzer = TrajectoryAnalyzer()
        self.risk_assessor = RiskAssessment()
        
        # Store results
        self.results = {}
        
    def run_enhanced_analysis(self, type1_dir, type2_dir, config=None):
        """
        Run complete enhanced analysis pipeline.
        
        Args:
            type1_dir: Directory with Type 1 data
            type2_dir: Directory with Type 2 data
            config: Optional configuration dictionary
            
        Returns:
            Comprehensive results dictionary
        """
        print("="*60)
        print("ENHANCED CRIMINAL ARCHETYPAL ANALYSIS")
        print("="*60)
        
        # Default configuration
        if config is None:
            config = {
                'markov_order': 3,
                'n_clusters': None,  # Auto-select
                'n_trajectories': 4,
                'use_ensemble': True,
                'n_permutations': 1000,
                'n_bootstrap': 1000
            }
        
        # 1. Data Loading and Preprocessing
        print("\n[1/8] Loading and preprocessing data...")
        data_results = self._load_and_preprocess(type1_dir, type2_dir)
        self.results['data'] = data_results
        
        # 2. Enhanced Clustering
        print("\n[2/8] Performing enhanced clustering...")
        clustering_results = self._enhanced_clustering(
            data_results['embeddings'],
            data_results['all_events'],
            config
        )
        self.results['clustering'] = clustering_results
        
        # 3. Higher-Order Markov Analysis
        print("\n[3/8] Building higher-order Markov models...")
        markov_results = self._higher_order_markov_analysis(
            data_results['sequences'],
            clustering_results['labels'],
            config
        )
        self.results['markov'] = markov_results
        
        # 4. Temporal Analysis
        print("\n[4/8] Analyzing temporal patterns...")
        temporal_results = self._temporal_analysis(
            data_results['sequences'],
            data_results.get('sequences_with_ages'),
            clustering_results['labels']
        )
        self.results['temporal'] = temporal_results
        
        # 5. Trajectory Analysis
        print("\n[5/8] Identifying criminal trajectories...")
        trajectory_results = self._trajectory_analysis(
            data_results['sequences'],
            data_results.get('ages'),
            clustering_results['embeddings']
        )
        self.results['trajectories'] = trajectory_results
        
        # 6. Statistical Validation
        print("\n[6/8] Performing statistical validation...")
        validation_results = self._statistical_validation(
            data_results['embeddings'],
            clustering_results['labels'],
            data_results['sequences'],
            config
        )
        self.results['validation'] = validation_results
        
        # 7. Interactive Visualizations
        print("\n[7/8] Creating interactive visualizations...")
        viz_results = self._create_visualizations(
            data_results,
            clustering_results,
            markov_results,
            temporal_results,
            trajectory_results
        )
        self.results['visualizations'] = viz_results
        
        # 8. Generate Comprehensive Report
        print("\n[8/8] Generating comprehensive report...")
        report_path = self._generate_report()
        self.results['report_path'] = report_path
        
        # Save all results
        self._save_results()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)
        
        return self.results
    
    def _load_and_preprocess(self, type1_dir, type2_dir):
        """Load and preprocess criminal data."""
        # Load Type 1 data
        type1_data = load_all_criminals_type1(type1_dir)
        
        # Load Type 2 data
        type2_df = load_all_type2_data(type2_dir)
        
        # Match criminals
        matched_criminals = []
        for crim_id, crim_data in type1_data.items():
            # Simple name matching - check if criminal exists in Type2
            if crim_id in type2_df['Name'].values or any(crim_id.lower() in name.lower() for name in type2_df['Name'].values):
                matched_criminals.append(crim_id)
        
        print(f"[INFO] Matched {len(matched_criminals)} criminals")
        
        # Process Type2 data
        df_clean = clean_type2_data(type2_df)
        
        # Create segments dict
        segments_dict = {}
        
        # Extract sequences and events
        all_events = []
        sequences = []
        sequences_with_ages = []
        
        for criminal_id, segments in segments_dict.items():
            criminal_events = []
            criminal_ages = []
            
            for segment in segments:
                events = segment['events']
                ages = segment.get('ages', [])
                
                criminal_events.extend(events)
                criminal_ages.extend(ages)
                all_events.extend(events)
            
            if criminal_events:
                sequences.append(criminal_events)
                if criminal_ages:
                    sequences_with_ages.append((criminal_events, criminal_ages))
        
        # Create embeddings
        from analysis_integration_improved import create_event_embeddings
        embeddings = create_event_embeddings(all_events, use_tfidf=True)
        
        return {
            'n_criminals': len(matched_data),
            'n_events': len(all_events),
            'all_events': all_events,
            'sequences': sequences,
            'sequences_with_ages': sequences_with_ages if sequences_with_ages else None,
            'embeddings': embeddings,
            'segments_dict': segments_dict
        }
    
    def _enhanced_clustering(self, embeddings, events, config):
        """Perform enhanced clustering with ensemble methods."""
        if config.get('use_ensemble', True):
            # Use ensemble clustering
            ensemble = create_robust_clusters(
                embeddings,
                n_clusters_range=(3, 10),
                methods=['kmeans', 'gmm', 'spectral', 'hierarchical']
            )
            
            labels = ensemble.final_labels
            metrics = ensemble.get_summary()
            confidence = ensemble.get_confidence_scores()
            
        else:
            # Use improved single method
            labels, clusterer, metrics = improved_clustering(
                embeddings,
                n_clusters=config.get('n_clusters'),
                method='kmeans',
                reduce_dims=True,
                auto_select_k=(config.get('n_clusters') is None)
            )
            confidence = None
        
        # Generate cluster names (simplified - would use LLM in production)
        n_clusters = len(np.unique(labels))
        cluster_names = [f"Archetype {i}" for i in range(n_clusters)]
        
        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'cluster_names': cluster_names,
            'metrics': metrics,
            'confidence_scores': confidence,
            'embeddings': embeddings
        }
    
    def _higher_order_markov_analysis(self, sequences, cluster_labels, config):
        """Analyze sequences with higher-order Markov models."""
        # Map events to clusters
        clustered_sequences = []
        event_idx = 0
        
        for seq in sequences:
            clustered_seq = []
            for _ in seq:
                if event_idx < len(cluster_labels):
                    clustered_seq.append(cluster_labels[event_idx])
                    event_idx += 1
            clustered_sequences.append(clustered_seq)
        
        # Build higher-order model
        order = config.get('markov_order', 3)
        self.higher_order_markov = HigherOrderMarkov(order=order)
        self.higher_order_markov.fit(clustered_sequences)
        
        # Find critical patterns
        critical_patterns = identify_critical_patterns(
            clustered_sequences,
            order=order,
            min_support=0.05
        )
        
        # Build time-varying model if age data available
        time_varying_results = None
        if hasattr(self, 'sequences_with_ages') and self.sequences_with_ages:
            self.time_varying_markov = TimeVaryingMarkov()
            # Convert to format needed
            time_sequences = []
            for seq, ages in self.sequences_with_ages:
                time_sequences.append((seq, ages))
            
            if time_sequences:
                self.time_varying_markov.fit(time_sequences)
                time_varying_results = self.time_varying_markov.compare_time_periods()
        
        return {
            'order': order,
            'common_patterns': self.higher_order_markov.find_common_patterns(),
            'critical_patterns': critical_patterns,
            'time_varying_analysis': time_varying_results
        }
    
    def _temporal_analysis(self, sequences, sequences_with_ages, cluster_labels):
        """Perform comprehensive temporal analysis."""
        # Change point detection
        self.change_detector = ChangePointDetector(method='bayesian')
        change_results = self.change_detector.analyze_change_points(sequences)
        
        # Sequential pattern mining
        self.pattern_miner = SequentialPatternMiner(min_support=0.1)
        frequent_patterns = self.pattern_miner.find_patterns(sequences)
        
        # Association rules
        rules = self.pattern_miner.find_association_rules(min_confidence=0.5)
        
        # Life phase segmentation if age data available
        life_phases = None
        if sequences_with_ages:
            segmenter = LifePhaseSegmenter(n_phases=5)
            phase_labels, event_info = segmenter.segment_life_phases(sequences_with_ages)
            life_phases = segmenter.phase_characteristics
            
            # Create visualization
            segmenter.plot_life_phases(
                os.path.join(self.output_dir, 'life_phases.png')
            )
        
        return {
            'change_points': change_results,
            'frequent_patterns': frequent_patterns[:20],  # Top 20
            'association_rules': rules[:20],  # Top 20
            'life_phases': life_phases
        }
    
    def _trajectory_analysis(self, sequences, ages, embeddings):
        """Analyze criminal development trajectories."""
        # Identify trajectories
        trajectory_labels = self.trajectory_analyzer.identify_trajectories(
            sequences, ages, embeddings
        )
        
        # Analyze transitions
        transition_analysis = self.trajectory_analyzer.analyze_trajectory_transitions(
            sequences, trajectory_labels
        )
        
        # Risk assessment for each criminal
        risk_scores = []
        for i, seq in enumerate(sequences):
            traj_type = trajectory_labels[i]
            traj_name = self.trajectory_analyzer.trajectory_profiles[traj_type]['name']
            
            current_age = ages[i][-1] if ages and i < len(ages) else None
            risk_score, components = self.risk_assessor.compute_risk_score(
                seq, traj_name, current_age
            )
            
            risk_level, action = self.risk_assessor.classify_risk_level(risk_score)
            
            risk_scores.append({
                'criminal_idx': i,
                'trajectory_type': traj_type,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'recommended_action': action,
                'risk_components': components
            })
        
        # Visualize trajectories
        visualize_trajectories(
            self.trajectory_analyzer,
            sequences,
            trajectory_labels,
            os.path.join(self.output_dir, 'trajectories.png')
        )
        
        return {
            'trajectory_labels': trajectory_labels,
            'trajectory_profiles': self.trajectory_analyzer.trajectory_profiles,
            'transition_analysis': transition_analysis,
            'risk_scores': risk_scores
        }
    
    def _statistical_validation(self, embeddings, cluster_labels, sequences, config):
        """Perform comprehensive statistical validation."""
        # Permutation test for clustering
        perm_test = PermutationTest(n_permutations=config.get('n_permutations', 1000))
        clustering_significance = perm_test.test_clustering_significance(
            embeddings, cluster_labels
        )
        
        # Bootstrap validation
        bootstrap = BootstrapValidation(n_bootstrap=config.get('n_bootstrap', 1000))
        
        # Bootstrap transition matrices
        transition_confidence = bootstrap.bootstrap_transition_matrix(sequences)
        
        # Multiple testing correction for patterns
        pattern_tests = {}
        if hasattr(self, 'pattern_miner') and self.pattern_miner.frequent_patterns:
            for i, pattern_info in enumerate(self.pattern_miner.frequent_patterns[:10]):
                pattern = pattern_info['pattern']
                test_result = perm_test.test_pattern_significance(sequences, pattern)
                pattern_tests[f'pattern_{i}'] = test_result
        
        # Apply multiple testing correction
        if pattern_tests:
            p_values = [result['permutation_p_value'] for result in pattern_tests.values()]
            mt_correction = MultipleTestingCorrection.correct_pvalues(p_values)
        else:
            mt_correction = None
        
        return {
            'clustering_significance': clustering_significance,
            'transition_confidence': transition_confidence,
            'pattern_tests': pattern_tests,
            'multiple_testing_correction': mt_correction
        }
    
    def _create_visualizations(self, data_results, clustering_results, 
                             markov_results, temporal_results, trajectory_results):
        """Create all interactive visualizations."""
        viz_paths = {}
        
        # 1. Sankey diagram
        sequences_indices = []
        for seq in data_results['sequences']:
            # Map sequence to indices
            seq_indices = list(range(len(seq)))
            sequences_indices.append(seq_indices)
        
        sankey_fig = create_sankey_diagram(
            sequences_indices,
            clustering_results['labels'],
            clustering_results['cluster_names'],
            os.path.join(self.output_dir, 'sankey_diagram.html')
        )
        viz_paths['sankey'] = 'sankey_diagram.html'
        
        # 2. 3D cluster visualization
        cluster_fig = create_3d_cluster_visualization(
            clustering_results['embeddings'],
            clustering_results['labels'],
            clustering_results['cluster_names'],
            method='tsne',
            save_path=os.path.join(self.output_dir, 'clusters_3d.html')
        )
        viz_paths['clusters_3d'] = 'clusters_3d.html'
        
        # 3. Comprehensive dashboard
        analysis_summary = {
            'cluster_sizes': dict(zip(*np.unique(clustering_results['labels'], 
                                               return_counts=True))),
            'change_points': temporal_results['change_points'],
            'cluster_labels': clustering_results['labels']
        }
        
        dashboard_fig = create_comprehensive_dashboard(
            analysis_summary,
            os.path.join(self.output_dir, 'dashboard.html')
        )
        viz_paths['dashboard'] = 'dashboard.html'
        
        return viz_paths
    
    def _generate_report(self):
        """Generate comprehensive analysis report."""
        report_content = f"""
# Enhanced Criminal Archetypal Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Executive Summary

This report presents the results of an enhanced criminal archetypal analysis using advanced machine learning and statistical methods.

### Key Findings

1. **Data Overview**
   - Analyzed {self.results['data']['n_criminals']} criminals
   - Total events: {self.results['data']['n_events']}

2. **Archetypal Patterns**
   - Identified {self.results['clustering']['n_clusters']} distinct criminal archetypes
   - Clustering quality (silhouette score): {self.results['clustering']['metrics'].get('final_silhouette', 'N/A'):.3f}

3. **Critical Patterns**
   - Found {len(self.results['markov']['critical_patterns'])} critical multi-step patterns
   - Most common pattern: {self.results['markov']['common_patterns'][0][0] if self.results['markov']['common_patterns'] else 'None'}

4. **Temporal Insights**
   - Average change points per criminal: {self.results['temporal']['change_points']['avg_change_points_per_sequence']:.2f}
   - Number of frequent sequential patterns: {len(self.results['temporal']['frequent_patterns'])}

5. **Trajectory Analysis**
   - Identified {len(self.results['trajectories']['trajectory_profiles'])} distinct criminal trajectories
   - High-risk individuals: {sum(1 for r in self.results['trajectories']['risk_scores'] if r['risk_level'] == 'Very High Risk')}

## Detailed Results

### 1. Enhanced Clustering Analysis

The ensemble clustering approach identified {self.results['clustering']['n_clusters']} robust archetypes:

"""
        
        # Add cluster details
        for i, name in enumerate(self.results['clustering']['cluster_names']):
            count = np.sum(self.results['clustering']['labels'] == i)
            report_content += f"- **{name}**: {count} events\n"
        
        # Add validation results
        if 'validation' in self.results:
            sig = self.results['validation']['clustering_significance']
            report_content += f"""

### 2. Statistical Validation

- Clustering significance (p-value): {sig['p_value']:.4f}
- Effect size: {sig['effect_size']:.3f}
- Result: {'Significant' if sig['significant'] else 'Not significant'}

"""
        
        # Add trajectory details
        report_content += "### 3. Criminal Trajectory Types\n\n"
        for traj_id, profile in self.results['trajectories']['trajectory_profiles'].items():
            report_content += f"""
**{profile['name']}**
- Number of criminals: {profile['n_criminals']}
- Average career length: {profile['avg_sequence_length']:.1f} events
- Common events: {', '.join(e[0] for e in profile['common_events'][:3])}

"""
        
        # Add recommendations
        report_content += """
## Recommendations

Based on the analysis, we recommend:

1. **Early Intervention**: Focus on individuals showing patterns associated with high-risk trajectories
2. **Targeted Programs**: Develop specific interventions for each identified archetype
3. **Monitoring**: Use change point detection to identify critical transitions
4. **Risk Assessment**: Apply trajectory-based risk scoring for resource allocation

## Interactive Visualizations

The following interactive visualizations are available:
- Sankey Diagram: `sankey_diagram.html`
- 3D Cluster Visualization: `clusters_3d.html`  
- Comprehensive Dashboard: `dashboard.html`
- Life Phases: `life_phases.png`
- Trajectory Analysis: `trajectories.png`

---
*This report was generated using enhanced criminal archetypal analysis with higher-order Markov models, 
ensemble clustering, temporal pattern mining, and trajectory analysis.*
"""
        
        # Save report
        report_path = os.path.join(self.output_dir, 'analysis_report.md')
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        return report_path
    
    def _save_results(self):
        """Save all results to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(self.results)
        
        # Save main results
        with open(os.path.join(self.output_dir, 'enhanced_results.json'), 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {self.output_dir}/enhanced_results.json")


def run_enhanced_analysis(type1_dir='type1csvs', type2_dir='type2csvs', 
                         output_dir='enhanced_results', config=None):
    """
    Convenience function to run enhanced analysis.
    
    Args:
        type1_dir: Directory with Type 1 data
        type2_dir: Directory with Type 2 data
        output_dir: Output directory
        config: Optional configuration
        
    Returns:
        Analysis results
    """
    analyzer = EnhancedCriminalAnalysis(output_dir)
    results = analyzer.run_enhanced_analysis(type1_dir, type2_dir, config)
    return results


if __name__ == "__main__":
    # Run enhanced analysis
    print("Starting Enhanced Criminal Archetypal Analysis...")
    
    config = {
        'markov_order': 3,
        'n_clusters': None,  # Auto-select
        'n_trajectories': 4,
        'use_ensemble': True,
        'n_permutations': 100,  # Reduced for demo
        'n_bootstrap': 100  # Reduced for demo
    }
    
    results = run_enhanced_analysis(config=config)
    
    print("\nEnhanced analysis complete!")
    print(f"Report available at: {results['report_path']}")