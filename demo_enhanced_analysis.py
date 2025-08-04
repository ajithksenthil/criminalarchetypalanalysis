#!/usr/bin/env python3
"""
demo_enhanced_analysis.py

Demonstration of enhanced analysis features without full integration.
Shows how each enhancement works independently.
"""

import numpy as np
import os
from datetime import datetime

# Import enhancement modules
from markov_models import HigherOrderMarkov, identify_critical_patterns
from temporal_analysis import ChangePointDetector, SequentialPatternMiner
from ensemble_clustering import EnsembleClustering
from statistical_validation import PermutationTest, BootstrapValidation
from trajectory_analysis import TrajectoryAnalyzer, RiskAssessment

# For visualization
from interactive_visualizations import create_sankey_diagram, create_3d_cluster_visualization


def generate_demo_data():
    """Generate realistic demo criminal data."""
    print("[INFO] Generating demo criminal data...")
    
    # Define event types
    event_types = [
        # Childhood/early events
        'childhood_trauma', 'parental_abuse', 'parental_divorce', 'foster_care',
        'school_problems', 'bullying_victim', 'social_isolation',
        
        # Adolescent events  
        'truancy', 'vandalism', 'petty_theft', 'drug_experimentation',
        'gang_involvement', 'juvenile_arrest', 'school_expulsion',
        
        # Adult criminal events
        'assault', 'drug_dealing', 'armed_robbery', 'domestic_violence',
        'weapons_charge', 'murder', 'imprisonment'
    ]
    
    # Generate different criminal trajectories
    sequences = []
    ages = []
    
    # Type 1: Early onset escalating violence (30 criminals)
    for _ in range(30):
        seq = ['childhood_trauma', 'school_problems', 'bullying_victim',
               'truancy', 'petty_theft', 'juvenile_arrest',
               'assault', 'weapons_charge', 'armed_robbery', 'murder']
        age = [6, 8, 10, 13, 14, 15, 18, 20, 22, 25]
        sequences.append(seq)
        ages.append(age)
    
    # Type 2: Drug-related trajectory (25 criminals)
    for _ in range(25):
        seq = ['parental_divorce', 'social_isolation', 'drug_experimentation',
               'truancy', 'drug_dealing', 'drug_dealing', 'imprisonment']
        age = [10, 14, 15, 16, 18, 20, 22]
        sequences.append(seq)
        ages.append(age)
    
    # Type 3: Late onset property crimes (20 criminals)
    for _ in range(20):
        seq = ['foster_care', 'school_problems', 'petty_theft', 
               'petty_theft', 'armed_robbery']
        age = [8, 12, 25, 28, 30]
        sequences.append(seq)
        ages.append(age)
    
    # Create embeddings (simple one-hot encoding for demo)
    all_events = [event for seq in sequences for event in seq]
    unique_events = list(set(all_events))
    event_to_idx = {event: i for i, event in enumerate(unique_events)}
    
    # Create embeddings
    embeddings = np.zeros((len(all_events), len(unique_events)))
    idx = 0
    for seq in sequences:
        for event in seq:
            embeddings[idx, event_to_idx[event]] = 1
            idx += 1
    
    # Add some noise
    embeddings += np.random.normal(0, 0.1, embeddings.shape)
    
    return {
        'sequences': sequences,
        'ages': ages,
        'embeddings': embeddings,
        'all_events': all_events,
        'event_types': unique_events,
        'n_criminals': len(sequences)
    }


def demo_higher_order_markov(sequences):
    """Demonstrate higher-order Markov chains."""
    print("\n" + "="*60)
    print("1. HIGHER-ORDER MARKOV CHAINS")
    print("="*60)
    
    # Build 3rd order model
    model = HigherOrderMarkov(order=3)
    model.fit(sequences)
    
    # Find common patterns
    patterns = model.find_common_patterns(min_frequency=5, top_n=5)
    print("\nTop 5 three-step patterns:")
    for i, (pattern, count) in enumerate(patterns):
        print(f"{i+1}. {' → '.join(pattern)} (occurs {count} times)")
    
    # Predict next event
    test_sequence = ['childhood_trauma', 'school_problems', 'truancy']
    predictions = model.predict_next_state(test_sequence, top_k=3)
    
    print(f"\nGiven sequence: {' → '.join(test_sequence)}")
    print("Predicted next events:")
    for event, prob in predictions:
        print(f"  - {event}: {prob:.2%} probability")
    
    # Find critical patterns
    critical = identify_critical_patterns(sequences, order=3, min_support=0.2)
    print(f"\nFound {len(critical)} critical patterns (>20% support)")


def demo_temporal_analysis(sequences, ages):
    """Demonstrate temporal analysis features."""
    print("\n" + "="*60)
    print("2. TEMPORAL ANALYSIS")
    print("="*60)
    
    # Change point detection
    detector = ChangePointDetector(method='bayesian')
    
    # Analyze first few sequences
    print("\nChange points for first 3 criminals:")
    for i in range(3):
        change_points = detector.detect_change_points(sequences[i])
        if change_points:
            print(f"Criminal {i+1}: Change points at positions {change_points}")
            for cp in change_points:
                if 0 < cp < len(sequences[i]):
                    print(f"  - Transition: {sequences[i][cp-1]} → {sequences[i][cp]}")
    
    # Pattern mining
    miner = SequentialPatternMiner(min_support=0.3)
    patterns = miner.find_patterns(sequences[:30])  # Use subset for speed
    
    print(f"\nFrequent sequential patterns (>30% support):")
    for i, pattern in enumerate(patterns[:5]):
        print(f"{i+1}. {' → '.join(pattern['pattern'])} ({pattern['support']:.1%} support)")


def demo_ensemble_clustering(embeddings):
    """Demonstrate ensemble clustering."""
    print("\n" + "="*60)
    print("3. ENSEMBLE CLUSTERING")
    print("="*60)
    
    # Use subset for speed
    subset_embeddings = embeddings[:200]
    
    # Create ensemble
    ensemble = EnsembleClustering(
        n_clusters=3,
        methods=['kmeans', 'gmm', 'spectral'],
        n_iterations=5
    )
    
    labels = ensemble.fit(subset_embeddings)
    summary = ensemble.get_summary()
    
    print(f"\nEnsemble clustering results:")
    print(f"  - Final silhouette: {summary['ensemble_metrics']['final_silhouette']:.3f}")
    print(f"  - Consensus strength: {summary['ensemble_metrics']['consensus_strength']:.3f}")
    print(f"  - Method agreement: {summary['ensemble_metrics']['method_agreement']:.3f}")
    
    # Get confidence scores
    confidence = ensemble.get_confidence_scores()
    print(f"  - Average confidence: {np.mean(confidence):.3f}")
    print(f"  - Low confidence samples: {np.sum(confidence < 0.5)}")
    
    return labels[:len(embeddings)]  # Extend labels for visualization


def demo_statistical_validation(embeddings, labels, sequences):
    """Demonstrate statistical validation."""
    print("\n" + "="*60)
    print("4. STATISTICAL VALIDATION")  
    print("="*60)
    
    # Permutation test
    perm_test = PermutationTest(n_permutations=100)  # Reduced for speed
    
    print("\nTesting clustering significance...")
    cluster_test = perm_test.test_clustering_significance(embeddings[:200], labels[:200])
    
    print(f"  - Observed silhouette: {cluster_test['observed_score']:.3f}")
    print(f"  - Random expectation: {cluster_test['permuted_mean']:.3f}")
    print(f"  - P-value: {cluster_test['p_value']:.4f}")
    print(f"  - Significant: {'Yes' if cluster_test['significant'] else 'No'}")
    
    # Bootstrap validation
    bootstrap = BootstrapValidation(n_bootstrap=100)  # Reduced for speed
    
    print("\nBootstrap confidence intervals for patterns...")
    # Test a specific pattern
    pattern = ['childhood_trauma', 'school_problems', 'truancy']
    pattern_test = perm_test.test_pattern_significance(sequences[:30], pattern)
    
    print(f"  - Pattern: {' → '.join(pattern)}")
    print(f"  - Observed frequency: {pattern_test['observed_frequency']:.2%}")
    print(f"  - Fold enrichment: {pattern_test['fold_enrichment']:.2f}x")
    print(f"  - P-value: {pattern_test['permutation_p_value']:.4f}")


def demo_trajectory_analysis(sequences, ages):
    """Demonstrate trajectory analysis."""
    print("\n" + "="*60)
    print("5. TRAJECTORY ANALYSIS")
    print("="*60)
    
    # Identify trajectories
    analyzer = TrajectoryAnalyzer(n_trajectories=3)
    trajectory_labels = analyzer.identify_trajectories(sequences, ages)
    
    print("\nIdentified criminal trajectories:")
    for traj_id, profile in analyzer.trajectory_profiles.items():
        print(f"\n{profile['name']}:")
        print(f"  - {profile['n_criminals']} criminals")
        print(f"  - Average length: {profile['avg_sequence_length']:.1f} events")
        print(f"  - Age range: {profile.get('age_range', 'N/A')}")
    
    # Risk assessment
    risk_assessor = RiskAssessment()
    
    print("\nRisk assessment examples:")
    for i in range(3):
        traj_type = trajectory_labels[i]
        traj_name = analyzer.trajectory_profiles[traj_type]['name']
        
        risk_score, components = risk_assessor.compute_risk_score(
            sequences[i], traj_name, ages[i][-1]
        )
        risk_level, action = risk_assessor.classify_risk_level(risk_score)
        
        print(f"\nCriminal {i+1}:")
        print(f"  - Trajectory: {traj_name}")
        print(f"  - Risk score: {risk_score:.2f}")
        print(f"  - Risk level: {risk_level}")
        print(f"  - Recommendation: {action}")


def demo_visualizations(sequences, embeddings, labels, output_dir):
    """Create demo visualizations."""
    print("\n" + "="*60)
    print("6. INTERACTIVE VISUALIZATIONS")
    print("="*60)
    
    # Create sequence indices
    sequence_indices = []
    event_idx = 0
    for seq in sequences:
        indices = list(range(event_idx, event_idx + len(seq)))
        sequence_indices.append(indices)
        event_idx += len(seq)
    
    # Sankey diagram
    print("\nCreating Sankey diagram...")
    sankey_path = os.path.join(output_dir, 'demo_sankey.html')
    create_sankey_diagram(
        sequence_indices[:30],  # Use subset
        labels,
        cluster_names=['Early Violence', 'Drug-Related', 'Property Crime'],
        save_path=sankey_path
    )
    print(f"  - Saved to: {sankey_path}")
    
    # 3D visualization
    print("\nCreating 3D cluster visualization...")
    viz3d_path = os.path.join(output_dir, 'demo_clusters_3d.html')
    create_3d_cluster_visualization(
        embeddings[:200],  # Use subset
        labels[:200],
        cluster_names=['Early Violence', 'Drug-Related', 'Property Crime'],
        method='pca',  # Faster than t-SNE
        save_path=viz3d_path
    )
    print(f"  - Saved to: {viz3d_path}")


def main():
    """Run all demonstrations."""
    print("="*60)
    print("ENHANCED CRIMINAL ANALYSIS - DEMONSTRATION")
    print("="*60)
    
    # Create output directory
    output_dir = f"demo_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate demo data
    data = generate_demo_data()
    print(f"\n[INFO] Generated data for {data['n_criminals']} criminals")
    print(f"[INFO] Total events: {len(data['all_events'])}")
    print(f"[INFO] Event types: {len(data['event_types'])}")
    
    # Run demonstrations
    demo_higher_order_markov(data['sequences'])
    demo_temporal_analysis(data['sequences'], data['ages'])
    labels = demo_ensemble_clustering(data['embeddings'])
    demo_statistical_validation(data['embeddings'], labels, data['sequences'])
    demo_trajectory_analysis(data['sequences'], data['ages'])
    demo_visualizations(data['sequences'], data['embeddings'], labels, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}/")
    print("\nKey insights from demo:")
    print("  1. Higher-order patterns capture multi-step criminal progressions")
    print("  2. Change points identify critical life transitions")
    print("  3. Ensemble clustering provides robust criminal archetypes")
    print("  4. Statistical validation ensures findings aren't due to chance")
    print("  5. Trajectory analysis enables risk assessment")
    print("  6. Interactive visualizations reveal complex relationships")
    
    print("\nTo run on real data:")
    print("  python run_enhanced_analysis.py --quick")


if __name__ == "__main__":
    main()