#!/usr/bin/env python3
"""
test_enhancements.py

Quick test to verify all enhancement modules work correctly.
"""

import numpy as np
import sys

def test_module(module_name, test_func):
    """Test a module and report results."""
    try:
        test_func()
        print(f"✓ {module_name:<30} - PASSED")
        return True
    except Exception as e:
        print(f"✗ {module_name:<30} - FAILED: {str(e)[:50]}...")
        return False

def test_markov_models():
    """Test higher-order Markov models."""
    from markov_models import HigherOrderMarkov
    
    sequences = [['A', 'B', 'C', 'D'], ['A', 'B', 'D', 'C']]
    model = HigherOrderMarkov(order=2)
    model.fit(sequences)
    patterns = model.find_common_patterns(min_frequency=1)
    assert len(patterns) > 0

def test_temporal_analysis():
    """Test temporal analysis."""
    from temporal_analysis import ChangePointDetector
    
    sequence = ['A', 'A', 'B', 'B', 'C', 'C']
    detector = ChangePointDetector()
    changes = detector.detect_change_points(sequence)
    assert isinstance(changes, list)

def test_ensemble_clustering():
    """Test ensemble clustering."""
    from ensemble_clustering import EnsembleClustering
    
    X = np.random.randn(100, 10)
    ensemble = EnsembleClustering(n_clusters=3, n_iterations=2)
    labels = ensemble.fit(X)
    assert len(labels) == 100

def test_statistical_validation():
    """Test statistical validation."""
    from statistical_validation import PermutationTest
    
    X = np.random.randn(100, 10)
    labels = np.random.randint(0, 3, 100)
    perm = PermutationTest(n_permutations=10)
    result = perm.test_clustering_significance(X, labels)
    assert 'p_value' in result

def test_trajectory_analysis():
    """Test trajectory analysis."""
    from trajectory_analysis import TrajectoryAnalyzer
    
    sequences = [['A', 'B', 'C'], ['D', 'E', 'F']] * 10
    analyzer = TrajectoryAnalyzer(n_trajectories=2)
    labels = analyzer.identify_trajectories(sequences)
    assert len(labels) == 20

def test_visualizations():
    """Test visualization creation."""
    from interactive_visualizations import create_sankey_diagram
    import os
    
    sequences = [[0, 1, 2], [1, 2, 0]]
    labels = [0, 1, 2]
    
    # Just test that function runs without error
    create_sankey_diagram(sequences, labels, save_path='test_sankey.html')
    os.remove('test_sankey.html')  # Clean up

def test_improved_clustering():
    """Test improved clustering."""
    from improved_clustering import improved_clustering
    
    X = np.random.randn(100, 10)
    labels, clusterer, metrics = improved_clustering(
        X, 
        n_clusters=3, 
        reduce_dims=True,
        n_components=10  # Appropriate for 10-feature data
    )
    assert len(labels) == 100
    assert 'silhouette_score' in metrics

def main():
    print("="*60)
    print("TESTING ENHANCEMENT MODULES")
    print("="*60)
    print()
    
    tests = [
        ("Higher-Order Markov Models", test_markov_models),
        ("Temporal Analysis", test_temporal_analysis),
        ("Ensemble Clustering", test_ensemble_clustering),
        ("Statistical Validation", test_statistical_validation),
        ("Trajectory Analysis", test_trajectory_analysis),
        ("Interactive Visualizations", test_visualizations),
        ("Improved Clustering", test_improved_clustering)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        if test_module(name, test_func):
            passed += 1
    
    print()
    print("="*60)
    print(f"RESULTS: {passed}/{total} modules passed")
    print("="*60)
    
    if passed == total:
        print("\n✓ All enhancement modules are working correctly!")
        print("\nYou can now run:")
        print("  - python run_analysis_improved.py --auto_k")
        print("  - python demo_enhanced_analysis.py")
        print("\nNote: The full enhanced_analysis_integration.py requires")
        print("      some refactoring to work with the current data format.")
        return 0
    else:
        print("\n✗ Some modules failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())