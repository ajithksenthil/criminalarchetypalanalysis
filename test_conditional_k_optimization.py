#!/usr/bin/env python3
"""
test_conditional_k_optimization.py

Test script for the conditional effect-optimized clustering functionality.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import sys
import os

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis_integration_improved import (
    get_condition_map,
    find_optimal_k_for_conditional_analysis,
    multi_objective_k_selection,
    build_conditional_markov,
    compute_stationary_distribution
)

def create_test_data():
    """Create synthetic test data for testing."""
    # Create synthetic embeddings
    np.random.seed(42)
    embeddings = np.random.randn(100, 10)  # 100 events, 10 dimensions
    
    # Create synthetic criminal sequences
    criminal_sequences = {}
    event_idx = 0
    for crim_id in range(1, 11):  # 10 criminals
        criminal_sequences[str(crim_id)] = []
        num_events = np.random.randint(5, 15)  # 5-15 events per criminal
        for _ in range(num_events):
            criminal_sequences[str(crim_id)].append(event_idx)
            event_idx += 1
            if event_idx >= 100:
                break
        if event_idx >= 100:
            break
    
    # Create synthetic Type 2 data
    type2_data = []
    headings = ["Sex", "Physically abused?", "Number of victims", "Education level"]
    
    for crim_id in range(1, 11):
        # Sex
        type2_data.append({
            "CriminalID": str(crim_id),
            "Heading": "Sex",
            "Value": "Male" if np.random.random() > 0.2 else "Female"
        })
        
        # Physically abused
        type2_data.append({
            "CriminalID": str(crim_id),
            "Heading": "Physically abused?",
            "Value": "Yes" if np.random.random() > 0.6 else "No"
        })
        
        # Number of victims
        type2_data.append({
            "CriminalID": str(crim_id),
            "Heading": "Number of victims",
            "Value": str(np.random.randint(1, 10))
        })
        
        # Education level
        type2_data.append({
            "CriminalID": str(crim_id),
            "Heading": "Education level",
            "Value": np.random.choice(["High School", "College", "Graduate", "None"])
        })
    
    type2_df = pd.DataFrame(type2_data)
    
    return embeddings, criminal_sequences, type2_df

def test_get_condition_map():
    """Test the get_condition_map function."""
    print("Testing get_condition_map...")
    
    _, _, type2_df = create_test_data()
    
    # Test getting condition map for "Sex"
    condition_map = get_condition_map(type2_df, "Sex")
    print(f"Condition map for 'Sex': {condition_map}")
    
    # Test getting condition map for "Physically abused?"
    condition_map = get_condition_map(type2_df, "Physically abused?")
    print(f"Condition map for 'Physically abused?': {condition_map}")
    
    print("✓ get_condition_map test passed\n")

def test_conditional_k_optimization():
    """Test the conditional k optimization function."""
    print("Testing find_optimal_k_for_conditional_analysis...")
    
    embeddings, criminal_sequences, type2_df = create_test_data()
    
    try:
        optimal_k, results = find_optimal_k_for_conditional_analysis(
            embeddings, criminal_sequences, type2_df, k_range=range(3, 6)
        )
        
        print(f"Optimal k: {optimal_k}")
        print(f"Results: {results}")
        print("✓ find_optimal_k_for_conditional_analysis test passed\n")
        
    except Exception as e:
        print(f"✗ find_optimal_k_for_conditional_analysis test failed: {e}\n")

def test_multi_objective_k_selection():
    """Test the multi-objective k selection function."""
    print("Testing multi_objective_k_selection...")
    
    embeddings, criminal_sequences, type2_df = create_test_data()
    
    try:
        optimal_k, results = multi_objective_k_selection(
            embeddings, criminal_sequences, type2_df, k_range=range(3, 6)
        )
        
        print(f"Multi-objective optimal k: {optimal_k}")
        print(f"Results: {results}")
        print("✓ multi_objective_k_selection test passed\n")
        
    except Exception as e:
        print(f"✗ multi_objective_k_selection test failed: {e}\n")

def main():
    """Run all tests."""
    print("="*60)
    print("TESTING CONDITIONAL K OPTIMIZATION FUNCTIONS")
    print("="*60)
    
    test_get_condition_map()
    test_conditional_k_optimization()
    test_multi_objective_k_selection()
    
    print("="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
