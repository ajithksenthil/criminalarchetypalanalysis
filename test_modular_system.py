#!/usr/bin/env python3
"""
test_modular_system.py

Test script to verify the modular system works correctly.
"""

import sys
import os
import tempfile
import shutil
import pandas as pd
import numpy as np

# Add modular_analysis to path
modular_path = os.path.join(os.path.dirname(__file__), 'modular_analysis')
sys.path.insert(0, modular_path)
sys.path.insert(0, os.path.dirname(__file__))

def create_test_data():
    """Create minimal test data for testing."""
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    type1_dir = os.path.join(temp_dir, "type1")
    type2_dir = os.path.join(temp_dir, "type2")
    output_dir = os.path.join(temp_dir, "output")
    
    os.makedirs(type1_dir)
    os.makedirs(type2_dir)
    
    # Create Type 1 data (life events)
    type1_data = {
        "1": [
            {"Date": "2000-01-01", "Age": "15", "Life Event": "childhood trauma"},
            {"Date": "2005-01-01", "Age": "20", "Life Event": "substance abuse"},
            {"Date": "2010-01-01", "Age": "25", "Life Event": "violent behavior"}
        ],
        "2": [
            {"Date": "1995-01-01", "Age": "18", "Life Event": "family problems"},
            {"Date": "2000-01-01", "Age": "23", "Life Event": "criminal activity"},
            {"Date": "2005-01-01", "Age": "28", "Life Event": "arrest"}
        ],
        "3": [
            {"Date": "1990-01-01", "Age": "16", "Life Event": "school dropout"},
            {"Date": "1995-01-01", "Age": "21", "Life Event": "drug dealing"},
            {"Date": "2000-01-01", "Age": "26", "Life Event": "violence"}
        ]
    }
    
    for crim_id, events in type1_data.items():
        df = pd.DataFrame(events)
        df.to_csv(os.path.join(type1_dir, f"Type1_{crim_id}.csv"), index=False)
    
    # Create Type 2 data (structured data)
    type2_data = []
    for crim_id in ["1", "2", "3"]:
        type2_data.extend([
            {"CriminalID": crim_id, "Heading": "Sex", "Value": "Male"},
            {"CriminalID": crim_id, "Heading": "Physically abused?", "Value": "Yes" if crim_id in ["1", "3"] else "No"},
            {"CriminalID": crim_id, "Heading": "Number of victims", "Value": str(int(crim_id) + 2)}
        ])
    
    for crim_id in ["1", "2", "3"]:
        crim_data = [row for row in type2_data if row["CriminalID"] == crim_id]
        df = pd.DataFrame(crim_data)
        df.to_csv(os.path.join(type2_dir, f"Type2_{crim_id}.csv"), index=False)
    
    return temp_dir, type1_dir, type2_dir, output_dir

def test_basic_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from core.config import AnalysisConfig, setup_environment
        from data.loaders import Type1DataLoader, Type2DataLoader
        from data.text_processing import TextPreprocessor, EmbeddingGenerator
        from clustering.basic_clustering import BasicClusterer
        from clustering.conditional_optimization import ConditionalEffectOptimizer
        from markov.transition_analysis import TransitionMatrixBuilder
        from visualization.diagrams import TransitionDiagramGenerator
        from integration.pipeline import CriminalArchetypalAnalysisPipeline
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    print("Testing data loading...")
    
    try:
        temp_dir, type1_dir, type2_dir, output_dir = create_test_data()
        
        # Test Type 1 loading
        from data.loaders import Type1DataLoader
        type1_loader = Type1DataLoader(type1_dir)
        criminals_data = type1_loader.load_all_criminals()
        
        assert len(criminals_data) == 3, f"Expected 3 criminals, got {len(criminals_data)}"
        assert all(len(data["events"]) == 3 for data in criminals_data.values()), "Each criminal should have 3 events"
        
        # Test Type 2 loading
        from data.loaders import Type2DataLoader
        type2_loader = Type2DataLoader(type2_dir)
        type2_df = type2_loader.load_data()
        
        assert len(type2_df) == 9, f"Expected 9 Type 2 records, got {len(type2_df)}"
        assert "CriminalID" in type2_df.columns, "Type 2 data should have CriminalID column"
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print("✓ Data loading test passed")
        return True
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return False

def test_text_processing():
    """Test text processing functionality."""
    print("Testing text processing...")
    
    try:
        from data.text_processing import TextPreprocessor, EmbeddingGenerator
        
        # Test preprocessing
        processor = TextPreprocessor()
        text = "This is a test sentence with numbers 123 and punctuation!"
        processed = processor.preprocess_text(text)
        
        assert isinstance(processed, str), "Processed text should be a string"
        assert "123" not in processed, "Numbers should be removed"
        
        # Test embedding generation (TF-IDF to avoid model download)
        generator = EmbeddingGenerator(use_tfidf=True)
        sentences = ["childhood trauma", "substance abuse", "violent behavior"]
        embeddings = generator.generate_embeddings(sentences)
        
        assert embeddings.shape[0] == 3, f"Expected 3 embeddings, got {embeddings.shape[0]}"
        assert embeddings.shape[1] > 0, "Embeddings should have features"
        
        print("✓ Text processing test passed")
        return True
        
    except Exception as e:
        print(f"✗ Text processing test failed: {e}")
        return False

def test_clustering():
    """Test clustering functionality."""
    print("Testing clustering...")
    
    try:
        from clustering.basic_clustering import BasicClusterer
        
        # Create simple test data
        np.random.seed(42)
        embeddings = np.random.randn(20, 5)  # 20 samples, 5 features
        
        clusterer = BasicClusterer()
        labels, model = clusterer.kmeans_cluster(embeddings, n_clusters=3)
        
        assert len(labels) == 20, f"Expected 20 labels, got {len(labels)}"
        assert len(set(labels)) <= 3, f"Expected at most 3 clusters, got {len(set(labels))}"
        
        # Test evaluation
        metrics = clusterer.evaluate_clustering(embeddings, labels)
        assert 'silhouette' in metrics, "Metrics should include silhouette score"
        assert 'n_clusters' in metrics, "Metrics should include number of clusters"
        
        print("✓ Clustering test passed")
        return True
        
    except Exception as e:
        print(f"✗ Clustering test failed: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    print("Testing configuration...")
    
    try:
        from core.config import AnalysisConfig
        
        # Test configuration creation
        config = AnalysisConfig(
            type1_dir="test_type1",
            type2_csv="test_type2.csv",
            output_dir="test_output",
            n_clusters=5,
            auto_k=True
        )
        
        assert config.type1_dir == "test_type1"
        assert config.n_clusters == 5
        assert config.auto_k == True
        
        # Test dictionary conversion
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['n_clusters'] == 5
        
        print("✓ Configuration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("TESTING MODULAR CRIMINAL ARCHETYPAL ANALYSIS SYSTEM")
    print("="*60)
    
    tests = [
        test_basic_imports,
        test_configuration,
        test_data_loading,
        test_text_processing,
        test_clustering
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! The modular system is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
