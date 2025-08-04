#!/usr/bin/env python3
"""
verify_modular_completeness.py

Verify that the modular system has all the functionality from the original script.
"""

import ast
import os
import re
from typing import Set, List, Dict

def extract_functions_from_file(filepath: str) -> Set[str]:
    """Extract all function names from a Python file."""
    if not os.path.exists(filepath):
        return set()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        functions = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.add(node.name)
        
        return functions
    except:
        return set()

def extract_functions_from_directory(directory: str) -> Set[str]:
    """Extract all function names from all Python files in a directory."""
    all_functions = set()
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                filepath = os.path.join(root, file)
                functions = extract_functions_from_file(filepath)
                all_functions.update(functions)
    
    return all_functions

def check_command_line_arguments():
    """Check if all command line arguments are supported."""
    original_args = {
        "--type1_dir", "--type2_csv", "--output_dir", "--n_clusters",
        "--no_llm", "--multi_modal", "--train_proto_net", "--use_tfidf",
        "--match_only", "--auto_k"
    }
    
    # Check modular main.py
    modular_main_path = "modular_analysis/main.py"
    if os.path.exists(modular_main_path):
        with open(modular_main_path, 'r') as f:
            content = f.read()
        
        modular_args = set(re.findall(r'--\w+', content))
        
        missing_args = original_args - modular_args
        extra_args = modular_args - original_args
        
        print("📋 COMMAND LINE ARGUMENTS")
        print("-" * 40)
        print(f"✅ Original arguments: {len(original_args)}")
        print(f"✅ Modular arguments: {len(modular_args)}")
        
        if missing_args:
            print(f"❌ Missing arguments: {missing_args}")
        else:
            print("✅ All original arguments supported")
        
        if extra_args:
            print(f"➕ Additional arguments: {extra_args}")
        
        return len(missing_args) == 0
    else:
        print("❌ Modular main.py not found")
        return False

def check_key_functionality():
    """Check if key functionality is present in the modular system."""
    key_functions = {
        # Data loading
        "load_all_criminals_type1": "data loading",
        "load_type2_data": "data loading", 
        "load_matched_criminal_data": "matched data loading",
        
        # Text processing
        "preprocess_text": "text preprocessing",
        "generate_embeddings": "embedding generation",
        "get_imputed_embedding": "lexical augmentation",
        
        # Clustering
        "kmeans_cluster": "basic clustering",
        "improved_clustering": "improved clustering",
        "find_optimal_k_for_conditional_analysis": "conditional k optimization",
        
        # Markov analysis
        "build_conditional_markov": "conditional markov",
        "compute_stationary_distribution": "stationary distribution",
        "analyze_all_conditional_insights": "conditional insights",
        
        # Visualization
        "plot_state_transition_diagram": "transition diagrams",
        "plot_tsne_embeddings": "t-SNE visualization",
        
        # LLM analysis
        "analyze_cluster_with_llm": "LLM cluster analysis",
        
        # Regression
        "integrated_logistic_regression_analysis": "logistic regression",
        
        # Prototypical networks
        "train_prototypical_network": "prototypical networks",
        
        # Multi-modal
        "get_extended_type2_vector": "extended multi-modal features"
    }
    
    # Get all functions from original file
    original_functions = extract_functions_from_file("analysis_integration_improved.py")
    
    # Get all functions from modular system
    modular_functions = extract_functions_from_directory("modular_analysis")
    
    print("\n🔧 KEY FUNCTIONALITY CHECK")
    print("-" * 40)
    
    missing_functions = []
    present_functions = []
    
    for func_name, description in key_functions.items():
        if func_name in original_functions:
            if func_name in modular_functions:
                present_functions.append((func_name, description))
                print(f"✅ {description}: {func_name}")
            else:
                missing_functions.append((func_name, description))
                print(f"❌ {description}: {func_name} - MISSING")
        else:
            print(f"⚠️  {description}: {func_name} - NOT IN ORIGINAL")
    
    print(f"\n📊 Summary: {len(present_functions)}/{len(key_functions)} key functions present")
    
    return len(missing_functions) == 0

def check_file_structure():
    """Check if the modular file structure is complete."""
    expected_structure = {
        "modular_analysis/main.py": "Main entry point",
        "modular_analysis/core/config.py": "Configuration management",
        "modular_analysis/data/loaders.py": "Data loading",
        "modular_analysis/data/text_processing.py": "Text processing",
        "modular_analysis/data/matching.py": "Data matching",
        "modular_analysis/clustering/basic_clustering.py": "Basic clustering",
        "modular_analysis/clustering/conditional_optimization.py": "Conditional optimization",
        "modular_analysis/clustering/improved_clustering.py": "Improved clustering",
        "modular_analysis/clustering/prototypical_network.py": "Prototypical networks",
        "modular_analysis/markov/transition_analysis.py": "Markov analysis",
        "modular_analysis/visualization/diagrams.py": "Visualizations",
        "modular_analysis/integration/pipeline.py": "Main pipeline",
        "modular_analysis/integration/llm_analysis.py": "LLM analysis",
        "modular_analysis/integration/regression_analysis.py": "Regression analysis",
        "modular_analysis/integration/report_generator.py": "Report generation",
        "modular_analysis/utils/helpers.py": "Utility functions"
    }
    
    print("\n📁 FILE STRUCTURE CHECK")
    print("-" * 40)
    
    missing_files = []
    present_files = []
    
    for filepath, description in expected_structure.items():
        if os.path.exists(filepath):
            present_files.append((filepath, description))
            print(f"✅ {description}: {filepath}")
        else:
            missing_files.append((filepath, description))
            print(f"❌ {description}: {filepath} - MISSING")
    
    print(f"\n📊 Summary: {len(present_files)}/{len(expected_structure)} files present")
    
    return len(missing_files) == 0

def check_imports():
    """Check if all necessary imports are available."""
    print("\n📦 IMPORT CHECK")
    print("-" * 40)
    
    try:
        # Test core imports
        import sys
        import os
        sys.path.append('modular_analysis')
        
        from core.config import AnalysisConfig
        print("✅ Core configuration")
        
        from data.loaders import Type1DataLoader, Type2DataLoader
        print("✅ Data loaders")
        
        from clustering.basic_clustering import BasicClusterer
        print("✅ Basic clustering")
        
        from markov.transition_analysis import TransitionMatrixBuilder
        print("✅ Markov analysis")
        
        from integration.pipeline import CriminalArchetypalAnalysisPipeline
        print("✅ Main pipeline")
        
        print("\n✅ All critical imports successful")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        return False

def main():
    """Run all verification checks."""
    print("="*60)
    print("MODULAR SYSTEM COMPLETENESS VERIFICATION")
    print("="*60)
    
    checks = [
        ("File Structure", check_file_structure),
        ("Command Line Arguments", check_command_line_arguments),
        ("Key Functionality", check_key_functionality),
        ("Import System", check_imports)
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n🔍 Running {check_name} check...")
        try:
            if check_func():
                passed_checks += 1
                print(f"✅ {check_name}: PASSED")
            else:
                print(f"❌ {check_name}: FAILED")
        except Exception as e:
            print(f"❌ {check_name}: ERROR - {e}")
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Checks passed: {passed_checks}/{total_checks}")
    
    if passed_checks == total_checks:
        print("🎉 ALL CHECKS PASSED!")
        print("The modular system appears to have complete functionality.")
    else:
        print("⚠️  Some checks failed.")
        print("The modular system may be missing some functionality.")
    
    print("\n📋 NEXT STEPS:")
    if passed_checks == total_checks:
        print("✅ The modular system is ready for testing")
        print("✅ Try running: python modular_analysis/main.py --help")
        print("✅ Test with your actual data")
    else:
        print("🔧 Fix the failing checks above")
        print("🔧 Ensure all modules are properly implemented")
        print("🔧 Test imports and functionality")

if __name__ == "__main__":
    main()
