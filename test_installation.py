#!/usr/bin/env python3
"""
test_installation.py

Test if all required packages are installed correctly.
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    if package_name is None:
        package_name = module_name
    
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name:<25} - OK")
        return True
    except ImportError as e:
        print(f"✗ {package_name:<25} - MISSING ({str(e)})")
        return False

def main():
    print("="*60)
    print("Testing Enhanced Criminal Analysis Installation")
    print("="*60)
    print()
    
    # Core requirements
    print("Core Requirements:")
    print("-"*30)
    core_modules = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('networkx', 'NetworkX'),
        ('nltk', 'NLTK'),
        ('torch', 'PyTorch'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('openai', 'OpenAI'),
        ('jinja2', 'Jinja2'),
    ]
    
    core_ok = all(test_import(mod, name) for mod, name in core_modules)
    
    # Enhancement requirements
    print("\nEnhancement Requirements:")
    print("-"*30)
    enhancement_modules = [
        ('plotly', 'Plotly'),
        ('kneed', 'Kneed'),
        ('umap', 'UMAP'),
        ('tqdm', 'tqdm'),
        ('statsmodels', 'Statsmodels'),
    ]
    
    enhancement_ok = all(test_import(mod, name) for mod, name in enhancement_modules)
    
    # Optional requirements
    print("\nOptional Requirements:")
    print("-"*30)
    optional_modules = [
        ('hmmlearn', 'HMMLlearn'),
        ('ruptures', 'Ruptures'),
        ('prefixspan', 'PrefixSpan'),
        ('numba', 'Numba'),
    ]
    
    optional_status = [test_import(mod, name) for mod, name in optional_modules]
    
    # Test custom modules
    print("\nCustom Modules:")
    print("-"*30)
    custom_modules = [
        'markov_models',
        'temporal_analysis',
        'interactive_visualizations',
        'ensemble_clustering',
        'statistical_validation',
        'trajectory_analysis',
        'enhanced_analysis_integration',
    ]
    
    custom_ok = all(test_import(mod) for mod in custom_modules)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if core_ok and enhancement_ok and custom_ok:
        print("✓ All required packages are installed!")
        print("✓ The enhanced analysis should work correctly.")
        
        if not all(optional_status):
            print("\n⚠ Some optional packages are missing:")
            print("  This may limit some advanced features, but core functionality will work.")
            
        print("\nYou can now run:")
        print("  python run_enhanced_analysis.py --quick")
        
    else:
        print("✗ Some required packages are missing!")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        print("\nOr try minimal installation:")
        print("  pip install -r requirements_minimal.txt")
        
        sys.exit(1)
    
    # Test data directories
    print("\nData Directories:")
    print("-"*30)
    import os
    
    dirs_exist = True
    for dir_name in ['type1csvs', 'type2csvs']:
        if os.path.exists(dir_name):
            n_files = len([f for f in os.listdir(dir_name) if f.endswith('.csv')])
            print(f"✓ {dir_name:<15} - Found ({n_files} CSV files)")
        else:
            print(f"✗ {dir_name:<15} - Not found")
            dirs_exist = False
    
    if not dirs_exist:
        print("\n⚠ Data directories not found. Make sure to specify them when running analysis:")
        print("  python run_enhanced_analysis.py --type1_dir path/to/type1 --type2_dir path/to/type2")

if __name__ == "__main__":
    main()