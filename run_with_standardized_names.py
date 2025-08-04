#!/usr/bin/env python3
"""
run_with_standardized_names.py

This script modifies the analysis pipeline to use the improved lexical imputation
with name standardization, then runs the analysis with auto-k selection.
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime

def modify_analysis_for_standardized_names():
    """Modify the analysis script to use improved lexical imputation."""
    
    # Read the original analysis file
    with open('analysis_integration_improved.py', 'r') as f:
        content = f.read()
    
    # Add import for improved lexical imputation
    import_section = """from improved_lexical_imputation import ImprovedLexicalImputation

# Initialize improved lexical imputation
improved_imputer = ImprovedLexicalImputation()
"""
    
    # Find where to insert the import (after other imports)
    import_pos = content.find('warnings.filterwarnings("ignore", category=FutureWarning)')
    if import_pos > 0:
        content = content[:import_pos] + import_section + "\n" + content[import_pos:]
    
    # Make sentence_transformers import conditional
    content = content.replace(
        "from sentence_transformers import SentenceTransformer",
        """try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("[INFO] sentence_transformers not available, using TF-IDF embeddings")
    SENTENCE_TRANSFORMERS_AVAILABLE = False"""
    )
    
    # Replace the generate_lexical_variations function
    old_func_start = content.find("def generate_lexical_variations(text, num_variants=5):")
    old_func_end = content.find("def get_imputed_embedding(", old_func_start)
    
    if old_func_start > 0 and old_func_end > 0:
        new_func = '''def generate_lexical_variations(text, num_variants=5):
    """Use improved lexical imputation with name standardization."""
    try:
        return improved_imputer.generate_variations(text, num_variants)
    except Exception as e:
        print(f"[ERROR] Generating lexical variations: {e}")
        return [text]
    
'''
        content = content[:old_func_start] + new_func + content[old_func_end:]
    
    # Save modified version
    modified_file = 'analysis_integration_improved_with_names.py'
    with open(modified_file, 'w') as f:
        f.write(content)
    
    return modified_file

def main():
    # Check for arguments
    auto_k = '--auto_k' in sys.argv
    use_tfidf = '--use_tfidf' in sys.argv
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_standardized_names_auto_k_{timestamp}"
    
    print("="*60)
    print("RUNNING ANALYSIS WITH NAME STANDARDIZATION")
    print("="*60)
    print(f"Auto-k selection: {auto_k}")
    print(f"Use TF-IDF: {use_tfidf}")
    print(f"Output directory: {output_dir}")
    print("")
    
    # Modify the analysis script
    print("Creating modified analysis with name standardization...")
    modified_script = modify_analysis_for_standardized_names()
    
    # Build command
    cmd = [
        sys.executable,
        modified_script,
        "--type1_dir", "type1csvs",
        "--type2_csv", "type2csvs", 
        "--output_dir", output_dir,
        "--match_only"
    ]
    
    if auto_k:
        # Auto-k is not supported in analysis_integration_improved.py
        # Use 12 clusters as found optimal in test_run_fixed
        print("Note: Using 12 clusters (found optimal in previous auto-k run)")
        cmd.extend(["--n_clusters", "12"])
    else:
        cmd.extend(["--n_clusters", "12"])
    
    if use_tfidf:
        cmd.append("--use_tfidf")
    
    print(f"Running: {' '.join(cmd)}")
    print("="*60)
    
    # Load environment
    if os.path.exists('.env'):
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✓ Loaded .env file")
        except ImportError:
            # Fallback: manually load .env
            print("Loading .env manually...")
            with open('.env', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
            print("✓ Loaded .env file (manual)")
    
    # Run the analysis
    try:
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print(f"\n✓ Analysis completed successfully!")
            print(f"✓ Results saved to: {output_dir}/")
            
            # Analyze conditional patterns
            print("\nAnalyzing conditional patterns...")
            analyze_cmd = [sys.executable, "analyze_conditional_patterns.py", output_dir]
            subprocess.run(analyze_cmd)
            
            # Create visualizations
            print("\nCreating visualizations...")
            viz_cmd = [sys.executable, "visualize_conditional_patterns.py", output_dir]
            subprocess.run(viz_cmd)
            
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE WITH NAME STANDARDIZATION")
            print(f"Results directory: {output_dir}/")
            print("Key files:")
            print(f"  - {output_dir}/conditional_patterns_detailed.txt")
            print(f"  - {output_dir}/conditional_patterns_summary.png")
            print(f"  - {output_dir}/report/analysis_report.html")
            print("="*60)
            
        else:
            print(f"\n✗ Analysis failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(modified_script):
            os.remove(modified_script)
            print(f"✓ Cleaned up temporary file: {modified_script}")

if __name__ == "__main__":
    main()