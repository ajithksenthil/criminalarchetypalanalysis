#!/usr/bin/env python3
"""
run_analysis_improved_names.py

Run improved analysis with name standardization in lexical imputation.
This prevents names from biasing the embeddings and clustering.
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime

def create_modified_analysis():
    """Create a modified version of analysis_integration_improved.py with name standardization."""
    
    # Read the original file
    with open('analysis_integration_improved.py', 'r') as f:
        content = f.read()
    
    # Find where to insert the import
    import_location = content.find('from sentence_transformers import SentenceTransformer')
    if import_location == -1:
        import_location = content.find('import numpy as np')
    
    # Insert the import for improved imputation
    insert_text = '\nfrom improved_lexical_imputation import ImprovedLexicalImputation\n'
    content = content[:import_location] + insert_text + content[import_location:]
    
    # Replace the generate_lexical_variations function
    old_func_start = content.find('def generate_lexical_variations(text, num_variants=5):')
    if old_func_start != -1:
        # Find the end of the function
        func_end = content.find('\ndef ', old_func_start + 1)
        
        # New function implementation
        new_func = '''def generate_lexical_variations(text, num_variants=5):
    """Use improved lexical variations with name standardization."""
    global _imputer
    if '_imputer' not in globals():
        _imputer = ImprovedLexicalImputation(client=client)
    return _imputer.generate_improved_variations(text, num_variants)
'''
        
        content = content[:old_func_start] + new_func + content[func_end:]
    
    # Save as new file
    modified_file = 'analysis_integration_improved_names.py'
    with open(modified_file, 'w') as f:
        f.write(content)
    
    return modified_file

def main():
    """Run the improved analysis with name standardization."""
    
    print("="*60)
    print("IMPROVED ANALYSIS WITH NAME STANDARDIZATION")
    print("="*60)
    print()
    print("This version standardizes all names to [PERSON] and")
    print("locations to [LOCATION] to prevent bias in clustering.")
    print()
    
    # Create modified analysis file
    print("Creating modified analysis with name standardization...")
    try:
        modified_file = create_modified_analysis()
        print(f"✓ Created {modified_file}")
    except Exception as e:
        print(f"✗ Error creating modified file: {e}")
        print("Running standard improved analysis instead...")
        modified_file = 'analysis_integration_improved.py'
    
    # Set up environment
    env = os.environ.copy()
    
    # Load .env if exists
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env[key.strip()] = value.strip().strip('"').strip("'")
        print("✓ Loaded .env file")
    
    # Set options
    env['USE_IMPROVED_CLUSTERING'] = '1'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_standardized_names_{timestamp}"
    
    # Build command
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # Default to sensible options if none provided
    if not any(arg.startswith('--') for arg in args):
        args.extend(['--auto_k', '--clustering_method', 'kmeans'])
    
    cmd = [
        sys.executable,
        modified_file,
        '--type1_dir', 'type1csvs',
        '--type2_csv', 'type2csvs', 
        '--output_dir', output_dir,
        '--match_only'
    ] + args
    
    # Add --no_llm if no API key (for faster processing)
    if 'OPENAI_API_KEY' not in env:
        cmd.append('--no_llm')
        print("⚠️  No OpenAI API key found - using standard embeddings")
    else:
        print("✓ OpenAI API key found - using improved lexical imputation")
        print("  Note: This will take ~30-45 minutes for full dataset")
        print("  Add --use_tfidf for faster processing")
    
    print()
    print("Running command:")
    print(" ".join(cmd))
    print("="*60)
    
    # Run the analysis
    result = subprocess.run(cmd, env=env)
    
    if result.returncode == 0:
        print()
        print("="*60)
        print("✓ Analysis completed successfully!")
        print(f"✓ Results saved to: {output_dir}/")
        print()
        print("Benefits of name standardization:")
        print("- Criminal behaviors are clustered by pattern, not by person")
        print("- More meaningful archetypes emerge") 
        print("- Better generalization to new cases")
    else:
        print()
        print("✗ Analysis failed with return code:", result.returncode)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())