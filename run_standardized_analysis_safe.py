#!/usr/bin/env python3
"""
run_standardized_analysis_safe.py

Runs the analysis with name standardization, lexical imputation, and auto-k selection.
Handles import errors gracefully and uses fallback options where needed.
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime

def create_minimal_analysis_script():
    """Create a version of the analysis that handles import errors gracefully."""
    
    script_content = '''#!/usr/bin/env python3
"""
Minimal analysis script with name standardization and import error handling.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Import core modules with fallbacks
try:
    import pandas as pd
except ImportError:
    print("[WARNING] pandas not available, using basic CSV reading")
    pd = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    print("[ERROR] scikit-learn is required but not installed")
    print("Please run: pip install scikit-learn")
    sys.exit(1)

# Import improved lexical imputation
try:
    from improved_lexical_imputation import ImprovedLexicalImputation
    IMPROVED_IMPUTATION_AVAILABLE = True
except ImportError:
    print("[WARNING] improved_lexical_imputation not available")
    IMPROVED_IMPUTATION_AVAILABLE = False

# Try to import the main analysis module
try:
    # Import only what we need, avoiding problematic imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_loading import load_matched_criminal_data
    from data_cleaning import clean_type2_data
except ImportError as e:
    print(f"[WARNING] Import error: {e}")
    print("[INFO] Will use basic data loading")

def load_data_basic(type1_dir, type2_dir):
    """Basic data loading without complex dependencies."""
    type1_data = {}
    type2_data = {}
    
    # Load Type 1 data
    for filename in os.listdir(type1_dir):
        if filename.endswith('.csv'):
            criminal_id = filename.replace('.csv', '')
            filepath = os.path.join(type1_dir, filename)
            events = []
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('Event'):
                        events.append(line)
            
            if events:
                type1_data[criminal_id] = events
    
    # Load Type 2 data
    for filename in os.listdir(type2_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(type2_dir, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    headers = lines[0].strip().split(',')
                    for line in lines[1:]:
                        values = line.strip().split(',')
                        if values and values[0]:
                            type2_data[values[0]] = dict(zip(headers[1:], values[1:]))
    
    return type1_data, type2_data

def run_analysis(args):
    """Run the analysis with name standardization."""
    
    print("\\n" + "="*60)
    print("CRIMINAL ARCHETYPAL ANALYSIS WITH NAME STANDARDIZATION")
    print("="*60)
    
    # Load data
    print("\\n[INFO] Loading data...")
    try:
        type1_data, type2_data = load_matched_criminal_data(
            args['type1_dir'], 
            args['type2_dir'], 
            match_only=True
        )
    except:
        print("[INFO] Using basic data loading...")
        type1_data, type2_data = load_data_basic(args['type1_dir'], args['type2_dir'])
    
    print(f"[INFO] Loaded {len(type1_data)} criminals with matched data")
    
    # Process events
    all_events = []
    criminal_events = {}
    
    if IMPROVED_IMPUTATION_AVAILABLE and args.get('use_improved_imputation', True):
        print("\\n[INFO] Using improved lexical imputation with name standardization...")
        imputer = ImprovedLexicalImputation()
        
        for criminal_id, events in type1_data.items():
            criminal_events[criminal_id] = []
            for event in events:
                # Standardize names and generate variations
                try:
                    variations = imputer.generate_variations(event, num_variants=3)
                    for var in variations:
                        all_events.append(var)
                        criminal_events[criminal_id].append(var)
                except:
                    all_events.append(event)
                    criminal_events[criminal_id].append(event)
    else:
        print("\\n[INFO] Using original events without imputation...")
        for criminal_id, events in type1_data.items():
            criminal_events[criminal_id] = events
            all_events.extend(events)
    
    print(f"[INFO] Total events after processing: {len(all_events)}")
    
    # Create embeddings using TF-IDF
    print("\\n[INFO] Creating TF-IDF embeddings...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    embeddings = vectorizer.fit_transform(all_events)
    embeddings_array = embeddings.toarray()
    
    # Determine optimal k if requested
    if args.get('auto_k', False):
        print("\\n[INFO] Finding optimal number of clusters...")
        silhouette_scores = []
        k_values = range(2, min(21, len(set(all_events))//2))
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_array)
            score = silhouette_score(embeddings_array, labels)
            silhouette_scores.append(score)
            print(f"  k={k}: silhouette score = {score:.4f}")
        
        optimal_k = k_values[np.argmax(silhouette_scores)]
        print(f"\\n[INFO] Optimal k = {optimal_k}")
        n_clusters = optimal_k
    else:
        n_clusters = args.get('n_clusters', 12)
    
    # Perform clustering
    print(f"\\n[INFO] Clustering with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    
    # Save results
    output_dir = args['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'n_criminals': len(type1_data),
        'n_events': len(all_events),
        'n_clusters': n_clusters,
        'cluster_sizes': {str(i): int(np.sum(cluster_labels == i)) for i in range(n_clusters)},
        'silhouette_score': float(silhouette_score(embeddings_array, cluster_labels)),
        'used_name_standardization': IMPROVED_IMPUTATION_AVAILABLE and args.get('use_improved_imputation', True)
    }
    
    with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n[INFO] Results saved to {output_dir}/")
    print("\\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"✓ Used name standardization: {results['used_name_standardization']}")
    print(f"✓ Number of clusters: {n_clusters}")
    print(f"✓ Silhouette score: {results['silhouette_score']:.4f}")
    
    return 0

if __name__ == "__main__":
    # Parse arguments
    args = {
        'type1_dir': 'type1csvs',
        'type2_dir': 'type2csvs',
        'output_dir': f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'auto_k': '--auto_k' in sys.argv,
        'use_improved_imputation': True,
        'n_clusters': 12
    }
    
    # Override with command line arguments
    for i, arg in enumerate(sys.argv):
        if arg == '--type1_dir' and i + 1 < len(sys.argv):
            args['type1_dir'] = sys.argv[i + 1]
        elif arg == '--type2_dir' and i + 1 < len(sys.argv):
            args['type2_dir'] = sys.argv[i + 1]
        elif arg == '--output_dir' and i + 1 < len(sys.argv):
            args['output_dir'] = sys.argv[i + 1]
        elif arg == '--n_clusters' and i + 1 < len(sys.argv):
            args['n_clusters'] = int(sys.argv[i + 1])
    
    sys.exit(run_analysis(args))
'''
    
    # Save the script
    script_path = 'minimal_analysis_with_names.py'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path

def main():
    """Main function to run the analysis."""
    
    # Parse arguments
    auto_k = '--auto_k' in sys.argv
    use_tfidf = '--use_tfidf' in sys.argv
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_standardized_safe_{timestamp}"
    
    print("="*60)
    print("SAFE STANDARDIZED ANALYSIS RUNNER")
    print("="*60)
    print(f"Auto-k selection: {auto_k}")
    print(f"Output directory: {output_dir}")
    print("")
    
    # First, try the full analysis
    print("Attempting to run full analysis with all features...")
    
    # Load environment
    if os.path.exists('.env'):
        print("Loading environment variables...")
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✓ Environment loaded")
    
    # Try full analysis first
    cmd = [
        sys.executable,
        "run_analysis_improved.py",
        "--output_dir", output_dir,
        "--use_tfidf"  # This avoids sentence-transformers
    ]
    
    if auto_k:
        cmd.append("--auto_k")
    else:
        cmd.extend(["--n_clusters", "12"])
    
    print(f"\\nCommand: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\\n✓ Full analysis completed successfully!")
        else:
            print("\\n✗ Full analysis failed, trying minimal version...")
            print(f"Error: {result.stderr}")
            
            # Create and run minimal version
            print("\\nCreating minimal analysis script...")
            minimal_script = create_minimal_analysis_script()
            
            cmd = [
                sys.executable,
                minimal_script,
                "--output_dir", output_dir
            ]
            
            if auto_k:
                cmd.append("--auto_k")
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd)
            
            if result.returncode == 0:
                print("\\n✓ Minimal analysis completed successfully!")
            
            # Clean up
            if os.path.exists(minimal_script):
                os.remove(minimal_script)
        
        # If successful, run additional analyses
        if result.returncode == 0:
            print("\\nRunning conditional patterns analysis...")
            subprocess.run([sys.executable, "analyze_conditional_patterns.py", output_dir])
            
            print("\\nCreating visualizations...")
            subprocess.run([sys.executable, "visualize_conditional_patterns.py", output_dir])
            
            print("\\n" + "="*60)
            print("ANALYSIS COMPLETE")
            print(f"Results saved to: {output_dir}/")
            print("="*60)
            
    except Exception as e:
        print(f"\\n✗ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())