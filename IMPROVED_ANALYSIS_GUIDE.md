# Improved Criminal Archetypal Analysis Guide

## Overview
This guide explains how to use the improved analysis with better clustering methods and LLM labeling enabled.

## Key Improvements

### 1. LLM Labeling for Archetypal Themes
- Uses OpenAI's GPT-4o-mini to generate meaningful labels for each cluster
- Analyzes representative samples from each cluster
- Provides criminal psychology-informed archetypal themes

### 2. Improved Clustering Methods
- **Multiple algorithms**: K-means, Hierarchical, DBSCAN, Spectral clustering
- **Automatic k selection**: Uses multiple metrics to find optimal number of clusters
- **Dimensionality reduction**: PCA, UMAP, or Truncated SVD before clustering
- **Better evaluation**: Comprehensive metrics and visualizations

## Setup

### 1. Install Additional Dependencies
```bash
source venv/bin/activate
pip install kneed==0.8.5 umap-learn==0.5.5
```

### 2. Set OpenAI API Key
You need an OpenAI API key to enable LLM labeling. Get one from https://platform.openai.com/api-keys

Set it as an environment variable:
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

Or pass it as an argument when running the analysis.

## Running the Improved Analysis

### Basic Usage (with LLM and improved clustering)
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Set your OpenAI API key
export OPENAI_API_KEY="sk-your-api-key-here"

# Run improved analysis
python run_analysis_improved.py
```

### Advanced Options

#### 1. Automatic Cluster Selection
```bash
python run_analysis_improved.py --auto_k
```
This will test multiple k values and select the optimal number of clusters.

#### 2. Different Clustering Methods
```bash
# Hierarchical clustering
python run_analysis_improved.py --clustering_method hierarchical --auto_k

# DBSCAN (density-based, finds clusters automatically)
python run_analysis_improved.py --clustering_method dbscan

# Spectral clustering
python run_analysis_improved.py --clustering_method spectral --auto_k
```

#### 3. Pass OpenAI Key as Argument
```bash
python run_analysis_improved.py --openai_key "sk-your-api-key-here"
```

#### 4. Fixed Number of Clusters
```bash
python run_analysis_improved.py --n_clusters 7
```

#### 5. Without Dimension Reduction
```bash
python run_analysis_improved.py --no-reduce_dims
```

### Full Example
```bash
python run_analysis_improved.py \
    --type1_dir type1csvs \
    --type2_dir type2csvs \
    --output_dir results_improved \
    --openai_key "sk-your-api-key-here" \
    --clustering_method hierarchical \
    --auto_k \
    --reduce_dims
```

## Understanding the Results

### 1. LLM-Generated Archetypal Themes
Check `global_clusters.json` in the output directory. Each cluster will have:
```json
{
    "cluster_id": 0,
    "size": 423,
    "representative_samples": [...],
    "archetypal_theme": "Early Childhood Trauma and Family Dysfunction"
}
```

### 2. Improved Clustering Metrics
The `cluster_metrics.json` will include:
- **silhouette_score**: Higher is better (range: -1 to 1)
- **calinski_harabasz_score**: Higher is better
- **davies_bouldin_score**: Lower is better
- **n_clusters**: Final number of clusters
- **cluster_sizes**: Distribution of events across clusters

### 3. Additional Visualizations
- `clustering_tsne.png`: t-SNE visualization of clusters
- `clustering_dendrogram.png`: Hierarchical clustering dendrogram
- `optimal_k_analysis.png`: Metrics across different k values (if auto_k used)

## Expected Improvements

1. **Better Cluster Quality**: Dimensionality reduction and alternative algorithms should improve silhouette scores
2. **Meaningful Labels**: LLM-generated themes provide interpretable cluster descriptions
3. **Optimal k**: Automatic selection prevents over/under-clustering
4. **Robust Analysis**: Multiple clustering methods provide validation

## Troubleshooting

### No LLM Labels Generated
- Check if `OPENAI_API_KEY` is set correctly
- Ensure you're not using `--no_llm` flag
- Check for API errors in the console output

### Poor Clustering Results
- Try different clustering methods
- Enable auto_k to find optimal clusters
- Experiment with different dimension reduction methods
- Consider using sentence embeddings instead of TF-IDF

### Memory Issues
- Use TF-IDF embeddings: Add `--use_tfidf` flag
- Reduce max_features for TF-IDF
- Use dimensionality reduction

## Example Output with LLM Labeling

When LLM labeling is enabled, you'll see output like:
```
[INFO] Analyzing cluster 0 with LLM...
[INFO] Archetypal theme: "Early trauma and abuse patterns"
[INFO] Analyzing cluster 1 with LLM...
[INFO] Archetypal theme: "Criminal escalation and violence"
...
```

The themes will be saved in `global_clusters.json` and used throughout the analysis to provide meaningful context for the Markov chain transitions.