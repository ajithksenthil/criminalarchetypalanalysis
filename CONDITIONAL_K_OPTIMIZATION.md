# Conditional Effect-Optimized Clustering

This document describes the new conditional effect-optimized clustering functionality that has been added to the criminal archetypal analysis system.

## Overview

The new functionality automatically selects the optimal number of clusters (k) by maximizing the ability to detect significant conditional effects in criminal behavior patterns. Instead of using generic clustering quality metrics, this approach optimizes specifically for the downstream conditional Markov chain analysis.

## New Functions

### 1. `get_condition_map(type2_df, heading)`

Creates a mapping from criminal ID to value for a specific heading in the Type 2 data.

**Parameters:**
- `type2_df`: Type 2 DataFrame with CriminalID, Heading, Value columns
- `heading`: The heading to extract values for

**Returns:**
- `dict`: Mapping from criminal ID to value for the heading

**Example:**
```python
condition_map = get_condition_map(type2_df, "Sex")
# Returns: {'1': 'Male', '2': 'Female', ...}
```

### 2. `find_optimal_k_for_conditional_analysis(embeddings, criminal_sequences, type2_df, k_range=None, min_effect_size=0.1)`

Finds the optimal number of clusters by maximizing significant conditional effects.

**Parameters:**
- `embeddings`: Event embeddings
- `criminal_sequences`: Criminal ID to sequence mapping
- `type2_df`: Type 2 data for conditional analysis
- `k_range`: Range of k values to test (default: 3 to min(20, len(embeddings)//50))
- `min_effect_size`: Minimum L1 difference to consider significant (default: 0.1)

**Returns:**
- `optimal_k`: Best k value
- `effect_metrics`: Dictionary with detailed results for each k

**Algorithm:**
1. For each k value:
   - Cluster events using k-means
   - Build global transition matrix
   - For each condition in Type 2 data:
     - Build conditional transition matrix for each value
     - Compute L1 distance between conditional and global stationary distributions
     - Count significant effects (L1 distance >= min_effect_size)
   - Score k based on: (significant_effects / total_effects) * mean_effect_size
2. Return k with highest score

### 3. `multi_objective_k_selection(embeddings, criminal_sequences, type2_df, k_range=None)`

Selects k using both clustering quality and conditional effect strength.

**Parameters:**
- `embeddings`: Event embeddings
- `criminal_sequences`: Criminal ID to sequence mapping
- `type2_df`: Type 2 data for conditional analysis
- `k_range`: Range of k values to test (default: 3 to 15)

**Returns:**
- `optimal_k`: Best k value
- `results`: Dictionary with detailed results for each k

**Algorithm:**
Combines silhouette score (clustering quality) with conditional effect score:
```
combined_score = 0.3 * silhouette_score + 0.7 * effect_score
```

## Usage

### Command Line

Add the `--auto_k` flag to enable automatic k optimization:

```bash
python analysis_integration_improved.py \
    --type1_dir=/path/to/type1_data \
    --type2_csv=/path/to/type2_data \
    --output_dir=/path/to/output \
    --auto_k \
    --match_only
```

### What Happens

1. **Early Type 2 Loading**: Type 2 data is loaded before clustering (if `--auto_k` is specified)
2. **K Optimization**: The system tests different k values and selects the one that maximizes conditional effects
3. **Results Saved**: Optimization results are saved to `k_optimization_results.json`
4. **Normal Analysis**: The analysis proceeds with the optimized k value

### Output Files

When using `--auto_k`, the following additional files are created:

- `k_optimization_results.json`: Detailed results for each k value tested
  ```json
  {
    "3": {
      "significant_effects": 5,
      "total_effects": 12,
      "significance_rate": 0.417,
      "mean_effect_size": 0.234,
      "score": 0.098
    },
    "4": {
      "significant_effects": 8,
      "total_effects": 12,
      "significance_rate": 0.667,
      "mean_effect_size": 0.189,
      "score": 0.126
    }
  }
  ```

## Benefits

1. **Targeted Optimization**: Optimizes specifically for detecting meaningful differences in criminal behavior patterns
2. **Data-Driven Selection**: Uses actual conditional effects rather than generic clustering metrics
3. **Automatic Operation**: No manual tuning required
4. **Comprehensive Analysis**: Tests multiple k values and provides detailed metrics
5. **Backward Compatible**: Existing functionality remains unchanged when `--auto_k` is not used

## Example Workflow

```bash
# Run analysis with automatic k optimization
python analysis_integration_improved.py \
    --type1_dir=data_csv \
    --type2_csv=data_csv \
    --output_dir=output_auto_k \
    --auto_k \
    --match_only

# Check the optimization results
cat output_auto_k/k_optimization_results.json

# Compare with fixed k analysis
python analysis_integration_improved.py \
    --type1_dir=data_csv \
    --type2_csv=data_csv \
    --output_dir=output_fixed_k \
    --n_clusters=5 \
    --match_only
```

## Technical Details

- **Minimum Group Size**: Groups with fewer than 5 criminals are skipped to ensure robust statistics
- **Effect Size Threshold**: Default minimum L1 difference of 0.1 for significance
- **K Range**: Default range from 3 to min(20, number_of_embeddings/50)
- **Scoring**: Combines effect rate and effect magnitude for balanced optimization
- **Memory Efficient**: Processes one k value at a time to minimize memory usage

## Integration

The conditional effect optimization is fully integrated into the existing analysis pipeline:

1. Preserves all existing functionality
2. Works with both `--match_only` and regular modes
3. Compatible with improved clustering methods
4. Saves all standard outputs plus optimization results
