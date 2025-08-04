# Modular Criminal Archetypal Analysis System

A refactored, modular implementation of the criminal archetypal analysis system that preserves all functionality while organizing code into logical, maintainable components.

## Architecture Overview

The system is organized into the following modules:

```
modular_analysis/
├── core/                    # Core configuration and settings
│   ├── config.py           # Configuration management
│   └── __init__.py
├── data/                    # Data loading and processing
│   ├── loaders.py          # Type 1 and Type 2 data loaders
│   ├── text_processing.py  # Text preprocessing and embeddings
│   └── __init__.py
├── clustering/              # Clustering algorithms and optimization
│   ├── basic_clustering.py # Standard clustering methods
│   ├── conditional_optimization.py # Conditional effect optimization
│   └── __init__.py
├── markov/                  # Markov chain analysis
│   ├── transition_analysis.py # Transition matrices and statistics
│   └── __init__.py
├── visualization/           # Visualization components
│   ├── diagrams.py         # Charts, plots, and diagrams
│   └── __init__.py
├── integration/             # Analysis integration and orchestration
│   ├── llm_analysis.py     # LLM-based analysis
│   ├── regression_analysis.py # Logistic regression
│   ├── pipeline.py         # Main analysis pipeline
│   └── __init__.py
├── utils/                   # Utility functions
│   ├── helpers.py          # Common helper functions
│   └── __init__.py
└── main.py                  # Main entry point
```

## Key Improvements

### 1. **Modularity**
- Each component has a single responsibility
- Clear separation of concerns
- Easy to test individual components
- Facilitates code reuse

### 2. **Maintainability**
- Reduced file size (from 1400+ lines to manageable modules)
- Clear interfaces between components
- Consistent error handling
- Comprehensive documentation

### 3. **Extensibility**
- Easy to add new clustering algorithms
- Simple to integrate new analysis methods
- Pluggable visualization components
- Configurable pipeline stages

### 4. **Robustness**
- Input validation
- Error handling and recovery
- Progress tracking
- Comprehensive logging

## Usage

### Basic Usage

```bash
# Run basic analysis
python main.py --type1_dir=../data_csv --type2_csv=../data_csv --output_dir=output

# With automatic k optimization
python main.py --type1_dir=../data_csv --type2_csv=../data_csv --output_dir=output --auto_k

# Matched data only
python main.py --type1_dir=../data_csv --type2_csv=../data_csv --output_dir=output --match_only

# Offline mode (no LLM)
python main.py --type1_dir=../data_csv --type2_csv=../data_csv --output_dir=output --no_llm --use_tfidf
```

### Advanced Options

```bash
# Multi-modal clustering with verbose output
python main.py --type1_dir=../data_csv --type2_csv=../data_csv --output_dir=output \
               --multi_modal --verbose

# Dry run to validate inputs
python main.py --type1_dir=../data_csv --type2_csv=../data_csv --output_dir=output --dry_run

# Custom number of clusters
python main.py --type1_dir=../data_csv --type2_csv=../data_csv --output_dir=output \
               --n_clusters=7
```

## Component Details

### Core Configuration (`core/config.py`)
- Global settings and constants
- Environment setup
- Configuration management
- NLTK data handling

### Data Processing (`data/`)
- **loaders.py**: Type 1/Type 2 data loading, matched data handling
- **text_processing.py**: Text preprocessing, embeddings, lexical augmentation

### Clustering (`clustering/`)
- **basic_clustering.py**: Standard clustering algorithms, k optimization
- **conditional_optimization.py**: Conditional effect-optimized clustering

### Markov Analysis (`markov/`)
- **transition_analysis.py**: Transition matrices, conditional analysis, statistics

### Visualization (`visualization/`)
- **diagrams.py**: All visualization components (t-SNE, transition diagrams, heatmaps)

### Integration (`integration/`)
- **llm_analysis.py**: LLM-based cluster labeling and insights
- **regression_analysis.py**: Logistic regression analysis
- **pipeline.py**: Main orchestration pipeline

### Utilities (`utils/`)
- **helpers.py**: Common utility functions, JSON handling, file operations

## Output Structure

The system creates a structured output directory:

```
output/
├── analysis_results.json    # Complete analysis results
├── config.json             # Analysis configuration
├── clustering/             # Clustering-related outputs
│   ├── cluster_info.json
│   └── k_optimization_results.json
├── markov/                 # Markov analysis outputs
│   └── state_transition_*.png
├── visualization/          # Visualizations
│   ├── tsne_visualization.png
│   ├── clustering_comprehensive.png
│   └── global_transition_diagram.png
├── analysis/               # Analysis results
│   └── conditional_insights.json
└── data/                   # Raw data outputs
    ├── embeddings.npy
    ├── labels.npy
    └── global_transition_matrix.npy
```

## Comparison with Original

| Aspect | Original (`analysis_integration_improved.py`) | Modular System |
|--------|-----------------------------------------------|----------------|
| **File Size** | 1400+ lines | ~300 lines per module |
| **Maintainability** | Difficult to navigate | Easy to understand |
| **Testing** | Hard to test individual components | Easy unit testing |
| **Extensibility** | Requires modifying large file | Add new modules |
| **Debugging** | Hard to isolate issues | Clear error sources |
| **Code Reuse** | Limited | High reusability |
| **Documentation** | Minimal | Comprehensive |

## Preserved Functionality

All functionality from the original system is preserved:

✅ Type 1 and Type 2 data loading  
✅ Text preprocessing and embedding generation  
✅ Standard and improved clustering  
✅ Conditional effect-optimized k selection  
✅ Markov chain transition analysis  
✅ Conditional insights analysis  
✅ LLM-based cluster labeling  
✅ Logistic regression analysis  
✅ Multi-modal clustering  
✅ Criminal-level transition analysis  
✅ Comprehensive visualizations  
✅ All command-line options  

## Development

### Adding New Components

1. **New Clustering Algorithm**: Add to `clustering/` module
2. **New Visualization**: Add to `visualization/diagrams.py`
3. **New Analysis Method**: Add to `integration/` module
4. **New Data Source**: Add to `data/loaders.py`

### Testing

Each module can be tested independently:

```python
# Test data loading
from data.loaders import Type1DataLoader
loader = Type1DataLoader("path/to/data")
data = loader.load_all_criminals()

# Test clustering
from clustering.basic_clustering import BasicClusterer
clusterer = BasicClusterer()
labels, model = clusterer.kmeans_cluster(embeddings, n_clusters=5)
```

## Dependencies

Same as the original system:
- numpy, pandas, scikit-learn
- matplotlib, seaborn, networkx
- nltk, sentence-transformers
- openai (optional)

## Migration

To migrate from the original system:

1. Use the same command-line arguments
2. Output format is identical
3. All analysis results are preserved
4. Configuration files are compatible

The modular system is a drop-in replacement that provides better organization and maintainability while preserving all existing functionality.
