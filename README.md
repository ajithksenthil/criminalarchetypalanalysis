# Criminal Archetypal Analysis

Advanced statistical psychometric analysis system for studying criminal life trajectories using Markov transition matrices and machine learning.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This system analyzes serial killers' life histories by:
1. Processing Type 1 data (chronological life events) and Type 2 data (standardized attributes)
2. Using NLP and clustering to identify archetypal life events
3. Building Markov transition matrices to model life event sequences
4. Performing conditional analysis based on Type 2 factors
5. Clustering criminals by their transition patterns
6. Generating comprehensive statistical reports with visualizations

## Key Features

- **Lexical Imputation**: Uses LLMs to handle semantic variations in life event descriptions
- **Archetypal Clustering**: K-means clustering with sentence embeddings to identify event patterns
- **Markov Analysis**: Transition matrices with stationary distribution calculations
- **Conditional Analysis**: Statistical tests (KS test, Wasserstein distance) for subgroup comparisons
- **Criminal Clustering**: Groups criminals by transition pattern similarity
- **Prototypical Networks**: Few-shot learning for event classification
- **Automated Reporting**: HTML reports with visualizations and statistical insights

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment support
- Git (for version control)

### Quick Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd criminalarchetypalanalysis
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

3. Activate the virtual environment:
```bash
source venv/bin/activate
```

### Manual Setup

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Usage

### Research-Grade Analysis (Recommended)

For reproducible, validated results:

```bash
# Activate virtual environment
source venv/bin/activate

# Run comprehensive analysis with validation
python run_analysis.py --type1_dir type1csvs --type2_dir type2csvs

# With all features enabled
python run_analysis.py \
    --type1_dir type1csvs \
    --type2_dir type2csvs \
    --n_clusters 5 \
    --train_proto_net \
    --multi_modal

# Offline mode (no LLM, using TF-IDF)
python run_analysis.py \
    --type1_dir type1csvs \
    --type2_dir type2csvs \
    --use_tfidf \
    --no_llm
```

### Basic Analysis
```bash
python analysis_cli.py --type1_dir path/to/type1csvs --type2_csv path/to/type2.csv --n_clusters 5 --diagram state_transition.png [--lexical_impute]
```

### Data Matching
The system can automatically match Type1 and Type2 files by criminal name:
```bash
# Test data matching
python data_matching.py --type1_dir type1csvs --type2_dir type2csvs
```

### Options
- `--n_clusters`: Number of archetypal event clusters (default: 5)
- `--no_llm`: Disable LLM features (for offline mode)
- `--use_tfidf`: Use TF-IDF instead of sentence embeddings
- `--train_proto_net`: Train prototypical network
- `--multi_modal`: Combine Type 1 and Type 2 features for clustering
- `--match_only`: Only analyze criminals with both Type1 and Type2 data (recommended for conditional analysis)

## Output

The system generates:
- State transition diagrams
- t-SNE visualizations of event clusters
- Criminal clustering dendrograms and PCA plots
- Conditional transition matrices for Type 2 subgroups
- Statistical significance tests
- Comprehensive HTML report with all results

## Validation and Reproducibility

The system includes comprehensive validation metrics:

### Clustering Validation
- **Silhouette Score**: Measures cluster cohesion and separation
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster variance
- **Davies-Bouldin Score**: Average similarity between clusters
- **Stability Analysis**: Bootstrap validation of cluster assignments
- **Cross-validation**: Tests generalization across criminals

### Statistical Tests
- **Kolmogorov-Smirnov Test**: Compares transition distributions
- **Wasserstein Distance**: Measures difference between stationary distributions
- **Anderson-Darling Test**: Tests for distribution differences
- **Bonferroni Correction**: Controls for multiple comparisons

### Reproducibility Features
- Fixed random seeds (42)
- Virtual environment for consistent dependencies
- Timestamped outputs
- Comprehensive logging
- Package version tracking
- Environment variable documentation

## Data Format

### Type 1 CSV (Life Events)
```csv
Date,Age,Life Event
1943,0,"Born in San Antonio, Texas"
1954,11,"Father abandoned family"
```

### Type 2 CSV (Attributes)
```csv
Heading,Value
Sex,Male
Number of victims,8
Physically abused?,Yes
```

