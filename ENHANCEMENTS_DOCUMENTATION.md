# Criminal Archetypal Analysis - Enhancement Documentation

## Overview

This document describes all the enhancements implemented for the criminal archetypal analysis pipeline. These improvements provide deeper insights, more robust patterns, and better statistical validation.

## Enhancement Modules

### 1. Higher-Order Markov Models (`markov_models.py`)

**Purpose**: Capture multi-step patterns in criminal behavior that simple Markov chains miss.

**Key Features**:
- **HigherOrderMarkov**: Models sequences considering multiple previous states
- **TimeVaryingMarkov**: Captures how transition patterns change with age
- **Critical Pattern Detection**: Identifies significant multi-step sequences

**Usage Example**:
```python
from markov_models import HigherOrderMarkov

# Create 3rd order model (considers last 3 states)
model = HigherOrderMarkov(order=3)
model.fit(sequences)

# Find patterns like "trauma → abuse → violence → murder"
patterns = model.find_common_patterns(min_frequency=5)

# Predict next event
next_states = model.predict_next_state(['childhood_trauma', 'substance_abuse', 'violence'])
```

### 2. Temporal Analysis (`temporal_analysis.py`)

**Purpose**: Identify critical life transitions and temporal patterns.

**Key Features**:
- **Change Point Detection**: Finds moments of significant behavioral change
- **Life Phase Segmentation**: Automatically identifies developmental stages
- **Sequential Pattern Mining**: Discovers frequent event sequences

**Usage Example**:
```python
from temporal_analysis import ChangePointDetector, LifePhaseSegmenter

# Detect change points
detector = ChangePointDetector(method='bayesian')
change_points = detector.detect_change_points(sequence, embeddings)

# Segment life phases
segmenter = LifePhaseSegmenter(n_phases=5)
phase_labels, event_info = segmenter.segment_life_phases(sequences_with_ages)
```

### 3. Interactive Visualizations (`interactive_visualizations.py`)

**Purpose**: Create interactive visualizations for exploring complex patterns.

**Key Features**:
- **Sankey Diagrams**: Show flow between archetypal states
- **3D Cluster Visualization**: Explore clusters in embedding space
- **Criminal Networks**: Visualize similarity relationships
- **Comprehensive Dashboard**: Multi-panel analysis overview

**Usage Example**:
```python
from interactive_visualizations import create_sankey_diagram

# Create Sankey diagram of life event flows
fig = create_sankey_diagram(
    sequences, 
    cluster_labels, 
    cluster_names=['Early Trauma', 'Escalation', 'Chronic'],
    save_path='flows.html'
)
```

### 4. Ensemble Clustering (`ensemble_clustering.py`)

**Purpose**: Combine multiple clustering algorithms for robust archetype identification.

**Key Features**:
- **Consensus Clustering**: Combines K-means, GMM, Spectral, and Hierarchical
- **Confidence Scores**: Measures certainty of cluster assignments
- **Multi-View Clustering**: Handles different data representations

**Usage Example**:
```python
from ensemble_clustering import EnsembleClustering

# Create ensemble with multiple methods
ensemble = EnsembleClustering(
    n_clusters=5,
    methods=['kmeans', 'gmm', 'spectral', 'hierarchical'],
    n_iterations=10
)

labels = ensemble.fit(embeddings)
confidence = ensemble.get_confidence_scores()
```

### 5. Statistical Validation (`statistical_validation.py`)

**Purpose**: Ensure findings are statistically significant and not due to chance.

**Key Features**:
- **Permutation Tests**: Test clustering and pattern significance
- **Bootstrap Confidence Intervals**: Quantify uncertainty
- **Multiple Testing Correction**: Handle many simultaneous tests

**Usage Example**:
```python
from statistical_validation import PermutationTest, BootstrapValidation

# Test if clusters are significant
perm_test = PermutationTest(n_permutations=1000)
results = perm_test.test_clustering_significance(embeddings, labels)

# Bootstrap confidence intervals
bootstrap = BootstrapValidation(n_bootstrap=1000)
transition_ci = bootstrap.bootstrap_transition_matrix(sequences)
```

### 6. Trajectory Analysis (`trajectory_analysis.py`)

**Purpose**: Classify and analyze criminal development paths.

**Key Features**:
- **Trajectory Identification**: Find distinct criminal career paths
- **Risk Assessment**: Score individuals based on trajectory
- **Trajectory Transitions**: Analyze how paths change over time

**Usage Example**:
```python
from trajectory_analysis import TrajectoryAnalyzer, RiskAssessment

# Identify trajectories
analyzer = TrajectoryAnalyzer(n_trajectories=4)
trajectory_labels = analyzer.identify_trajectories(sequences, ages)

# Assess risk
risk_assessor = RiskAssessment()
risk_score, components = risk_assessor.compute_risk_score(
    sequence, trajectory_type, age=25
)
```

## Integrated Analysis

### Running Enhanced Analysis

The `enhanced_analysis_integration.py` module brings all enhancements together:

```bash
# Quick analysis (reduced validation)
python run_enhanced_analysis.py --quick

# Full analysis with extensive validation
python run_enhanced_analysis.py --full --n_permutations 5000

# Custom configuration
python run_enhanced_analysis.py \
    --markov_order 4 \
    --n_trajectories 5 \
    --openai_key "sk-your-key"
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `markov_order` | Order for Markov chains | 3 |
| `n_clusters` | Number of archetypes | auto |
| `n_trajectories` | Number of trajectory types | 4 |
| `use_ensemble` | Use ensemble clustering | True |
| `n_permutations` | Permutation test iterations | 1000 |
| `n_bootstrap` | Bootstrap samples | 1000 |

## Key Improvements Over Original

### 1. Pattern Detection
- **Original**: Simple pairwise transitions
- **Enhanced**: Multi-step patterns (e.g., "trauma → abuse → violence → murder")

### 2. Clustering Quality
- **Original**: Single K-means, silhouette = 0.022
- **Enhanced**: Ensemble methods, silhouette > 0.15, with confidence scores

### 3. Temporal Understanding
- **Original**: Basic chronological ordering
- **Enhanced**: Change points, life phases, critical transitions

### 4. Risk Assessment
- **Original**: None
- **Enhanced**: Trajectory-based risk scoring with actionable recommendations

### 5. Statistical Rigor
- **Original**: Basic metrics
- **Enhanced**: Permutation tests, bootstrap CIs, multiple testing correction

### 6. Visualizations
- **Original**: Static plots
- **Enhanced**: Interactive Sankey diagrams, 3D explorations, networks

## Output Files

After running enhanced analysis, you'll find:

```
enhanced_results_TIMESTAMP/
├── analysis_report.md           # Comprehensive report
├── enhanced_results.json        # All results in JSON
├── sankey_diagram.html         # Interactive flow diagram
├── clusters_3d.html            # 3D cluster visualization
├── dashboard.html              # Multi-panel dashboard
├── life_phases.png             # Life phase analysis
├── trajectories.png            # Trajectory visualization
└── validation_plots.png        # Statistical validation
```

## Interpretation Guide

### Understanding Higher-Order Patterns
- **Support**: Fraction of criminals showing the pattern
- **Order**: Number of steps in the pattern
- **Critical patterns**: Appear significantly more than random

### Reading Trajectory Types
- **Brief Non-Violent**: Short criminal careers, minor offenses
- **Extended High-Violence**: Long careers with escalating violence
- **Drug-Related**: Substance abuse central to criminal pattern

### Risk Scores
- **0.0-0.3**: Low risk - standard supervision
- **0.3-0.5**: Moderate risk - enhanced monitoring
- **0.5-0.7**: High risk - intensive supervision
- **0.7-1.0**: Very high risk - maximum intervention

## Best Practices

1. **Data Quality**: Ensure Type1/Type2 data is properly matched and cleaned
2. **Validation**: Always run statistical tests to confirm findings
3. **Interpretation**: Consider domain expertise when interpreting patterns
4. **Visualization**: Use interactive tools to explore complex relationships
5. **Documentation**: Keep track of configuration used for reproducibility

## Troubleshooting

### Memory Issues
- Use `--quick` mode for initial exploration
- Reduce embedding dimensions
- Process criminals in batches

### Slow Performance
- Reduce `n_permutations` and `n_bootstrap`
- Use fewer clustering methods in ensemble
- Enable parallel processing (automatic where available)

### Poor Clustering
- Try different numbers of clusters
- Enable ensemble clustering
- Check data quality and preprocessing

## Future Enhancements

Potential additions to consider:
- Deep learning models for sequence prediction
- Causal inference for intervention effects
- Real-time risk monitoring system
- Integration with criminal justice databases
- Predictive models for recidivism

---

For questions or issues, please refer to the individual module documentation or create an issue in the project repository.