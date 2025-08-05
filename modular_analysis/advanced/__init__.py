"""
Advanced prototype optimization module.

This module provides two-layer prototype optimization for criminal archetypal analysis:

Layer 1: Individual event prototypes (existing system)
- Entity replacement (names → [PERSON], locations → [LOCATION])
- LLM lexical variations (reduce word choice bias)
- Prototype embeddings (average variations)

Layer 2: Cluster-level archetypal optimization (new system)
- Train/validation splits for generalizability
- Archetypal prototype creation with outlier removal
- Cross-validation for robust archetype representation
- Optimization metrics and validation scores

Key Components:
- ClusterPrototypeOptimizer: Core optimization algorithm
- AdvancedPrototypePipeline: Integration with existing modular system
- Command-line interface: run_advanced_prototype_optimization.py

Usage:
    # Run on existing analysis results
    python run_advanced_prototype_optimization.py output_best_sota
    
    # Run complete pipeline (Layer 1 + Layer 2)
    python run_advanced_prototype_optimization.py --run_complete_pipeline
"""

from .cluster_prototype_optimizer import ClusterPrototypeOptimizer, save_optimized_clusters
from .advanced_pipeline import AdvancedPrototypePipeline, run_advanced_pipeline

__all__ = [
    'ClusterPrototypeOptimizer',
    'save_optimized_clusters', 
    'AdvancedPrototypePipeline',
    'run_advanced_pipeline'
]
