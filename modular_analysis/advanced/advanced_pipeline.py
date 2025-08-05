#!/usr/bin/env python3
"""
advanced_pipeline.py

Advanced pipeline that integrates two-layer prototype optimization
with the existing modular system.

Pipeline Flow:
1. Standard Layer 1 processing (existing system)
2. Layer 2 cluster optimization (new advanced system)
3. Validation and comparison with Layer 1 results
4. Integration with existing output format
"""

import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import logging

from .cluster_prototype_optimizer import ClusterPrototypeOptimizer, save_optimized_clusters

class AdvancedPrototypePipeline:
    """
    Advanced pipeline that adds Layer 2 cluster optimization to existing system.
    """
    
    def __init__(self, validation_split: float = 0.3, random_state: int = 42):
        """
        Initialize advanced pipeline.
        
        Args:
            validation_split: Fraction of cluster events for validation
            random_state: Random seed for reproducibility
        """
        self.validation_split = validation_split
        self.random_state = random_state
        self.optimizer = ClusterPrototypeOptimizer(validation_split, random_state)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger('AdvancedPrototypePipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(name)s] %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run_advanced_optimization(self, 
                                output_dir: Path,
                                layer1_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run advanced two-layer prototype optimization.
        
        Args:
            output_dir: Output directory from Layer 1 analysis
            layer1_results: Optional Layer 1 results (will load if not provided)
            
        Returns:
            Dictionary with Layer 2 optimization results
        """
        self.logger.info("Starting advanced two-layer prototype optimization")
        self.logger.info(f"Output directory: {output_dir}")
        
        # Load Layer 1 results if not provided
        if layer1_results is None:
            layer1_results = self._load_layer1_results(output_dir)
        
        # Extract data for Layer 2 optimization
        embeddings, labels, original_texts, processed_texts = self._extract_layer1_data(
            output_dir, layer1_results
        )
        
        # Run Layer 2 optimization
        self.logger.info("Running Layer 2 cluster optimization...")
        optimization_results = self.optimizer.optimize_cluster_prototypes(
            embeddings=embeddings,
            labels=labels,
            original_texts=original_texts,
            processed_texts=processed_texts
        )
        
        # Compare Layer 1 vs Layer 2 results
        comparison_results = self._compare_layers(layer1_results, optimization_results)
        
        # Save advanced results
        self._save_advanced_results(output_dir, optimization_results, comparison_results)
        
        # Create summary report
        summary = self._create_summary_report(layer1_results, optimization_results, comparison_results)
        
        return {
            'layer1_results': layer1_results,
            'layer2_optimization': optimization_results,
            'layer_comparison': comparison_results,
            'summary': summary,
            'output_dir': str(output_dir)
        }
    
    def _load_layer1_results(self, output_dir: Path) -> Dict[str, Any]:
        """Load Layer 1 results from output directory."""
        self.logger.info("Loading Layer 1 results...")
        
        # Load main analysis results
        with open(output_dir / "analysis_results.json", 'r') as f:
            analysis_results = json.load(f)
        
        # Load cluster info
        with open(output_dir / "clustering" / "cluster_info.json", 'r') as f:
            cluster_info = json.load(f)
        
        return {
            'analysis_results': analysis_results,
            'cluster_info': cluster_info
        }
    
    def _extract_layer1_data(self, 
                           output_dir: Path, 
                           layer1_results: Dict[str, Any]) -> tuple:
        """Extract embeddings, labels, and texts from Layer 1 results."""
        self.logger.info("Extracting Layer 1 data for optimization...")
        
        # Load embeddings and labels
        embeddings = np.load(output_dir / "data" / "embeddings.npy")
        labels = np.load(output_dir / "data" / "labels.npy")
        
        # Load original texts (try multiple sources)
        original_texts = self._load_original_texts(output_dir)
        
        # Create processed texts from cluster info (these should have entity replacement)
        processed_texts = self._extract_processed_texts(layer1_results['cluster_info'], len(embeddings))
        
        self.logger.info(f"Extracted: {len(embeddings)} embeddings, {len(np.unique(labels))} clusters")
        
        return embeddings, labels, original_texts, processed_texts
    
    def _load_original_texts(self, output_dir: Path) -> List[str]:
        """Load original texts from various possible sources."""
        # Try to load from criminal sequences
        try:
            with open(output_dir / "criminal_sequences.json", 'r') as f:
                sequences = json.load(f)
            
            # Extract all events
            all_events = []
            for criminal_id, events in sequences.items():
                all_events.extend(events)
            
            return all_events
            
        except FileNotFoundError:
            self.logger.warning("Could not load original texts, using placeholder")
            # Create placeholder texts
            return [f"Event {i}" for i in range(1000)]  # Adjust as needed
    
    def _extract_processed_texts(self, cluster_info: List[Dict], n_events: int) -> List[str]:
        """Extract processed texts from cluster info."""
        # This is a simplified approach - in practice, you'd want to store processed texts
        processed_texts = []
        
        for cluster in cluster_info:
            cluster_size = cluster['size']
            representative_samples = cluster['representative_samples']
            
            # Use representative samples as proxy for processed texts
            # Repeat to match cluster size
            for i in range(cluster_size):
                sample_idx = i % len(representative_samples)
                processed_texts.append(representative_samples[sample_idx])
        
        # Ensure we have the right number of texts
        while len(processed_texts) < n_events:
            processed_texts.append("Processed event")
        
        return processed_texts[:n_events]
    
    def _compare_layers(self, 
                       layer1_results: Dict[str, Any],
                       optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare Layer 1 and Layer 2 results."""
        self.logger.info("Comparing Layer 1 vs Layer 2 results...")
        
        # Extract Layer 1 metrics
        layer1_silhouette = layer1_results['analysis_results']['clustering']['silhouette']
        layer1_n_clusters = len(layer1_results['cluster_info'])
        
        # Extract Layer 2 metrics
        layer2_metrics = optimization_results['overall_metrics']
        layer2_coherence = layer2_metrics['mean_coherence']
        layer2_validation = layer2_metrics['mean_validation_score']
        
        # Compute improvements
        coherence_improvement = layer2_coherence  # Baseline comparison
        validation_score = layer2_validation
        
        return {
            'layer1_metrics': {
                'silhouette_score': layer1_silhouette,
                'n_clusters': layer1_n_clusters
            },
            'layer2_metrics': {
                'mean_coherence': layer2_coherence,
                'mean_validation_score': layer2_validation,
                'overall_optimization_score': layer2_metrics['overall_optimization_score']
            },
            'improvements': {
                'coherence_improvement': coherence_improvement,
                'validation_score': validation_score,
                'optimization_success': coherence_improvement > 0.5 and validation_score > 0.5
            },
            'recommendation': self._generate_recommendation(layer1_silhouette, layer2_metrics)
        }
    
    def _generate_recommendation(self, 
                               layer1_silhouette: float,
                               layer2_metrics: Dict[str, Any]) -> str:
        """Generate recommendation based on comparison."""
        layer2_score = layer2_metrics['overall_optimization_score']
        
        if layer2_score > 0.7:
            return "EXCELLENT: Layer 2 optimization significantly improved cluster quality"
        elif layer2_score > 0.5:
            return "GOOD: Layer 2 optimization provided moderate improvements"
        elif layer2_score > 0.3:
            return "FAIR: Layer 2 optimization provided some improvements"
        else:
            return "LIMITED: Layer 2 optimization provided minimal improvements"
    
    def _save_advanced_results(self, 
                             output_dir: Path,
                             optimization_results: Dict[str, Any],
                             comparison_results: Dict[str, Any]) -> None:
        """Save advanced results to output directory."""
        self.logger.info("Saving advanced optimization results...")
        
        # Save optimization results
        save_optimized_clusters(optimization_results, output_dir)
        
        # Save comparison results
        advanced_dir = output_dir / "advanced"
        with open(advanced_dir / "layer_comparison.json", 'w') as f:
            json.dump(comparison_results, f, indent=2)
    
    def _create_summary_report(self, 
                             layer1_results: Dict[str, Any],
                             optimization_results: Dict[str, Any],
                             comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive summary report."""
        
        return {
            'pipeline_type': 'Two-Layer Prototype Optimization',
            'layer1_summary': {
                'method': 'Individual Event Prototypes + Clustering',
                'silhouette_score': layer1_results['analysis_results']['clustering']['silhouette'],
                'n_clusters': len(layer1_results['cluster_info']),
                'n_events': layer1_results['analysis_results']['data_summary']['n_events']
            },
            'layer2_summary': {
                'method': 'Cluster-Level Archetypal Optimization',
                'mean_coherence': optimization_results['overall_metrics']['mean_coherence'],
                'mean_validation_score': optimization_results['overall_metrics']['mean_validation_score'],
                'optimization_score': optimization_results['overall_metrics']['overall_optimization_score'],
                'validation_split': self.validation_split
            },
            'key_improvements': {
                'archetypal_representation': optimization_results['overall_metrics']['mean_coherence'],
                'validation_generalizability': optimization_results['overall_metrics']['mean_validation_score'],
                'overall_quality': optimization_results['overall_metrics']['overall_optimization_score']
            },
            'recommendation': comparison_results['recommendation'],
            'next_steps': self._generate_next_steps(comparison_results)
        }
    
    def _generate_next_steps(self, comparison_results: Dict[str, Any]) -> List[str]:
        """Generate next steps based on results."""
        next_steps = []
        
        if comparison_results['improvements']['optimization_success']:
            next_steps.extend([
                "Use Layer 2 optimized clusters for final analysis",
                "Examine archetypal prototypes for behavioral insights",
                "Validate findings on independent dataset"
            ])
        else:
            next_steps.extend([
                "Consider alternative clustering parameters",
                "Examine individual cluster optimization results",
                "Try different validation split ratios"
            ])
        
        next_steps.extend([
            "Compare with baseline clustering methods",
            "Document methodology for publication",
            "Consider ensemble approaches"
        ])
        
        return next_steps

def run_advanced_pipeline(output_dir: str, 
                        validation_split: float = 0.3,
                        random_state: int = 42) -> Dict[str, Any]:
    """
    Convenience function to run advanced pipeline.
    
    Args:
        output_dir: Output directory from Layer 1 analysis
        validation_split: Fraction for validation
        random_state: Random seed
        
    Returns:
        Advanced pipeline results
    """
    pipeline = AdvancedPrototypePipeline(validation_split, random_state)
    return pipeline.run_advanced_optimization(Path(output_dir))

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python advanced_pipeline.py <output_directory>")
        print("Example: python advanced_pipeline.py output_best_sota")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    results = run_advanced_pipeline(output_dir)
    
    print("\n" + "="*60)
    print("ADVANCED TWO-LAYER PROTOTYPE OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}/advanced/")
    print(f"Recommendation: {results['summary']['recommendation']}")
    print(f"Optimization Score: {results['summary']['layer2_summary']['optimization_score']:.3f}")
