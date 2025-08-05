#!/usr/bin/env python3
"""
pipeline.py

Main analysis pipeline orchestrating all components.
"""

import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Import all required modules with fallback handling
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import modules with error handling
def import_with_fallback(module_name, class_name):
    """Import a class with fallback error handling."""
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError:
        print(f"Warning: Could not import {class_name} from {module_name}")
        return None

# Import all required classes
AnalysisConfig = import_with_fallback('core.config', 'AnalysisConfig')
Type1DataLoader = import_with_fallback('data.loaders', 'Type1DataLoader')
Type2DataLoader = import_with_fallback('data.loaders', 'Type2DataLoader')
DataProcessor = import_with_fallback('data.loaders', 'DataProcessor')
TextPreprocessor = import_with_fallback('data.text_processing', 'TextPreprocessor')
AdvancedEmbeddingGenerator = import_with_fallback('data.text_processing', 'AdvancedEmbeddingGenerator')
TextAnalyzer = import_with_fallback('data.text_processing', 'TextAnalyzer')
BasicClusterer = import_with_fallback('clustering.basic_clustering', 'BasicClusterer')
MultiModalClusterer = import_with_fallback('clustering.basic_clustering', 'MultiModalClusterer')
ImprovedClusterer = import_with_fallback('clustering.improved_clustering', 'ImprovedClusterer')
ClusteringVisualizer = import_with_fallback('clustering.improved_clustering', 'ClusteringVisualizer')
ConditionalClusteringPipeline = import_with_fallback('clustering.conditional_optimization', 'ConditionalClusteringPipeline')
TransitionMatrixBuilder = import_with_fallback('markov.transition_analysis', 'TransitionMatrixBuilder')
ConditionalAnalyzer = import_with_fallback('markov.transition_analysis', 'ConditionalAnalyzer')
CriminalTransitionAnalyzer = import_with_fallback('markov.transition_analysis', 'CriminalTransitionAnalyzer')
ComprehensiveVisualizer = import_with_fallback('visualization.diagrams', 'ComprehensiveVisualizer')
LLMAnalyzer = import_with_fallback('integration.llm_analysis', 'LLMAnalyzer')
IntegratedRegressionAnalyzer = import_with_fallback('integration.regression_analysis', 'IntegratedRegressionAnalyzer')
AnalysisReportGenerator = import_with_fallback('integration.report_generator', 'AnalysisReportGenerator')

# Import functions
try:
    from core.config import setup_environment, get_clustering_config
except ImportError:
    def setup_environment():
        return None
    def get_clustering_config():
        return {'use_improved': False, 'method': 'kmeans', 'auto_select_k': False, 'reduce_dims': True}

try:
    from data.matching import load_matched_criminal_data, find_matching_pairs
except ImportError:
    def load_matched_criminal_data(type1_dir, type2_dir):
        print("Warning: Matched data loading not available")
        return {}, None
    def find_matching_pairs(type1_dir, type2_dir):
        return []

# Import utility functions
# Import prototypical network function
try:
    from clustering.prototypical_network import train_prototypical_network
except ImportError:
    def train_prototypical_network(embeddings, labels):
        print("Warning: Prototypical network training not available")
        return None, None, 0.0

try:
    from utils.helpers import save_json, create_output_structure
except ImportError:
    import json
    def save_json(data, filepath):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, default=str)
    def create_output_structure(base_dir):
        subdirs = ['clustering', 'markov', 'visualization', 'analysis', 'data']
        result = {'base': base_dir}
        for subdir in subdirs:
            path = os.path.join(base_dir, subdir)
            os.makedirs(path, exist_ok=True)
            result[subdir] = path
        return result

class CriminalArchetypalAnalysisPipeline:
    """Main analysis pipeline for criminal archetypal analysis."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.client = setup_environment()
        
        # Initialize components with fallback handling
        self.text_processor = TextPreprocessor() if TextPreprocessor else None
        self.embedding_generator = AdvancedEmbeddingGenerator(
            use_tfidf=config.use_tfidf,
            use_augmentation=not config.no_llm,
            client=self.client,
            model_name=config.openai_model if config.use_openai else config.embedding_model,
            use_openai=config.use_openai,
            use_prototype=config.use_prototype,
            use_lexical_bias_reduction=config.use_lexical_bias_reduction
        ) if AdvancedEmbeddingGenerator else None
        self.text_analyzer = TextAnalyzer() if TextAnalyzer else None
        self.basic_clusterer = BasicClusterer() if BasicClusterer else None
        self.improved_clusterer = ImprovedClusterer() if ImprovedClusterer else None
        self.clustering_visualizer = ClusteringVisualizer() if ClusteringVisualizer else None
        self.conditional_pipeline = ConditionalClusteringPipeline() if ConditionalClusteringPipeline else None
        self.multimodal_clusterer = MultiModalClusterer() if MultiModalClusterer else None
        self.transition_builder = TransitionMatrixBuilder() if TransitionMatrixBuilder else None
        self.conditional_analyzer = ConditionalAnalyzer() if ConditionalAnalyzer else None
        self.criminal_analyzer = CriminalTransitionAnalyzer() if CriminalTransitionAnalyzer else None
        self.visualizer = ComprehensiveVisualizer() if ComprehensiveVisualizer else None
        self.llm_analyzer = LLMAnalyzer(self.client) if LLMAnalyzer else None
        self.regression_analyzer = IntegratedRegressionAnalyzer() if IntegratedRegressionAnalyzer else None
        self.report_generator = AnalysisReportGenerator() if AnalysisReportGenerator else None
        
        # Create output structure
        self.output_dirs = create_output_structure(config.output_dir)
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run the complete criminal archetypal analysis pipeline.
        
        Returns:
            Dictionary containing all analysis results
        """
        print("="*70)
        print("CRIMINAL ARCHETYPAL ANALYSIS PIPELINE")
        print("="*70)
        
        results = {}
        
        # Step 1: Load and preprocess data
        print("\n[STEP 1] Loading and preprocessing data...")
        criminals_data, type2_df = self._load_data()
        global_events, event_criminal_ids = DataProcessor.aggregate_events(criminals_data)
        
        if not global_events:
            raise ValueError("No life events found in the data")
        
        print(f"[INFO] Loaded {len(global_events)} events from {len(criminals_data)} criminals")
        results['data_summary'] = {
            'n_criminals': len(criminals_data),
            'n_events': len(global_events),
            'has_type2_data': type2_df is not None
        }
        
        # Step 2: Text processing and embedding
        print("\n[STEP 2] Processing text and generating embeddings...")
        processed_events = self.text_processor.preprocess_batch(global_events)
        embeddings = self.embedding_generator.generate_embeddings(processed_events)
        
        print(f"[INFO] Generated embeddings with shape: {embeddings.shape}")
        results['embeddings_shape'] = embeddings.shape
        
        # Step 3: Conditional k optimization (if enabled)
        print("\n[STEP 3] Clustering optimization...")
        criminal_sequences_temp = self._create_temp_sequences(criminals_data)
        
        optimization_results = self.conditional_pipeline.run_conditional_optimization(
            embeddings, criminal_sequences_temp, type2_df, self.config.to_dict()
        )
        
        optimal_k = optimization_results['optimal_k']
        self.config.n_clusters = optimal_k
        results['optimization'] = optimization_results
        
        # Step 4: Final clustering
        print(f"\n[STEP 4] Final clustering with k={optimal_k}...")

        # Check if we should use improved clustering
        clustering_config = get_clustering_config()

        if clustering_config['use_improved']:
            print("[INFO] Using IMPROVED clustering methods...")
            labels, clusterer, clustering_metrics = self.improved_clusterer.improved_clustering(
                embeddings,
                n_clusters=optimal_k if not clustering_config['auto_select_k'] else None,
                method=clustering_config['method'],
                reduce_dims=clustering_config['reduce_dims'],
                dim_reduction='pca' if not self.config.use_tfidf else 'truncated_svd',
                n_components=50,
                auto_select_k=clustering_config['auto_select_k']
            )

            # Update n_clusters if auto-selected
            if clustering_config['auto_select_k']:
                optimal_k = clustering_metrics['n_clusters']
                self.config.n_clusters = optimal_k

            # Create additional visualizations
            self.clustering_visualizer.plot_clustering_results(
                embeddings[:1000] if len(embeddings) > 1000 else embeddings,
                labels[:1000] if len(labels) > 1000 else labels,
                os.path.join(self.output_dirs['clustering'], "clustering_tsne.png")
            )

            if clustering_config['method'] == 'hierarchical' or clustering_config['use_improved']:
                self.clustering_visualizer.hierarchical_clustering_dendrogram(
                    embeddings[:100] if len(embeddings) > 100 else embeddings,
                    os.path.join(self.output_dirs['clustering'], "clustering_dendrogram.png")
                )
        else:
            # Standard clustering
            labels, clustering_metrics = self.conditional_pipeline.apply_optimized_clustering(
                embeddings, optimal_k
            )

        results['clustering'] = clustering_metrics

        # Save embeddings and labels for validation
        np.save(os.path.join(self.output_dirs['data'], "embeddings.npy"), embeddings)
        np.save(os.path.join(self.output_dirs['data'], "labels.npy"), labels)
        print(f"[INFO] Saved embeddings and labels for validation")
        
        # Step 5: Cluster analysis
        print("\n[STEP 5] Analyzing clusters...")
        cluster_info = self.text_analyzer.find_representative_samples(
            global_events, embeddings, labels, n_reps=3
        )
        
        # LLM analysis of clusters
        cluster_info = self.llm_analyzer.analyze_all_clusters(cluster_info)
        results['clusters'] = cluster_info

        # Prototypical network training (if requested)
        if self.config.train_proto_net:
            print("\n[INFO] Training prototypical network on event embeddings...")
            proto_model, prototypes, val_acc = train_prototypical_network(embeddings, labels)

            # Save prototypes
            proto_path = os.path.join(self.output_dirs['data'], "prototypes.npy")
            np.save(proto_path, prototypes)

            results['prototypical_network'] = {
                'validation_accuracy': val_acc,
                'n_prototypes': len(prototypes),
                'prototypes_path': proto_path
            }

            print(f"[INFO] Prototypical network validation accuracy: {val_acc:.3f}")
            print(f"[INFO] Prototypes saved to {proto_path}")
        else:
            results['prototypical_network'] = None
        
        # Step 6: Build criminal sequences and transition matrices
        print("\n[STEP 6] Building transition matrices...")
        criminal_sequences = DataProcessor.build_criminal_sequences(event_criminal_ids, labels)
        self.criminal_sequences = criminal_sequences  # Store for later use
        
        global_transition_matrix = self.transition_builder.build_global_transition_matrix(
            criminal_sequences, optimal_k
        )
        global_stationary = self.transition_builder.compute_stationary_distribution(
            global_transition_matrix
        )
        
        results['markov'] = {
            'global_stationary': global_stationary.tolist(),
            'transition_entropy': self.conditional_analyzer.stats.transition_entropy(global_transition_matrix)
        }
        
        # Step 7: Conditional analysis
        print("\n[STEP 7] Conditional Markov analysis...")
        if type2_df is not None:
            insights = self.conditional_analyzer.analyze_all_conditional_insights(
                type2_df, criminal_sequences, optimal_k, global_stationary,
                global_transition_matrix
            )
            results['conditional_insights'] = insights
            
            # Generate conditional diagrams
            self.conditional_analyzer.run_conditional_markov_analysis(
                type2_df, criminal_sequences, optimal_k, self.output_dirs['markov']
            )
        else:
            print("[INFO] Skipping conditional analysis (no Type 2 data)")
            results['conditional_insights'] = {}
        
        # Step 8: Regression analysis
        print("\n[STEP 8] Regression analysis...")
        if type2_df is not None and self.regression_analyzer:
            regression_results = self.regression_analyzer.run_comprehensive_regression_analysis(
                criminal_sequences, type2_df
            )
            results['regression'] = regression_results
        else:
            if type2_df is None:
                print("[INFO] Skipping regression analysis (no Type 2 data)")
            else:
                print("[INFO] Skipping regression analysis (module not available)")
            results['regression'] = {}
        
        # Step 9: Multi-modal clustering
        print("\n[STEP 9] Multi-modal clustering...")
        if self.config.multi_modal and type2_df is not None:
            # Standard multi-modal clustering
            multimodal_results = self.multimodal_clusterer.cluster_multimodal(
                criminal_sequences, type2_df, optimal_k
            )

            # Extended multi-modal clustering at criminal level
            print("[INFO] Running extended multi-modal clustering...")
            criminal_embeddings = {}
            for crim_id in criminals_data.keys():
                indices = [i for i, cid in enumerate(event_criminal_ids) if cid == crim_id]
                if indices:
                    criminal_embeddings[crim_id] = np.mean(embeddings[indices], axis=0)

            from ..data.loaders import Type2DataProcessor
            multi_modal_vectors = []
            modal_criminal_ids = []

            for crim_id, emb in criminal_embeddings.items():
                extended_vec = Type2DataProcessor.get_extended_type2_vector(crim_id, type2_df)
                if extended_vec is not None:
                    # Concatenate the average embedding with the extended Type 2 feature vector
                    combined = np.concatenate([emb, extended_vec])
                    multi_modal_vectors.append(combined)
                    modal_criminal_ids.append(crim_id)

            if multi_modal_vectors:
                multi_modal_vectors = np.array(multi_modal_vectors)
                mm_labels, mm_model = self.basic_clusterer.kmeans_cluster(
                    multi_modal_vectors, n_clusters=optimal_k
                )

                extended_multimodal_results = {
                    'criminal_ids': modal_criminal_ids,
                    'labels': mm_labels.tolist(),
                    'cluster_assignments': {cid: int(label) for cid, label in zip(modal_criminal_ids, mm_labels)}
                }

                print("[INFO] Extended multi-modal clustering (criminal level) complete. Cluster assignments:")
                for cid, label in zip(modal_criminal_ids, mm_labels):
                    print(f"  Criminal {cid}: Cluster {label}")

                multimodal_results['extended'] = extended_multimodal_results
            else:
                print("[INFO] No criminals with extended multi-modal features available.")
                multimodal_results['extended'] = {}

            results['multimodal'] = multimodal_results
        else:
            print("[INFO] Skipping multi-modal clustering")
            results['multimodal'] = {}
        
        # Step 10: Criminal-level analysis
        print("\n[STEP 10] Criminal-level transition analysis...")
        criminal_clustering_results = self.criminal_analyzer.cluster_criminals_by_transition_patterns(
            criminal_sequences, optimal_k
        )
        results['criminal_clustering'] = criminal_clustering_results
        
        # Step 11: Visualizations
        print("\n[STEP 11] Creating visualizations...")
        self.visualizer.create_analysis_dashboard(
            embeddings, labels, global_transition_matrix, self.output_dirs['visualization']
        )
        
        # Step 12: Generate comprehensive summary
        print("\n[STEP 12] Generating analysis summary...")
        if type2_df is not None:
            summary = self.llm_analyzer.generate_analysis_summary(
                results.get('conditional_insights', {}), cluster_info
            )
            results['llm_summary'] = summary
        
        # Step 13: Save all results
        print("\n[STEP 13] Saving results...")
        self._save_results(results, embeddings, labels, global_transition_matrix, global_stationary)

        # Step 14: Generate comprehensive report
        print("\n[STEP 14] Generating comprehensive HTML report...")
        if self.report_generator:
            try:
                loaded_results = self.report_generator.load_analysis_results(self.config.output_dir)
                stats = self.report_generator.generate_summary_statistics(loaded_results)
                report_path = self.report_generator.generate_html_report(loaded_results, stats, self.config.output_dir)
                print(f"[SUCCESS] Report available at: {report_path}")
                results['report_path'] = report_path
            except Exception as e:
                print(f"[WARNING] Could not generate HTML report: {e}")
                results['report_path'] = None
        else:
            print("[INFO] HTML report generation not available")
            results['report_path'] = None
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"Results saved to: {self.config.output_dir}")
        print(f"Number of clusters: {optimal_k}")
        print(f"Clustering quality (silhouette): {clustering_metrics.get('silhouette', 'N/A'):.3f}")
        if type2_df is not None:
            n_insights = len(results.get('conditional_insights', {}))
            print(f"Conditional insights found: {n_insights}")
        
        return results
    
    def _load_data(self) -> Tuple[Dict[str, Any], Optional[Any]]:
        """Load Type 1 and Type 2 data."""
        if self.config.match_only:
            # Load matched data
            print("[INFO] Loading only criminals with both Type1 and Type2 data...")

            # First check if Type2 path is a directory
            if not os.path.isdir(self.config.type2_csv):
                raise ValueError("When using --match_only, --type2_csv must be a directory")

            # Show matching statistics
            matching_pairs = find_matching_pairs(self.config.type1_dir, self.config.type2_csv)

            # Load matched data
            criminals_data, type2_df = load_matched_criminal_data(self.config.type1_dir, self.config.type2_csv)

            return criminals_data, type2_df
        else:
            # Load Type 1 data
            type1_loader = Type1DataLoader(self.config.type1_dir)
            criminals_data = type1_loader.load_all_criminals()

            if not criminals_data:
                raise ValueError(f"No Type1 files found in {self.config.type1_dir}")

            print(f"[INFO] Loaded Type 1 data for {len(criminals_data)} criminals.")

            # Load Type 2 data if available
            type2_df = None
            if self.config.type2_csv:
                try:
                    type2_loader = Type2DataLoader(self.config.type2_csv)
                    type2_df = type2_loader.load_data()
                except Exception as e:
                    print(f"[WARNING] Could not load Type 2 data: {e}")

            return criminals_data, type2_df
    
    def _create_temp_sequences(self, criminals_data: Dict[str, Any]) -> Dict[str, List[int]]:
        """Create temporary sequences for k optimization."""
        criminal_sequences_temp = {}
        event_idx = 0
        
        for crim_id, data in criminals_data.items():
            criminal_sequences_temp[crim_id] = []
            for event in data["events"]:
                criminal_sequences_temp[crim_id].append(event_idx)
                event_idx += 1
        
        return criminal_sequences_temp
    
    def _save_results(self, results: Dict[str, Any], embeddings: np.ndarray, 
                     labels: np.ndarray, transition_matrix: np.ndarray, 
                     stationary: np.ndarray) -> None:
        """Save all analysis results."""
        # Save main results
        save_json(results, os.path.join(self.config.output_dir, "analysis_results.json"))
        
        # Save numpy arrays
        np.save(os.path.join(self.output_dirs['data'], "embeddings.npy"), embeddings)
        np.save(os.path.join(self.output_dirs['data'], "labels.npy"), labels)
        np.save(os.path.join(self.output_dirs['data'], "global_transition_matrix.npy"), transition_matrix)
        np.save(os.path.join(self.output_dirs['data'], "global_stationary_distribution.npy"), stationary)
        
        # Save individual components
        if 'clusters' in results:
            save_json(results['clusters'], os.path.join(self.output_dirs['clustering'], "cluster_info.json"))
        
        if 'conditional_insights' in results:
            save_json(results['conditional_insights'], os.path.join(self.output_dirs['analysis'], "conditional_insights.json"))
        
        if 'optimization' in results and 'optimization_results' in results['optimization']:
            save_json(results['optimization']['optimization_results'], 
                     os.path.join(self.output_dirs['clustering'], "k_optimization_results.json"))
        
        # Save configuration
        save_json(self.config.to_dict(), os.path.join(self.config.output_dir, "config.json"))

        # Save criminal sequences for reference (already computed earlier)
        # criminal_sequences was already computed in step 6

        # Convert numpy types to Python types for JSON serialization
        criminal_sequences_json = {}
        if hasattr(self, 'criminal_sequences') and self.criminal_sequences:
            for crim_id, seq in self.criminal_sequences.items():
                criminal_sequences_json[crim_id] = [int(x) for x in seq]

        sequences_path = os.path.join(self.config.output_dir, "criminal_sequences.json")
        save_json(criminal_sequences_json, sequences_path)
        print(f"[INFO] Criminal sequences saved to {sequences_path}")

        print(f"[INFO] All results saved to {self.config.output_dir}")
