#!/usr/bin/env python3
"""
embedding_comparison_system.py

Easy system to test different state-of-the-art embeddings on your criminal archetypal analysis.
Allows iterative testing and comparison of embedding quality.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class EmbeddingComparator:
    """Compare different embedding models for criminal life event analysis."""
    
    def __init__(self, base_output_dir: str = "output_semantic_embeddings"):
        self.base_output_dir = Path(base_output_dir)
        self.comparison_dir = Path("embedding_comparisons")
        self.comparison_dir.mkdir(exist_ok=True)
        
        # Load original data for comparison
        self.original_events = self._load_original_events()
        
    def _load_original_events(self) -> List[str]:
        """Load the original event texts for re-embedding."""
        try:
            # Try to load from criminal sequences or find event texts
            with open(self.base_output_dir / "criminal_sequences.json", 'r') as f:
                sequences = json.load(f)
            
            # Extract event texts (this might need adjustment based on your data structure)
            events = []
            print("[INFO] Loading original event texts...")
            print("[WARNING] You may need to adjust this method to load your actual event texts")
            return events
            
        except Exception as e:
            print(f"[ERROR] Could not load original events: {e}")
            print("[INFO] You'll need to provide the original event texts manually")
            return []
    
    def test_embedding_models(self, event_texts: List[str], 
                            models_to_test: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Test multiple embedding models and compare their clustering quality.
        
        Args:
            event_texts: List of life event descriptions
            models_to_test: List of model names to test
            
        Returns:
            Comparison results
        """
        if models_to_test is None:
            models_to_test = self._get_recommended_models()
        
        print(f"[INFO] Testing {len(models_to_test)} embedding models on {len(event_texts)} events")
        
        results = {}
        
        for model_name in models_to_test:
            print(f"\n[TESTING] {model_name}")
            
            try:
                # Generate embeddings
                start_time = time.time()
                embeddings = self._generate_embeddings(event_texts, model_name)
                embedding_time = time.time() - start_time
                
                # Test clustering quality
                clustering_results = self._test_clustering_quality(embeddings, model_name)
                
                # Store results
                results[model_name] = {
                    'embedding_time': embedding_time,
                    'embedding_shape': embeddings.shape,
                    'clustering_results': clustering_results,
                    'embeddings': embeddings  # Store for later use
                }
                
                print(f"  ✅ Completed in {embedding_time:.1f}s")
                print(f"  📊 Best silhouette: {clustering_results['best_silhouette']:.4f}")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                results[model_name] = {'error': str(e)}
        
        # Save comparison results
        self._save_comparison_results(results, event_texts)
        
        # Generate comparison report
        self._generate_comparison_report(results)
        
        return results
    
    def _get_recommended_models(self) -> List[str]:
        """Get list of recommended state-of-the-art models."""
        return [
            # Current baseline
            "all-MiniLM-L6-v2",
            
            # State-of-the-art Sentence-BERT
            "all-mpnet-base-v2",           # Best general performance
            "all-MiniLM-L12-v2",          # Improved version of current
            "paraphrase-mpnet-base-v2",   # Good for similar events
            
            # Larger models (if computational resources allow)
            "all-distilroberta-v1",       # Fast and good
            "multi-qa-mpnet-base-dot-v1", # Good for diverse text
            
            # Legal/Criminal domain (if available)
            # "nlpaueb/legal-bert-base-uncased",  # Uncomment if needed
        ]
    
    def _generate_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """Generate embeddings using specified model."""
        
        if model_name.startswith("text-embedding-"):
            # OpenAI embeddings
            return self._generate_openai_embeddings(texts, model_name)
        else:
            # Sentence-BERT embeddings
            return self._generate_sentence_bert_embeddings(texts, model_name)
    
    def _generate_sentence_bert_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """Generate Sentence-BERT embeddings."""
        from sentence_transformers import SentenceTransformer
        
        print(f"    Loading {model_name}...")
        model = SentenceTransformer(model_name)
        
        print(f"    Encoding {len(texts)} texts...")
        embeddings = model.encode(texts, show_progress_bar=True)
        
        return embeddings
    
    def _generate_openai_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """Generate OpenAI embeddings."""
        try:
            import openai
            
            print(f"    Using OpenAI {model_name}...")
            client = openai.OpenAI()  # Requires OPENAI_API_KEY environment variable
            
            embeddings = []
            batch_size = 100  # OpenAI rate limits
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                print(f"    Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
                response = client.embeddings.create(
                    input=batch,
                    model=model_name
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
                time.sleep(0.1)  # Rate limiting
            
            return np.array(embeddings)
            
        except ImportError:
            raise Exception("OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
    
    def _test_clustering_quality(self, embeddings: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Test clustering quality for different k values."""
        
        k_range = range(3, min(12, len(embeddings) // 50))
        results = {}
        best_silhouette = -1
        best_k = 3
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                
                silhouette = silhouette_score(embeddings, labels)
                
                results[k] = {
                    'silhouette': silhouette,
                    'labels': labels.tolist()
                }
                
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_k = k
                    
            except Exception as e:
                results[k] = {'error': str(e)}
        
        return {
            'k_results': results,
            'best_k': best_k,
            'best_silhouette': best_silhouette
        }
    
    def _save_comparison_results(self, results: Dict[str, Any], event_texts: List[str]):
        """Save detailed comparison results."""
        
        # Save full results (without embeddings to save space)
        results_to_save = {}
        for model_name, result in results.items():
            if 'embeddings' in result:
                # Save embeddings separately
                np.save(self.comparison_dir / f"{model_name.replace('/', '_')}_embeddings.npy", 
                       result['embeddings'])
                
                # Remove embeddings from JSON
                result_copy = result.copy()
                del result_copy['embeddings']
                results_to_save[model_name] = result_copy
            else:
                results_to_save[model_name] = result
        
        with open(self.comparison_dir / "comparison_results.json", 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        # Save event texts for reference
        with open(self.comparison_dir / "event_texts.json", 'w') as f:
            json.dump(event_texts, f, indent=2)
    
    def _generate_comparison_report(self, results: Dict[str, Any]):
        """Generate human-readable comparison report."""
        
        # Create summary table
        summary = []
        for model_name, result in results.items():
            if 'error' not in result:
                summary.append({
                    'model': model_name,
                    'dimensions': result['embedding_shape'][1],
                    'time': result['embedding_time'],
                    'best_k': result['clustering_results']['best_k'],
                    'best_silhouette': result['clustering_results']['best_silhouette']
                })
        
        # Sort by silhouette score
        summary.sort(key=lambda x: x['best_silhouette'], reverse=True)
        
        # Create report
        report = """
EMBEDDING MODEL COMPARISON REPORT
=================================

RANKING BY CLUSTERING QUALITY
-----------------------------
"""
        
        for i, model_result in enumerate(summary, 1):
            report += f"""
{i}. {model_result['model']}
   • Dimensions: {model_result['dimensions']}
   • Best silhouette: {model_result['best_silhouette']:.4f}
   • Best k: {model_result['best_k']}
   • Embedding time: {model_result['time']:.1f}s
"""
        
        report += f"""

RECOMMENDATIONS
--------------
"""
        
        if summary:
            best_model = summary[0]
            report += f"""
🏆 BEST OVERALL: {best_model['model']}
   • Highest clustering quality (silhouette: {best_model['best_silhouette']:.4f})
   • Optimal clusters: {best_model['best_k']}
   
💡 TO USE THIS MODEL:
   python run_modular_analysis.py \\
     --type1_dir=type1csvs \\
     --type2_csv=type2csvs \\
     --output_dir=output_best_embeddings \\
     --embedding_model={best_model['model']} \\
     --auto_k --match_only --no_llm --n_clusters={best_model['best_k']}
"""
        
        # Performance vs Quality tradeoff
        if len(summary) > 1:
            fastest = min(summary, key=lambda x: x['time'])
            report += f"""
⚡ FASTEST: {fastest['model']} ({fastest['time']:.1f}s)
   • Good for rapid iteration
   • Silhouette: {fastest['best_silhouette']:.4f}
"""
        
        with open(self.comparison_dir / "comparison_report.txt", 'w') as f:
            f.write(report)
        
        print(f"\n📊 Comparison report saved to: {self.comparison_dir}/comparison_report.txt")
        
        # Create visualization
        self._create_comparison_visualization(summary)
    
    def _create_comparison_visualization(self, summary: List[Dict]):
        """Create visualization comparing models."""
        
        if not summary:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Silhouette scores
        models = [s['model'].replace('/', '\n') for s in summary]
        silhouettes = [s['best_silhouette'] for s in summary]
        
        bars1 = ax1.bar(range(len(models)), silhouettes, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Best Silhouette Score')
        ax1.set_title('Clustering Quality Comparison')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, score in zip(bars1, silhouettes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Time vs Quality scatter
        times = [s['time'] for s in summary]
        ax2.scatter(times, silhouettes, s=100, alpha=0.7, c='coral')
        
        for i, model in enumerate([s['model'] for s in summary]):
            ax2.annotate(model.split('/')[-1], (times[i], silhouettes[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Embedding Time (seconds)')
        ax2.set_ylabel('Best Silhouette Score')
        ax2.set_title('Performance vs Quality Trade-off')
        
        plt.tight_layout()
        plt.savefig(self.comparison_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Visualization saved to: {self.comparison_dir}/model_comparison.png")

def run_embedding_comparison(event_texts: List[str], 
                           models_to_test: Optional[List[str]] = None) -> str:
    """
    Easy function to run embedding comparison.
    
    Args:
        event_texts: List of life event descriptions
        models_to_test: Optional list of models to test
        
    Returns:
        Path to best model name
    """
    comparator = EmbeddingComparator()
    results = comparator.test_embedding_models(event_texts, models_to_test)
    
    # Find best model
    best_model = None
    best_score = -1
    
    for model_name, result in results.items():
        if 'error' not in result:
            score = result['clustering_results']['best_silhouette']
            if score > best_score:
                best_score = score
                best_model = model_name
    
    print(f"\n🏆 BEST MODEL: {best_model} (silhouette: {best_score:.4f})")
    return best_model

def main():
    """Example usage."""
    print("EMBEDDING COMPARISON SYSTEM")
    print("=" * 50)
    
    # Example event texts (replace with your actual data)
    example_events = [
        "Born in Chicago, Illinois to working-class parents",
        "Arrested for burglary at age 16",
        "Served 3 years in state prison",
        "Released on parole, moved back with mother",
        "First murder committed in downtown area",
        "Victim found in abandoned warehouse",
        "Police investigation leads to suspect",
        "Trial begins with jury selection"
    ]
    
    print(f"[EXAMPLE] Testing with {len(example_events)} sample events")
    print("[INFO] Replace 'example_events' with your actual event texts")
    
    # Test a subset of models for demo
    test_models = [
        "all-MiniLM-L6-v2",      # Current baseline
        "all-mpnet-base-v2",     # State-of-the-art
        "all-MiniLM-L12-v2"      # Improved version
    ]
    
    best_model = run_embedding_comparison(example_events, test_models)
    
    print(f"\n💡 To use the best model in your analysis:")
    print(f"python run_modular_analysis.py --embedding_model={best_model} [other args]")

if __name__ == "__main__":
    main()
