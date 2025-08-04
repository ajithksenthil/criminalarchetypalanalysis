#!/usr/bin/env python3
"""
diagrams.py

Visualization functionality for criminal archetypal analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from typing import Optional, List, Dict, Any
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class TransitionDiagramGenerator:
    """Generate transition diagrams and network visualizations."""
    
    def plot_state_transition_diagram(self, transition_matrix: np.ndarray, 
                                    out_path: str = "state_transition.png", 
                                    title: str = "State Transition Diagram") -> None:
        """
        Plot a state transition diagram using networkx.
        
        Args:
            transition_matrix: Transition matrix
            out_path: Output file path
            title: Diagram title
        """
        G = nx.DiGraph()
        num_states = transition_matrix.shape[0]
        
        # Add nodes
        for i in range(num_states):
            G.add_node(i)
        
        # Add edges with weights
        for i in range(num_states):
            for j in range(num_states):
                weight = transition_matrix[i, j]
                if weight > 0.01:  # Only show significant transitions
                    G.add_edge(i, j, weight=weight)
        
        # Create layout
        pos = nx.circular_layout(G)
        
        # Plot
        plt.figure(figsize=(10, 10))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.7)
        
        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights], alpha=0.6, edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # Draw edge labels
        edge_labels = {(i, j): f"{w:.2f}" for (i, j), w in zip(edges, weights) if w > 0.05}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] State transition diagram saved to {out_path}")

class ClusteringVisualizer:
    """Visualize clustering results."""
    
    def plot_tsne_embeddings(self, embeddings: np.ndarray, labels: np.ndarray, 
                           out_path: str = "tsne.png", title: str = "t-SNE Clustering Visualization") -> None:
        """
        Plot t-SNE visualization of embeddings with cluster colors.
        
        Args:
            embeddings: Input embeddings
            labels: Cluster labels
            out_path: Output file path
            title: Plot title
        """
        # Subsample if too many points
        if len(embeddings) > 1000:
            indices = np.random.choice(len(embeddings), 1000, replace=False)
            embeddings_sub = embeddings[indices]
            labels_sub = labels[indices]
        else:
            embeddings_sub = embeddings
            labels_sub = labels
        
        # Compute t-SNE
        print("[INFO] Computing t-SNE embedding...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_sub)-1))
        embeddings_2d = tsne.fit_transform(embeddings_sub)
        
        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels_sub, cmap='tab10', alpha=0.7, s=50)
        plt.colorbar(scatter, label='Cluster')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] t-SNE visualization saved to {out_path}")
    
    def plot_clustering_results(self, embeddings: np.ndarray, labels: np.ndarray, 
                              out_path: str = "clustering_results.png") -> None:
        """
        Plot comprehensive clustering results with multiple visualizations.
        
        Args:
            embeddings: Input embeddings
            labels: Cluster labels
            out_path: Output file path
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Subsample for visualization
        if len(embeddings) > 1000:
            indices = np.random.choice(len(embeddings), 1000, replace=False)
            embeddings_sub = embeddings[indices]
            labels_sub = labels[indices]
        else:
            embeddings_sub = embeddings
            labels_sub = labels
        
        # t-SNE plot
        print("[INFO] Computing t-SNE for clustering results...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_sub)-1))
        embeddings_tsne = tsne.fit_transform(embeddings_sub)
        
        axes[0, 0].scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                          c=labels_sub, cmap='tab10', alpha=0.7, s=30)
        axes[0, 0].set_title('t-SNE Visualization')
        axes[0, 0].set_xlabel('t-SNE 1')
        axes[0, 0].set_ylabel('t-SNE 2')
        
        # PCA plot
        if embeddings_sub.shape[1] > 2:
            pca = PCA(n_components=2)
            embeddings_pca = pca.fit_transform(embeddings_sub)
            
            axes[0, 1].scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                              c=labels_sub, cmap='tab10', alpha=0.7, s=30)
            axes[0, 1].set_title(f'PCA Visualization (explained var: {pca.explained_variance_ratio_.sum():.2%})')
            axes[0, 1].set_xlabel('PC1')
            axes[0, 1].set_ylabel('PC2')
        
        # Cluster size distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        axes[1, 0].bar(unique_labels, counts, alpha=0.7)
        axes[1, 0].set_title('Cluster Size Distribution')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('Number of Points')
        
        # Cluster statistics
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        try:
            sil_score = silhouette_score(embeddings_sub, labels_sub)
            cal_score = calinski_harabasz_score(embeddings_sub, labels_sub)
            
            stats_text = f'Silhouette Score: {sil_score:.3f}\nCalinski-Harabasz: {cal_score:.1f}\nNumber of Clusters: {len(unique_labels)}'
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
            axes[1, 1].set_title('Clustering Statistics')
            axes[1, 1].axis('off')
        except:
            axes[1, 1].text(0.1, 0.5, 'Statistics unavailable', fontsize=12, verticalalignment='center')
            axes[1, 1].set_title('Clustering Statistics')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Clustering results visualization saved to {out_path}")

class HeatmapGenerator:
    """Generate heatmaps for matrix visualizations."""
    
    def plot_transition_matrix_heatmap(self, transition_matrix: np.ndarray, 
                                     out_path: str = "transition_heatmap.png",
                                     title: str = "Transition Matrix Heatmap") -> None:
        """
        Plot transition matrix as a heatmap.
        
        Args:
            transition_matrix: Transition matrix
            out_path: Output file path
            title: Plot title
        """
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(transition_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=range(transition_matrix.shape[1]),
                   yticklabels=range(transition_matrix.shape[0]),
                   cbar_kws={'label': 'Transition Probability'})
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('To State')
        plt.ylabel('From State')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Transition matrix heatmap saved to {out_path}")
    
    def plot_correlation_matrix(self, data: np.ndarray, labels: List[str],
                              out_path: str = "correlation_matrix.png",
                              title: str = "Correlation Matrix") -> None:
        """
        Plot correlation matrix heatmap.
        
        Args:
            data: Input data matrix
            labels: Feature labels
            out_path: Output file path
            title: Plot title
        """
        # Compute correlation matrix
        corr_matrix = np.corrcoef(data.T)
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   xticklabels=labels, yticklabels=labels,
                   center=0, vmin=-1, vmax=1,
                   cbar_kws={'label': 'Correlation'})
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Correlation matrix saved to {out_path}")

class DendrogramGenerator:
    """Generate dendrograms for hierarchical clustering."""
    
    def plot_hierarchical_dendrogram(self, embeddings: np.ndarray, 
                                   out_path: str = "dendrogram.png",
                                   title: str = "Hierarchical Clustering Dendrogram") -> None:
        """
        Plot hierarchical clustering dendrogram.
        
        Args:
            embeddings: Input embeddings
            out_path: Output file path
            title: Plot title
        """
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import pdist
        
        # Subsample if too many points
        if len(embeddings) > 100:
            indices = np.random.choice(len(embeddings), 100, replace=False)
            embeddings_sub = embeddings[indices]
            labels = [f"Point {i}" for i in indices]
        else:
            embeddings_sub = embeddings
            labels = [f"Point {i}" for i in range(len(embeddings))]
        
        # Compute distance matrix and linkage
        print("[INFO] Computing hierarchical clustering...")
        dist_matrix = pdist(embeddings_sub, metric='euclidean')
        linkage_matrix = linkage(dist_matrix, method='ward')
        
        # Plot dendrogram
        plt.figure(figsize=(15, 8))
        dendrogram(linkage_matrix, labels=labels, orientation='top', leaf_rotation=90)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Data Points')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Dendrogram saved to {out_path}")

class ComprehensiveVisualizer:
    """Comprehensive visualization suite."""
    
    def __init__(self):
        self.transition_gen = TransitionDiagramGenerator()
        self.clustering_vis = ClusteringVisualizer()
        self.heatmap_gen = HeatmapGenerator()
        self.dendrogram_gen = DendrogramGenerator()
    
    def create_analysis_dashboard(self, embeddings: np.ndarray, labels: np.ndarray,
                                transition_matrix: np.ndarray, output_dir: str) -> None:
        """
        Create a comprehensive analysis dashboard.
        
        Args:
            embeddings: Input embeddings
            labels: Cluster labels
            transition_matrix: Global transition matrix
            output_dir: Output directory
        """
        print("[INFO] Creating comprehensive analysis dashboard...")
        
        # Clustering visualizations
        self.clustering_vis.plot_tsne_embeddings(
            embeddings, labels, f"{output_dir}/tsne_visualization.png"
        )
        
        self.clustering_vis.plot_clustering_results(
            embeddings, labels, f"{output_dir}/clustering_comprehensive.png"
        )
        
        # Transition matrix visualizations
        self.transition_gen.plot_state_transition_diagram(
            transition_matrix, f"{output_dir}/global_transition_diagram.png"
        )
        
        self.heatmap_gen.plot_transition_matrix_heatmap(
            transition_matrix, f"{output_dir}/transition_heatmap.png"
        )
        
        # Hierarchical clustering
        self.dendrogram_gen.plot_hierarchical_dendrogram(
            embeddings, f"{output_dir}/hierarchical_dendrogram.png"
        )
        
        print(f"[INFO] Analysis dashboard created in {output_dir}")

# Backward compatibility aliases
def plot_state_transition_diagram(transition_matrix: np.ndarray,
                                out_path: str = "state_transition.png",
                                title: str = "State Transition Diagram") -> None:
    """Backward compatibility alias."""
    generator = TransitionDiagramGenerator()
    generator.plot_state_transition_diagram(transition_matrix, out_path, title)

def plot_tsne_embeddings(embeddings: np.ndarray, labels: np.ndarray,
                       out_path: str = "tsne.png", title: str = "t-SNE Clustering Visualization") -> None:
    """Backward compatibility alias."""
    visualizer = ClusteringVisualizer()
    visualizer.plot_tsne_embeddings(embeddings, labels, out_path, title)

def plot_clustering_results(embeddings: np.ndarray, labels: np.ndarray,
                          out_path: str = "clustering_results.png") -> None:
    """Backward compatibility alias."""
    visualizer = ClusteringVisualizer()
    visualizer.plot_clustering_results(embeddings, labels, out_path)

def hierarchical_clustering_dendrogram(embeddings: np.ndarray,
                                     out_path: str = "dendrogram.png") -> None:
    """Backward compatibility alias."""
    generator = DendrogramGenerator()
    generator.plot_hierarchical_dendrogram(embeddings, out_path)
