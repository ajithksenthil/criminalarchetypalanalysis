#!/usr/bin/env python3
"""
interactive_visualizations.py

Create interactive visualizations for criminal archetypal analysis.
Includes Sankey diagrams, 3D cluster visualizations, and network graphs.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')


def create_sankey_diagram(sequences, cluster_labels, cluster_names=None, 
                         save_path='sankey_diagram.html'):
    """
    Create an interactive Sankey diagram showing life event flows between archetypes.
    
    Args:
        sequences: List of event sequences (indices)
        cluster_labels: Cluster label for each event
        cluster_names: Optional names for clusters
        save_path: Path to save the HTML file
    """
    # Prepare cluster names
    if cluster_names is None:
        n_clusters = max(cluster_labels) + 1
        cluster_names = [f"Archetype {i}" for i in range(n_clusters)]
    
    # Build flow data
    flows = defaultdict(lambda: defaultdict(int))
    
    for seq in sequences:
        for i in range(len(seq) - 1):
            from_event = seq[i]
            to_event = seq[i + 1]
            
            from_cluster = cluster_labels[from_event]
            to_cluster = cluster_labels[to_event]
            
            flows[from_cluster][to_cluster] += 1
    
    # Create nodes and links
    nodes = []
    node_labels = []
    
    # Add nodes for each time step
    max_seq_length = max(len(seq) for seq in sequences)
    time_steps = min(5, max_seq_length)  # Limit to 5 time steps for clarity
    
    node_mapping = {}
    node_idx = 0
    
    for t in range(time_steps):
        for cluster_idx, cluster_name in enumerate(cluster_names):
            node_key = f"T{t}_{cluster_idx}"
            node_mapping[node_key] = node_idx
            node_labels.append(f"{cluster_name} (T{t})")
            node_idx += 1
    
    # Build links
    sources = []
    targets = []
    values = []
    link_labels = []
    
    # Sample sequences to match time steps
    sampled_sequences = []
    for seq in sequences:
        if len(seq) >= time_steps:
            # Sample evenly across the sequence
            indices = np.linspace(0, len(seq)-1, time_steps, dtype=int)
            sampled_seq = [seq[i] for i in indices]
            sampled_sequences.append(sampled_seq)
    
    # Count transitions
    for seq in sampled_sequences:
        for t in range(len(seq) - 1):
            from_event = seq[t]
            to_event = seq[t + 1]
            
            from_cluster = cluster_labels[from_event]
            to_cluster = cluster_labels[to_event]
            
            source_key = f"T{t}_{from_cluster}"
            target_key = f"T{t+1}_{to_cluster}"
            
            if source_key in node_mapping and target_key in node_mapping:
                source_idx = node_mapping[source_key]
                target_idx = node_mapping[target_key]
                
                # Check if this link already exists
                link_exists = False
                for i, (s, t) in enumerate(zip(sources, targets)):
                    if s == source_idx and t == target_idx:
                        values[i] += 1
                        link_exists = True
                        break
                
                if not link_exists:
                    sources.append(source_idx)
                    targets.append(target_idx)
                    values.append(1)
                    link_labels.append(f"{cluster_names[from_cluster]} → {cluster_names[to_cluster]}")
    
    # Create color scheme
    n_clusters = len(cluster_names)
    colors = px.colors.qualitative.Set3[:n_clusters]
    node_colors = []
    for t in range(time_steps):
        node_colors.extend(colors)
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels,
            color="rgba(0,0,0,0.2)"
        )
    )])
    
    fig.update_layout(
        title={
            'text': "Criminal Life Event Flows Between Archetypes",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        font_size=12,
        height=800,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    # Save as HTML
    fig.write_html(save_path)
    print(f"Sankey diagram saved to {save_path}")
    
    return fig


def create_3d_cluster_visualization(embeddings, labels, cluster_names=None,
                                  method='tsne', save_path='clusters_3d.html'):
    """
    Create interactive 3D visualization of clusters.
    
    Args:
        embeddings: Event embeddings
        labels: Cluster labels
        cluster_names: Optional names for clusters
        method: Dimensionality reduction method ('tsne' or 'pca')
        save_path: Path to save HTML file
    """
    print(f"Creating 3D visualization using {method.upper()}...")
    
    # Reduce to 3D
    if method == 'tsne':
        reducer = TSNE(n_components=3, random_state=42, perplexity=30)
        coords_3d = reducer.fit_transform(embeddings)
    else:  # PCA
        reducer = PCA(n_components=3, random_state=42)
        coords_3d = reducer.fit_transform(embeddings)
    
    # Prepare cluster names
    if cluster_names is None:
        n_clusters = max(labels) + 1
        cluster_names = [f"Archetype {i}" for i in range(n_clusters)]
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': coords_3d[:, 0],
        'y': coords_3d[:, 1],
        'z': coords_3d[:, 2],
        'cluster': labels,
        'cluster_name': [cluster_names[l] for l in labels]
    })
    
    # Create 3D scatter plot
    fig = px.scatter_3d(df, x='x', y='y', z='z', 
                       color='cluster_name',
                       labels={'cluster_name': 'Archetype'},
                       title='3D Visualization of Criminal Archetypes',
                       color_discrete_sequence=px.colors.qualitative.Set3)
    
    # Update layout
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(
        scene=dict(
            xaxis_title=f'{method.upper()} 1',
            yaxis_title=f'{method.upper()} 2',
            zaxis_title=f'{method.upper()} 3'
        ),
        height=800,
        font_size=12
    )
    
    # Save as HTML
    fig.write_html(save_path)
    print(f"3D visualization saved to {save_path}")
    
    return fig


def create_criminal_network_graph(embeddings, criminal_names=None, 
                                similarity_threshold=0.7,
                                save_path='criminal_network.html'):
    """
    Create an interactive network graph of criminals based on similarity.
    
    Args:
        embeddings: Criminal embeddings (one per criminal)
        criminal_names: Optional names for criminals
        similarity_threshold: Minimum similarity to create edge
        save_path: Path to save HTML file
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    n_criminals = len(embeddings)
    
    if criminal_names is None:
        criminal_names = [f"Criminal {i}" for i in range(n_criminals)]
    
    # Compute similarity matrix
    similarities = cosine_similarity(embeddings)
    
    # Create network
    G = nx.Graph()
    
    # Add nodes
    for i, name in enumerate(criminal_names):
        G.add_node(i, name=name)
    
    # Add edges based on similarity
    edge_weights = []
    for i in range(n_criminals):
        for j in range(i + 1, n_criminals):
            if similarities[i, j] > similarity_threshold:
                G.add_edge(i, j, weight=similarities[i, j])
                edge_weights.append(similarities[i, j])
    
    # Calculate layout
    pos = nx.spring_layout(G, k=1/np.sqrt(n_criminals), iterations=50)
    
    # Extract node positions
    node_x = []
    node_y = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{G.nodes[node]['name']}<br>Degree: {G.degree(node)}")
    
    # Extract edge positions
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        weight = G[edge[0]][edge[1]]['weight']
        edge_text.append(f"Similarity: {weight:.3f}")
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(125,125,125,0.5)'),
        hoverinfo='none',
        mode='lines')
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            size=10,
            color=[G.degree(node) for node in G.nodes()],
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Criminal Similarity Network',
                       titlefont_size=20,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=800
                   ))
    
    # Add network statistics
    avg_degree = sum(dict(G.degree()).values()) / n_criminals
    clustering_coef = nx.average_clustering(G)
    
    fig.add_annotation(
        text=f"Avg Degree: {avg_degree:.2f} | Clustering Coefficient: {clustering_coef:.3f}",
        xref="paper", yref="paper",
        x=0.5, y=0.05,
        showarrow=False,
        font=dict(size=12)
    )
    
    # Save as HTML
    fig.write_html(save_path)
    print(f"Criminal network saved to {save_path}")
    
    return fig, G


def create_transition_heatmap(transition_matrix, state_names=None,
                            save_path='transition_heatmap.html'):
    """
    Create an interactive heatmap of transition probabilities.
    
    Args:
        transition_matrix: Markov chain transition matrix
        state_names: Names for states
        save_path: Path to save HTML file
    """
    n_states = transition_matrix.shape[0]
    
    if state_names is None:
        state_names = [f"State {i}" for i in range(n_states)]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=transition_matrix,
        x=state_names,
        y=state_names,
        colorscale='Viridis',
        text=np.round(transition_matrix, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Probability")
    ))
    
    fig.update_layout(
        title='Criminal Event Transition Probabilities',
        xaxis_title='To State',
        yaxis_title='From State',
        height=800,
        width=800
    )
    
    # Save as HTML
    fig.write_html(save_path)
    print(f"Transition heatmap saved to {save_path}")
    
    return fig


def create_comprehensive_dashboard(analysis_results, save_path='dashboard.html'):
    """
    Create a comprehensive interactive dashboard with multiple visualizations.
    
    Args:
        analysis_results: Dictionary containing analysis results
        save_path: Path to save HTML file
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cluster Distribution', 'Silhouette Scores by Cluster',
                       'Temporal Patterns', 'Change Point Distribution'),
        specs=[[{'type': 'pie'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'histogram'}]]
    )
    
    # 1. Cluster distribution (pie chart)
    if 'cluster_sizes' in analysis_results:
        cluster_sizes = analysis_results['cluster_sizes']
        fig.add_trace(
            go.Pie(
                labels=[f"Cluster {k}" for k in cluster_sizes.keys()],
                values=list(cluster_sizes.values()),
                hole=0.3
            ),
            row=1, col=1
        )
    
    # 2. Silhouette scores by cluster (bar chart)
    if 'silhouette_samples' in analysis_results:
        silhouette_vals = analysis_results['silhouette_samples']
        cluster_labels = analysis_results['cluster_labels']
        
        silhouette_by_cluster = defaultdict(list)
        for val, label in zip(silhouette_vals, cluster_labels):
            silhouette_by_cluster[label].append(val)
        
        avg_silhouettes = {k: np.mean(v) for k, v in silhouette_by_cluster.items()}
        
        fig.add_trace(
            go.Bar(
                x=[f"Cluster {k}" for k in avg_silhouettes.keys()],
                y=list(avg_silhouettes.values()),
                marker_color='lightblue'
            ),
            row=1, col=2
        )
    
    # 3. Temporal patterns (scatter plot)
    if 'temporal_patterns' in analysis_results:
        patterns = analysis_results['temporal_patterns']
        fig.add_trace(
            go.Scatter(
                x=patterns['ages'],
                y=patterns['event_counts'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=patterns['phases'],
                    colorscale='Viridis',
                    showscale=True
                )
            ),
            row=2, col=1
        )
    
    # 4. Change point distribution (histogram)
    if 'change_points' in analysis_results:
        change_positions = analysis_results['change_points']['change_point_positions']
        fig.add_trace(
            go.Histogram(
                x=change_positions,
                nbinsx=20,
                marker_color='salmon'
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text="Criminal Archetypal Analysis Dashboard",
        showlegend=False,
        height=1000,
        font_size=12
    )
    
    # Save as HTML
    fig.write_html(save_path)
    print(f"Dashboard saved to {save_path}")
    
    return fig


def create_parallel_coordinates(features_df, cluster_column='cluster',
                              save_path='parallel_coords.html'):
    """
    Create parallel coordinates plot for exploring feature relationships.
    
    Args:
        features_df: DataFrame with features and cluster labels
        cluster_column: Name of cluster column
        save_path: Path to save HTML file
    """
    # Select numeric columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != cluster_column]
    
    # Normalize features to 0-1 range
    df_normalized = features_df.copy()
    for col in numeric_cols:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        if max_val > min_val:
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    
    # Create parallel coordinates plot
    fig = px.parallel_coordinates(
        df_normalized,
        dimensions=numeric_cols,
        color=cluster_column,
        labels={col: col.replace('_', ' ').title() for col in numeric_cols},
        title="Criminal Feature Patterns by Archetype",
        color_continuous_scale=px.colors.diverging.Tealrose
    )
    
    fig.update_layout(height=600)
    
    # Save as HTML
    fig.write_html(save_path)
    print(f"Parallel coordinates saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Test visualizations with dummy data
    print("Testing Interactive Visualizations...")
    
    # Generate dummy data
    n_events = 1000
    n_clusters = 5
    
    # Random cluster labels
    cluster_labels = np.random.randint(0, n_clusters, n_events)
    cluster_names = [f"Archetype {i}" for i in range(n_clusters)]
    
    # Generate dummy sequences
    sequences = []
    for _ in range(100):
        seq_length = np.random.randint(5, 20)
        seq = np.random.choice(n_events, seq_length, replace=False)
        sequences.append(list(seq))
    
    # Create Sankey diagram
    print("\nCreating Sankey diagram...")
    sankey_fig = create_sankey_diagram(sequences, cluster_labels, cluster_names)
    
    print("\nInteractive visualizations ready!")
    print("Open the generated HTML files in your browser to explore.")