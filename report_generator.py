#!/usr/bin/env python3
"""
report_generator.py

Generates comprehensive analysis reports from the criminal archetypal analysis results.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from jinja2 import Template


def load_analysis_results(output_dir):
    """Load all analysis results from the output directory."""
    results = {}
    
    # Load JSON files
    json_files = [
        'global_clusters.json',
        'cluster_metrics.json',
        'conditional_insights.json',
        'criminal_clustering_results.json',
        'criminal_sequences.json'
    ]
    
    for file in json_files:
        path = os.path.join(output_dir, file)
        if os.path.exists(path):
            with open(path, 'r') as f:
                results[file.replace('.json', '')] = json.load(f)
    
    # Load numpy arrays
    npy_files = [
        'global_transition_matrix.npy',
        'global_stationary_distribution.npy'
    ]
    
    for file in npy_files:
        path = os.path.join(output_dir, file)
        if os.path.exists(path):
            results[file.replace('.npy', '')] = np.load(path)
    
    return results


def generate_summary_statistics(results):
    """Generate summary statistics from the analysis results."""
    stats = {}
    
    # Basic counts
    if 'criminal_sequences' in results:
        stats['n_criminals'] = len(results['criminal_sequences'])
        stats['total_events'] = sum(len(seq) for seq in results['criminal_sequences'].values())
        stats['avg_events_per_criminal'] = stats['total_events'] / stats['n_criminals']
    
    # Cluster metrics
    if 'cluster_metrics' in results:
        stats['silhouette_score'] = results['cluster_metrics'].get('silhouette', 0)
        stats['davies_bouldin_score'] = results['cluster_metrics'].get('davies_bouldin', 0)
    
    # Criminal clustering
    if 'criminal_clustering_results' in results:
        stats['criminal_cluster_silhouette'] = results['criminal_clustering_results'].get('silhouette_score', 0)
        stats['criminal_cluster_sizes'] = results['criminal_clustering_results'].get('cluster_sizes', {})
    
    # Conditional insights
    if 'conditional_insights' in results:
        stats['n_significant_conditions'] = sum(
            1 for insight in results['conditional_insights'].values()
            if insight.get('significant', False)
        )
        stats['total_conditions_analyzed'] = len(results['conditional_insights'])
    
    return stats


def create_visualization_summary(output_dir, report_dir):
    """Create a summary visualization combining key plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Load and display key visualizations
    viz_files = [
        ('global_state_transition.png', 'Global State Transitions'),
        ('tsne.png', 'Event Clustering (t-SNE)'),
        ('criminal_clustering_pca.png', 'Criminal Clustering (PCA)'),
        ('criminal_clustering_dendrogram.png', 'Criminal Hierarchy')
    ]
    
    for idx, (file, title) in enumerate(viz_files):
        ax = axes[idx // 2, idx % 2]
        path = os.path.join(output_dir, file)
        if os.path.exists(path):
            img = plt.imread(path)
            ax.imshow(img)
            ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    summary_path = os.path.join(report_dir, 'visualization_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return summary_path


def generate_html_report(results, stats, output_dir):
    """Generate an HTML report from the analysis results."""
    
    report_template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>Criminal Archetypal Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        h1, h2, h3 { color: #333; }
        .summary-box { background: #f4f4f4; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .metric { display: inline-block; margin: 10px 20px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .metric-label { font-size: 14px; color: #7f8c8d; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .significant { color: #e74c3c; font-weight: bold; }
        .cluster-info { margin: 20px 0; padding: 15px; background: #ecf0f1; border-left: 4px solid #3498db; }
        img { max-width: 100%; height: auto; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Criminal Archetypal Analysis Report</h1>
    <p>Generated on: {{ timestamp }}</p>
    
    <div class="summary-box">
        <h2>Executive Summary</h2>
        <div class="metric">
            <div class="metric-value">{{ stats.n_criminals }}</div>
            <div class="metric-label">Criminals Analyzed</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ stats.total_events }}</div>
            <div class="metric-label">Total Life Events</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ "%.1f"|format(stats.avg_events_per_criminal) }}</div>
            <div class="metric-label">Avg Events/Criminal</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ n_event_clusters }}</div>
            <div class="metric-label">Event Clusters</div>
        </div>
    </div>
    
    <h2>Clustering Quality Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Interpretation</th>
        </tr>
        <tr>
            <td>Event Clustering Silhouette Score</td>
            <td>{{ "%.3f"|format(stats.silhouette_score) }}</td>
            <td>{{ interpret_silhouette(stats.silhouette_score) }}</td>
        </tr>
        <tr>
            <td>Davies-Bouldin Score</td>
            <td>{{ "%.3f"|format(stats.davies_bouldin_score) }}</td>
            <td>Lower is better (cluster separation)</td>
        </tr>
        <tr>
            <td>Criminal Clustering Silhouette Score</td>
            <td>{{ "%.3f"|format(stats.criminal_cluster_silhouette) }}</td>
            <td>{{ interpret_silhouette(stats.criminal_cluster_silhouette) }}</td>
        </tr>
    </table>
    
    <h2>Archetypal Life Event Clusters</h2>
    {% for cluster in event_clusters %}
    <div class="cluster-info">
        <h3>Cluster {{ cluster.cluster_id }}: {{ cluster.archetypal_theme }}</h3>
        <p><strong>Size:</strong> {{ cluster.size }} events</p>
        <p><strong>Representative Examples:</strong></p>
        <ul>
        {% for example in cluster.representative_samples[:3] %}
            <li>{{ example }}</li>
        {% endfor %}
        </ul>
    </div>
    {% endfor %}
    
    <h2>Criminal Clustering Results</h2>
    <p>Criminals were clustered into {{ criminal_cluster_sizes|length }} groups based on their life event transition patterns:</p>
    <table>
        <tr>
            <th>Cluster</th>
            <th>Number of Criminals</th>
            <th>Percentage</th>
        </tr>
        {% for cluster, size in criminal_cluster_sizes.items() %}
        <tr>
            <td>Cluster {{ cluster }}</td>
            <td>{{ size }}</td>
            <td>{{ "%.1f"|format(100 * size / stats.n_criminals) }}%</td>
        </tr>
        {% endfor %}
    </table>
    
    <h2>Significant Conditional Insights</h2>
    <p>Found <span class="significant">{{ stats.n_significant_conditions }}</span> statistically significant conditions out of {{ stats.total_conditions_analyzed }} analyzed.</p>
    
    {% if significant_insights %}
    <table>
        <tr>
            <th>Condition</th>
            <th>Sample Size</th>
            <th>KS p-value</th>
            <th>Wasserstein Distance</th>
            <th>Effect Size</th>
        </tr>
        {% for name, insight in significant_insights %}
        <tr>
            <td>{{ name }}</td>
            <td>{{ insight.n_criminals }}</td>
            <td class="significant">{{ "%.4f"|format(insight.statistics.ks_pvalue) }}</td>
            <td>{{ "%.3f"|format(insight.statistics.wasserstein_distance) }}</td>
            <td>{{ "%.3f"|format(insight.difference) }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}
    
    <h2>Visualizations</h2>
    <img src="visualization_summary.png" alt="Analysis Visualizations Summary">
    
    <h2>Global Transition Analysis</h2>
    <h3>Stationary Distribution</h3>
    <p>The long-term probability distribution across event clusters:</p>
    <table>
        <tr>
            <th>Cluster</th>
            {% for i in range(n_event_clusters) %}
            <th>{{ i }}</th>
            {% endfor %}
        </tr>
        <tr>
            <td><strong>Probability</strong></td>
            {% for prob in stationary_dist %}
            <td>{{ "%.3f"|format(prob) }}</td>
            {% endfor %}
        </tr>
    </table>
    
    <h2>Recommendations for Further Analysis</h2>
    <ul>
        <li>Investigate the {{ stats.n_significant_conditions }} significant conditional patterns in detail</li>
        <li>Examine temporal patterns within each criminal cluster</li>
        <li>Cross-reference criminal clusters with additional Type 2 variables</li>
        <li>Perform survival analysis on time-to-crime based on early life events</li>
        <li>Develop predictive models using the identified archetypal patterns</li>
    </ul>
    
    <footer>
        <p><em>This report was automatically generated by the Criminal Archetypal Analysis system.</em></p>
    </footer>
</body>
</html>
""")
    
    # Helper functions for the template
    def interpret_silhouette(score):
        if score > 0.7:
            return "Excellent clustering"
        elif score > 0.5:
            return "Good clustering"
        elif score > 0.25:
            return "Acceptable clustering"
        else:
            return "Poor clustering - consider adjusting parameters"
    
    # Prepare template data
    template_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stats': stats,
        'n_event_clusters': len(results.get('global_clusters', [])),
        'event_clusters': results.get('global_clusters', []),
        'criminal_cluster_sizes': stats.get('criminal_cluster_sizes', {}),
        'significant_insights': [
            (name, data) for name, data in results.get('conditional_insights', {}).items()
            if data.get('significant', False)
        ][:10],  # Top 10 significant insights
        'stationary_dist': results.get('global_stationary_distribution', []),
        'interpret_silhouette': interpret_silhouette
    }
    
    # Generate report
    html_content = report_template.render(**template_data)
    
    # Create report directory
    report_dir = os.path.join(output_dir, 'report')
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate visualization summary
    create_visualization_summary(output_dir, report_dir)
    
    # Save HTML report
    report_path = os.path.join(report_dir, 'analysis_report.html')
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"[INFO] Comprehensive report generated at {report_path}")
    return report_path


def main():
    """Generate report from existing analysis results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate analysis report")
    parser.add_argument("--output_dir", required=True, help="Directory containing analysis outputs")
    args = parser.parse_args()
    
    # Load results
    results = load_analysis_results(args.output_dir)
    
    # Generate statistics
    stats = generate_summary_statistics(results)
    
    # Generate HTML report
    report_path = generate_html_report(results, stats, args.output_dir)
    
    print(f"Report generation complete. Open {report_path} in a web browser to view.")


if __name__ == "__main__":
    main()