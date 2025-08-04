#!/usr/bin/env python3
"""
report_generator.py

Generate comprehensive HTML reports for analysis results.
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..utils.helpers import load_json

class AnalysisReportGenerator:
    """Generate comprehensive analysis reports."""
    
    def __init__(self):
        self.template = self._get_html_template()
    
    def load_analysis_results(self, output_dir: str) -> Dict[str, Any]:
        """
        Load all analysis results from output directory.
        
        Args:
            output_dir: Output directory containing analysis results
            
        Returns:
            Dictionary containing all loaded results
        """
        results = {}
        
        # Load main results
        main_results_path = os.path.join(output_dir, "analysis_results.json")
        if os.path.exists(main_results_path):
            results['main'] = load_json(main_results_path)
        
        # Load cluster info
        cluster_info_path = os.path.join(output_dir, "clustering", "cluster_info.json")
        if os.path.exists(cluster_info_path):
            results['clusters'] = load_json(cluster_info_path)
        
        # Load conditional insights
        insights_path = os.path.join(output_dir, "analysis", "conditional_insights.json")
        if os.path.exists(insights_path):
            results['insights'] = load_json(insights_path)
        
        # Load k optimization results
        k_opt_path = os.path.join(output_dir, "clustering", "k_optimization_results.json")
        if os.path.exists(k_opt_path):
            results['k_optimization'] = load_json(k_opt_path)
        
        # Load configuration
        config_path = os.path.join(output_dir, "config.json")
        if os.path.exists(config_path):
            results['config'] = load_json(config_path)
        
        # Load criminal sequences
        sequences_path = os.path.join(output_dir, "criminal_sequences.json")
        if os.path.exists(sequences_path):
            results['sequences'] = load_json(sequences_path)
        
        return results
    
    def generate_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary statistics from analysis results.
        
        Args:
            results: Loaded analysis results
            
        Returns:
            Summary statistics
        """
        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_criminals': 0,
            'total_events': 0,
            'n_clusters': 0,
            'clustering_quality': {},
            'conditional_insights': {},
            'top_insights': []
        }
        
        # Basic statistics
        if 'main' in results and 'data_summary' in results['main']:
            data_summary = results['main']['data_summary']
            stats['total_criminals'] = data_summary.get('n_criminals', 0)
            stats['total_events'] = data_summary.get('n_events', 0)
        
        # Clustering statistics
        if 'main' in results and 'clustering' in results['main']:
            clustering = results['main']['clustering']
            stats['n_clusters'] = clustering.get('n_clusters', 0)
            stats['clustering_quality'] = {
                'silhouette': clustering.get('silhouette', 0),
                'calinski_harabasz': clustering.get('calinski_harabasz', 0),
                'davies_bouldin': clustering.get('davies_bouldin', float('inf'))
            }
        
        # Conditional insights statistics
        if 'insights' in results:
            insights = results['insights']
            significant_insights = [k for k, v in insights.items() if v.get('significant', False)]
            
            stats['conditional_insights'] = {
                'total_insights': len(insights),
                'significant_insights': len(significant_insights),
                'significance_rate': len(significant_insights) / len(insights) if insights else 0
            }
            
            # Top insights by L1 difference
            insight_items = [(k, v) for k, v in insights.items() if 'l1_difference' in v]
            insight_items.sort(key=lambda x: x[1]['l1_difference'], reverse=True)
            stats['top_insights'] = insight_items[:5]
        
        # K optimization statistics
        if 'k_optimization' in results:
            k_opt = results['k_optimization']
            if isinstance(k_opt, dict):
                best_k = max(k_opt.keys(), key=lambda k: k_opt[k].get('score', 0))
                stats['k_optimization'] = {
                    'optimal_k': best_k,
                    'best_score': k_opt[best_k].get('score', 0),
                    'tested_k_values': len(k_opt)
                }
        
        return stats
    
    def generate_html_report(self, results: Dict[str, Any], stats: Dict[str, Any], 
                           output_dir: str) -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            results: Analysis results
            stats: Summary statistics
            output_dir: Output directory
            
        Returns:
            Path to generated HTML report
        """
        # Generate report sections
        overview_html = self._generate_overview_section(stats)
        clustering_html = self._generate_clustering_section(results, stats)
        insights_html = self._generate_insights_section(results, stats)
        visualizations_html = self._generate_visualizations_section(output_dir)
        
        # Combine all sections
        report_content = self.template.format(
            title="Criminal Archetypal Analysis Report",
            timestamp=stats['timestamp'],
            overview=overview_html,
            clustering=clustering_html,
            insights=insights_html,
            visualizations=visualizations_html
        )
        
        # Save report
        report_path = os.path.join(output_dir, "analysis_report.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_path
    
    def _generate_overview_section(self, stats: Dict[str, Any]) -> str:
        """Generate overview section HTML."""
        return f"""
        <div class="section">
            <h2>Analysis Overview</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>{stats['total_criminals']}</h3>
                    <p>Total Criminals</p>
                </div>
                <div class="stat-card">
                    <h3>{stats['total_events']}</h3>
                    <p>Total Life Events</p>
                </div>
                <div class="stat-card">
                    <h3>{stats['n_clusters']}</h3>
                    <p>Event Clusters</p>
                </div>
                <div class="stat-card">
                    <h3>{stats['conditional_insights'].get('significant_insights', 0)}</h3>
                    <p>Significant Insights</p>
                </div>
            </div>
        </div>
        """
    
    def _generate_clustering_section(self, results: Dict[str, Any], stats: Dict[str, Any]) -> str:
        """Generate clustering section HTML."""
        quality = stats['clustering_quality']
        
        cluster_html = f"""
        <div class="section">
            <h2>Clustering Analysis</h2>
            <div class="metrics">
                <p><strong>Silhouette Score:</strong> {quality.get('silhouette', 0):.3f}</p>
                <p><strong>Calinski-Harabasz Score:</strong> {quality.get('calinski_harabasz', 0):.1f}</p>
                <p><strong>Davies-Bouldin Score:</strong> {quality.get('davies_bouldin', 0):.3f}</p>
            </div>
        """
        
        # Add cluster information if available
        if 'clusters' in results:
            cluster_html += "<h3>Cluster Themes</h3><ul>"
            for cluster in results['clusters']:
                theme = cluster.get('archetypal_theme', 'Unknown')
                size = cluster.get('size', 0)
                cluster_html += f"<li><strong>Cluster {cluster['cluster_id']}:</strong> {theme} ({size} events)</li>"
            cluster_html += "</ul>"
        
        cluster_html += "</div>"
        return cluster_html
    
    def _generate_insights_section(self, results: Dict[str, Any], stats: Dict[str, Any]) -> str:
        """Generate insights section HTML."""
        insights_html = """
        <div class="section">
            <h2>Conditional Insights</h2>
        """
        
        if 'top_insights' in stats and stats['top_insights']:
            insights_html += "<h3>Top Significant Patterns</h3><ul>"
            for insight_key, insight_data in stats['top_insights']:
                l1_diff = insight_data.get('l1_difference', 0)
                n_criminals = insight_data.get('n_criminals', 0)
                insights_html += f"<li><strong>{insight_key}:</strong> L1 difference = {l1_diff:.3f} (n={n_criminals})</li>"
            insights_html += "</ul>"
        
        insights_html += "</div>"
        return insights_html
    
    def _generate_visualizations_section(self, output_dir: str) -> str:
        """Generate visualizations section HTML."""
        viz_dir = os.path.join(output_dir, "visualization")
        
        visualizations = [
            ("tsne_visualization.png", "t-SNE Clustering Visualization"),
            ("clustering_comprehensive.png", "Comprehensive Clustering Results"),
            ("global_transition_diagram.png", "Global Transition Diagram"),
            ("transition_heatmap.png", "Transition Matrix Heatmap")
        ]
        
        viz_html = """
        <div class="section">
            <h2>Visualizations</h2>
            <div class="viz-grid">
        """
        
        for filename, title in visualizations:
            filepath = os.path.join(viz_dir, filename)
            if os.path.exists(filepath):
                rel_path = os.path.relpath(filepath, output_dir)
                viz_html += f"""
                <div class="viz-card">
                    <h4>{title}</h4>
                    <img src="{rel_path}" alt="{title}" style="max-width: 100%; height: auto;">
                </div>
                """
        
        viz_html += "</div></div>"
        return viz_html
    
    def _get_html_template(self) -> str:
        """Get HTML template for reports."""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; text-align: center; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
                h2 {{ color: #007bff; margin-top: 30px; }}
                .timestamp {{ text-align: center; color: #666; margin-bottom: 30px; }}
                .section {{ margin-bottom: 40px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: linear-gradient(135deg, #007bff, #0056b3); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
                .stat-card h3 {{ margin: 0; font-size: 2em; }}
                .stat-card p {{ margin: 10px 0 0 0; }}
                .metrics {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                .viz-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
                .viz-card {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; }}
                .viz-card h4 {{ margin-top: 0; color: #333; }}
                ul {{ padding-left: 20px; }}
                li {{ margin-bottom: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                <div class="timestamp">Generated on: {timestamp}</div>
                {overview}
                {clustering}
                {insights}
                {visualizations}
            </div>
        </body>
        </html>
        """

# Convenience functions for backward compatibility
def load_analysis_results(output_dir: str) -> Dict[str, Any]:
    """Load analysis results from output directory."""
    generator = AnalysisReportGenerator()
    return generator.load_analysis_results(output_dir)

def generate_summary_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary statistics from results."""
    generator = AnalysisReportGenerator()
    return generator.generate_summary_statistics(results)

def generate_html_report(results: Dict[str, Any], stats: Dict[str, Any], 
                        output_dir: str) -> str:
    """Generate HTML report."""
    generator = AnalysisReportGenerator()
    return generator.generate_html_report(results, stats, output_dir)
