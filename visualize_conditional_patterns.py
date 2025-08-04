#!/usr/bin/env python3
"""
visualize_conditional_patterns.py

Create visualizations for conditional patterns analysis.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict


def create_pattern_summary_plot(insights, output_file='conditional_patterns_summary.png'):
    """Create a summary visualization of conditional patterns."""
    
    # Group by category
    patterns_by_category = defaultdict(list)
    for key, value in insights.items():
        category = key.split('=')[0]
        patterns_by_category[category].append({
            'pattern': key,
            'significant': value.get('significant', False),
            'p_value': value['statistics']['ks_pvalue'],
            'effect_size': value['difference'],
            'n_criminals': value['n_criminals']
        })
    
    # Calculate summary statistics per category
    category_stats = []
    for category, patterns in patterns_by_category.items():
        n_total = len(patterns)
        n_significant = sum(1 for p in patterns if p['significant'])
        avg_effect = np.mean([p['effect_size'] for p in patterns])
        
        category_stats.append({
            'Category': category,
            'Total Patterns': n_total,
            'Significant': n_significant,
            'Proportion Significant': n_significant / n_total if n_total > 0 else 0,
            'Avg Effect Size': avg_effect
        })
    
    # Create DataFrame and sort
    df = pd.DataFrame(category_stats)
    df = df.sort_values('Proportion Significant', ascending=False)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Top categories by significance rate
    ax1 = axes[0, 0]
    top_categories = df.head(15)
    ax1.barh(top_categories['Category'], top_categories['Proportion Significant'])
    ax1.set_xlabel('Proportion of Significant Patterns')
    ax1.set_title('Top 15 Categories by Significance Rate')
    ax1.set_xlim(0, 1.1)
    
    # Add value labels
    for i, (idx, row) in enumerate(top_categories.iterrows()):
        ax1.text(row['Proportion Significant'] + 0.02, i, 
                f"{row['Significant']}/{row['Total Patterns']}", 
                va='center', fontsize=8)
    
    # 2. Distribution of p-values
    ax2 = axes[0, 1]
    all_p_values = [p['p_value'] for patterns in patterns_by_category.values() 
                    for p in patterns]
    ax2.hist(all_p_values, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
    ax2.set_xlabel('P-value')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of P-values')
    ax2.legend()
    ax2.set_xlim(0, 1)
    
    # 3. Effect size vs sample size
    ax3 = axes[1, 0]
    effect_sizes = []
    sample_sizes = []
    colors = []
    
    for patterns in patterns_by_category.values():
        for p in patterns:
            effect_sizes.append(p['effect_size'])
            sample_sizes.append(p['n_criminals'])
            colors.append('red' if p['significant'] else 'blue')
    
    ax3.scatter(sample_sizes, effect_sizes, c=colors, alpha=0.6)
    ax3.set_xlabel('Number of Criminals')
    ax3.set_ylabel('Effect Size (L1 Distance)')
    ax3.set_title('Effect Size vs Sample Size')
    ax3.legend(['Significant', 'Not Significant'])
    
    # 4. Most significant patterns
    ax4 = axes[1, 1]
    sig_patterns = [(k, v) for k, v in insights.items() if v.get('significant', False)]
    sig_patterns.sort(key=lambda x: x[1]['statistics']['ks_pvalue'])
    
    top_patterns = sig_patterns[:10]
    pattern_names = [p[0] for p in top_patterns]
    p_values = [-np.log10(p[1]['statistics']['ks_pvalue'] + 1e-10) for p in top_patterns]
    
    ax4.barh(range(len(pattern_names)), p_values)
    ax4.set_yticks(range(len(pattern_names)))
    ax4.set_yticklabels(pattern_names, fontsize=8)
    ax4.set_xlabel('-log10(p-value)')
    ax4.set_title('Top 10 Most Significant Patterns')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved to: {output_file}")


def create_effect_size_heatmap(insights, output_file='effect_size_heatmap.png'):
    """Create a heatmap showing effect sizes for different conditions."""
    
    # Get top significant patterns
    sig_patterns = [(k, v) for k, v in insights.items() if v.get('significant', False)]
    sig_patterns.sort(key=lambda x: x[1]['difference'], reverse=True)
    
    # Take top 20 patterns
    top_patterns = sig_patterns[:20]
    
    if not top_patterns:
        print("No significant patterns to visualize")
        return
    
    # Extract cluster distributions
    n_clusters = len(top_patterns[0][1]['stationary_cond'])
    
    # Create matrix of deviations from global
    deviations = []
    pattern_labels = []
    
    for pattern, data in top_patterns:
        cond_dist = np.array(data['stationary_cond'])
        global_dist = np.array(data['global_stationary'])
        deviation = cond_dist - global_dist
        deviations.append(deviation)
        pattern_labels.append(f"{pattern} (n={data['n_criminals']})")
    
    deviations = np.array(deviations)
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(deviations, 
                xticklabels=[f"C{i}" for i in range(n_clusters)],
                yticklabels=pattern_labels,
                cmap='RdBu_r',
                center=0,
                vmin=-0.2,
                vmax=0.2,
                cbar_kws={'label': 'Deviation from Global Distribution'})
    
    plt.xlabel('Criminal Archetype Cluster')
    plt.ylabel('Conditional Pattern')
    plt.title('Deviations from Global Distribution for Top Significant Patterns')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to: {output_file}")


def create_category_analysis(insights, output_file='category_analysis.png'):
    """Analyze patterns by category type."""
    
    # Categorize patterns
    categories = {
        'Demographics': ['Sex', 'Race', 'Age', 'Birth_order', 'Number_of_siblings'],
        'Family Background': ['Parent_s_marital_status', 'Living_with', 'Mother_s_occupation', 
                             'Father_s_occupation', 'Did_serial_killer_spend_time_in_an_orphanage_',
                             'Did_serial_killer_spend_time_in_a_foster_home_'],
        'Education': ['Highest_grade_in_school', 'Highest_degree', 'Grades_in_school', 
                      'Problems_in_school_', 'Teased_while_in_school_'],
        'Criminal Behavior': ['Method_of_killing', 'Type_of_killer', 'Type_of_serial_killer',
                             'Number_of_victims', 'Gender_of_victims', 'Victim_abducted_or_killed_at_contact_'],
        'Psychology': ['Bed_wetting', 'Been_to_a_psychologist_prior_to_killing_', 
                       'Psychologically_abused_', 'Sexually_abused', 'Physical_defect_'],
        'Substance Use': ['Abused_drugs', 'Abused_alcohol'],
        'Legal Outcomes': ['Sentence', 'Confessed', 'Plead_NGRI', 'Executed', 'Committed_suicide']
    }
    
    # Calculate statistics per category type
    category_type_stats = {}
    
    for cat_type, cat_list in categories.items():
        total = 0
        significant = 0
        
        for key, value in insights.items():
            heading = key.split('=')[0]
            if heading in cat_list:
                total += 1
                if value.get('significant', False):
                    significant += 1
        
        if total > 0:
            category_type_stats[cat_type] = {
                'total': total,
                'significant': significant,
                'rate': significant / total
            }
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart of significance rates
    cat_types = list(category_type_stats.keys())
    rates = [category_type_stats[ct]['rate'] for ct in cat_types]
    
    ax1.bar(cat_types, rates, color='skyblue', edgecolor='navy')
    ax1.set_ylabel('Proportion Significant')
    ax1.set_title('Significance Rate by Category Type')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add counts on bars
    for i, ct in enumerate(cat_types):
        stats = category_type_stats[ct]
        ax1.text(i, rates[i] + 0.02, f"{stats['significant']}/{stats['total']}", 
                ha='center', fontsize=10)
    
    # Pie chart of distribution
    sizes = [stats['significant'] for stats in category_type_stats.values()]
    ax2.pie(sizes, labels=cat_types, autopct='%1.0f%%', startangle=90)
    ax2.set_title('Distribution of Significant Patterns by Category Type')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Category analysis saved to: {output_file}")


def main():
    """Create all visualizations."""
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = 'test_run_fixed'
    
    # Load insights
    insights_file = f'{results_dir}/conditional_insights.json'
    with open(insights_file, 'r') as f:
        insights = json.load(f)
    
    # Create visualizations
    create_pattern_summary_plot(insights, f'{results_dir}/conditional_patterns_summary.png')
    create_effect_size_heatmap(insights, f'{results_dir}/effect_size_heatmap.png')
    create_category_analysis(insights, f'{results_dir}/category_analysis.png')
    
    print(f"\nAll visualizations saved to {results_dir}/")


if __name__ == "__main__":
    main()