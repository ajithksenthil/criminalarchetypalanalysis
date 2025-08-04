#!/usr/bin/env python3
"""
analyze_conditional_patterns.py

Analyze conditional patterns from the criminal archetypal analysis in detail.
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict
import pandas as pd


def load_conditional_insights(results_dir):
    """Load conditional insights from results directory."""
    insights_file = os.path.join(results_dir, 'conditional_insights.json')
    
    if not os.path.exists(insights_file):
        print(f"Error: No conditional insights found at {insights_file}")
        return None
    
    with open(insights_file, 'r') as f:
        return json.load(f)


def analyze_insights(insights):
    """Analyze the conditional insights in detail."""
    print("="*70)
    print("CONDITIONAL PATTERNS ANALYSIS")
    print("="*70)
    
    # Basic statistics
    total_insights = len(insights)
    significant_insights = sum(1 for v in insights.values() if v.get('significant', False))
    
    print(f"\nTotal conditional patterns found: {total_insights}")
    print(f"Statistically significant patterns: {significant_insights}")
    print(f"Percentage significant: {significant_insights/total_insights*100:.1f}%")
    
    # Group by heading type
    patterns_by_type = defaultdict(list)
    for key, value in insights.items():
        heading = key.split('=')[0]
        patterns_by_type[heading].append((key, value))
    
    print(f"\nPatterns by category: {len(patterns_by_type)} categories")
    
    # Find most significant patterns
    significant_patterns = [(k, v) for k, v in insights.items() if v.get('significant', False)]
    significant_patterns.sort(key=lambda x: x[1]['statistics']['ks_pvalue'])
    
    print("\n" + "-"*70)
    print("TOP 10 MOST SIGNIFICANT PATTERNS (by p-value):")
    print("-"*70)
    
    for i, (pattern, data) in enumerate(significant_patterns[:10], 1):
        p_value = data['statistics']['ks_pvalue']
        n_criminals = data['n_criminals']
        l1_diff = data['difference']
        
        print(f"\n{i}. {pattern}")
        print(f"   - p-value: {p_value:.6f}")
        print(f"   - L1 difference: {l1_diff:.3f}")
        print(f"   - N criminals: {n_criminals}")
        
        # Show top deviations from global distribution
        if 'stationary_cond' in data and 'global_stationary' in data:
            cond_dist = np.array(data['stationary_cond'])
            global_dist = np.array(data['global_stationary'])
            deviations = cond_dist - global_dist
            
            # Find clusters with largest positive and negative deviations
            sorted_idx = np.argsort(np.abs(deviations))[::-1]
            
            print("   - Top deviations from global pattern:")
            for j in range(min(3, len(sorted_idx))):
                idx = sorted_idx[j]
                dev = deviations[idx]
                if abs(dev) > 0.01:  # Only show meaningful deviations
                    direction = "more" if dev > 0 else "less"
                    print(f"     * Cluster {idx}: {dev:+.3f} ({direction} frequent)")
    
    # Analyze by category
    print("\n" + "-"*70)
    print("SIGNIFICANT PATTERNS BY CATEGORY:")
    print("-"*70)
    
    for heading, patterns in patterns_by_type.items():
        sig_count = sum(1 for _, v in patterns if v.get('significant', False))
        if sig_count > 0:
            print(f"\n{heading}: {sig_count}/{len(patterns)} significant")
            
            # Show most significant in this category
            sig_patterns = [(k, v) for k, v in patterns if v.get('significant', False)]
            sig_patterns.sort(key=lambda x: x[1]['statistics']['ks_pvalue'])
            
            for pattern, data in sig_patterns[:3]:
                value = pattern.split('=')[1]
                p_value = data['statistics']['ks_pvalue']
                n = data['n_criminals']
                print(f"  - {value}: p={p_value:.4f}, n={n}")
    
    # Look for interesting patterns
    print("\n" + "-"*70)
    print("INTERESTING FINDINGS:")
    print("-"*70)
    
    # Find patterns with largest effect sizes
    effect_sizes = [(k, v['difference'], v) for k, v in insights.items()]
    effect_sizes.sort(key=lambda x: x[1], reverse=True)
    
    print("\nLargest effect sizes (L1 distance from global):")
    for pattern, diff, data in effect_sizes[:5]:
        n = data['n_criminals']
        sig = " (SIGNIFICANT)" if data.get('significant', False) else ""
        print(f"  - {pattern}: L1={diff:.3f}, n={n}{sig}")
    
    # Find categories with most significant patterns
    category_significance = {}
    for heading, patterns in patterns_by_type.items():
        sig_count = sum(1 for _, v in patterns if v.get('significant', False))
        total_count = len(patterns)
        if total_count > 0:
            category_significance[heading] = sig_count / total_count
    
    sorted_categories = sorted(category_significance.items(), key=lambda x: x[1], reverse=True)
    
    print("\nCategories with highest proportion of significant patterns:")
    for category, prop in sorted_categories[:5]:
        if prop > 0:
            print(f"  - {category}: {prop*100:.1f}% significant")
    
    return {
        'total': total_insights,
        'significant': significant_insights,
        'categories': len(patterns_by_type),
        'top_patterns': significant_patterns[:10] if significant_patterns else []
    }


def generate_detailed_report(insights, output_file='conditional_patterns_report.txt'):
    """Generate a detailed report of conditional patterns."""
    with open(output_file, 'w') as f:
        f.write("DETAILED CONDITIONAL PATTERNS REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Summary statistics
        total = len(insights)
        significant = sum(1 for v in insights.values() if v.get('significant', False))
        
        f.write(f"Total patterns analyzed: {total}\n")
        f.write(f"Significant patterns: {significant}\n")
        f.write(f"Significance rate: {significant/total*100:.1f}%\n\n")
        
        # All significant patterns
        f.write("ALL SIGNIFICANT PATTERNS:\n")
        f.write("-"*70 + "\n\n")
        
        sig_patterns = [(k, v) for k, v in insights.items() if v.get('significant', False)]
        sig_patterns.sort(key=lambda x: x[1]['statistics']['ks_pvalue'])
        
        for i, (pattern, data) in enumerate(sig_patterns, 1):
            f.write(f"{i}. {pattern}\n")
            f.write(f"   p-value: {data['statistics']['ks_pvalue']:.6f}\n")
            f.write(f"   L1 distance: {data['difference']:.3f}\n")
            f.write(f"   N criminals: {data['n_criminals']}\n")
            f.write(f"   KS statistic: {data['statistics']['ks_statistic']:.3f}\n")
            f.write("\n")
    
    print(f"\nDetailed report saved to: {output_file}")


def main():
    """Main analysis function."""
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Try to find most recent results
        results_dirs = [d for d in os.listdir('.') if d.startswith(('output_', 'results_', 'test_'))]
        if results_dirs:
            results_dir = sorted(results_dirs)[-1]
            print(f"Using results directory: {results_dir}")
        else:
            print("Error: No results directory found. Please specify one.")
            sys.exit(1)
    
    # Load insights
    insights = load_conditional_insights(results_dir)
    if insights is None:
        sys.exit(1)
    
    # Analyze
    summary = analyze_insights(insights)
    
    # Generate detailed report
    output_file = os.path.join(results_dir, 'conditional_patterns_detailed.txt')
    generate_detailed_report(insights, output_file)
    
    # Create visualization data
    viz_data = {
        'summary': summary,
        'insights': insights
    }
    
    viz_file = os.path.join(results_dir, 'conditional_patterns_viz_data.json')
    with open(viz_file, 'w') as f:
        json.dump(viz_data, f, indent=2)
    
    print(f"\nVisualization data saved to: {viz_file}")
    
    # Key takeaway
    print("\n" + "="*70)
    print("KEY TAKEAWAY:")
    print("="*70)
    if summary['significant'] > 0:
        print(f"Found {summary['significant']} significant conditional patterns out of {summary['total']} tested.")
        print("This indicates that certain criminal characteristics (Type 2 data) are associated")
        print("with different behavioral transition patterns in their life events.")
    else:
        print("No significant conditional patterns found.")
        print("This suggests criminal behavior patterns are similar across different characteristics.")


if __name__ == "__main__":
    main()