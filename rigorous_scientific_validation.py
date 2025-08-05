#!/usr/bin/env python3
"""
rigorous_scientific_validation.py

Rigorous scientific validation of the conditional matrix analysis results.
Checks for statistical validity, multiple testing corrections, effect sizes, and methodological soundness.
"""

import json
import numpy as np
from scipy import stats
import pandas as pd

def validate_scientific_rigor():
    """Perform rigorous scientific validation of the conditional analysis."""
    
    print("="*80)
    print("RIGOROUS SCIENTIFIC VALIDATION OF CONDITIONAL ANALYSIS")
    print("="*80)
    
    # Load results
    with open('output_proper_fixed/analysis_results.json', 'r') as f:
        results = json.load(f)

    with open('output_proper_fixed/analysis/conditional_insights.json', 'r') as f:
        insights = json.load(f)
    
    print(f"Dataset: {results['data_summary']['n_criminals']} criminals, {results['data_summary']['n_events']} events")
    print(f"Number of claimed significant effects: {len(insights)}")
    print()
    
    # 1. MULTIPLE TESTING CORRECTION
    print("1. MULTIPLE TESTING CORRECTION ANALYSIS")
    print("-" * 50)
    
    # Extract all p-values
    p_values = []
    effect_names = []
    sample_sizes = []
    effect_strengths = []
    
    for effect_name, data in insights.items():
        p_val = data['statistics']['ks_pvalue']
        n_criminals = data['n_criminals']
        l1_diff = data['l1_difference']
        
        p_values.append(p_val)
        effect_names.append(effect_name)
        sample_sizes.append(n_criminals)
        effect_strengths.append(l1_diff)
    
    p_values = np.array(p_values)
    
    # Bonferroni correction
    alpha = 0.05
    bonferroni_threshold = alpha / len(p_values)
    bonferroni_significant = np.sum(p_values < bonferroni_threshold)
    
    # Benjamini-Hochberg (FDR) correction
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    fdr_threshold = 0.05
    fdr_significant = 0
    for i, p in enumerate(sorted_p):
        threshold = (i + 1) / len(sorted_p) * fdr_threshold
        if p <= threshold:
            fdr_significant = i + 1
        else:
            break
    
    print(f"Original significant effects: {len(insights)}")
    print(f"Bonferroni correction (α = {alpha}):")
    print(f"  Threshold: {bonferroni_threshold:.2e}")
    print(f"  Significant after correction: {bonferroni_significant}")
    print(f"FDR correction (q = {fdr_threshold}):")
    print(f"  Significant after correction: {fdr_significant}")
    
    # 2. EFFECT SIZE VALIDATION
    print(f"\n2. EFFECT SIZE VALIDATION")
    print("-" * 50)
    
    effect_strengths = np.array(effect_strengths)
    
    print(f"L1 difference statistics:")
    print(f"  Mean: {np.mean(effect_strengths):.3f}")
    print(f"  Median: {np.median(effect_strengths):.3f}")
    print(f"  Std: {np.std(effect_strengths):.3f}")
    print(f"  Range: {np.min(effect_strengths):.3f} - {np.max(effect_strengths):.3f}")
    
    # Cohen's conventions for effect sizes (adapted for L1 distance)
    small_effect = 0.2
    medium_effect = 0.5
    large_effect = 0.8
    
    small_effects = np.sum(effect_strengths >= small_effect)
    medium_effects = np.sum(effect_strengths >= medium_effect)
    large_effects = np.sum(effect_strengths >= large_effect)
    
    print(f"\nEffect size classification:")
    print(f"  Small effects (L1 ≥ {small_effect}): {small_effects}")
    print(f"  Medium effects (L1 ≥ {medium_effect}): {medium_effects}")
    print(f"  Large effects (L1 ≥ {large_effect}): {large_effects}")
    
    # 3. SAMPLE SIZE ADEQUACY
    print(f"\n3. SAMPLE SIZE ADEQUACY")
    print("-" * 50)
    
    sample_sizes = np.array(sample_sizes)
    
    print(f"Sample size statistics:")
    print(f"  Mean: {np.mean(sample_sizes):.1f}")
    print(f"  Median: {np.median(sample_sizes):.1f}")
    print(f"  Range: {np.min(sample_sizes)} - {np.max(sample_sizes)}")
    
    # Check for adequate sample sizes (rule of thumb: n ≥ 5 per group)
    adequate_samples = np.sum(sample_sizes >= 5)
    small_samples = np.sum(sample_sizes < 5)
    
    print(f"\nSample size adequacy:")
    print(f"  Adequate samples (n ≥ 5): {adequate_samples}")
    print(f"  Small samples (n < 5): {small_samples}")
    
    if small_samples > 0:
        print(f"  ⚠️  WARNING: {small_samples} effects have small sample sizes")
    
    # 4. STATISTICAL ASSUMPTIONS CHECK
    print(f"\n4. STATISTICAL ASSUMPTIONS CHECK")
    print("-" * 50)
    
    # Check for identical p-values (suggests potential issues)
    unique_p_values = len(np.unique(p_values))
    print(f"Unique p-values: {unique_p_values} out of {len(p_values)} tests")
    
    if unique_p_values < len(p_values) * 0.8:
        print("⚠️  WARNING: Many identical p-values detected")
        
        # Find most common p-values
        p_value_counts = {}
        for p in p_values:
            p_str = f"{p:.2e}"
            p_value_counts[p_str] = p_value_counts.get(p_str, 0) + 1
        
        print("Most common p-values:")
        for p_val, count in sorted(p_value_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {p_val}: {count} occurrences")
    
    # 5. CLUSTERING VALIDATION
    print(f"\n5. CLUSTERING VALIDATION")
    print("-" * 50)
    
    silhouette = results['clustering']['silhouette']
    n_clusters = results['optimization']['optimal_k']
    
    print(f"Clustering quality:")
    print(f"  Silhouette score: {silhouette:.4f}")
    print(f"  Number of clusters: {n_clusters}")
    
    # Silhouette score interpretation
    if silhouette > 0.7:
        quality = "EXCELLENT"
    elif silhouette > 0.5:
        quality = "GOOD"
    elif silhouette > 0.25:
        quality = "WEAK but acceptable"
    else:
        quality = "POOR"
    
    print(f"  Quality assessment: {quality}")
    
    # 6. METHODOLOGICAL CONCERNS
    print(f"\n6. METHODOLOGICAL CONCERNS")
    print("-" * 50)
    
    concerns = []
    
    # Check for overfitting
    n_features = len(insights)
    n_samples = results['data_summary']['n_criminals']
    feature_to_sample_ratio = n_features / n_samples
    
    if feature_to_sample_ratio > 0.1:
        concerns.append(f"High feature-to-sample ratio: {feature_to_sample_ratio:.2f}")
    
    # Check for multiple testing burden
    if len(insights) > 100:
        concerns.append(f"Large number of tests: {len(insights)} (increases Type I error risk)")
    
    # Check for small effect sizes
    if np.mean(effect_strengths) < 0.5:
        concerns.append(f"Small average effect size: {np.mean(effect_strengths):.3f}")
    
    # Check clustering quality
    if silhouette < 0.25:
        concerns.append(f"Poor clustering quality: silhouette = {silhouette:.3f}")
    
    if concerns:
        print("⚠️  METHODOLOGICAL CONCERNS IDENTIFIED:")
        for i, concern in enumerate(concerns, 1):
            print(f"  {i}. {concern}")
    else:
        print("✅ No major methodological concerns identified")
    
    # 7. SCIENTIFIC VALIDITY ASSESSMENT
    print(f"\n7. OVERALL SCIENTIFIC VALIDITY ASSESSMENT")
    print("=" * 50)
    
    validity_score = 0
    max_score = 6
    
    # Criterion 1: Multiple testing correction
    if bonferroni_significant > 0:
        validity_score += 1
        print("✅ Criterion 1: Effects survive multiple testing correction")
    else:
        print("❌ Criterion 1: No effects survive Bonferroni correction")
    
    # Criterion 2: Adequate effect sizes
    if large_effects > 0:
        validity_score += 1
        print("✅ Criterion 2: Large effect sizes present")
    elif medium_effects > 0:
        validity_score += 0.5
        print("⚠️  Criterion 2: Medium effect sizes present")
    else:
        print("❌ Criterion 2: No large effect sizes")
    
    # Criterion 3: Adequate sample sizes
    if small_samples == 0:
        validity_score += 1
        print("✅ Criterion 3: All groups have adequate sample sizes")
    elif small_samples < len(insights) * 0.2:
        validity_score += 0.5
        print("⚠️  Criterion 3: Most groups have adequate sample sizes")
    else:
        print("❌ Criterion 3: Many groups have inadequate sample sizes")
    
    # Criterion 4: Statistical diversity
    if unique_p_values > len(p_values) * 0.8:
        validity_score += 1
        print("✅ Criterion 4: Diverse p-values suggest robust testing")
    else:
        print("❌ Criterion 4: Many identical p-values suggest potential issues")
    
    # Criterion 5: Clustering quality
    if silhouette > 0.25:
        validity_score += 1
        print("✅ Criterion 5: Acceptable clustering quality")
    else:
        print("❌ Criterion 5: Poor clustering quality")
    
    # Criterion 6: Methodological soundness
    if len(concerns) <= 1:
        validity_score += 1
        print("✅ Criterion 6: Methodologically sound")
    else:
        print("❌ Criterion 6: Multiple methodological concerns")
    
    print(f"\nOVERALL VALIDITY SCORE: {validity_score}/{max_score} ({validity_score/max_score*100:.1f}%)")
    
    if validity_score >= 5:
        verdict = "HIGH SCIENTIFIC VALIDITY"
    elif validity_score >= 3:
        verdict = "MODERATE SCIENTIFIC VALIDITY"
    else:
        verdict = "LOW SCIENTIFIC VALIDITY"
    
    print(f"SCIENTIFIC VERDICT: {verdict}")
    
    # 8. RECOMMENDATIONS
    print(f"\n8. RECOMMENDATIONS FOR IMPROVEMENT")
    print("-" * 50)
    
    recommendations = []
    
    if bonferroni_significant == 0:
        recommendations.append("Apply less conservative multiple testing correction (e.g., FDR)")
    
    if small_samples > 0:
        recommendations.append("Collect more data for small demographic groups")
    
    if silhouette < 0.5:
        recommendations.append("Improve clustering algorithm or feature engineering")
    
    if np.mean(effect_strengths) < 0.5:
        recommendations.append("Focus on larger, more meaningful effect sizes")
    
    if len(insights) > n_samples:
        recommendations.append("Reduce number of tested variables to avoid overfitting")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("  No major improvements needed - analysis appears robust")

if __name__ == "__main__":
    validate_scientific_rigor()
