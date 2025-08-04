#!/usr/bin/env python3
"""
preview_cleaning.py

Preview the effects of data cleaning on Type2 data.
"""

import os
import pandas as pd
from collections import defaultdict
from data_loading import load_all_type2_data
from data_cleaning import standardize_missing_values, standardize_yes_no


def analyze_cleaning_effects(type2_dir):
    """Analyze what values will be consolidated by cleaning."""
    
    # Load without cleaning
    print("Loading Type2 data without cleaning...")
    df_original = load_all_type2_data(type2_dir, clean_data=False)
    
    # Load with cleaning
    print("\nLoading Type2 data with cleaning...")
    df_cleaned = load_all_type2_data(type2_dir, clean_data=True)
    
    print(f"\nOriginal shape: {df_original.shape}")
    print(f"Cleaned shape: {df_cleaned.shape}")
    print(f"Rows removed: {len(df_original) - len(df_cleaned)}")
    
    # Analyze by heading
    consolidations = defaultdict(lambda: {'original': set(), 'cleaned': set()})
    
    for heading in sorted(df_original['Heading'].unique()):
        orig_values = set(df_original[df_original['Heading'] == heading]['Value'].astype(str))
        
        # Check what these would be cleaned to
        cleaned_values = set()
        for val in orig_values:
            # Apply the same cleaning logic
            cleaned = standardize_missing_values(val)
            if cleaned != "Unknown" and any(h in heading for h in [
                'Physically abused?', 'Sexually abused?', 'Animal torture', 
                'Fire setting', 'Bed wetting', 'Abused alcohol?', 'Abused drugs?'
            ]):
                cleaned = standardize_yes_no(val)
            cleaned_values.add(cleaned)
        
        if len(orig_values) > len(cleaned_values) and len(cleaned_values) > 1:  # Exclude if everything becomes Unknown
            consolidations[heading]['original'] = orig_values
            consolidations[heading]['cleaned'] = cleaned_values
    
    # Print consolidations
    print("\n" + "="*80)
    print("VALUES THAT WILL BE CONSOLIDATED")
    print("="*80)
    
    for heading, values in sorted(consolidations.items()):
        if len(values['original']) > len(values['cleaned']):
            print(f"\n{heading}:")
            print(f"  From {len(values['original'])} unique values to {len(values['cleaned'])}")
            
            # Group original values by their cleaned value
            mapping = defaultdict(list)
            for orig in values['original']:
                cleaned = standardize_missing_values(orig)
                if cleaned != "Unknown" and any(h in heading for h in [
                    'Physically abused?', 'Sexually abused?', 'Animal torture', 
                    'Fire setting', 'Bed wetting', 'Abused alcohol?', 'Abused drugs?'
                ]):
                    cleaned = standardize_yes_no(orig)
                mapping[cleaned].append(orig)
            
            for cleaned, originals in sorted(mapping.items()):
                if len(originals) > 1:
                    print(f"    '{cleaned}' <- {originals}")
    
    # Check unique values per heading
    print("\n" + "="*80)
    print("UNIQUE VALUES PER HEADING (AFTER CLEANING)")
    print("="*80)
    
    problem_headings = []
    for heading in sorted(df_cleaned['Heading'].unique()):
        values = df_cleaned[df_cleaned['Heading'] == heading]['Value'].unique()
        if len(values) > 10:  # Flag headings with many unique values
            problem_headings.append((heading, len(values)))
        
    for heading, count in sorted(problem_headings, key=lambda x: x[1], reverse=True):
        print(f"\n{heading}: {count} unique values")
        values = df_cleaned[df_cleaned['Heading'] == heading]['Value'].value_counts()
        print("  Top 10 values:")
        for val, cnt in values.head(10).items():
            print(f"    {val}: {cnt}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preview data cleaning effects")
    parser.add_argument("--type2_dir", default="type2csvs", help="Directory with Type2 CSV files")
    args = parser.parse_args()
    
    analyze_cleaning_effects(args.type2_dir)