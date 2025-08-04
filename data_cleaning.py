#!/usr/bin/env python3
"""
data_cleaning.py

Data cleaning and standardization for Type 2 data to handle formatting inconsistencies.
"""

import re
import pandas as pd
import numpy as np


def standardize_missing_values(value):
    """
    Standardize various representations of missing data.
    """
    if pd.isna(value) or value is None:
        return "Unknown"
    
    value_str = str(value).strip()
    
    # Check for various missing value representations
    missing_patterns = [
        r'^missing$',
        r'^none$',
        r'^nan$',
        r'^n/a$',
        r'^na$',
        r'^unknown$',
        r'^not found$',
        r'^not available$',
        r'^not reported$',
        r'^\?+$',
        r'^-+$',
        r'^_+$',
        r'^\s*$'  # Empty or whitespace only
    ]
    
    for pattern in missing_patterns:
        if re.match(pattern, value_str, re.IGNORECASE):
            return "Unknown"
    
    return value_str


def standardize_yes_no(value):
    """
    Standardize yes/no values.
    """
    value_std = standardize_missing_values(value)
    if value_std == "Unknown":
        return value_std
    
    value_lower = value_std.lower()
    
    # Yes patterns
    yes_patterns = [r'^yes', r'^y$', r'^true', r'^1$', r'^1\.0$']
    for pattern in yes_patterns:
        if re.match(pattern, value_lower):
            return "Yes"
    
    # No patterns
    no_patterns = [r'^no', r'^n$', r'^false', r'^0$', r'^0\.0$']
    for pattern in no_patterns:
        if re.match(pattern, value_lower):
            return "No"
    
    # Special cases for complex answers
    if "no" in value_lower and "yes" not in value_lower:
        return "No"
    if "yes" in value_lower and "no" not in value_lower:
        return "Yes"
    
    return value_std


def standardize_numeric(value):
    """
    Standardize numeric values.
    """
    value_std = standardize_missing_values(value)
    if value_std == "Unknown":
        return value_std
    
    # Try to extract first number from string
    match = re.search(r'(\d+(?:\.\d+)?)', value_std)
    if match:
        return match.group(1)
    
    return value_std


def standardize_location(value):
    """
    Standardize location values.
    """
    value_std = standardize_missing_values(value)
    if value_std == "Unknown":
        return value_std
    
    # Remove extra whitespace and standardize capitalization
    value_std = ' '.join(value_std.split())
    
    # Common replacements
    replacements = {
        'USA': 'United States',
        'US': 'United States',
        'U.S.': 'United States',
        'U.S.A.': 'United States',
        'UK': 'United Kingdom',
        'U.K.': 'United Kingdom'
    }
    
    for old, new in replacements.items():
        if value_std.upper() == old:
            return new
    
    # Title case for proper nouns
    return value_std.title()


def standardize_category(value, category_map=None):
    """
    Standardize categorical values using a predefined mapping.
    """
    value_std = standardize_missing_values(value)
    if value_std == "Unknown":
        return value_std
    
    if category_map:
        value_lower = value_std.lower()
        for key, standard_value in category_map.items():
            if key.lower() == value_lower:
                return standard_value
    
    return value_std


def clean_type2_data(df):
    """
    Clean and standardize Type 2 data based on heading types.
    """
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Define heading-specific cleaning rules
    yes_no_headings = [
        'Physically abused?', 'Sexually abused?', 'Emotionally abused?',
        'Animal torture', 'Fire setting', 'Bed wetting',
        'Abused alcohol?', 'Abused drugs?',
        'Been to a psychologist?', 'Ate part of the body?',
        'Drank victim\'s blood?', 'Bound the victims?',
        'Tortured the victims?', 'Killed prior to series?',
        'Applied for job as a cop?', 'Attempted suicide?',
        'Fired from jobs?', 'Killed enemy during service?',
        'Head injury?', 'Dumped body in lake, river, etc?',
        'Burned body', 'Cut up and disposed of',
        'Did killer have a partner?', 'Did killer plead NGRI?',
        'Did serial killer confess?', 'Killer committed suicide?',
        'Killer executed?', 'Killer killed in prison?',
        'Lives with his children', 'Lives with her children',
        'Committed previous crimes?'
    ]
    
    numeric_headings = [
        'Number of victims', 'IQ', 'Birth order',
        'Killer age at start of series', 'Killer age at end of series',
        'Age of first kill', 'Age when first had intercourse',
        'Age of first sexual experience', 'Highest grade in school'
    ]
    
    location_headings = [
        'Country where killing occurred', 'States where killing occurred',
        'Cities where killing occurred', 'Counties where killing occurred'
    ]
    
    # Category mappings
    sex_map = {
        'male': 'Male',
        'm': 'Male',
        'female': 'Female',
        'f': 'Female',
        'female/ male': 'Both',
        'male/female': 'Both'
    }
    
    race_map = {
        'white': 'White',
        'caucasian': 'White',
        'black': 'Black',
        'african american': 'Black',
        'african-american': 'Black',
        'hispanic': 'Hispanic',
        'latino': 'Hispanic',
        'asian': 'Asian',
        'native american': 'Native American',
        'mixed': 'Mixed',
        'other': 'Other'
    }
    
    employment_map = {
        'employed': 'Employed',
        'unemployed': 'Unemployed',
        'part-time': 'Part-time',
        'part time': 'Part-time',
        'self-employed': 'Self-employed',
        'self employed': 'Self-employed'
    }
    
    # Clean values based on heading
    for idx, row in df_clean.iterrows():
        heading = str(row['Heading']).strip()
        value = row['Value']
        
        # First, always check for missing values
        cleaned_value = standardize_missing_values(value)
        
        # Then apply specific cleaning based on heading type
        if cleaned_value != "Unknown":
            if any(h in heading for h in yes_no_headings):
                cleaned_value = standardize_yes_no(value)
            elif any(h in heading for h in numeric_headings):
                cleaned_value = standardize_numeric(value)
            elif any(h in heading for h in location_headings):
                cleaned_value = standardize_location(value)
            elif heading == 'Sex':
                cleaned_value = standardize_category(value, sex_map)
            elif heading == 'Race':
                cleaned_value = standardize_category(value, race_map)
            elif 'Employment' in heading:
                cleaned_value = standardize_category(value, employment_map)
            else:
                # For other headings, just clean whitespace and standardize case
                cleaned_value = ' '.join(str(value).split())
        
        df_clean.at[idx, 'Value'] = cleaned_value
    
    # Standardize heading names too
    heading_replacements = {
        'Physically abused?': 'Physically abused',
        'Sexually abused?': 'Sexually abused',
        'Emotionally abused?': 'Emotionally abused',
        'Abused alcohol?': 'Abused alcohol',
        'Abused drugs?': 'Abused drugs',
        'Been to a psychologist?': 'Been to psychologist',
        'Been to a psychologist prior to killing?': 'Been to psychologist',
        'Applied for job as a cop?': 'Applied for cop job',
        'Attempted suicide?': 'Attempted suicide',
        'Fired from jobs?': 'Fired from jobs',
        'Killed enemy during service?': 'Killed in service',
        'Head injury?': 'Head injury',
        'Did killer have a partner?': 'Had partner',
        'Did killer plead NGRI?': 'Plead NGRI',
        'Did serial killer confess?': 'Confessed',
        'Killer committed suicide?': 'Committed suicide',
        'Killer executed?': 'Executed',
        'Killer killed in prison?': 'Killed in prison'
    }
    
    for old_heading, new_heading in heading_replacements.items():
        df_clean.loc[df_clean['Heading'] == old_heading, 'Heading'] = new_heading
    
    # Don't remove Unknown values - they are valid responses
    # Only remove if we want to focus on known values
    # df_clean = df_clean[df_clean['Value'] != 'Unknown'].copy()
    
    return df_clean


def merge_similar_categories(df, heading, similar_groups):
    """
    Merge similar categories within a specific heading.
    
    Args:
        df: DataFrame with cleaned data
        heading: The heading to process
        similar_groups: List of lists, where each inner list contains values to merge
    
    Example:
        similar_groups = [
            ['Unknown', 'Not reported', 'Missing'],
            ['Very high', 'High', 'Above average']
        ]
    """
    df_copy = df.copy()
    
    for group in similar_groups:
        if len(group) < 2:
            continue
        
        # Use the first value as the standard
        standard_value = group[0]
        
        # Replace all similar values
        mask = (df_copy['Heading'] == heading) & (df_copy['Value'].isin(group))
        df_copy.loc[mask, 'Value'] = standard_value
    
    return df_copy


def get_data_quality_report(df_original, df_cleaned):
    """
    Generate a report on data cleaning results.
    """
    report = {
        'original_rows': len(df_original),
        'cleaned_rows': len(df_cleaned),
        'rows_removed': len(df_original) - len(df_cleaned),
        'unique_headings_original': df_original['Heading'].nunique(),
        'unique_headings_cleaned': df_cleaned['Heading'].nunique(),
        'changes_by_heading': {}
    }
    
    # Analyze changes by heading
    for heading in df_original['Heading'].unique():
        orig_values = df_original[df_original['Heading'] == heading]['Value'].value_counts()
        clean_values = df_cleaned[df_cleaned['Heading'] == heading]['Value'].value_counts()
        
        if len(orig_values) != len(clean_values):
            report['changes_by_heading'][heading] = {
                'original_unique_values': len(orig_values),
                'cleaned_unique_values': len(clean_values),
                'reduction': len(orig_values) - len(clean_values)
            }
    
    return report


if __name__ == "__main__":
    # Test the cleaning functions
    import argparse
    
    parser = argparse.ArgumentParser(description="Test data cleaning on Type2 data")
    parser.add_argument("--type2_file", required=True, help="Path to a Type2 CSV file")
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.type2_file)
    print(f"Original data shape: {df.shape}")
    print(f"Original unique values in 'Value' column: {df['Value'].nunique()}")
    
    # Clean data
    df_cleaned = clean_type2_data(df)
    print(f"\nCleaned data shape: {df_cleaned.shape}")
    print(f"Cleaned unique values in 'Value' column: {df_cleaned['Value'].nunique()}")
    
    # Show some examples of cleaning
    print("\nExample cleanings:")
    for heading in df['Heading'].unique()[:5]:
        orig = df[df['Heading'] == heading]['Value'].unique()
        clean = df_cleaned[df_cleaned['Heading'] == heading]['Value'].unique()
        if len(orig) != len(clean):
            print(f"\n{heading}:")
            print(f"  Original: {orig}")
            print(f"  Cleaned: {clean}")