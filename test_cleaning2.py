#!/usr/bin/env python3

import pandas as pd
from data_loading import load_all_type2_data
from data_cleaning import clean_type2_data

# Load without cleaning
print("Loading Type2 data WITHOUT cleaning...")
df_uncleaned = load_all_type2_data('type2csvs', clean_data=False)

# Check headings that contain "abused"
print("\nHeadings containing 'abused' BEFORE cleaning:")
abused_headings = df_uncleaned[df_uncleaned['Heading'].str.contains('abused', case=False, na=False)]['Heading'].unique()
for h in sorted(abused_headings):
    count = (df_uncleaned['Heading'] == h).sum()
    print(f"  '{h}': {count} rows")

# Clean
df_cleaned = clean_type2_data(df_uncleaned.copy())

print("\nHeadings containing 'abused' AFTER cleaning:")
abused_headings_clean = df_cleaned[df_cleaned['Heading'].str.contains('abused', case=False, na=False)]['Heading'].unique()
for h in sorted(abused_headings_clean):
    count = (df_cleaned['Heading'] == h).sum()
    values = df_cleaned[df_cleaned['Heading'] == h]['Value'].value_counts()
    print(f"\n  '{h}': {count} rows")
    print(f"  Values: {dict(values)}")

# Check what happened to specific headings
print("\n\nChecking heading transformations:")
print("'Abused drugs?' -> ", end="")
if 'Abused drugs' in df_cleaned['Heading'].values:
    print("'Abused drugs'")
elif 'Abused drugs?' in df_cleaned['Heading'].values:
    print("'Abused drugs?' (unchanged)")
else:
    print("NOT FOUND")

# Check all unique headings after cleaning that contain "Abused"
print("\nAll headings with 'Abused' in cleaned data:")
for h in sorted(df_cleaned['Heading'].unique()):
    if 'bused' in h:
        print(f"  '{h}'")