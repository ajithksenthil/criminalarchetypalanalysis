#!/usr/bin/env python3

import pandas as pd
from data_loading import load_all_type2_data
from data_cleaning import standardize_missing_values, standardize_yes_no, clean_type2_data

# Load without cleaning first
print("Loading Type2 data WITHOUT cleaning...")
df_uncleaned = load_all_type2_data('type2csvs', clean_data=False)

# Check original values
print('\nBEFORE CLEANING:')
print('Abused drugs? unique values:')
abused_drugs = df_uncleaned[df_uncleaned['Heading'] == 'Abused drugs?']['Value'].value_counts()
print(abused_drugs.head(10))
print(f'Total unique values: {len(abused_drugs)}')

print('\n\nAbused alcohol? unique values:')
abused_alcohol = df_uncleaned[df_uncleaned['Heading'] == 'Abused alcohol?']['Value'].value_counts()
print(abused_alcohol.head(10))
print(f'Total unique values: {len(abused_alcohol)}')

# Test the mapping
print('\n\nMAPPING TEST:')
test_values = ['No', 'None', 'MISSING', 'Unknown', '?', 'nan', 'YES', 'yes', 'No ', ' No', 'no']
for val in test_values:
    # First standardize missing
    cleaned = standardize_missing_values(str(val))
    # Then apply yes/no if not Unknown
    if cleaned != 'Unknown':
        cleaned = standardize_yes_no(str(val))
    print(f"'{val}' -> '{cleaned}'")

# Now load WITH cleaning
print("\n\nLoading Type2 data WITH cleaning...")
df_cleaned = clean_type2_data(df_uncleaned.copy())

print('\nAFTER CLEANING:')
print('Abused drugs? unique values:')
abused_drugs_clean = df_cleaned[df_cleaned['Heading'] == 'Abused drugs?']['Value'].value_counts()
print(abused_drugs_clean)

print('\nAbused alcohol? unique values:')
abused_alcohol_clean = df_cleaned[df_cleaned['Heading'] == 'Abused alcohol?']['Value'].value_counts()
print(abused_alcohol_clean)

# Check what happened to "None" values
print('\n\nChecking specific None/Unknown handling:')
mask = (df_uncleaned['Heading'] == 'Abused drugs?') & (df_uncleaned['Value'].astype(str).str.lower() == 'none')
print(f"Rows with 'Abused drugs?' = 'None' in original: {mask.sum()}")

# Check if they're being removed
print('\nTotal rows before/after:')
print(f"Abused drugs? rows before: {(df_uncleaned['Heading'] == 'Abused drugs?').sum()}")
print(f"Abused drugs? rows after: {(df_cleaned['Heading'] == 'Abused drugs?').sum()}")