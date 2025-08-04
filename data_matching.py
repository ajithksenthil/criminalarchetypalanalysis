#!/usr/bin/env python3
"""
data_matching.py

Utilities for matching Type1 and Type2 files by criminal name.
"""

import os
import re


def extract_criminal_id_from_filename(filename, file_type='Type1'):
    """
    Extract a normalized criminal ID from a filename.
    
    Args:
        filename: The CSV filename
        file_type: Either 'Type1' or 'Type2'
    
    Returns:
        A normalized criminal ID string, or None if extraction fails
    """
    # Remove the file extension
    name_part = filename.replace('.csv', '')
    
    # Remove the Type prefix
    if file_type == 'Type1' and name_part.startswith('Type1_'):
        name_part = name_part[6:]  # Remove 'Type1_'
    elif file_type == 'Type2' and name_part.startswith('Type2_'):
        name_part = name_part[6:]  # Remove 'Type2_'
    else:
        return None
    
    # For Type2, remove the '_clean' suffix (and handle '_clean_clean' cases)
    if file_type == 'Type2':
        name_part = re.sub(r'_clean(?:_clean)*$', '', name_part)
    
    # Normalize the name by converting to lowercase for matching
    # This handles cases like 'campbell_charles' vs 'Campbell_Charles'
    normalized = name_part.lower()
    
    return normalized


def find_matching_pairs(type1_dir, type2_dir):
    """
    Find Type1 and Type2 files that match by criminal name.
    
    Args:
        type1_dir: Directory containing Type1 CSV files
        type2_dir: Directory containing Type2 CSV files
    
    Returns:
        A dictionary mapping criminal IDs to their file paths:
        {
            'criminal_id': {
                'type1_file': 'full/path/to/Type1_file.csv',
                'type2_file': 'full/path/to/Type2_file.csv',
                'type1_original': 'Type1_file.csv',
                'type2_original': 'Type2_file.csv'
            }
        }
    """
    # Get all Type1 files
    type1_files = {}
    for filename in os.listdir(type1_dir):
        if filename.startswith('Type1_') and filename.endswith('.csv'):
            criminal_id = extract_criminal_id_from_filename(filename, 'Type1')
            if criminal_id:
                type1_files[criminal_id] = filename
    
    # Get all Type2 files
    type2_files = {}
    for filename in os.listdir(type2_dir):
        if filename.startswith('Type2_') and filename.endswith('.csv'):
            criminal_id = extract_criminal_id_from_filename(filename, 'Type2')
            if criminal_id:
                type2_files[criminal_id] = filename
    
    # Find matching pairs
    matching_pairs = {}
    for criminal_id in type1_files:
        if criminal_id in type2_files:
            matching_pairs[criminal_id] = {
                'type1_file': os.path.join(type1_dir, type1_files[criminal_id]),
                'type2_file': os.path.join(type2_dir, type2_files[criminal_id]),
                'type1_original': type1_files[criminal_id],
                'type2_original': type2_files[criminal_id]
            }
    
    # Report statistics
    print(f"[INFO] Data matching results:")
    print(f"  - Type1 files found: {len(type1_files)}")
    print(f"  - Type2 files found: {len(type2_files)}")
    print(f"  - Matching pairs: {len(matching_pairs)}")
    print(f"  - Type1 only: {len(type1_files) - len(matching_pairs)}")
    print(f"  - Type2 only: {len(type2_files) - len(matching_pairs)}")
    
    # Show examples of unmatched files
    type1_only = set(type1_files.keys()) - set(matching_pairs.keys())
    type2_only = set(type2_files.keys()) - set(matching_pairs.keys())
    
    if type1_only:
        print("\n[INFO] Examples of Type1 files without Type2 match:")
        for criminal_id in list(type1_only)[:5]:
            print(f"  - {type1_files[criminal_id]}")
    
    if type2_only:
        print("\n[INFO] Examples of Type2 files without Type1 match:")
        for criminal_id in list(type2_only)[:5]:
            print(f"  - {type2_files[criminal_id]}")
    
    return matching_pairs


def get_matched_criminal_id(original_id, matching_pairs):
    """
    Get the normalized criminal ID that can be used to look up matched data.
    
    Args:
        original_id: The original criminal ID from the filename
        matching_pairs: The dictionary of matching pairs
    
    Returns:
        The normalized criminal ID if found in matching pairs, otherwise None
    """
    # First try direct lookup
    normalized_id = original_id.lower()
    if normalized_id in matching_pairs:
        return normalized_id
    
    # Try without year/season suffixes
    # Remove patterns like '_2005', '_spring_2006', etc.
    cleaned_id = re.sub(r'_(?:spring|fall|summer|winter)?_?\d{4}(?:_(?:spring|fall|summer|winter))?', '', normalized_id)
    cleaned_id = re.sub(r'_-_\d{4}(?:_(?:spring|fall|summer|winter))?', '', cleaned_id)
    
    if cleaned_id in matching_pairs:
        return cleaned_id
    
    # Try matching by partial name (last name + first name only)
    parts = cleaned_id.split('_')
    if len(parts) >= 2:
        partial_id = '_'.join(parts[:2])
        if partial_id in matching_pairs:
            return partial_id
    
    return None


if __name__ == "__main__":
    # Test the matching functionality
    import argparse
    
    parser = argparse.ArgumentParser(description="Test data matching functionality")
    parser.add_argument("--type1_dir", required=True, help="Directory with Type1 CSV files")
    parser.add_argument("--type2_dir", required=True, help="Directory with Type2 CSV files")
    args = parser.parse_args()
    
    matching_pairs = find_matching_pairs(args.type1_dir, args.type2_dir)
    
    print(f"\n[INFO] Found {len(matching_pairs)} matching criminal records")
    print("\nFirst 5 matches:")
    for i, (criminal_id, files) in enumerate(matching_pairs.items()):
        if i >= 5:
            break
        print(f"\nCriminal ID: {criminal_id}")
        print(f"  Type1: {files['type1_original']}")
        print(f"  Type2: {files['type2_original']}")