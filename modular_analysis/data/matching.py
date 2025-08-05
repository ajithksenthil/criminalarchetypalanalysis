#!/usr/bin/env python3
"""
matching.py

Data matching functionality for finding criminals with both Type 1 and Type 2 data.
"""

import os
import re
from typing import List, Tuple, Dict, Any
import pandas as pd

from .loaders import Type1DataLoader, Type2DataLoader

def find_matching_pairs(type1_dir: str, type2_dir: str) -> List[Tuple[str, str]]:
    """
    Find matching Type1 and Type2 files based on criminal IDs.
    
    Args:
        type1_dir: Directory containing Type1_*.csv files
        type2_dir: Directory containing Type2_*.csv files
        
    Returns:
        List of tuples (type1_file, type2_file) for matching criminals
    """
    if not os.path.exists(type1_dir) or not os.path.exists(type2_dir):
        return []
    
    # Get all Type1 files and extract criminal IDs
    type1_files = {}
    for filename in os.listdir(type1_dir):
        if filename.startswith("Type1_") and filename.endswith(".csv"):
            match = re.search(r"Type1_(.+)\.csv", filename)
            if match:
                criminal_id = match.group(1)
                type1_files[criminal_id] = filename
    
    # Get all Type2 files and extract criminal IDs
    type2_files = {}
    for filename in os.listdir(type2_dir):
        if filename.startswith("Type2_") and filename.endswith(".csv"):
            match = re.search(r"Type2_(.+)\.csv", filename)
            if match:
                criminal_id = match.group(1)
                # Remove '_clean' suffix if present
                if criminal_id.endswith('_clean'):
                    criminal_id = criminal_id[:-6]  # Remove '_clean'
                type2_files[criminal_id] = filename
    
    # Find matching pairs
    matching_pairs = []
    for criminal_id in type1_files:
        if criminal_id in type2_files:
            type1_path = os.path.join(type1_dir, type1_files[criminal_id])
            type2_path = os.path.join(type2_dir, type2_files[criminal_id])
            matching_pairs.append((type1_path, type2_path))
    
    print(f"[INFO] Found {len(matching_pairs)} criminals with both Type1 and Type2 data")
    print(f"[INFO] Type1 files: {len(type1_files)}, Type2 files: {len(type2_files)}")
    
    return matching_pairs

def load_matched_criminal_data(type1_dir: str, type2_dir: str) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Load only criminals that have both Type1 and Type2 data.
    
    Args:
        type1_dir: Directory containing Type1_*.csv files
        type2_dir: Directory containing Type2_*.csv files
        
    Returns:
        Tuple of (criminals_data, type2_df) for matched criminals only
    """
    matching_pairs = find_matching_pairs(type1_dir, type2_dir)
    
    if not matching_pairs:
        print("[WARNING] No matching criminals found")
        return {}, pd.DataFrame()
    
    # Load Type1 data for matching criminals
    type1_loader = Type1DataLoader(type1_dir)
    all_type1_data = type1_loader.load_all_criminals()
    
    # Load Type2 data for matching criminals
    type2_loader = Type2DataLoader(type2_dir)
    type2_df = type2_loader.load_data()
    
    # Extract criminal IDs from matching pairs
    matching_criminal_ids = set()
    for type1_path, type2_path in matching_pairs:
        # Extract criminal ID from Type1 filename
        type1_filename = os.path.basename(type1_path)
        match = re.search(r"Type1_(.+)\.csv", type1_filename)
        if match:
            matching_criminal_ids.add(match.group(1))
    
    # Filter Type1 data to only include matching criminals
    matched_criminals_data = {
        crim_id: data for crim_id, data in all_type1_data.items()
        if crim_id in matching_criminal_ids
    }
    
    # Filter Type2 data to only include matching criminals
    # Need to account for "_clean" suffix in Type2 criminal IDs
    def matches_criminal_id(type2_id, type1_ids):
        """Check if Type2 ID matches any Type1 ID (accounting for _clean suffix)."""
        # Remove _clean suffix if present
        clean_type2_id = type2_id.replace('_clean', '') if type2_id.endswith('_clean') else type2_id
        return clean_type2_id in type1_ids

    matched_type2_df = type2_df[
        type2_df["CriminalID"].astype(str).apply(lambda x: matches_criminal_id(x, matching_criminal_ids))
    ].copy()
    
    print(f"[INFO] Loaded matched data for {len(matched_criminals_data)} criminals")
    print(f"[INFO] Type2 data shape after filtering: {matched_type2_df.shape}")
    
    return matched_criminals_data, matched_type2_df

def get_matching_statistics(type1_dir: str, type2_dir: str) -> Dict[str, Any]:
    """
    Get statistics about data matching between Type1 and Type2.
    
    Args:
        type1_dir: Directory containing Type1_*.csv files
        type2_dir: Directory containing Type2_*.csv files
        
    Returns:
        Dictionary with matching statistics
    """
    if not os.path.exists(type1_dir) or not os.path.exists(type2_dir):
        return {"error": "One or both directories do not exist"}
    
    # Count Type1 files
    type1_files = [f for f in os.listdir(type1_dir) 
                   if f.startswith("Type1_") and f.endswith(".csv")]
    
    # Count Type2 files
    type2_files = [f for f in os.listdir(type2_dir) 
                   if f.startswith("Type2_") and f.endswith(".csv")]
    
    # Find matches
    matching_pairs = find_matching_pairs(type1_dir, type2_dir)
    
    # Extract criminal IDs
    type1_ids = set()
    for filename in type1_files:
        match = re.search(r"Type1_(.+)\.csv", filename)
        if match:
            type1_ids.add(match.group(1))
    
    type2_ids = set()
    for filename in type2_files:
        match = re.search(r"Type2_(.+)\.csv", filename)
        if match:
            type2_ids.add(match.group(1))
    
    # Calculate statistics
    matched_ids = type1_ids.intersection(type2_ids)
    type1_only = type1_ids - type2_ids
    type2_only = type2_ids - type1_ids
    
    stats = {
        "total_type1_files": len(type1_files),
        "total_type2_files": len(type2_files),
        "total_type1_criminals": len(type1_ids),
        "total_type2_criminals": len(type2_ids),
        "matched_criminals": len(matched_ids),
        "type1_only_criminals": len(type1_only),
        "type2_only_criminals": len(type2_only),
        "match_rate_type1": len(matched_ids) / len(type1_ids) if type1_ids else 0,
        "match_rate_type2": len(matched_ids) / len(type2_ids) if type2_ids else 0,
        "matching_pairs": len(matching_pairs)
    }
    
    return stats

def validate_matched_data(criminals_data: Dict[str, Dict[str, Any]], 
                         type2_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that matched data is consistent.
    
    Args:
        criminals_data: Type1 criminal data
        type2_df: Type2 DataFrame
        
    Returns:
        Validation results
    """
    type1_ids = set(criminals_data.keys())
    type2_ids = set(type2_df["CriminalID"].astype(str).unique())
    
    # Check for consistency
    common_ids = type1_ids.intersection(type2_ids)
    type1_missing = type2_ids - type1_ids
    type2_missing = type1_ids - type2_ids
    
    # Count events per criminal
    events_per_criminal = {crim_id: len(data["events"]) 
                          for crim_id, data in criminals_data.items()}
    
    # Count Type2 records per criminal
    type2_counts = type2_df["CriminalID"].value_counts().to_dict()
    type2_records_per_criminal = {str(k): v for k, v in type2_counts.items()}
    
    validation = {
        "is_consistent": len(type1_missing) == 0 and len(type2_missing) == 0,
        "common_criminals": len(common_ids),
        "type1_missing_from_type2": list(type1_missing),
        "type2_missing_from_type1": list(type2_missing),
        "total_events": sum(events_per_criminal.values()),
        "avg_events_per_criminal": sum(events_per_criminal.values()) / len(events_per_criminal) if events_per_criminal else 0,
        "total_type2_records": len(type2_df),
        "avg_type2_records_per_criminal": len(type2_df) / len(common_ids) if common_ids else 0,
        "events_distribution": events_per_criminal,
        "type2_distribution": type2_records_per_criminal
    }
    
    return validation
