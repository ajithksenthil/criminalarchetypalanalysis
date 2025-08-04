#!/usr/bin/env python3
"""
helpers.py

Utility functions and helpers for the criminal archetypal analysis system.
"""

import numpy as np
import json
import re
from typing import List, Dict, Any, Optional

def numpy_to_python_default(obj):
    """
    Helper for JSON dumping of NumPy types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Type not serializable: {type(obj)}")

def safe_filename(text: str, max_length: int = 100) -> str:
    """
    Create a safe filename from text by removing/replacing problematic characters.
    
    Args:
        text: Input text
        max_length: Maximum filename length
        
    Returns:
        Safe filename string
    """
    # Replace problematic characters with underscores
    safe_text = re.sub(r'\W+', '_', str(text))
    
    # Truncate if too long
    if len(safe_text) > max_length:
        safe_text = safe_text[:max_length - 4] + "..."
    
    return safe_text

def save_json(data: Dict[str, Any], filepath: str, indent: int = 4) -> None:
    """
    Save data to JSON file with proper numpy handling.
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=numpy_to_python_default)

def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def parse_age(age_str: str) -> float:
    """
    Parse age string to float, handling ranges and invalid values.
    
    Args:
        age_str: Age string (e.g., "25", "20-25", "unknown")
        
    Returns:
        Parsed age as float, or 9999 for invalid values
    """
    try:
        if "-" in age_str:
            return float(age_str.split("-")[0])
        return float(age_str)
    except (ValueError, AttributeError):
        return 9999.0

def extract_numeric_value(text: str, default: float = 0.0) -> float:
    """
    Extract the first numeric value from a text string.
    
    Args:
        text: Input text
        default: Default value if no number found
        
    Returns:
        Extracted numeric value
    """
    if text is None:
        return default
    
    match = re.search(r"(\d+)", str(text))
    return float(match.group(1)) if match else default

def normalize_value(value: Any) -> str:
    """
    Normalize a value to a consistent string representation.
    
    Args:
        value: Input value
        
    Returns:
        Normalized string value
    """
    if value is None:
        return "Unknown"
    
    str_val = str(value).strip()
    if str_val.lower() in ['none', 'nan', '', 'null']:
        return "Unknown"
    
    return str_val

def group_by_condition(criminal_ids: List[str], condition_map: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Group criminal IDs by condition values.
    
    Args:
        criminal_ids: List of criminal IDs
        condition_map: Mapping from criminal ID to condition value
        
    Returns:
        Dictionary mapping condition values to lists of criminal IDs
    """
    groups = {}
    for crim_id in criminal_ids:
        value = condition_map.get(crim_id, "Unknown")
        groups.setdefault(value, []).append(crim_id)
    return groups

def filter_small_groups(groups: Dict[str, List[str]], min_size: int = 5) -> Dict[str, List[str]]:
    """
    Filter out groups that are too small for analysis.
    
    Args:
        groups: Dictionary of groups
        min_size: Minimum group size
        
    Returns:
        Filtered groups dictionary
    """
    return {k: v for k, v in groups.items() if len(v) >= min_size}

def compute_group_statistics(groups: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    """
    Compute basic statistics for groups.
    
    Args:
        groups: Dictionary of groups
        
    Returns:
        Statistics for each group
    """
    stats = {}
    total_size = sum(len(group) for group in groups.values())
    
    for name, group in groups.items():
        stats[name] = {
            'size': len(group),
            'percentage': len(group) / total_size * 100 if total_size > 0 else 0,
            'members': group
        }
    
    return stats

class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.last_percent = -1
    
    def update(self, increment: int = 1) -> None:
        """Update progress and print if significant change."""
        self.current += increment
        percent = int(self.current / self.total * 100) if self.total > 0 else 0
        
        if percent != self.last_percent and percent % 10 == 0:
            print(f"[INFO] {self.description}: {percent}% ({self.current}/{self.total})")
            self.last_percent = percent
    
    def finish(self) -> None:
        """Mark as complete."""
        print(f"[INFO] {self.description}: Complete ({self.total}/{self.total})")

def validate_file_path(filepath: str, must_exist: bool = True) -> bool:
    """
    Validate a file path.
    
    Args:
        filepath: Path to validate
        must_exist: Whether the file must already exist
        
    Returns:
        True if valid, False otherwise
    """
    import os
    
    if not filepath:
        return False
    
    if must_exist:
        return os.path.exists(filepath)
    else:
        # Check if parent directory exists
        parent_dir = os.path.dirname(filepath)
        return os.path.exists(parent_dir) if parent_dir else True

def create_output_structure(base_dir: str) -> Dict[str, str]:
    """
    Create standard output directory structure.
    
    Args:
        base_dir: Base output directory
        
    Returns:
        Dictionary mapping subdirectory names to paths
    """
    import os
    
    subdirs = {
        'base': base_dir,
        'clustering': os.path.join(base_dir, 'clustering'),
        'markov': os.path.join(base_dir, 'markov'),
        'visualization': os.path.join(base_dir, 'visualization'),
        'analysis': os.path.join(base_dir, 'analysis'),
        'data': os.path.join(base_dir, 'data')
    }
    
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    return subdirs
