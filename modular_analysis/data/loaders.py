#!/usr/bin/env python3
"""
loaders.py

Data loading functionality for Type 1 and Type 2 criminal data.
"""

import os
import csv
import re
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

# Define helper functions locally to avoid import issues
def parse_age(age_str: str) -> float:
    """Parse age string to float, handling ranges and invalid values."""
    try:
        if "-" in age_str:
            return float(age_str.split("-")[0])
        return float(age_str)
    except (ValueError, AttributeError):
        return 9999.0

def normalize_value(value) -> str:
    """Normalize a value to a consistent string representation."""
    if value is None:
        return "Unknown"
    str_val = str(value).strip()
    if str_val.lower() in ['none', 'nan', '', 'null']:
        return "Unknown"
    return str_val

def extract_numeric_value(text: str, default: float = 0.0) -> float:
    """Extract the first numeric value from a text string."""
    if text is None:
        return default

    match = re.search(r"(\d+)", str(text))
    return float(match.group(1)) if match else default

# Constants
TYPE1_FILE_PATTERN = "Type1_*.csv"
TYPE2_FILE_PATTERN = "Type2_*.csv"

class Type1DataLoader:
    """Loader for Type 1 (life event) data."""
    
    def __init__(self, directory: str):
        self.directory = directory
    
    def load_all_criminals(self) -> Dict[str, Dict[str, Any]]:
        """
        Load life events from CSV files in the directory.
        
        Returns:
            Dictionary mapping criminal IDs to their data
        """
        criminals = {}
        
        if not os.path.exists(self.directory):
            raise FileNotFoundError(f"Type 1 directory not found: {self.directory}")
        
        for filename in os.listdir(self.directory):
            if filename.startswith("Type1_") and filename.endswith(".csv"):
                criminal_id = self._extract_criminal_id(filename)
                file_path = os.path.join(self.directory, filename)
                
                try:
                    criminal_data = self._load_criminal_file(file_path)
                    if criminal_data['events']:  # Only add if has events
                        criminals[criminal_id] = criminal_data
                except Exception as e:
                    print(f"[WARNING] Failed to load {filename}: {e}")
        
        return criminals
    
    def _extract_criminal_id(self, filename: str) -> str:
        """Extract criminal ID from filename."""
        match = re.search(r"Type1_(.+)\.csv", filename)
        return match.group(1) if match else filename
    
    def _load_criminal_file(self, file_path: str) -> Dict[str, Any]:
        """Load a single criminal's Type 1 data file."""
        rows = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                event_text = (row.get("Life Event", "") or row.get("Life Event ", "")).strip()
                date_str = row.get("Date", "")
                age_str = row.get("Age", "")
                
                if event_text:  # Only include non-empty events
                    rows.append({
                        "Date": date_str,
                        "Age": age_str,
                        "LifeEvent": event_text
                    })
        
        # Sort by age if possible
        rows.sort(key=lambda x: parse_age(x["Age"]))
        
        # Extract just the events in chronological order
        events = [r["LifeEvent"] for r in rows]
        
        return {
            "events": events,
            "rows": rows
        }

class Type2DataLoader:
    """Loader for Type 2 (structured) data."""
    
    def __init__(self, path: str):
        self.path = path
    
    def load_data(self) -> pd.DataFrame:
        """
        Load Type 2 data from either a single file or directory.
        
        Returns:
            DataFrame with Type 2 data
        """
        if os.path.isdir(self.path):
            return self._load_from_directory()
        elif os.path.isfile(self.path):
            return self._load_from_file()
        else:
            raise FileNotFoundError(f"Type 2 path not found: {self.path}")
    
    def _load_from_directory(self) -> pd.DataFrame:
        """Load Type 2 data from multiple CSV files in a directory."""
        type2_dfs = []
        
        for filename in os.listdir(self.path):
            if filename.startswith("Type2_") and filename.endswith(".csv"):
                file_path = os.path.join(self.path, filename)
                criminal_id = filename[len("Type2_"):-len(".csv")]
                
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', quotechar='"', skipinitialspace=True)
                    df["CriminalID"] = criminal_id
                    type2_dfs.append(df)
                except Exception as e:
                    print(f"[WARNING] Could not read {filename}: {e}")
        
        if not type2_dfs:
            raise FileNotFoundError("No Type2 CSV files found in the directory.")
        
        return pd.concat(type2_dfs, ignore_index=True)
    
    def _load_from_file(self) -> pd.DataFrame:
        """Load Type 2 data from a single CSV file."""
        return pd.read_csv(self.path, encoding='utf-8', quotechar='"', skipinitialspace=True)

class MatchedDataLoader:
    """Loader for matched Type 1 and Type 2 data."""
    
    def __init__(self, type1_dir: str, type2_dir: str):
        self.type1_dir = type1_dir
        self.type2_dir = type2_dir
    
    def find_matching_pairs(self) -> List[Tuple[str, str]]:
        """Find matching Type 1 and Type 2 files."""
        from data_matching import find_matching_pairs
        return find_matching_pairs(self.type1_dir, self.type2_dir)
    
    def load_matched_data(self) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
        """Load only criminals with both Type 1 and Type 2 data."""
        from data_loading import load_matched_criminal_data
        return load_matched_criminal_data(self.type1_dir, self.type2_dir)

class DataProcessor:
    """Process and aggregate loaded data."""
    
    @staticmethod
    def aggregate_events(criminals_data: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        Aggregate all events and track their criminal IDs.
        
        Args:
            criminals_data: Dictionary of criminal data
            
        Returns:
            Tuple of (all_events, event_criminal_ids)
        """
        global_events = []
        event_criminal_ids = []
        
        for crim_id, data in criminals_data.items():
            for event in data["events"]:
                global_events.append(event)
                event_criminal_ids.append(crim_id)
        
        return global_events, event_criminal_ids
    
    @staticmethod
    def build_criminal_sequences(event_criminal_ids: List[str], labels: List[int]) -> Dict[str, List[int]]:
        """
        Build sequences of cluster labels for each criminal.
        
        Args:
            event_criminal_ids: List mapping events to criminal IDs
            labels: Cluster labels for each event
            
        Returns:
            Dictionary mapping criminal IDs to their cluster sequences
        """
        criminal_sequences = {}
        
        for idx, crim_id in enumerate(event_criminal_ids):
            if idx < len(labels):
                criminal_sequences.setdefault(crim_id, []).append(labels[idx])
        
        return criminal_sequences

class Type2DataProcessor:
    """Process Type 2 data for analysis."""
    
    @staticmethod
    def get_condition_map(type2_df: pd.DataFrame, heading: str) -> Dict[str, str]:
        """
        Build a mapping from criminal ID to value for a specific heading.
        
        Args:
            type2_df: Type 2 DataFrame with CriminalID, Heading, Value columns
            heading: The heading to extract values for
            
        Returns:
            Dictionary mapping criminal ID to value for the heading
        """
        condition_map = {}
        
        for _, row in type2_df.iterrows():
            if row["Heading"].strip().lower() == heading.strip().lower():
                crim_id = str(row["CriminalID"])
                val = normalize_value(row["Value"])
                condition_map[crim_id] = val
        
        return condition_map
    
    @staticmethod
    def get_value_for_heading(crim_id: str, type2_df: pd.DataFrame, heading_query: str) -> Optional[str]:
        """
        Get value for a specific criminal and heading.
        
        Args:
            crim_id: Criminal ID
            type2_df: Type 2 DataFrame
            heading_query: Heading to search for
            
        Returns:
            Value for the heading, or None if not found
        """
        rows = type2_df[type2_df["CriminalID"] == crim_id]
        if rows.empty:
            return None
        
        matched = rows[rows["Heading"].str.strip().str.lower() == heading_query.strip().lower()]
        if not matched.empty:
            return normalize_value(matched.iloc[0]["Value"])
        
        return None
    
    @staticmethod
    def extract_feature_vector(crim_id: str, type2_df: pd.DataFrame, features: List[str]) -> Optional[List[float]]:
        """
        Extract a feature vector for a criminal based on specified Type 2 fields.
        
        Args:
            crim_id: Criminal ID
            type2_df: Type 2 DataFrame
            features: List of feature names to extract
            
        Returns:
            Feature vector or None if criminal not found
        """
        vector = []
        
        for feature in features:
            value = Type2DataProcessor.get_value_for_heading(crim_id, type2_df, feature)
            
            if feature.lower() in ["physically abused?", "physically abused"]:
                vector.append(1.0 if value and value.lower().startswith("yes") else 0.0)
            elif feature.lower() == "sex":
                vector.append(1.0 if value and value.strip().lower() == "male" else 0.0)
            elif "victim" in feature.lower():
                vector.append(extract_numeric_value(value))
            else:
                # For other features, try to extract numeric value or use binary encoding
                vector.append(extract_numeric_value(value))
        
        return vector if vector else None

    @staticmethod
    def get_extended_type2_vector(crim_id: str, type2_df) -> Optional[List[float]]:
        """
        Extract an extended feature vector for multi-modal analysis.

        Args:
            crim_id: Criminal ID
            type2_df: Type 2 DataFrame

        Returns:
            Extended feature vector or None if criminal not found
        """
        features = [
            "Physically abused?",
            "Sex",
            "Number of victims",
            "Education level",
            "Employment status",
            "Substance abuse",
            "Mental health issues"
        ]

        vector = []

        for feature in features:
            value = Type2DataProcessor.get_value_for_heading(crim_id, type2_df, feature)

            if feature.lower() in ["physically abused?", "physically abused"]:
                vector.append(1.0 if value and value.lower().startswith("yes") else 0.0)
            elif feature.lower() == "sex":
                vector.append(1.0 if value and value.strip().lower() == "male" else 0.0)
            elif "victim" in feature.lower():
                vector.append(extract_numeric_value(value))
            elif feature.lower() == "education level":
                # Encode education as ordinal: None=0, High School=1, College=2, Graduate=3
                if value:
                    val_lower = value.lower()
                    if "graduate" in val_lower or "phd" in val_lower or "master" in val_lower:
                        vector.append(3.0)
                    elif "college" in val_lower or "university" in val_lower or "bachelor" in val_lower:
                        vector.append(2.0)
                    elif "high school" in val_lower or "secondary" in val_lower:
                        vector.append(1.0)
                    else:
                        vector.append(0.0)
                else:
                    vector.append(0.0)
            elif feature.lower() in ["employment status", "employment"]:
                # Binary: employed=1, unemployed=0
                if value:
                    val_lower = value.lower()
                    vector.append(1.0 if "employ" in val_lower and "un" not in val_lower else 0.0)
                else:
                    vector.append(0.0)
            elif feature.lower() in ["substance abuse", "drug", "alcohol"]:
                # Binary: yes=1, no=0
                vector.append(1.0 if value and value.lower().startswith("yes") else 0.0)
            elif feature.lower() in ["mental health", "mental health issues"]:
                # Binary: yes=1, no=0
                vector.append(1.0 if value and value.lower().startswith("yes") else 0.0)
            else:
                # For other features, try to extract numeric value or use binary encoding
                vector.append(extract_numeric_value(value))

        return vector if vector else None

# Backward compatibility aliases
def load_all_criminals_type1(directory: str) -> Dict[str, Dict[str, Any]]:
    """Backward compatibility alias for Type1DataLoader.load_all_criminals()."""
    loader = Type1DataLoader(directory)
    return loader.load_all_criminals()

def load_type2_data(path: str):
    """Backward compatibility alias for Type2DataLoader.load_data()."""
    loader = Type2DataLoader(path)
    return loader.load_data()

def get_extended_type2_vector(crim_id: str, type2_df):
    """Backward compatibility alias."""
    return Type2DataProcessor.get_extended_type2_vector(crim_id, type2_df)
