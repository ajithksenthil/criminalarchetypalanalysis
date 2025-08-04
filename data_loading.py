import os
import csv
import re
import pandas as pd
from data_matching import find_matching_pairs, extract_criminal_id_from_filename
from data_cleaning import clean_type2_data


def load_all_criminals_type1(directory: str) -> dict:
    """Load life-event CSV files for each criminal."""
    criminals = {}
    for filename in os.listdir(directory):
        if filename.startswith("Type1_") and filename.endswith(".csv"):
            match = re.search(r"Type1_(.+)\.csv", filename)
            criminal_id = match.group(1) if match else filename
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = []
                for row in reader:
                    event_text = (row.get("Life Event", "") or row.get("Life Event ", "")).strip()
                    date_str = row.get("Date", "")
                    age_str = row.get("Age", "")
                    rows.append({"Date": date_str, "Age": age_str, "LifeEvent": event_text})
            def parse_age(age_str):
                try:
                    if "-" in age_str:
                        return float(age_str.split("-")[0])
                    return float(age_str)
                except Exception:
                    return 9999
            rows.sort(key=lambda x: parse_age(x["Age"]))
            events = [r["LifeEvent"] for r in rows if r["LifeEvent"]]
            criminals[criminal_id] = {"events": events, "rows": rows}
    return criminals


def load_all_type2_data(directory: str, clean_data: bool = True) -> pd.DataFrame:
    """Load all Type2 CSV files in a directory and concatenate them."""
    type2_dfs = []
    for filename in os.listdir(directory):
        if filename.startswith("Type2_") and filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(file_path, encoding="utf-8", quotechar='"', skipinitialspace=True)
            except Exception as e:
                print(f"[WARNING] Could not read {filename}: {e}")
                continue
            criminal_id = filename[len("Type2_"):-len(".csv")]
            df["CriminalID"] = criminal_id
            type2_dfs.append(df)
    if type2_dfs:
        combined_df = pd.concat(type2_dfs, ignore_index=True)
        if clean_data:
            print(f"[INFO] Cleaning Type2 data (before: {len(combined_df)} rows)...")
            combined_df = clean_type2_data(combined_df)
            print(f"[INFO] Cleaning complete (after: {len(combined_df)} rows)")
        return combined_df
    raise FileNotFoundError("No Type2 CSV files found in the provided directory.")


def load_type2_data(path: str, clean_data: bool = True) -> pd.DataFrame:
    """Load Type2 data from a directory of CSVs or a single CSV file."""
    if os.path.isdir(path):
        return load_all_type2_data(path, clean_data=clean_data)
    if os.path.isfile(path):
        df = pd.read_csv(path, encoding="utf-8", quotechar='"', skipinitialspace=True)
        if clean_data:
            print(f"[INFO] Cleaning Type2 data (before: {len(df)} rows)...")
            df = clean_type2_data(df)
            print(f"[INFO] Cleaning complete (after: {len(df)} rows)")
        return df
    raise FileNotFoundError(f"Provided Type2 path not found: {path}")


def load_matched_criminal_data(type1_dir: str, type2_dir: str) -> tuple:
    """
    Load only criminals that have both Type1 and Type2 data.
    
    Args:
        type1_dir: Directory containing Type1 CSV files
        type2_dir: Directory containing Type2 CSV files
    
    Returns:
        Tuple of (criminals_type1_dict, type2_dataframe) containing only matched data
    """
    # Find matching pairs
    matching_pairs = find_matching_pairs(type1_dir, type2_dir)
    
    if not matching_pairs:
        raise ValueError("No matching Type1 and Type2 files found!")
    
    # Load Type1 data for matched criminals only
    criminals_type1 = {}
    for criminal_id, file_paths in matching_pairs.items():
        file_path = file_paths['type1_file']
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                event_text = (row.get("Life Event", "") or row.get("Life Event ", "")).strip()
                date_str = row.get("Date", "")
                age_str = row.get("Age", "")
                rows.append({"Date": date_str, "Age": age_str, "LifeEvent": event_text})
        
        # Sort by age
        def parse_age(age_str):
            try:
                if "-" in age_str:
                    return float(age_str.split("-")[0])
                return float(age_str)
            except Exception:
                return 9999
        
        rows.sort(key=lambda x: parse_age(x["Age"]))
        events = [r["LifeEvent"] for r in rows if r["LifeEvent"]]
        
        # Use the original filename pattern as the criminal ID for consistency
        original_filename = file_paths['type1_original']
        original_id = extract_criminal_id_from_filename(original_filename, 'Type1')
        
        # Extract a more readable ID from the filename
        readable_id = re.sub(r'type1_', '', original_filename.lower())
        readable_id = re.sub(r'\.csv$', '', readable_id)
        readable_id = re.sub(r'_-_\d{4}.*$', '', readable_id)
        readable_id = re.sub(r'__\d{4}.*$', '', readable_id)
        
        criminals_type1[readable_id] = {"events": events, "rows": rows}
    
    # Load Type2 data for matched criminals only
    type2_dfs = []
    for criminal_id, file_paths in matching_pairs.items():
        file_path = file_paths['type2_file']
        try:
            df = pd.read_csv(file_path, encoding='utf-8', quotechar='"', skipinitialspace=True)
            
            # Extract readable ID to match Type1
            original_filename = file_paths['type1_original']
            readable_id = re.sub(r'type1_', '', original_filename.lower())
            readable_id = re.sub(r'\.csv$', '', readable_id)
            readable_id = re.sub(r'_-_\d{4}.*$', '', readable_id)
            readable_id = re.sub(r'__\d{4}.*$', '', readable_id)
            
            df["CriminalID"] = readable_id
            type2_dfs.append(df)
        except Exception as e:
            print(f"[WARNING] Could not read {file_paths['type2_original']}: {e}")
    
    if not type2_dfs:
        raise ValueError("No Type2 data could be loaded!")
    
    type2_df = pd.concat(type2_dfs, ignore_index=True)
    
    # Clean the Type2 data
    print(f"[INFO] Cleaning Type2 data (before: {len(type2_df)} rows)...")
    type2_df = clean_type2_data(type2_df)
    print(f"[INFO] Cleaning complete (after: {len(type2_df)} rows)")
    
    print(f"[INFO] Loaded matched data for {len(criminals_type1)} criminals")
    print(f"[INFO] Type1 events: {sum(len(c['events']) for c in criminals_type1.values())}")
    print(f"[INFO] Type2 records: {len(type2_df)}")
    
    return criminals_type1, type2_df
