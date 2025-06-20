import os
import csv
import re
import pandas as pd


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


def load_all_type2_data(directory: str) -> pd.DataFrame:
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
        return pd.concat(type2_dfs, ignore_index=True)
    raise FileNotFoundError("No Type2 CSV files found in the provided directory.")


def load_type2_data(path: str) -> pd.DataFrame:
    """Load Type2 data from a directory of CSVs or a single CSV file."""
    if os.path.isdir(path):
        return load_all_type2_data(path)
    if os.path.isfile(path):
        return pd.read_csv(path, encoding="utf-8", quotechar='"', skipinitialspace=True)
    raise FileNotFoundError(f"Provided Type2 path not found: {path}")
