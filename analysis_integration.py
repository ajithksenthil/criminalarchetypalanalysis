#!/usr/bin/env python3
"""
analysis_integration.py

A complete script that:
  1) Loads and processes Type 1 (life-event) data for multiple criminals.
  2) Embeds and clusters all life events globally (KMeans).
  3) Optionally calls an LLM to label clusters with an archetypal theme.
  4) Reconstructs per–criminal chronological theme sequences and builds a global Markov chain.
  5) Loads Type 2 (structured) data—either from a single CSV file or from a directory of Type2_*.csv files—and demonstrates an integrated logistic regression analysis.
  6) Optionally performs multi–modal clustering at the criminal level (combining Type 1 and Type 2 data).

Usage:
  python analysis_integration.py --type1_dir=/path/to/data_csv \
                                 --type2_csv=/path/to/data_csv \
                                 --output_dir=/path/to/output \
                                 [--n_clusters=5] [--no_llm] [--multi_modal]

Adjust file paths, column names, and other settings as necessary.
"""

import os
import sys
import csv
import json
import argparse
import warnings
import re
import random
import nltk
import openai
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from analysis import (
    evaluate_clustering,
    plot_tsne_embeddings,
    transition_entropy,
)
from prototype_network import train_prototypical_network

# Optional: For additional clustering methods
# from sklearn.cluster import AgglomerativeClustering, DBSCAN

warnings.filterwarnings("ignore", category=FutureWarning)


# --------------------------------------------------
# 1. GLOBAL SETTINGS
# --------------------------------------------------

def ensure_nltk_data():
    """Download required NLTK data if not present."""
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab/english/",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
    }
    for pkg, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


ensure_nltk_data()

# Reproducibility
random.seed(42)
np.random.seed(42)

# Set your OpenAI API key from the environment
client = None
if "OPENAI_API_KEY" in os.environ:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    client = openai.OpenAI()

# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def load_all_criminals_type1(directory: str) -> dict:
    """
    Loads life events from CSV files in the given directory.
    Assumes each file is named like 'Type1_<criminal_id>.csv'.
    Returns:
      criminals (dict): keys are criminal IDs and values are dicts with:
          "events": a list of life event texts (sorted chronologically if possible),
          "rows": the full row data.
    """
    criminals = {}
    for filename in os.listdir(directory):
        if filename.startswith("Type1_") and filename.endswith(".csv"):
            # Extract criminal id from filename (everything between "Type1_" and ".csv")
            match = re.search(r"Type1_(.+)\.csv", filename)
            if match:
                criminal_id = match.group(1)
            else:
                criminal_id = filename
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.DictReader(f)
                rows = []
                for row in csv_reader:
                    event_text = (row.get("Life Event", "") or row.get("Life Event ", "")).strip()
                    date_str = row.get("Date", "")
                    age_str = row.get("Age", "")
                    rows.append({
                        "Date": date_str,
                        "Age": age_str,
                        "LifeEvent": event_text
                    })
            # Naively sort rows by Age if possible.
            def parse_age(age_str):
                try:
                    if "-" in age_str:
                        return float(age_str.split("-")[0])
                    return float(age_str)
                except:
                    return 9999
            rows.sort(key=lambda x: parse_age(x["Age"]))
            events = [r["LifeEvent"] for r in rows if r["LifeEvent"]]
            criminals[criminal_id] = {"events": events, "rows": rows}
    return criminals


def preprocess_text(text: str) -> str:
    """
    Preprocess the text: lowercasing, remove digits/punctuation, tokenize,
    remove stopwords, and lemmatize.
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def generate_embeddings(sentences: list, model_name='all-MiniLM-L6-v2') -> np.ndarray:
    """
    Generate sentence embeddings using SentenceTransformer.
    """
    model = SentenceTransformer(model_name)
    return model.encode(sentences)


def generate_tfidf_embeddings(sentences: list, max_features: int = 500) -> np.ndarray:
    """Generate embeddings using a simple TF-IDF vectorizer."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectors = vectorizer.fit_transform(sentences)
    return vectors.toarray()


def kmeans_cluster(embeddings: np.ndarray, n_clusters=5, random_state=42):
    """
    Runs KMeans clustering on the embeddings and returns labels.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels, km


def find_representative_samples(sentences, embeddings, labels, n_reps=3):
    """
    For each cluster, find the n_reps most "central" samples based on cosine similarity.
    Returns a list of dictionaries with cluster_id, size, and representative samples.
    """
    clusters_dict = {}
    for idx, label in enumerate(labels):
        clusters_dict.setdefault(label, []).append(idx)

    results = []
    for label, indices in clusters_dict.items():
        cluster_embs = embeddings[indices]
        centroid = np.mean(cluster_embs, axis=0)
        sims = cosine_similarity([centroid], cluster_embs)[0]
        sorted_idx = np.argsort(sims)[::-1]
        rep_indices = [indices[i] for i in sorted_idx[:n_reps]]
        rep_texts = [sentences[i] for i in rep_indices]
        results.append({
            "cluster_id": label,
            "size": len(indices),
            "representative_samples": rep_texts
        })
    results.sort(key=lambda x: x["cluster_id"])
    return results


def analyze_cluster_with_llm(representative_samples: list) -> str:
    """
    Calls an LLM (e.g. GPT-3.5-turbo) to label the cluster with an archetypal theme.
    Requires openai.api_key to be set.
    """
    if not openai.api_key:
        return "No OpenAI API key found. (LLM analysis skipped)"
    prompt_template = (
        "You are an expert in criminal psychology and behavioral analysis.\n"
        "Given these representative life events of a serial killer, identify\n"
        "the archetypal pattern or theme they represent. Be concise and specific.\n\n"
        "Life events:\n{events}\n\nArchetypal theme:"
    )
    joined_events = "\n".join(representative_samples)
    prompt_text = prompt_template.format(events=joined_events)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # or "gpt-4" if available
            messages=[
                {"role": "system", "content": "You are an expert in criminal psychology and behavioral analysis."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=200,
            temperature=0.7
        )
        reply = response.choices[0].message['content'].strip()
        return reply
    except Exception as e:
        print(f"[ERROR in LLM] {e}")
        return "Unknown (LLM error)"


# --------------------------------------------------
# Markov Chain Utilities
# --------------------------------------------------
def plot_state_transition_diagram(transition_matrix: np.ndarray, out_path="state_transition.png"):
    """
    Plot a state transition diagram using networkx.
    """
    G = nx.DiGraph()
    num_states = transition_matrix.shape[0]
    for i in range(num_states):
        G.add_node(i)
    for i in range(num_states):
        for j in range(num_states):
            w = transition_matrix[i, j]
            if w > 0:
                G.add_edge(i, j, weight=w)
    pos = nx.circular_layout(G)
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {(i, j): f"{w:.2f}" for (i, j), w in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    plt.title("State Transition Diagram")
    plt.savefig(out_path)
    plt.clf()
    print(f"[INFO] State transition diagram saved to {out_path}")


def compute_stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the stationary distribution by finding the eigenvector of the transpose with eigenvalue=1.
    """
    eigvals, eigvecs = np.linalg.eig(transition_matrix.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    stat_dist = eigvecs[:, idx].real
    stat_dist /= stat_dist.sum()
    return stat_dist


def numpy_to_python_default(obj):
    """
    Helper for JSON dumping of NumPy types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Type not serializable: {type(obj)}")


# --------------------------------------------------
# Conditional Markov Chain Analysis
# --------------------------------------------------
def build_conditional_markov(selected_criminal_ids, criminal_sequences, n_clusters):
    """
    Builds a Markov transition matrix for a subset of criminals.
    Args:
        selected_criminal_ids (list): Criminal IDs for the subgroup.
        criminal_sequences (dict): Mapping from CriminalID to their sequence of cluster labels.
        n_clusters (int): Number of clusters.
    Returns:
        transition_matrix (np.ndarray): Row-normalized transition matrix.
    """
    matrix = np.zeros((n_clusters, n_clusters))
    for cid in selected_criminal_ids:
        seq = criminal_sequences.get(cid, [])
        if len(seq) < 2:
            continue
        for s1, s2 in zip(seq[:-1], seq[1:]):
            matrix[s1, s2] += 1
    # Normalize rows to probabilities:
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        matrix = np.divide(matrix, row_sums, where=row_sums != 0)
        for i in range(n_clusters):
            if row_sums[i] == 0:
                matrix[i] = 1.0 / n_clusters
    return matrix


def analyze_all_conditional_insights(type2_df, criminal_sequences, n_clusters, output_dir, global_stationary, diff_threshold=0.1):
    """
    For each unique heading in the Type 2 data, group criminals by their value,
    compute the conditional Markov chain (and its stationary distribution),
    and compare that distribution to the global stationary distribution using the L1 norm.
    
    If the difference exceeds diff_threshold, print and record an insight.
    
    Returns a dictionary of insights.
    """
    import re
    insights = {}
    # Get all unique headings from the Type2 data
    headings = type2_df["Heading"].unique()
    for heading in headings:
        print(f"[INFO] Processing conditional analysis for heading: {heading}")
        # Build a mapping from criminal id to its value for this heading
        condition_map = {}
        for crim_id in criminal_sequences.keys():
            # Use our helper that searches in the key-value pairs
            val = get_value_for_heading(crim_id, type2_df, heading)
            condition_map[crim_id] = val
        # Get all unique values (including None)
        unique_values = set(condition_map.values())
        for val in unique_values:
            # Get the list of criminals with this particular value
            selected_ids = [cid for cid, v in condition_map.items() if v == val]
            if not selected_ids:
                continue
            # Optionally, skip groups with very few criminals (adjust threshold as desired)
            if len(selected_ids) < 3:
                print(f"[INFO] Skipping {heading} = {val} due to insufficient criminals (n={len(selected_ids)})")
                continue
            # Build the conditional Markov chain for these criminals
            matrix = build_conditional_markov(selected_ids, criminal_sequences, n_clusters)
            # Compute the stationary distribution for this conditional chain
            stationary_cond = compute_stationary_distribution(matrix)
            # Compute the L1 difference with the global stationary distribution
            diff = np.sum(np.abs(stationary_cond - global_stationary))
            # Record an insight if the difference is large enough
            if diff > diff_threshold:
                safe_heading = re.sub(r'\W+', '_', heading)
                safe_val = re.sub(r'\W+', '_', str(val))
                insights[f"{safe_heading}={safe_val}"] = {
                    "n_criminals": len(selected_ids),
                    "stationary_cond": stationary_cond.tolist(),
                    "global_stationary": global_stationary.tolist(),
                    "difference": diff,
                }
                print(f"[INSIGHT] For condition {heading} = {val}, the L1 difference in stationary distribution is {diff:.3f} (n={len(selected_ids)}).")
                # Optionally, you can also save the state transition diagram for this condition
                out_path = os.path.join(output_dir, f"state_transition_{safe_heading}_{safe_val}.png")
                plot_state_transition_diagram(matrix, out_path=out_path)
                print(f"[INFO] Diagram for {heading} = {val} saved to {out_path}")
    return insights

# --------------------------------------------------
# Type 2 Integration Functions
# --------------------------------------------------
def load_all_type2_data(directory: str) -> pd.DataFrame:
    """
    Loads all Type2 CSV files from the specified directory.
    Assumes each file is named like 'Type2_<criminal_id>.csv' and adds a "CriminalID" column.
    Returns a concatenated DataFrame.
    """
    type2_dfs = []
    for filename in os.listdir(directory):
        if filename.startswith("Type2_") and filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(file_path, encoding='utf-8', quotechar='"', skipinitialspace=True)
            except Exception as e:
                print(f"[WARNING] Could not read {filename}: {e}")
                continue
            # Extract criminal id from filename.
            criminal_id = filename[len("Type2_"):-len(".csv")]
            df["CriminalID"] = criminal_id
            type2_dfs.append(df)
    if type2_dfs:
        return pd.concat(type2_dfs, ignore_index=True)
    else:
        raise FileNotFoundError("No Type2 CSV files found in the provided directory.")


def load_type2_data(path: str) -> pd.DataFrame:
    """
    If 'path' is a directory, load all Type2 CSV files.
    Otherwise, assume it is a single CSV file.
    """
    if os.path.isdir(path):
        return load_all_type2_data(path)
    elif os.path.isfile(path):
        return pd.read_csv(path, encoding='utf-8', quotechar='"', skipinitialspace=True)
    else:
        raise FileNotFoundError(f"Provided Type2 path not found: {path}")
    
def get_type2_vector(crim_id, type2_df):
    """
    Extracts and returns a feature vector for a criminal based on Type 2 data.
    Modify this function to include additional Type 2 features as needed.
    """
    row = type2_df[type2_df["CriminalID"] == crim_id]
    if row.empty:
        return None
    # Example: Use the "Physically abused?" flag (1 for yes, 0 for no)
    abused = 1 if str(row["Physically abused?"].values[0]).lower().startswith("yes") else 0
    # You can add more features here, e.g., number of victims, substance abuse, etc.
    return np.array([abused])

def get_value_for_heading(crim_id, type2_df, heading_query):
    """
    For a given criminal, searches the Type 2 DataFrame (which is in key-value format)
    for a row where the "Heading" matches the heading_query (case-insensitive).
    Returns the corresponding value as a string (or None if not found).
    """
    rows = type2_df[type2_df["CriminalID"] == crim_id]
    if rows.empty:
        return None
    # Use case-insensitive matching on the "Heading" column.
    matched = rows[rows["Heading"].str.strip().str.lower() == heading_query.strip().lower()]
    if not matched.empty:
        return str(matched.iloc[0]["Value"]).strip()
    else:
        return None

def get_extended_type2_vector(crim_id, type2_df):
    """
    Extracts and returns a feature vector for a criminal based on multiple Type 2 fields.
    This version looks for the desired key in the "Heading" column of the key-value pair data.
    It extracts:
      - Physically abused? (1 if the corresponding value starts with "yes", 0 otherwise)
      - Sex (1 for male, 0 for female)
      - Number of victims (numeric value extracted from the first number in the value)
    """
    # Extract value for "Physically abused?"
    abused_val = get_value_for_heading(crim_id, type2_df, "Physically abused?")
    if abused_val is None:
        abused = 0
    else:
        abused = 1 if abused_val.lower().startswith("yes") else 0

    # Extract value for "Sex"
    sex_val = get_value_for_heading(crim_id, type2_df, "Sex")
    if sex_val is None:
        # Default to Male if not found
        sex = 1
    else:
        sex = 1 if sex_val.strip().lower() == "male" else 0

    # Extract value for "Number of victims"
    victims_val = get_value_for_heading(crim_id, type2_df, "Number of victims")
    import re
    if victims_val is None:
        num_victims = 0.0
    else:
        match = re.search(r"(\d+)", victims_val)
        num_victims = float(match.group(1)) if match else 0.0

    return np.array([abused, sex, num_victims])


def run_all_conditional_markov_analysis(type2_df, criminal_sequences, n_clusters, output_dir, max_filename_length=100):
    """
    Runs conditional Markov chain analysis for every unique heading (condition)
    in the Type 2 data. For each heading, it groups criminals by the value of
    that heading (using get_value_for_heading) and then plots a state transition
    diagram for each group (if there are enough criminals in that group).
    
    It also computes a simple L1 difference with the global stationary distribution
    to automatically detect interesting insights.
    """
    import re
    # Get all unique headings from the Type2 data
    headings = type2_df["Heading"].unique()
    for heading in headings:
        print(f"[INFO] Processing conditional analysis for heading: {heading}")
        # Build a dictionary mapping each criminal id to its value for this heading.
        condition_map = {}
        for crim_id in criminal_sequences.keys():
            # Use our helper that searches in the key-value pairs
            val = get_value_for_heading(crim_id, type2_df, heading)
            condition_map[crim_id] = val
        # Get all unique values (including None)
        unique_values = set(condition_map.values())
        for val in unique_values:
            # Get the list of criminals with this particular value
            selected_ids = [cid for cid, v in condition_map.items() if v == val]
            if not selected_ids:
                continue
            # Optionally, skip groups with very few criminals (you can adjust the threshold)
            if len(selected_ids) < 3:
                print(f"[INFO] Skipping {heading} = {val} due to insufficient criminals (n={len(selected_ids)})")
                continue
            # Build the conditional Markov chain for these criminals
            matrix = build_conditional_markov(selected_ids, criminal_sequences, n_clusters)
            
            # Create safe file names from heading and value
            safe_heading = re.sub(r'\W+', '_', heading)
            safe_val = re.sub(r'\W+', '_', str(val))
            base_filename = f"state_transition_{safe_heading}_{safe_val}.png"
            
            # Debug: print the raw and safe names
            print(f"[DEBUG] Raw heading: {heading}, Raw value: {val}")
            print(f"[DEBUG] Safe heading: {safe_heading}, Safe value: {safe_val}")
            print(f"[DEBUG] Base filename before truncation: {base_filename} (length {len(base_filename)})")
            
            # Truncate the filename if too long. You can either simply slice it,
            # or use a hash of the combined string for a fixed-length alternative.
            if len(base_filename) > max_filename_length:
                # Option 1: simple truncation
                base_filename = base_filename[:max_filename_length - 4] + ".png"
                print(f"[DEBUG] Truncated filename to: {base_filename} (length {len(base_filename)})")
                # Option 2 (alternative): use a hash (uncomment the next three lines to use hashing)
                # import hashlib
                # hash_val = hashlib.md5((safe_heading + safe_val).encode('utf-8')).hexdigest()[:8]
                # base_filename = f"state_transition_{hash_val}.png"
            
            out_path = os.path.join(output_dir, base_filename)
            try:
                plot_state_transition_diagram(matrix, out_path=out_path)
                print(f"[INFO] Conditional Markov chain for {heading} = {val} saved to {out_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save diagram for {heading} = {val}: {e}")

def integrated_logistic_regression_extended(criminal_sequences: dict, type2_df: pd.DataFrame, target_cluster: int):
    """
    For each criminal, use an extended Type 2 feature vector (multiple fields)
    to predict membership in a target cluster.
    Expects the DataFrame to have a 'CriminalID' column.
    """
    if "CriminalID" not in type2_df.columns:
        print("[WARNING] 'CriminalID' column not found in Type 2 data. Skipping extended logistic regression.")
        return

    X = []
    y = []
    valid_ids = []
    for crim_id, seq in criminal_sequences.items():
        vec = get_extended_type2_vector(crim_id, type2_df)
        if vec is not None:
            # Define target: 1 if any event in the criminal's sequence belongs to the target cluster
            target_val = 1 if target_cluster in seq else 0
            X.append(vec.tolist())
            y.append(target_val)
            valid_ids.append(crim_id)
    if len(X) < 2:
        print("[WARNING] Not enough criminals for extended logistic regression.")
        return

    logreg = LogisticRegression()
    try:
        acc_scores = cross_val_score(logreg, X, y, cv=5, scoring="accuracy")
        f1_scores = cross_val_score(logreg, X, y, cv=5, scoring="f1")
        logreg.fit(X, y)
        preds = logreg.predict(X)
        print("[INFO] Extended Logistic Regression (predicting presence of target cluster):")
        print(f"CV Accuracy: {acc_scores.mean():.3f} +/- {acc_scores.std():.3f}")
        print(f"CV F1 Score: {f1_scores.mean():.3f}")
        print("CriminalID | Feature Vector (abused, sex, #victims) | Actual | Predicted")
        for cid, xi, actual, pred in zip(valid_ids, X, y, preds):
            print(f"{cid:20} | {xi} | {actual}      | {pred}")
    except Exception as e:
        print(f"[WARNING] Extended logistic regression failed: {e}")


def generate_lexical_variations(text, num_variants=5):
    prompt = (
        "Generate {} alternative versions of the following sentence, "
        "using synonyms and varied phrasing, while preserving the meaning:\n\n"
        "{}\n\nAlternative versions:".format(num_variants, text)
    )
    print(f"[DEBUG] Sending prompt for lexical variations:\n{prompt}\n")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4" if available
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        reply = response.choices[0].message.content.strip()
        print(f"[DEBUG] Received lexical variations reply:\n{reply}\n")
        variations = [line.strip() for line in reply.split("\n") if line.strip()]
        if not variations:
            return [text]
        return variations
    except Exception as e:
        print(f"[ERROR] Generating lexical variations: {e}")
        return [text]
    
def get_imputed_embedding(event_text, model, num_variants=5):
    """
    For a given event text, generates lexical variations, computes the embeddings for
    each variant (and optionally includes the original), and returns the centroid (average)
    embedding.
    """
    # Generate alternative versions using the LLM
    variants = generate_lexical_variations(event_text, num_variants=num_variants)
    # Optionally include the original text in the set of versions
    all_versions = variants + [event_text]
    # Compute embeddings for all versions
    embeddings = model.encode(all_versions)
    # Compute the average (centroid) embedding
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding

def integrated_logistic_regression_analysis(criminal_sequences: dict, type2_df: pd.DataFrame, target_cluster: int):
    """
    For each criminal, use a Type 2 feature ("Physically abused?") to predict membership in a target cluster.
    Expects the DataFrame to have a 'CriminalID' column.
    """
    if "CriminalID" not in type2_df.columns:
        print("[WARNING] 'CriminalID' column not found in Type 2 data. Skipping integrated logistic regression.")
        return
    # Build a mapping from CriminalID to the binary flag for "Physically abused?"
    type2_features = {}
    for _, row in type2_df.iterrows():
        crim_id = str(row["CriminalID"])
        abused_str = str(row.get("Physically abused?", "")).lower().strip()
        abused_flag = 1 if abused_str.startswith("yes") else 0
        type2_features[crim_id] = abused_flag

    X = []
    y = []
    valid_ids = []
    for crim_id, seq in criminal_sequences.items():
        if crim_id in type2_features:
            # Define target as 1 if any event for this criminal falls in target_cluster, else 0.
            target_val = 1 if target_cluster in seq else 0
            X.append([type2_features[crim_id]])
            y.append(target_val)
            valid_ids.append(crim_id)
    if len(X) < 2:
        print("[WARNING] Not enough criminals with both Type 1 and Type 2 data for logistic regression.")
        return
    logreg = LogisticRegression()
    try:
        acc_scores = cross_val_score(logreg, X, y, cv=5, scoring="accuracy")
        f1_scores = cross_val_score(logreg, X, y, cv=5, scoring="f1")
        logreg.fit(X, y)
        preds = logreg.predict(X)
        print("[INFO] Integrated Logistic Regression (predicting presence of target cluster):")
        print(f"CV Accuracy: {acc_scores.mean():.3f} +/- {acc_scores.std():.3f}")
        print(f"CV F1 Score: {f1_scores.mean():.3f}")
        print("CriminalID | Type2 (Physically abused?) | Actual | Predicted")
        for cid, xi, actual, pred in zip(valid_ids, X, y, preds):
            print(f"{cid:20} | {xi[0]:25} | {actual}      | {pred}")
    except Exception as e:
        print(f"[WARNING] Logistic regression failed: {e}")


# --------------------------------------------------
# Main Function
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Integrate Type 1 & Type 2 data analysis across multiple criminals.")
    parser.add_argument("--type1_dir", type=str, required=True,
                        help="Directory containing Type1_*.csv files (one per criminal).")
    parser.add_argument("--type2_csv", type=str, required=True,
                        help="Path to a Type2 CSV file or directory containing Type2_*.csv files.")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save output artifacts (json, diagrams, etc.).")
    parser.add_argument("--n_clusters", type=int, default=5,
                        help="Number of clusters for KMeans.")
    parser.add_argument("--no_llm", action="store_true",
                        help="Disable LLM calls if you have no OpenAI key.")
    parser.add_argument("--multi_modal", action="store_true",
                        help="Perform multi-modal clustering at the criminal level using Type 1 & Type 2 data.")
    parser.add_argument("--train_proto_net", action="store_true",
                        help="Train a prototypical network on clustered event embeddings.")
    parser.add_argument("--use_tfidf", action="store_true",
                        help="Use TF-IDF embeddings instead of SentenceTransformer (offline mode).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------
    # Load Type 1 data for all criminals
    # -------------------------------
    criminals_data = load_all_criminals_type1(args.type1_dir)
    if not criminals_data:
        print(f"[ERROR] No Type1 files found in {args.type1_dir}. Exiting.")
        sys.exit(1)
    print(f"[INFO] Loaded Type 1 data for {len(criminals_data)} criminals.")

    # Aggregate all events and record originating criminal IDs.
    global_events = []
    event_criminal_ids = []
    for crim_id, data in criminals_data.items():
        for event in data["events"]:
            global_events.append(event)
            event_criminal_ids.append(crim_id)
    if not global_events:
        print("[ERROR] No life events found. Exiting.")
        sys.exit(1)
    print(f"[INFO] Total life events loaded: {len(global_events)}")

    # -------------------------------
    # Preprocess and Embed Type 1 Events
    # -------------------------------
    # processed_events = [preprocess_text(e) for e in global_events]
    # embeddings = generate_embeddings(processed_events)
        # -------------------------------
    # Preprocess and Embed Type 1 Events with Lexical Imputation
    # -------------------------------
    processed_events = [preprocess_text(e) for e in global_events]
    print(f"[DEBUG] Preprocessing complete. Number of processed events: {len(processed_events)}")

    if args.use_tfidf:
        embeddings = generate_tfidf_embeddings(processed_events)
        print("[DEBUG] Using TF-IDF embeddings. Shape:", embeddings.shape)
    else:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        imputed_embeddings = []
        total_events = len(processed_events)
        print("[DEBUG] Starting lexical imputation and embedding computation...")
        for idx, event in enumerate(processed_events):
            if idx % 50 == 0:
                print(f"[DEBUG] Processing event {idx+1}/{total_events}")
            avg_emb = get_imputed_embedding(event, model, num_variants=5)
            imputed_embeddings.append(avg_emb)
        embeddings = np.array(imputed_embeddings)
        print("[DEBUG] Lexical imputation complete. Shape of embeddings:", embeddings.shape)

    # Global KMeans clustering on all events.
    labels, kmeans_model = kmeans_cluster(embeddings, n_clusters=args.n_clusters)
    print(f"[INFO] Global KMeans clustering complete with {args.n_clusters} clusters.")

    metrics = evaluate_clustering(embeddings, labels)
    metrics_path = os.path.join(args.output_dir, "cluster_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Silhouette score: {metrics['silhouette']:.3f}, Davies-Bouldin: {metrics['davies_bouldin']:.3f}")
    tsne_path = os.path.join(args.output_dir, "tsne.png")
    plot_tsne_embeddings(embeddings, labels, out_path=tsne_path)

    # Optionally, label clusters using LLM.
    cluster_info = find_representative_samples(global_events, embeddings, labels, n_reps=3)
    if not args.no_llm and openai.api_key:
        for cinfo in cluster_info:
            cinfo["archetypal_theme"] = analyze_cluster_with_llm(cinfo["representative_samples"])
    else:
        print("[INFO] Skipping LLM archetype labeling.")
        for cinfo in cluster_info:
            cinfo["archetypal_theme"] = "N/A (LLM disabled)"
    clusters_json_path = os.path.join(args.output_dir, "global_clusters.json")
    with open(clusters_json_path, "w", encoding="utf-8") as jf:
        json.dump(cluster_info, jf, indent=4, ensure_ascii=False, default=numpy_to_python_default)
    print(f"[INFO] Global cluster info saved to {clusters_json_path}")

    if args.train_proto_net:
        print("[INFO] Training prototypical network on event embeddings...")
        _, prototypes, val_acc = train_prototypical_network(embeddings, labels)
        proto_path = os.path.join(args.output_dir, "prototypes.npy")
        np.save(proto_path, prototypes)
        print(f"[INFO] Prototypical network validation accuracy: {val_acc:.3f}")
        print(f"[INFO] Prototypes saved to {proto_path}")

    # -------------------------------
    # Reconstruct Per-Criminal Cluster Sequences
    # -------------------------------
    criminal_sequences = {}
    for idx, cid in enumerate(event_criminal_ids):
        criminal_sequences.setdefault(cid, []).append(labels[idx])

    # -------------------------------
    # Build Global Markov Chain Transition Matrix
    # -------------------------------
    global_transition_matrix = np.zeros((args.n_clusters, args.n_clusters))
    for seq in criminal_sequences.values():
        if len(seq) < 2:
            continue
        for s1, s2 in zip(seq[:-1], seq[1:]):
            global_transition_matrix[s1, s2] += 1
    # Normalize rows
    row_sums = global_transition_matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        global_transition_matrix = np.divide(global_transition_matrix, row_sums, where=row_sums != 0)
        for i in range(args.n_clusters):
            if row_sums[i] == 0:
                global_transition_matrix[i] = 1.0 / args.n_clusters

    diagram_path = os.path.join(args.output_dir, "global_state_transition.png")
    plot_state_transition_diagram(global_transition_matrix, out_path=diagram_path)
    stationary = compute_stationary_distribution(global_transition_matrix)
    entropy = transition_entropy(global_transition_matrix)
    print("[INFO] Global stationary distribution:", stationary)
    print(f"[INFO] Transition matrix entropy: {entropy:.3f}")
    np.save(os.path.join(args.output_dir, "global_transition_matrix.npy"), global_transition_matrix)
    np.save(os.path.join(args.output_dir, "global_stationary_distribution.npy"), stationary)

    # -------------------------------
    # Load Type 2 Data and Perform Integrated Analysis
    # -------------------------------
    try:
        type2_df = load_type2_data(args.type2_csv)
        print(f"[INFO] Loaded Type 2 data from {args.type2_csv} with shape {type2_df.shape}")
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")
        type2_df = None
    except Exception as e:
        print(f"[WARNING] Could not process Type 2 data: {e}")
        type2_df = None

    # Run integrated logistic regression analysis (predict presence of target cluster, e.g. cluster 0).
    target_cluster = 0
    if type2_df is not None:
        integrated_logistic_regression_analysis(criminal_sequences, type2_df, target_cluster)
    else:
        print("[INFO] Skipping integrated logistic regression due to missing Type 2 data.")

    # Extended integrated logistic regression analysis using multiple Type 2 features
    if type2_df is not None:
        print("\n[INFO] Running extended logistic regression with additional Type 2 features...")
        extended_target_cluster = 0  # or choose another target cluster
        integrated_logistic_regression_extended(criminal_sequences, type2_df, extended_target_cluster)
    
    # -------------------------------
    # Conditional Markov Chain Analysis by "Physically abused?" status
    # -------------------------------
    if type2_df is not None:
        # Build a mapping for the "Physically abused?" flag (1 if Yes, 0 if not)
        type2_features = {}
        for _, row in type2_df.iterrows():
            crim_id = str(row["CriminalID"])
            abused_str = str(row.get("Physically abused?", "")).lower().strip()
            abused_flag = 1 if abused_str.startswith("yes") else 0
            type2_features[crim_id] = abused_flag

        # Split criminals into two groups:
        abused_ids = [cid for cid, flag in type2_features.items() if flag == 1]
        non_abused_ids = [cid for cid, flag in type2_features.items() if flag == 0]

        # Build conditional transition matrices:
        transition_abused = build_conditional_markov(abused_ids, criminal_sequences, args.n_clusters)
        transition_non_abused = build_conditional_markov(non_abused_ids, criminal_sequences, args.n_clusters)

        # Plot and save the conditional Markov chain diagrams:
        abused_diagram_path = os.path.join(args.output_dir, "state_transition_abused.png")
        non_abused_diagram_path = os.path.join(args.output_dir, "state_transition_non_abused.png")
        plot_state_transition_diagram(transition_abused, out_path=abused_diagram_path)
        plot_state_transition_diagram(transition_non_abused, out_path=non_abused_diagram_path)
        print("[INFO] Conditional Markov Chains for 'Physically abused?' subgroups saved.")

    # -------------------------------
    # Optional: Multi-modal Clustering at Criminal Level
    # -------------------------------
    if args.multi_modal and type2_df is not None:
        # Compute per-criminal average embedding.
        criminal_embeddings = {}
        for crim_id in criminals_data.keys():
            indices = [i for i, cid in enumerate(event_criminal_ids) if cid == crim_id]
            if indices:
                criminal_embeddings[crim_id] = np.mean(embeddings[indices], axis=0)
        
        multi_modal_vectors = []
        modal_criminal_ids = []
        for crim_id, emb in criminal_embeddings.items():
            extended_vec = get_extended_type2_vector(crim_id, type2_df)
            if extended_vec is not None:
                # Concatenate the average embedding with the extended Type 2 feature vector.
                combined = np.concatenate([emb, extended_vec])
                multi_modal_vectors.append(combined)
                modal_criminal_ids.append(crim_id)
        if multi_modal_vectors:
            multi_modal_vectors = np.array(multi_modal_vectors)
            mm_labels, mm_model = kmeans_cluster(multi_modal_vectors, n_clusters=args.n_clusters)
            print("[INFO] Extended multi-modal clustering (criminal level) complete. Cluster assignments:")
            for cid, label in zip(modal_criminal_ids, mm_labels):
                print(f"  Criminal {cid}: Cluster {label}")
        else:
            print("[INFO] No criminals with extended multi-modal features available.")


    if type2_df is not None:
        print("\n[INFO] Running conditional Markov chain analysis for all conditions...")
        run_all_conditional_markov_analysis(type2_df, criminal_sequences, args.n_clusters, args.output_dir)
    if type2_df is not None:
        print("\n[INFO] Running automated conditional Markov chain insight analysis for all conditions...")
        insights = analyze_all_conditional_insights(type2_df, criminal_sequences, args.n_clusters, args.output_dir, stationary, diff_threshold=0.1)
        insights_path = os.path.join(args.output_dir, "conditional_insights.json")
        with open(insights_path, "w") as f:
            json.dump(insights, f, indent=4)
        print(f"[INFO] Conditional insights saved to {insights_path}")

    # if args.multi_modal and type2_df is not None:
    #     # Build mapping from CriminalID to a numeric Type 2 feature (e.g., "Physically abused?")
    #     type2_features = {}
    #     for _, row in type2_df.iterrows():
    #         crim_id = str(row["CriminalID"])
    #         abused_str = str(row.get("Physically abused?", "")).lower().strip()
    #         abused_flag = 1 if abused_str.startswith("yes") else 0
    #         type2_features[crim_id] = abused_flag
    #     # Compute per-criminal average embedding.
    #     criminal_embeddings = {}
    #     for crim_id in criminals_data.keys():
    #         indices = [i for i, cid in enumerate(event_criminal_ids) if cid == crim_id]
    #         if indices:
    #             criminal_embeddings[crim_id] = np.mean(embeddings[indices], axis=0)
    #     # Concatenate average embedding with the Type 2 feature.
    #     multi_modal_vectors = []
    #     modal_criminal_ids = []
    #     for crim_id, emb in criminal_embeddings.items():
    #         if crim_id in type2_features:
    #             combined = np.concatenate([emb, [type2_features[crim_id]]])
    #             multi_modal_vectors.append(combined)
    #             modal_criminal_ids.append(crim_id)
    #     if multi_modal_vectors:
    #         multi_modal_vectors = np.array(multi_modal_vectors)
    #         mm_labels, mm_model = kmeans_cluster(multi_modal_vectors, n_clusters=args.n_clusters)
    #         print("[INFO] Multi-modal clustering (criminal level) complete. Cluster assignments:")
    #         for cid, label in zip(modal_criminal_ids, mm_labels):
    #             print(f"  Criminal {cid}: Cluster {label}")
    #     else:
    #         print("[INFO] No criminals with multi-modal features available.")

if __name__ == "__main__":
    main()