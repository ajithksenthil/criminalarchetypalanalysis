#!/usr/bin/env python3
import argparse
import numpy as np
from data_loading import load_all_criminals_type1, load_type2_data, load_matched_criminal_data
from text_processing import (
    preprocess_text,
    generate_embeddings,
    generate_imputed_embeddings,
)
from analysis import kmeans_cluster, build_conditional_markov, compute_stationary_distribution, plot_state_transition_diagram


def main():
    parser = argparse.ArgumentParser(description="Run criminal archetypal analysis")
    parser.add_argument("--type1_dir", required=True, help="Directory with Type1 CSV files")
    parser.add_argument("--type2_csv", help="Path to Type2 CSV file or directory")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--diagram", default="state_transition.png", help="Output path for transition diagram")
    parser.add_argument(
        "--lexical_impute",
        action="store_true",
        help="Use LLM-generated lexical variations when computing embeddings",
    )
    parser.add_argument(
        "--match_only",
        action="store_true",
        help="Only analyze criminals with both Type1 and Type2 data",
    )
    args = parser.parse_args()

    # Load data based on match_only flag
    if args.match_only and args.type2_csv:
        import os
        if not os.path.isdir(args.type2_csv):
            print("[ERROR] When using --match_only, --type2_csv must be a directory")
            return
        criminals, type2_df = load_matched_criminal_data(args.type1_dir, args.type2_csv)
    else:
        criminals = load_all_criminals_type1(args.type1_dir)
        type2_df = None
    
    events = []
    event_ids = []
    for cid, data in criminals.items():
        for ev in data["events"]:
            events.append(ev)
            event_ids.append(cid)

    processed = [preprocess_text(e) for e in events]
    if args.lexical_impute:
        embeddings = generate_imputed_embeddings(processed)
    else:
        embeddings = generate_embeddings(processed)
    labels, _ = kmeans_cluster(embeddings, args.n_clusters)

    sequences = {}
    for cid, label in zip(event_ids, labels):
        sequences.setdefault(cid, []).append(label)

    matrix = build_conditional_markov(list(sequences.keys()), sequences, args.n_clusters)
    plot_state_transition_diagram(matrix, out_path=args.diagram)
    stationary = compute_stationary_distribution(matrix)
    print("Stationary distribution:", stationary)

    if args.type2_csv and not args.match_only:
        df = load_type2_data(args.type2_csv)
        print("Loaded Type2 data with shape", df.shape)
    elif args.match_only and type2_df is not None:
        print("Using matched Type2 data with shape", type2_df.shape)


if __name__ == "__main__":
    main()
