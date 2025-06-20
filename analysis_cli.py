#!/usr/bin/env python3
import argparse
import numpy as np
from data_loading import load_all_criminals_type1, load_type2_data
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
    args = parser.parse_args()

    criminals = load_all_criminals_type1(args.type1_dir)
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

    if args.type2_csv:
        df = load_type2_data(args.type2_csv)
        print("Loaded Type2 data with shape", df.shape)


if __name__ == "__main__":
    main()
