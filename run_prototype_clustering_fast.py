#!/usr/bin/env python3
"""
run_prototype_clustering_fast.py

A faster version of prototype clustering that uses simple text augmentation
instead of LLM-based lexical imputation for testing purposes.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from data_loading import load_matched_criminal_data
import re

class FastPrototypeGenerator:
    """Fast prototype generation without LLM calls."""
    
    def __init__(self):
        # Synonym mappings for common words
        self.synonyms = {
            'killed': ['murdered', 'slayed', 'executed', 'ended life of'],
            'died': ['passed away', 'deceased', 'perished', 'expired'],
            'born': ['came into world', 'entered life', 'arrived', 'delivered'],
            'married': ['wed', 'united with', 'joined in matrimony', 'tied knot with'],
            'divorced': ['separated from', 'split from', 'ended marriage with', 'left'],
            'arrested': ['apprehended', 'detained', 'taken into custody', 'caught'],
            'convicted': ['found guilty', 'sentenced', 'judged', 'condemned'],
            'abused': ['mistreated', 'harmed', 'hurt', 'victimized'],
            'mother': ['mom', 'maternal parent', 'female parent'],
            'father': ['dad', 'paternal parent', 'male parent'],
            'parents': ['mother and father', 'mom and dad', 'guardians'],
            'child': ['kid', 'youngster', 'offspring', 'youth'],
            'young': ['youthful', 'early age', 'juvenile', 'adolescent']
        }
        
        # Common name patterns to replace
        self.name_pattern = re.compile(
            r'\b(John|James|Robert|Michael|William|David|Mary|Patricia|Jennifer|Linda|Smith|Johnson|Williams|Brown|Jones)\b',
            re.IGNORECASE
        )
    
    def standardize_names(self, text):
        """Replace names with [PERSON] placeholder."""
        standardized = self.name_pattern.sub('[PERSON]', text)
        # Clean up multiple placeholders
        standardized = re.sub(r'\[PERSON\]\s+\[PERSON\]', '[PERSON]', standardized)
        return standardized.strip(), []
    
    def generate_variations(self, text, num_variants=3):
        """Generate simple variations using synonyms."""
        variations = []
        
        # Standardize names first
        standardized, _ = self.standardize_names(text)
        
        # Generate variations by replacing synonyms
        words = standardized.lower().split()
        
        for i in range(min(num_variants, 3)):
            variant_words = []
            for word in words:
                # Check if word has synonyms
                if word in self.synonyms and np.random.random() > 0.5:
                    # Randomly choose a synonym
                    synonym = np.random.choice(self.synonyms[word])
                    variant_words.append(synonym)
                else:
                    variant_words.append(word)
            
            variation = ' '.join(variant_words)
            if variation != standardized.lower():
                variations.append(variation)
        
        # If we couldn't generate enough variations, add slight modifications
        while len(variations) < num_variants:
            if np.random.random() > 0.5:
                variations.append(f"The event: {standardized.lower()}")
            else:
                variations.append(f"{standardized.lower()} occurred")
        
        return variations[:num_variants]

def create_prototype_embeddings_fast(events, generator, num_variants=3):
    """Create prototype embeddings quickly."""
    print(f"\n[INFO] Creating fast prototype representations for {len(events)} events...")
    
    all_texts = []
    variation_groups = []
    event_mapping = {}
    all_variations = []
    
    for idx, event in enumerate(events):
        if idx % 500 == 0:
            print(f"  Processing event {idx}/{len(events)}...")
        
        # Generate variations
        variations = generator.generate_variations(event, num_variants=num_variants)
        standardized_original, _ = generator.standardize_names(event)
        all_versions = variations + [standardized_original.lower()]
        
        variation_groups.append(all_versions)
        all_texts.extend(all_versions)
        
        event_mapping[idx] = event
        all_variations.append({
            'original': event,
            'standardized': standardized_original,
            'variations': variations
        })
    
    # Create TF-IDF embeddings
    print("\n[INFO] Creating TF-IDF embeddings...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    all_embeddings = vectorizer.fit_transform(all_texts).toarray()
    
    # Create prototypes
    print("\n[INFO] Creating prototype embeddings...")
    prototype_embeddings = []
    start_idx = 0
    
    for group in variation_groups:
        end_idx = start_idx + len(group)
        group_embeddings = all_embeddings[start_idx:end_idx]
        prototype = np.mean(group_embeddings, axis=0)
        prototype_embeddings.append(prototype)
        start_idx = end_idx
    
    prototype_embeddings = np.array(prototype_embeddings)
    print(f"[INFO] Created {len(prototype_embeddings)} prototype embeddings")
    
    return prototype_embeddings, event_mapping, all_variations

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', help='Output directory')
    parser.add_argument('--n_clusters', type=int, default=12)
    parser.add_argument('--auto_k', action='store_true')
    parser.add_argument('--num_variants', type=int, default=3)
    
    args = parser.parse_args()
    
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results_fast_prototype_{timestamp}"
    
    print("="*60)
    print("FAST PROTOTYPE-BASED CLUSTERING")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Number of clusters: {args.n_clusters}")
    print(f"Number of variations: {args.num_variants}")
    
    # Load data
    print("\n[INFO] Loading data...")
    type1_data, type2_data = load_matched_criminal_data('type1csvs', 'type2csvs')
    
    all_events = []
    criminal_events = {}
    
    for criminal_id, criminal_data in type1_data.items():
        if 'events' in criminal_data:
            events = criminal_data['events']
            criminal_events[criminal_id] = events
            all_events.extend(events)
    
    print(f"[INFO] Loaded {len(all_events)} events from {len(type1_data)} criminals")
    
    # Create prototypes
    generator = FastPrototypeGenerator()
    prototype_embeddings, event_mapping, all_variations = create_prototype_embeddings_fast(
        all_events, generator, num_variants=args.num_variants
    )
    
    # Cluster
    if args.auto_k:
        print("\n[INFO] Finding optimal k...")
        scores = []
        k_range = range(5, min(21, len(all_events)//50))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(prototype_embeddings)
            score = silhouette_score(prototype_embeddings, labels)
            scores.append(score)
            print(f"  k={k}: score={score:.4f}")
        
        n_clusters = k_range[np.argmax(scores)]
        print(f"[INFO] Optimal k = {n_clusters}")
    else:
        n_clusters = args.n_clusters
    
    print(f"\n[INFO] Clustering with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(prototype_embeddings)
    
    sil_score = silhouette_score(prototype_embeddings, cluster_labels)
    print(f"[INFO] Silhouette score: {sil_score:.4f}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create cluster report
    cluster_info = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(cluster_labels):
        cluster_info[label].append({
            'original': event_mapping[idx],
            'standardized': all_variations[idx]['standardized']
        })
    
    cluster_report = {
        'n_clusters': n_clusters,
        'n_events': len(cluster_labels),
        'silhouette_score': float(sil_score),
        'cluster_info': {}
    }
    
    for cluster_id, events in cluster_info.items():
        cluster_report['cluster_info'][str(cluster_id)] = {
            'size': len(events),
            'sample_events': events[:10]  # First 10 examples
        }
    
    with open(os.path.join(args.output_dir, 'prototype_cluster_report.json'), 'w') as f:
        json.dump(cluster_report, f, indent=2)
    
    # Save criminal sequences
    criminal_sequences = {}
    event_idx = 0
    for criminal_id, events in criminal_events.items():
        sequence = []
        for _ in events:
            sequence.append(int(cluster_labels[event_idx]))
            event_idx += 1
        criminal_sequences[criminal_id] = sequence
    
    with open(os.path.join(args.output_dir, 'criminal_sequences.json'), 'w') as f:
        json.dump(criminal_sequences, f, indent=2)
    
    print(f"\n[INFO] Results saved to {args.output_dir}/")
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    # Show improvement
    if sil_score > 0.1:
        print(f"\n✓ IMPROVED CLUSTERING! Silhouette score: {sil_score:.4f}")
        print("  (Previous non-prototype clustering: ~0.064)")
    else:
        print(f"\nSilhouette score: {sil_score:.4f}")
        print("Consider adjusting number of clusters or variations.")

if __name__ == "__main__":
    main()