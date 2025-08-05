#!/usr/bin/env python3
"""
prototypical_network.py

Prototypical network implementation for few-shot learning on clustered embeddings.
"""

import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestCentroid

# Constants to avoid import issues
RANDOM_SEED = 42

class PrototypicalNetwork:
    """Simple prototypical network implementation using scikit-learn."""
    
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.classifier = NearestCentroid()
        self.prototypes = None
        self.class_labels = None
    
    def train(self, embeddings: np.ndarray, labels: np.ndarray, 
             test_size: float = 0.2) -> Tuple[float, np.ndarray]:
        """
        Train the prototypical network.
        
        Args:
            embeddings: Input embeddings
            labels: Cluster labels
            test_size: Fraction of data to use for testing
            
        Returns:
            Tuple of (validation_accuracy, prototypes)
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=test_size, 
            random_state=self.random_state, stratify=labels
        )
        
        # Train classifier (compute centroids)
        self.classifier.fit(X_train, y_train)
        
        # Get prototypes (centroids)
        self.prototypes = self.classifier.centroids_
        self.class_labels = self.classifier.classes_
        
        # Evaluate on test set
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"[INFO] Prototypical network trained with {len(self.class_labels)} prototypes")
        print(f"[INFO] Validation accuracy: {accuracy:.3f}")
        
        return accuracy, self.prototypes
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new embeddings."""
        if self.classifier is None:
            raise ValueError("Model not trained yet")
        return self.classifier.predict(embeddings)
    
    def get_prototype_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """Get distances to all prototypes."""
        if self.prototypes is None:
            raise ValueError("Model not trained yet")
        
        from sklearn.metrics.pairwise import euclidean_distances
        return euclidean_distances(embeddings, self.prototypes)

class AdvancedPrototypicalNetwork:
    """More advanced prototypical network with support vector and episodic training."""
    
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.prototypes = None
        self.class_labels = None
        self.support_vectors = None
    
    def episodic_training(self, embeddings: np.ndarray, labels: np.ndarray,
                         n_episodes: int = 100, n_way: int = 5, 
                         n_support: int = 5, n_query: int = 10) -> float:
        """
        Episodic training for few-shot learning.
        
        Args:
            embeddings: Input embeddings
            labels: Cluster labels
            n_episodes: Number of training episodes
            n_way: Number of classes per episode
            n_support: Number of support examples per class
            n_query: Number of query examples per class
            
        Returns:
            Average accuracy across episodes
        """
        unique_labels = np.unique(labels)
        
        if len(unique_labels) < n_way:
            print(f"[WARNING] Not enough classes ({len(unique_labels)}) for {n_way}-way learning")
            n_way = len(unique_labels)
        
        episode_accuracies = []
        
        for episode in range(n_episodes):
            # Sample classes for this episode
            episode_classes = np.random.choice(unique_labels, n_way, replace=False)
            
            support_embeddings = []
            support_labels = []
            query_embeddings = []
            query_labels = []
            
            # For each class, sample support and query examples
            for class_idx, class_label in enumerate(episode_classes):
                class_mask = labels == class_label
                class_embeddings = embeddings[class_mask]
                
                if len(class_embeddings) < n_support + n_query:
                    # Not enough examples, skip this episode
                    continue
                
                # Sample support and query examples
                indices = np.random.choice(len(class_embeddings), 
                                         n_support + n_query, replace=False)
                
                support_indices = indices[:n_support]
                query_indices = indices[n_support:]
                
                support_embeddings.extend(class_embeddings[support_indices])
                support_labels.extend([class_idx] * n_support)
                
                query_embeddings.extend(class_embeddings[query_indices])
                query_labels.extend([class_idx] * n_query)
            
            if not support_embeddings:
                continue
            
            # Convert to arrays
            support_embeddings = np.array(support_embeddings)
            support_labels = np.array(support_labels)
            query_embeddings = np.array(query_embeddings)
            query_labels = np.array(query_labels)
            
            # Compute prototypes for this episode
            episode_prototypes = []
            for class_idx in range(n_way):
                class_mask = support_labels == class_idx
                if np.any(class_mask):
                    prototype = np.mean(support_embeddings[class_mask], axis=0)
                    episode_prototypes.append(prototype)
            
            if len(episode_prototypes) != n_way:
                continue
            
            episode_prototypes = np.array(episode_prototypes)
            
            # Classify query examples
            predictions = []
            for query_emb in query_embeddings:
                distances = np.linalg.norm(episode_prototypes - query_emb, axis=1)
                predictions.append(np.argmin(distances))
            
            # Compute accuracy for this episode
            accuracy = accuracy_score(query_labels, predictions)
            episode_accuracies.append(accuracy)
        
        avg_accuracy = np.mean(episode_accuracies) if episode_accuracies else 0.0
        print(f"[INFO] Episodic training complete. Average accuracy: {avg_accuracy:.3f}")
        
        return avg_accuracy
    
    def compute_final_prototypes(self, embeddings: np.ndarray, 
                               labels: np.ndarray) -> np.ndarray:
        """Compute final prototypes from all data."""
        unique_labels = np.unique(labels)
        prototypes = []
        
        for label in unique_labels:
            mask = labels == label
            prototype = np.mean(embeddings[mask], axis=0)
            prototypes.append(prototype)
        
        self.prototypes = np.array(prototypes)
        self.class_labels = unique_labels
        
        return self.prototypes

def train_prototypical_network(embeddings: np.ndarray, labels: np.ndarray,
                             advanced: bool = False) -> Tuple[Any, np.ndarray, float]:
    """
    Train a prototypical network on clustered embeddings.
    
    Args:
        embeddings: Input embeddings
        labels: Cluster labels
        advanced: Whether to use advanced episodic training
        
    Returns:
        Tuple of (model, prototypes, validation_accuracy)
    """
    if advanced:
        model = AdvancedPrototypicalNetwork()
        
        # Episodic training
        val_accuracy = model.episodic_training(embeddings, labels)
        
        # Compute final prototypes
        prototypes = model.compute_final_prototypes(embeddings, labels)
        
    else:
        model = PrototypicalNetwork()
        val_accuracy, prototypes = model.train(embeddings, labels)
    
    return model, prototypes, val_accuracy

def evaluate_prototypical_network(model: PrototypicalNetwork, 
                                embeddings: np.ndarray, 
                                labels: np.ndarray) -> Dict[str, float]:
    """
    Evaluate a trained prototypical network.
    
    Args:
        model: Trained prototypical network
        embeddings: Test embeddings
        labels: True labels
        
    Returns:
        Evaluation metrics
    """
    predictions = model.predict(embeddings)
    accuracy = accuracy_score(labels, predictions)
    
    # Compute per-class accuracy
    unique_labels = np.unique(labels)
    per_class_accuracy = {}
    
    for label in unique_labels:
        mask = labels == label
        if np.any(mask):
            class_accuracy = accuracy_score(labels[mask], predictions[mask])
            per_class_accuracy[f"class_{label}"] = class_accuracy
    
    metrics = {
        "overall_accuracy": accuracy,
        "n_classes": len(unique_labels),
        "n_samples": len(embeddings)
    }
    metrics.update(per_class_accuracy)
    
    return metrics
