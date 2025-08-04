#!/usr/bin/env python3
"""
regression_analysis.py

Logistic regression analysis integrating Type 1 and Type 2 data.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from ..data.loaders import Type2DataProcessor

class IntegratedRegressionAnalyzer:
    """Integrated regression analysis for Type 1 and Type 2 data."""
    
    def __init__(self):
        self.processor = Type2DataProcessor()
    
    def basic_logistic_regression(self, criminal_sequences: Dict[str, List[int]], 
                                type2_df, target_cluster: int) -> Dict[str, Any]:
        """
        Basic logistic regression using "Physically abused?" feature.
        
        Args:
            criminal_sequences: Criminal sequences
            type2_df: Type 2 DataFrame
            target_cluster: Target cluster to predict
            
        Returns:
            Analysis results
        """
        if "CriminalID" not in type2_df.columns:
            print("[WARNING] 'CriminalID' column not found in Type 2 data.")
            return {}
        
        # Build feature mapping
        type2_features = {}
        for _, row in type2_df.iterrows():
            crim_id = str(row["CriminalID"])
            abused_str = str(row.get("Physically abused?", "")).lower().strip()
            abused_flag = 1 if abused_str.startswith("yes") else 0
            type2_features[crim_id] = abused_flag
        
        # Prepare data
        X, y, valid_ids = self._prepare_regression_data(
            criminal_sequences, type2_features, target_cluster
        )
        
        if len(X) < 2:
            print("[WARNING] Not enough data for logistic regression.")
            return {}
        
        # Run regression
        results = self._run_logistic_regression(X, y, valid_ids, ["Physically abused?"])
        results['target_cluster'] = target_cluster
        results['feature_type'] = 'basic'
        
        return results
    
    def extended_logistic_regression(self, criminal_sequences: Dict[str, List[int]], 
                                   type2_df, target_cluster: int) -> Dict[str, Any]:
        """
        Extended logistic regression using multiple Type 2 features.
        
        Args:
            criminal_sequences: Criminal sequences
            type2_df: Type 2 DataFrame
            target_cluster: Target cluster to predict
            
        Returns:
            Analysis results
        """
        features = ["Physically abused?", "Sex", "Number of victims"]
        
        # Extract features for each criminal
        feature_vectors = {}
        for crim_id in criminal_sequences.keys():
            vector = self.processor.extract_feature_vector(crim_id, type2_df, features)
            if vector is not None:
                feature_vectors[crim_id] = vector
        
        # Prepare data
        X, y, valid_ids = self._prepare_regression_data(
            criminal_sequences, feature_vectors, target_cluster
        )
        
        if len(X) < 2:
            print("[WARNING] Not enough data for extended logistic regression.")
            return {}
        
        # Run regression
        results = self._run_logistic_regression(X, y, valid_ids, features)
        results['target_cluster'] = target_cluster
        results['feature_type'] = 'extended'
        
        return results
    
    def _prepare_regression_data(self, criminal_sequences: Dict[str, List[int]], 
                               features_dict: Dict[str, Any], 
                               target_cluster: int) -> tuple:
        """Prepare data for regression analysis."""
        X = []
        y = []
        valid_ids = []
        
        for crim_id, seq in criminal_sequences.items():
            if crim_id in features_dict:
                # Target: 1 if any event in sequence belongs to target cluster
                target_val = 1 if target_cluster in seq else 0
                
                feature_val = features_dict[crim_id]
                if isinstance(feature_val, (int, float)):
                    X.append([feature_val])
                else:
                    X.append(feature_val)
                
                y.append(target_val)
                valid_ids.append(crim_id)
        
        return X, y, valid_ids
    
    def _run_logistic_regression(self, X: List, y: List, valid_ids: List[str], 
                               feature_names: List[str]) -> Dict[str, Any]:
        """Run logistic regression and return results."""
        logreg = LogisticRegression(random_state=42)
        
        try:
            # Cross-validation scores
            acc_scores = cross_val_score(logreg, X, y, cv=min(5, len(X)), scoring="accuracy")
            f1_scores = cross_val_score(logreg, X, y, cv=min(5, len(X)), scoring="f1")
            
            # Fit model and get predictions
            logreg.fit(X, y)
            preds = logreg.predict(X)
            probas = logreg.predict_proba(X)[:, 1] if len(set(y)) > 1 else [0.5] * len(y)
            
            # Prepare results
            results = {
                'cv_accuracy_mean': float(acc_scores.mean()),
                'cv_accuracy_std': float(acc_scores.std()),
                'cv_f1_mean': float(f1_scores.mean()),
                'cv_f1_std': float(f1_scores.std()),
                'feature_names': feature_names,
                'n_samples': len(X),
                'predictions': []
            }
            
            # Add individual predictions
            for cid, xi, actual, pred, proba in zip(valid_ids, X, y, preds, probas):
                results['predictions'].append({
                    'criminal_id': cid,
                    'features': xi if isinstance(xi, list) else [xi],
                    'actual': int(actual),
                    'predicted': int(pred),
                    'probability': float(proba)
                })
            
            # Print summary
            print(f"[INFO] Logistic Regression Results ({', '.join(feature_names)}):")
            print(f"  CV Accuracy: {acc_scores.mean():.3f} +/- {acc_scores.std():.3f}")
            print(f"  CV F1 Score: {f1_scores.mean():.3f} +/- {f1_scores.std():.3f}")
            print(f"  Sample size: {len(X)}")
            
            return results
            
        except Exception as e:
            print(f"[WARNING] Logistic regression failed: {e}")
            return {'error': str(e)}
    
    def run_comprehensive_regression_analysis(self, criminal_sequences: Dict[str, List[int]], 
                                            type2_df, target_clusters: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run comprehensive regression analysis for multiple target clusters.
        
        Args:
            criminal_sequences: Criminal sequences
            type2_df: Type 2 DataFrame
            target_clusters: List of target clusters (default: [0, 1])
            
        Returns:
            Comprehensive analysis results
        """
        if target_clusters is None:
            target_clusters = [0, 1]
        
        results = {
            'basic_regression': {},
            'extended_regression': {},
            'summary': {}
        }
        
        print("\n[INFO] Running comprehensive regression analysis...")
        
        for target_cluster in target_clusters:
            print(f"\n--- Target Cluster: {target_cluster} ---")
            
            # Basic regression
            basic_results = self.basic_logistic_regression(
                criminal_sequences, type2_df, target_cluster
            )
            if basic_results:
                results['basic_regression'][target_cluster] = basic_results
            
            # Extended regression
            extended_results = self.extended_logistic_regression(
                criminal_sequences, type2_df, target_cluster
            )
            if extended_results:
                results['extended_regression'][target_cluster] = extended_results
        
        # Generate summary
        results['summary'] = self._generate_regression_summary(results)
        
        return results
    
    def _generate_regression_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of regression results."""
        summary = {
            'best_basic_accuracy': 0,
            'best_extended_accuracy': 0,
            'best_basic_target': None,
            'best_extended_target': None,
            'total_analyses': 0
        }
        
        # Find best performing models
        for target, result in results.get('basic_regression', {}).items():
            if 'cv_accuracy_mean' in result:
                summary['total_analyses'] += 1
                if result['cv_accuracy_mean'] > summary['best_basic_accuracy']:
                    summary['best_basic_accuracy'] = result['cv_accuracy_mean']
                    summary['best_basic_target'] = target
        
        for target, result in results.get('extended_regression', {}).items():
            if 'cv_accuracy_mean' in result:
                if result['cv_accuracy_mean'] > summary['best_extended_accuracy']:
                    summary['best_extended_accuracy'] = result['cv_accuracy_mean']
                    summary['best_extended_target'] = target
        
        return summary

# Backward compatibility aliases
def integrated_logistic_regression_analysis(criminal_sequences: Dict[str, List[int]],
                                           type2_df, target_cluster: int) -> None:
    """Backward compatibility alias for basic logistic regression."""
    analyzer = IntegratedRegressionAnalyzer()
    results = analyzer.basic_logistic_regression(criminal_sequences, type2_df, target_cluster)

    if results:
        print(f"\n[INFO] Integrated Logistic Regression Analysis (Target Cluster: {target_cluster})")
        print(f"  CV Accuracy: {results['cv_accuracy_mean']:.3f} +/- {results['cv_accuracy_std']:.3f}")
        print(f"  CV F1 Score: {results['cv_f1_mean']:.3f} +/- {results['cv_f1_std']:.3f}")

def integrated_logistic_regression_extended(criminal_sequences: Dict[str, List[int]],
                                          type2_df, target_cluster: int) -> None:
    """Backward compatibility alias for extended logistic regression."""
    analyzer = IntegratedRegressionAnalyzer()
    results = analyzer.extended_logistic_regression(criminal_sequences, type2_df, target_cluster)

    if results:
        print(f"\n[INFO] Extended Logistic Regression Analysis (Target Cluster: {target_cluster})")
        print(f"  CV Accuracy: {results['cv_accuracy_mean']:.3f} +/- {results['cv_accuracy_std']:.3f}")
        print(f"  CV F1 Score: {results['cv_f1_mean']:.3f} +/- {results['cv_f1_std']:.3f}")
