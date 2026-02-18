"""Model comparison utilities for MLModelSelect.

This module provides the unique model_compare feature that analyzes datasets
and intelligently compares different models to determine which will perform best.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score
)
import time

from .base import BaseModel


class DatasetAnalyzer:
    """Analyzes dataset characteristics to inform model selection."""
    
    @staticmethod
    def analyze_dataset(X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze dataset characteristics.
        
        Args:
            X: Feature matrix
            y: Target vector (optional for clustering)
            
        Returns:
            Dictionary containing dataset characteristics
        """
        n_samples, n_features = X.shape
        
        analysis = {
            'n_samples': n_samples,
            'n_features': n_features,
            'feature_means': np.mean(X, axis=0).tolist(),
            'feature_stds': np.std(X, axis=0).tolist(),
            'has_missing': np.isnan(X).any(),
            'sparsity': np.sum(X == 0) / X.size,
        }
        
        if y is not None:
            unique_labels = np.unique(y)
            analysis['n_classes'] = len(unique_labels)
            analysis['class_distribution'] = {
                str(label): int(np.sum(y == label)) 
                for label in unique_labels
            }
            analysis['is_balanced'] = DatasetAnalyzer._check_balance(y)
            
            # Determine task type
            if len(unique_labels) < 20 and all(isinstance(val, (int, np.integer)) for val in unique_labels[:min(10, len(unique_labels))]):
                analysis['task_type'] = 'classification'
            else:
                analysis['task_type'] = 'regression'
        else:
            analysis['task_type'] = 'clustering'
            
        # Dataset size category
        if n_samples < 100:
            analysis['size_category'] = 'tiny'
        elif n_samples < 1000:
            analysis['size_category'] = 'small'
        elif n_samples < 10000:
            analysis['size_category'] = 'medium'
        else:
            analysis['size_category'] = 'large'
            
        # Feature dimensionality
        if n_features < 10:
            analysis['dimensionality'] = 'low'
        elif n_features < 100:
            analysis['dimensionality'] = 'medium'
        else:
            analysis['dimensionality'] = 'high'
            
        return analysis
    
    @staticmethod
    def _check_balance(y: np.ndarray, threshold: float = 0.3) -> bool:
        """Check if classes are balanced."""
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            return True
        ratios = counts / len(y)
        return all(r > threshold / len(unique) for r in ratios)


class ModelCompare:
    """Compare multiple ML models on a dataset.
    
    This class provides intelligent model comparison by:
    1. Analyzing dataset characteristics
    2. Training models with cross-validation
    3. Computing comprehensive performance metrics
    4. Providing recommendations based on results
    """
    
    def __init__(self, models: List[BaseModel], cv: int = 5, 
                 test_size: float = 0.2, random_state: int = 42):
        """Initialize ModelCompare.
        
        Args:
            models: List of models to compare
            cv: Number of cross-validation folds
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.models = models
        self.cv = cv
        self.test_size = test_size
        self.random_state = random_state
        self.results = None
        self.dataset_analysis = None
        
    def compare(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                metric: Optional[str] = None) -> Dict[str, Any]:
        """Compare models on the dataset.
        
        Args:
            X: Feature matrix
            y: Target vector (optional for clustering)
            metric: Specific metric to optimize (auto-detected if None)
            
        Returns:
            Dictionary containing comparison results and recommendations
        """
        # Analyze dataset
        self.dataset_analysis = DatasetAnalyzer.analyze_dataset(X, y)
        task_type = self.dataset_analysis['task_type']
        
        print(f"Dataset Analysis:")
        print(f"  Task Type: {task_type}")
        print(f"  Samples: {self.dataset_analysis['n_samples']}")
        print(f"  Features: {self.dataset_analysis['n_features']}")
        print(f"  Size: {self.dataset_analysis['size_category']}")
        print(f"  Dimensionality: {self.dataset_analysis['dimensionality']}")
        
        # Compare models based on task type
        if task_type == 'classification':
            results = self._compare_classification(X, y, metric)
        elif task_type == 'regression':
            results = self._compare_regression(X, y, metric)
        else:  # clustering
            results = self._compare_clustering(X, metric)
            
        self.results = results
        return results
        
    def _compare_classification(self, X: np.ndarray, y: np.ndarray,
                               metric: Optional[str]) -> Dict[str, Any]:
        """Compare classification models."""
        if metric is None:
            metric = 'accuracy'
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        results = {
            'task_type': 'classification',
            'metric_used': metric,
            'models': {},
            'dataset_analysis': self.dataset_analysis
        }
        
        print(f"\nComparing {len(self.models)} models...")
        
        for model in self.models:
            model_name = model.get_model_info()['algorithm']
            print(f"  Evaluating {model_name}...")
            
            try:
                start_time = time.time()
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Compute metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'training_time': time.time() - start_time,
                }
                
                # Add additional metrics if binary/multiclass
                try:
                    metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    metrics['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                except (ValueError, AttributeError) as e:
                    pass
                
                # Cross-validation score
                try:
                    cv_scores = cross_val_score(model.model, X_train, y_train, 
                                               cv=min(self.cv, len(X_train) // 2))
                    metrics['cv_mean'] = cv_scores.mean()
                    metrics['cv_std'] = cv_scores.std()
                except (ValueError, AttributeError) as e:
                    metrics['cv_mean'] = None
                    metrics['cv_std'] = None
                
                results['models'][model_name] = {
                    'metrics': metrics,
                    'model_info': model.get_model_info()
                }
                
            except Exception as e:
                print(f"    Error with {model_name}: {str(e)}")
                results['models'][model_name] = {
                    'error': str(e),
                    'model_info': model.get_model_info()
                }
        
        # Find best model
        best_model = self._find_best_model(results, metric)
        results['best_model'] = best_model
        results['recommendation'] = self._generate_recommendation(results)
        
        return results
        
    def _compare_regression(self, X: np.ndarray, y: np.ndarray,
                           metric: Optional[str]) -> Dict[str, Any]:
        """Compare regression models."""
        if metric is None:
            metric = 'r2'
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        results = {
            'task_type': 'regression',
            'metric_used': metric,
            'models': {},
            'dataset_analysis': self.dataset_analysis
        }
        
        print(f"\nComparing {len(self.models)} models...")
        
        for model in self.models:
            model_name = model.get_model_info()['algorithm']
            print(f"  Evaluating {model_name}...")
            
            try:
                start_time = time.time()
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Compute metrics
                metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'training_time': time.time() - start_time,
                }
                
                # Cross-validation score
                try:
                    cv_scores = cross_val_score(model.model, X_train, y_train,
                                               cv=min(self.cv, len(X_train) // 2),
                                               scoring='r2')
                    metrics['cv_mean'] = cv_scores.mean()
                    metrics['cv_std'] = cv_scores.std()
                except (ValueError, AttributeError) as e:
                    metrics['cv_mean'] = None
                    metrics['cv_std'] = None
                
                results['models'][model_name] = {
                    'metrics': metrics,
                    'model_info': model.get_model_info()
                }
                
            except Exception as e:
                print(f"    Error with {model_name}: {str(e)}")
                results['models'][model_name] = {
                    'error': str(e),
                    'model_info': model.get_model_info()
                }
        
        # Find best model
        best_model = self._find_best_model(results, metric)
        results['best_model'] = best_model
        results['recommendation'] = self._generate_recommendation(results)
        
        return results
        
    def _compare_clustering(self, X: np.ndarray, metric: Optional[str]) -> Dict[str, Any]:
        """Compare clustering models."""
        if metric is None:
            metric = 'silhouette'
            
        results = {
            'task_type': 'clustering',
            'metric_used': metric,
            'models': {},
            'dataset_analysis': self.dataset_analysis
        }
        
        print(f"\nComparing {len(self.models)} models...")
        
        for model in self.models:
            model_name = model.get_model_info()['algorithm']
            print(f"  Evaluating {model_name}...")
            
            try:
                start_time = time.time()
                
                # Fit and predict
                labels = model.fit_predict(X)
                
                # Compute metrics
                metrics = {
                    'training_time': time.time() - start_time,
                    'n_clusters': len(np.unique(labels[labels >= 0])),  # Exclude noise points (-1)
                }
                
                # Only compute scores if we have valid clusters
                if metrics['n_clusters'] > 1 and metrics['n_clusters'] < len(X):
                    try:
                        metrics['silhouette'] = silhouette_score(X, labels)
                    except (ValueError, RuntimeError) as e:
                        metrics['silhouette'] = None
                        
                    try:
                        metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
                    except (ValueError, RuntimeError) as e:
                        metrics['davies_bouldin'] = None
                else:
                    metrics['silhouette'] = None
                    metrics['davies_bouldin'] = None
                
                results['models'][model_name] = {
                    'metrics': metrics,
                    'model_info': model.get_model_info()
                }
                
            except Exception as e:
                print(f"    Error with {model_name}: {str(e)}")
                results['models'][model_name] = {
                    'error': str(e),
                    'model_info': model.get_model_info()
                }
        
        # Find best model
        best_model = self._find_best_model(results, metric)
        results['best_model'] = best_model
        results['recommendation'] = self._generate_recommendation(results)
        
        return results
        
    def _find_best_model(self, results: Dict[str, Any], metric: str) -> Optional[str]:
        """Find the best performing model based on metric."""
        valid_models = {
            name: data for name, data in results['models'].items()
            if 'error' not in data and metric in data.get('metrics', {})
            and data['metrics'][metric] is not None
        }
        
        if not valid_models:
            return None
        
        # Determine if higher or lower is better
        maximize_metrics = {'accuracy', 'precision', 'recall', 'f1', 'r2', 'silhouette', 'cv_mean'}
        minimize_metrics = {'mse', 'rmse', 'mae', 'davies_bouldin', 'training_time'}
        
        if metric in maximize_metrics:
            best_name = max(valid_models.items(), 
                          key=lambda x: x[1]['metrics'][metric])[0]
        elif metric in minimize_metrics:
            best_name = min(valid_models.items(),
                          key=lambda x: x[1]['metrics'][metric])[0]
        else:
            best_name = max(valid_models.items(),
                          key=lambda x: x[1]['metrics'][metric])[0]
        
        return best_name
        
    def _generate_recommendation(self, results: Dict[str, Any]) -> str:
        """Generate intelligent recommendation based on results and dataset analysis."""
        best_model = results.get('best_model')
        if not best_model:
            return "Unable to determine best model. All models failed or produced invalid results."
        
        analysis = self.dataset_analysis
        task_type = results['task_type']
        metric = results['metric_used']
        
        best_metrics = results['models'][best_model]['metrics']
        
        recommendation = f"Recommendation: {best_model}\n\n"
        recommendation += f"Based on {task_type} task with:\n"
        recommendation += f"  - {analysis['n_samples']} samples ({analysis['size_category']} dataset)\n"
        recommendation += f"  - {analysis['n_features']} features ({analysis['dimensionality']} dimensionality)\n"
        
        if task_type == 'classification':
            recommendation += f"  - {analysis.get('n_classes', 'N/A')} classes\n"
            recommendation += f"  - {'Balanced' if analysis.get('is_balanced') else 'Imbalanced'} distribution\n\n"
            recommendation += f"{best_model} achieved:\n"
            recommendation += f"  - {metric}: {best_metrics.get(metric, 'N/A'):.4f}\n"
            if 'cv_mean' in best_metrics and best_metrics['cv_mean'] is not None:
                recommendation += f"  - CV Score: {best_metrics['cv_mean']:.4f} (±{best_metrics['cv_std']:.4f})\n"
        elif task_type == 'regression':
            recommendation += f"\n{best_model} achieved:\n"
            recommendation += f"  - {metric}: {best_metrics.get(metric, 'N/A'):.4f}\n"
            if 'cv_mean' in best_metrics and best_metrics['cv_mean'] is not None:
                recommendation += f"  - CV Score: {best_metrics['cv_mean']:.4f} (±{best_metrics['cv_std']:.4f})\n"
        else:  # clustering
            recommendation += f"\n{best_model} achieved:\n"
            recommendation += f"  - {metric}: {best_metrics.get(metric, 'N/A'):.4f}\n"
            recommendation += f"  - Clusters found: {best_metrics.get('n_clusters', 'N/A')}\n"
        
        recommendation += f"  - Training time: {best_metrics.get('training_time', 'N/A'):.4f}s"
        
        return recommendation
    
    def print_summary(self):
        """Print a formatted summary of comparison results."""
        if not self.results:
            print("No results available. Run compare() first.")
            return
        
        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)
        print(self.results['recommendation'])
        print("\n" + "-"*70)
        print("All Models Performance:")
        print("-"*70)
        
        for model_name, data in self.results['models'].items():
            if 'error' in data:
                print(f"\n{model_name}: FAILED")
                print(f"  Error: {data['error']}")
            else:
                print(f"\n{model_name}:")
                for metric, value in data['metrics'].items():
                    if value is not None:
                        if isinstance(value, float):
                            print(f"  {metric}: {value:.4f}")
                        else:
                            print(f"  {metric}: {value}")
        print("="*70)


def compare_models(models: List[BaseModel], X: np.ndarray, 
                   y: Optional[np.ndarray] = None, cv: int = 5,
                   test_size: float = 0.2, metric: Optional[str] = None,
                   print_summary: bool = True) -> Dict[str, Any]:
    """Quick function to compare multiple models.
    
    Args:
        models: List of models to compare
        X: Feature matrix
        y: Target vector (optional for clustering)
        cv: Number of cross-validation folds
        test_size: Proportion of data for testing
        metric: Specific metric to optimize (auto-detected if None)
        print_summary: Whether to print summary
        
    Returns:
        Dictionary containing comparison results
    """
    comparer = ModelCompare(models, cv=cv, test_size=test_size)
    results = comparer.compare(X, y, metric=metric)
    
    if print_summary:
        comparer.print_summary()
    
    return results
