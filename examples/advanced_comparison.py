"""
Example: Advanced Model Comparison

This example demonstrates advanced features of model comparison:
- Using specific metrics
- Analyzing results programmatically
- Comparing only specific model types
"""

import numpy as np
from sklearn.datasets import make_classification
from mlmodelselect.models.classification import (
    LogisticRegression, RandomForestClassifier, SVM, KNN
)
from mlmodelselect import ModelCompare


def main():
    # Generate a challenging imbalanced dataset
    print("Generating imbalanced classification dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=2,
        weights=[0.9, 0.1],  # Imbalanced classes
        random_state=42
    )
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    print()
    
    # Create models to compare
    models = [
        LogisticRegression(max_iter=1000, C=0.1),
        LogisticRegression(max_iter=1000, C=1.0),
        LogisticRegression(max_iter=1000, C=10.0),
        RandomForestClassifier(n_estimators=50, max_depth=5),
        RandomForestClassifier(n_estimators=100, max_depth=10),
        SVM(kernel='rbf', C=0.1),
        SVM(kernel='rbf', C=1.0),
        KNN(n_neighbors=3),
        KNN(n_neighbors=7),
    ]
    
    # Use ModelCompare for more control
    comparer = ModelCompare(models, cv=5, test_size=0.3, random_state=42)
    results = comparer.compare(X, y, metric='f1')
    
    # Print summary
    comparer.print_summary()
    
    # Programmatic analysis of results
    print("\n" + "="*70)
    print("PROGRAMMATIC ANALYSIS")
    print("="*70)
    
    # Find models with F1 > 0.30
    print("\nModels with F1 > 0.30:")
    for model_name, data in results['models'].items():
        if 'error' not in data:
            f1 = data['metrics'].get('f1', 0)
            if f1 > 0.30:
                print(f"  {model_name}: {f1:.4f}")
    
    # Find fastest model
    print("\nFastest training models:")
    times = []
    for model_name, data in results['models'].items():
        if 'error' not in data:
            time = data['metrics']['training_time']
            times.append((model_name, time))
    times.sort(key=lambda x: x[1])
    for model_name, time in times[:3]:
        print(f"  {model_name}: {time:.4f}s")
    
    # Dataset characteristics
    print("\nDataset characteristics:")
    ds_analysis = results['dataset_analysis']
    print(f"  Task: {ds_analysis['task_type']}")
    print(f"  Samples: {ds_analysis['n_samples']}")
    print(f"  Features: {ds_analysis['n_features']}")
    print(f"  Classes: {ds_analysis['n_classes']}")
    print(f"  Balanced: {ds_analysis['is_balanced']}")
    print(f"  Size category: {ds_analysis['size_category']}")
    print(f"  Dimensionality: {ds_analysis['dimensionality']}")
    
    # Best model details
    print(f"\nBest model: {results['best_model']}")
    if results['best_model']:
        best_data = results['models'][results['best_model']]
        print("Metrics:")
        for metric, value in best_data['metrics'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
