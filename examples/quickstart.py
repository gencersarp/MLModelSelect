"""
Example: Quick Start Guide

A simple example showing the basic usage of MLModelSelect.
"""

from sklearn.datasets import load_iris
from mlmodelselect.models.classification import RandomForestClassifier, SVM, KNN
from mlmodelselect import compare_models


def main():
    # Load a real dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print("Quick Start with Iris Dataset")
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(set(y))} classes\n")
    
    # Create a few models
    models = [
        RandomForestClassifier(n_estimators=50),
        SVM(kernel='rbf'),
        KNN(n_neighbors=3),
    ]
    
    # Compare them!
    results = compare_models(models, X, y, print_summary=True)
    
    print(f"\n✓ Best model: {results['best_model']}")


if __name__ == "__main__":
    main()
