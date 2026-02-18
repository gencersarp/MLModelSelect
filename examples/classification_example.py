"""
Example: Classification with Model Comparison

This example demonstrates how to use MLModelSelect for classification
tasks and compare multiple models to find the best one.
"""

import numpy as np
from sklearn.datasets import make_classification
from mlmodelselect.models.classification import (
    LogisticRegression, RandomForestClassifier, GradientBoostingClassifier,
    SVM, KNN, DecisionTree, NeuralNetwork
)
from mlmodelselect import compare_models


def main():
    # Generate a synthetic classification dataset
    print("Generating synthetic classification dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}\n")
    
    # Create multiple models with different configurations
    models = [
        LogisticRegression(max_iter=1000),
        RandomForestClassifier(n_estimators=100, max_depth=10),
        GradientBoostingClassifier(n_estimators=50, learning_rate=0.1),
        SVM(kernel='rbf', C=1.0),
        KNN(n_neighbors=5),
        DecisionTree(max_depth=10),
        NeuralNetwork(hidden_layer_sizes=(50, 30), max_iter=500),
    ]
    
    # Compare models
    print("Comparing models on the dataset...\n")
    results = compare_models(models, X, y, cv=5, test_size=0.2, print_summary=True)
    
    # Access specific results
    best_model_name = results['best_model']
    print(f"\n\nBest model for this dataset: {best_model_name}")
    
    # You can also access detailed metrics
    if best_model_name:
        best_metrics = results['models'][best_model_name]['metrics']
        print(f"Accuracy: {best_metrics['accuracy']:.4f}")
        if 'f1' in best_metrics:
            print(f"F1 Score: {best_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
