"""
Example: Clustering with Model Comparison

This example demonstrates how to use MLModelSelect for clustering
tasks and compare different clustering algorithms.
"""

import numpy as np
from sklearn.datasets import make_blobs
from mlmodelselect.models.clustering import (
    KMeans, DBSCAN, AgglomerativeClustering
)
from mlmodelselect import compare_models


def main():
    # Generate a synthetic clustering dataset
    print("Generating synthetic clustering dataset...")
    X, _ = make_blobs(
        n_samples=500,
        n_features=10,
        centers=4,
        cluster_std=1.5,
        random_state=42
    )
    print(f"Dataset shape: {X.shape}\n")
    
    # Create multiple clustering models
    models = [
        KMeans(n_clusters=4, n_init=10),
        KMeans(n_clusters=3, n_init=10),
        KMeans(n_clusters=5, n_init=10),
        DBSCAN(eps=2.0, min_samples=5),
        AgglomerativeClustering(n_clusters=4, linkage='ward'),
    ]
    
    # Compare models (no y needed for clustering)
    print("Comparing clustering models...\n")
    results = compare_models(models, X, y=None, print_summary=True)
    
    # Access specific results
    best_model_name = results['best_model']
    print(f"\n\nBest clustering model: {best_model_name}")
    
    if best_model_name:
        best_metrics = results['models'][best_model_name]['metrics']
        print(f"Silhouette Score: {best_metrics.get('silhouette', 'N/A')}")
        print(f"Number of clusters: {best_metrics.get('n_clusters', 'N/A')}")


if __name__ == "__main__":
    main()
