"""
Example: Regression with Model Comparison

This example demonstrates how to use MLModelSelect for regression
tasks and compare multiple models.
"""

import numpy as np
from sklearn.datasets import make_regression
from mlmodelselect.models.regression import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    RandomForestRegressor, GradientBoostingRegressor,
    SVR, KNeighborsRegressor, DecisionTreeRegressor
)
from mlmodelselect import compare_models


def main():
    # Generate a synthetic regression dataset
    print("Generating synthetic regression dataset...")
    X, y = make_regression(
        n_samples=500,
        n_features=15,
        n_informative=10,
        noise=10.0,
        random_state=42
    )
    print(f"Dataset shape: {X.shape}\n")
    
    # Create multiple regression models
    models = [
        LinearRegression(),
        Ridge(alpha=1.0),
        Lasso(alpha=0.1),
        ElasticNet(alpha=0.1, l1_ratio=0.5),
        RandomForestRegressor(n_estimators=100),
        GradientBoostingRegressor(n_estimators=50),
        SVR(kernel='rbf'),
        KNeighborsRegressor(n_neighbors=5),
        DecisionTreeRegressor(max_depth=10),
    ]
    
    # Compare models
    print("Comparing models on the dataset...\n")
    results = compare_models(models, X, y, cv=5, test_size=0.2, 
                           metric='r2', print_summary=True)
    
    # Access specific results
    best_model_name = results['best_model']
    print(f"\n\nBest model for this dataset: {best_model_name}")
    
    if best_model_name:
        best_metrics = results['models'][best_model_name]['metrics']
        print(f"R² Score: {best_metrics['r2']:.4f}")
        print(f"RMSE: {best_metrics['rmse']:.4f}")
        print(f"MAE: {best_metrics['mae']:.4f}")


if __name__ == "__main__":
    main()
