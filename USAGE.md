# MLModelSelect Usage Guide

## Installation

### From source:
```bash
git clone https://github.com/gencersarp/MLModelSelect.git
cd MLModelSelect
pip install -e .
```

### Requirements:
- Python 3.7+
- numpy >= 1.19.0
- scikit-learn >= 0.24.0
- pandas >= 1.2.0
- scipy >= 1.6.0

## Quick Start

```python
from sklearn.datasets import load_iris
from mlmodelselect.models.classification import RandomForestClassifier, SVM
from mlmodelselect import compare_models

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Compare models
models = [
    RandomForestClassifier(n_estimators=50),
    SVM(kernel='rbf'),
]

results = compare_models(models, X, y, print_summary=True)
```

## Using Individual Models

### Classification

```python
from mlmodelselect.models.classification import RandomForestClassifier

# Create model with custom parameters
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Get model info
info = model.get_model_info()
```

### Regression

```python
from mlmodelselect.models.regression import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Clustering

```python
from mlmodelselect.models.clustering import KMeans

model = KMeans(n_clusters=3)
model.fit(X)
labels = model.predict(X)
# or
labels = model.fit_predict(X)
```

## Model Comparison Features

### Basic Comparison

```python
from mlmodelselect import compare_models

results = compare_models(
    models=[model1, model2, model3],
    X=X,
    y=y,
    cv=5,              # cross-validation folds
    test_size=0.2,     # test split ratio
    print_summary=True # print formatted results
)
```

### Advanced Comparison

```python
from mlmodelselect import ModelCompare

comparer = ModelCompare(
    models=[model1, model2],
    cv=5,
    test_size=0.2,
    random_state=42
)

# Compare with specific metric
results = comparer.compare(X, y, metric='f1')

# Access results programmatically
best_model = results['best_model']
dataset_info = results['dataset_analysis']

# Print formatted summary
comparer.print_summary()
```

## Available Models

### Classification (7 models)
- `LogisticRegression` - Linear classification
- `RandomForestClassifier` - Ensemble of trees
- `GradientBoostingClassifier` - Gradient boosting
- `SVM` - Support Vector Machine
- `KNN` - K-Nearest Neighbors
- `DecisionTree` - Single decision tree
- `NeuralNetwork` - Multi-layer perceptron

### Regression (9 models)
- `LinearRegression` - OLS regression
- `Ridge` - L2 regularization
- `Lasso` - L1 regularization
- `ElasticNet` - L1 + L2 regularization
- `RandomForestRegressor` - Ensemble regression
- `GradientBoostingRegressor` - Gradient boosting
- `SVR` - Support Vector Regression
- `KNeighborsRegressor` - KNN regression
- `DecisionTreeRegressor` - Decision tree

### Clustering (3 models)
- `KMeans` - K-means clustering
- `DBSCAN` - Density-based clustering
- `AgglomerativeClustering` - Hierarchical clustering

## Model Parameters

Each model supports extensive parameters. Examples:

### RandomForestClassifier
```python
RandomForestClassifier(
    n_estimators=100,        # number of trees
    max_depth=10,            # max tree depth
    min_samples_split=2,     # min samples to split
    min_samples_leaf=1,      # min samples in leaf
    max_features='sqrt',     # max features per split
    bootstrap=True,          # bootstrap samples
    random_state=42          # random seed
)
```

### NeuralNetwork
```python
NeuralNetwork(
    hidden_layer_sizes=(100, 50, 25),  # layer sizes
    activation='relu',                  # activation function
    solver='adam',                      # optimizer
    alpha=0.0001,                      # regularization
    learning_rate='adaptive',          # learning rate schedule
    max_iter=500                       # max iterations
)
```

### SVM
```python
SVM(
    kernel='rbf',           # kernel type
    C=1.0,                  # regularization
    degree=3,               # polynomial degree
    gamma='scale'           # kernel coefficient
)
```

## Dataset Analysis

The library automatically analyzes your dataset:

```python
from mlmodelselect.model_compare import DatasetAnalyzer

analysis = DatasetAnalyzer.analyze_dataset(X, y)

# Available information:
# - n_samples: number of samples
# - n_features: number of features
# - n_classes: number of classes (classification)
# - task_type: 'classification', 'regression', or 'clustering'
# - size_category: 'tiny', 'small', 'medium', or 'large'
# - dimensionality: 'low', 'medium', or 'high'
# - is_balanced: whether classes are balanced
# - class_distribution: samples per class
```

## Metrics

### Classification Metrics
- accuracy
- precision (weighted)
- recall (weighted)
- f1 (weighted)
- cross-validation score

### Regression Metrics
- mse (mean squared error)
- rmse (root mean squared error)
- mae (mean absolute error)
- r2 (R² score)
- cross-validation score

### Clustering Metrics
- silhouette score
- davies_bouldin score
- number of clusters found

## Examples

See the `examples/` directory for complete working examples:
- `quickstart.py` - Basic usage
- `classification_example.py` - Classification with 7 models
- `regression_example.py` - Regression with 9 models
- `clustering_example.py` - Clustering comparison
- `advanced_comparison.py` - Advanced features
- `complete_demo.py` - Full feature showcase

## Tips

1. **Start with compare_models()** - Let the library find the best model
2. **Use appropriate metrics** - f1 for imbalanced data, r2 for regression
3. **Adjust cross-validation folds** - Use cv=3 for small datasets, cv=10 for large
4. **Parameter tuning** - After finding best model, tune its parameters
5. **Check dataset analysis** - Understand your data characteristics

## Common Use Cases

### Finding the best classifier
```python
from mlmodelselect.models.classification import *

models = [
    LogisticRegression(max_iter=1000),
    RandomForestClassifier(),
    SVM(),
    KNN(),
]

results = compare_models(models, X, y, print_summary=True)
best = results['best_model']
```

### Comparing regularization strengths
```python
from mlmodelselect.models.regression import Ridge

models = [
    Ridge(alpha=0.1),
    Ridge(alpha=1.0),
    Ridge(alpha=10.0),
]

results = compare_models(models, X, y, metric='r2')
```

### Finding optimal cluster count
```python
from mlmodelselect.models.clustering import KMeans

models = [
    KMeans(n_clusters=k) 
    for k in range(2, 8)
]

results = compare_models(models, X)
```

## Support

For issues, questions, or contributions, visit:
https://github.com/gencersarp/MLModelSelect
