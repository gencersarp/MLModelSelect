# MLModelSelect

**Vast range of ML Models for plug and use with intelligent model comparison**

MLModelSelect is a Python library that provides a comprehensive collection of machine learning models with a unique feature: **intelligent model comparison**. Given multiple models and a dataset, MLModelSelect automatically analyzes the dataset characteristics and determines which model will perform best.

## 🌟 Key Features

- **20+ Ready-to-Use Models**: Classification, Regression, and Clustering algorithms
- **Intelligent Model Comparison**: Unique `model_compare` feature that analyzes datasets and recommends the best model
- **Extensive Parameters**: Each model comes with comprehensive configuration options
- **Plug-and-Play Interface**: Consistent API across all models
- **Dataset Analysis**: Automatic analysis of dataset characteristics (size, dimensionality, balance, etc.)
- **Performance Metrics**: Comprehensive evaluation with multiple metrics and cross-validation
- **Not a PyTorch Clone**: Built on scikit-learn with unique features for rapid prototyping

## 📦 Installation

```bash
pip install -r requirements.txt
```

Or install directly:
```bash
pip install numpy scikit-learn pandas scipy
```

## 🚀 Quick Start

```python
from sklearn.datasets import load_iris
from mlmodelselect.models.classification import RandomForestClassifier, SVM, KNN
from mlmodelselect import compare_models

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create models
models = [
    RandomForestClassifier(n_estimators=50),
    SVM(kernel='rbf'),
    KNN(n_neighbors=3),
]

# Compare and find the best model!
results = compare_models(models, X, y, print_summary=True)
print(f"Best model: {results['best_model']}")
```

## 📚 Available Models

### Classification Models
- `LogisticRegression` - Linear classification with regularization
- `RandomForestClassifier` - Ensemble of decision trees
- `GradientBoostingClassifier` - Gradient boosting for classification
- `SVM` - Support Vector Machine with multiple kernels
- `KNN` - K-Nearest Neighbors
- `DecisionTree` - Single decision tree classifier
- `NeuralNetwork` - Multi-layer perceptron

### Regression Models
- `LinearRegression` - Ordinary least squares
- `Ridge` - Linear regression with L2 regularization
- `Lasso` - Linear regression with L1 regularization
- `ElasticNet` - Linear regression with L1 + L2 regularization
- `RandomForestRegressor` - Ensemble regression
- `GradientBoostingRegressor` - Gradient boosting for regression
- `SVR` - Support Vector Regression
- `KNeighborsRegressor` - K-Nearest Neighbors regression
- `DecisionTreeRegressor` - Single decision tree regressor

### Clustering Models
- `KMeans` - K-Means clustering
- `DBSCAN` - Density-based clustering
- `AgglomerativeClustering` - Hierarchical clustering

## 🎯 Model Comparison - The Killer Feature

The `model_compare` feature is what sets MLModelSelect apart. It doesn't just run models - it intelligently analyzes your dataset and provides recommendations:

```python
from mlmodelselect import ModelCompare
from mlmodelselect.models.classification import *

# Create multiple models
models = [
    LogisticRegression(max_iter=1000),
    RandomForestClassifier(n_estimators=100),
    GradientBoostingClassifier(n_estimators=50),
    SVM(kernel='rbf'),
    NeuralNetwork(hidden_layer_sizes=(100, 50)),
]

# Compare models
comparer = ModelCompare(models, cv=5, test_size=0.2)
results = comparer.compare(X, y)
comparer.print_summary()
```

**What it does:**
1. ✅ Analyzes dataset characteristics (size, dimensionality, class balance)
2. ✅ Trains all models with cross-validation
3. ✅ Computes comprehensive metrics (accuracy, precision, recall, F1, etc.)
4. ✅ Identifies the best performing model
5. ✅ Provides intelligent recommendations based on dataset properties

## 💡 Examples

### Classification Example

```python
from sklearn.datasets import make_classification
from mlmodelselect.models.classification import RandomForestClassifier, SVM
from mlmodelselect import compare_models

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3)

# Compare models
models = [
    RandomForestClassifier(n_estimators=100),
    SVM(kernel='rbf', C=1.0),
]

results = compare_models(models, X, y, print_summary=True)
```

### Regression Example

```python
from sklearn.datasets import make_regression
from mlmodelselect.models.regression import Ridge, RandomForestRegressor
from mlmodelselect import compare_models

# Generate dataset
X, y = make_regression(n_samples=500, n_features=15)

# Compare models
models = [
    Ridge(alpha=1.0),
    RandomForestRegressor(n_estimators=100),
]

results = compare_models(models, X, y, metric='r2', print_summary=True)
```

### Clustering Example

```python
from sklearn.datasets import make_blobs
from mlmodelselect.models.clustering import KMeans, DBSCAN
from mlmodelselect import compare_models

# Generate dataset
X, _ = make_blobs(n_samples=500, centers=4)

# Compare clustering algorithms
models = [
    KMeans(n_clusters=4),
    DBSCAN(eps=2.0, min_samples=5),
]

results = compare_models(models, X, print_summary=True)
```

## 🔧 Model Parameters

All models support extensive parameters. Examples:

```python
# Classification with many options
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)

# Regression with regularization
ridge = Ridge(
    alpha=1.0,
    fit_intercept=True,
    solver='auto',
    max_iter=1000
)

# Neural network with custom architecture
nn = NeuralNetwork(
    hidden_layer_sizes=(100, 50, 25),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate='adaptive',
    max_iter=500
)
```

## 📊 Dataset Analysis

The library automatically analyzes your dataset:

```python
from mlmodelselect.model_compare import DatasetAnalyzer

analysis = DatasetAnalyzer.analyze_dataset(X, y)
print(f"Task type: {analysis['task_type']}")
print(f"Samples: {analysis['n_samples']}")
print(f"Features: {analysis['n_features']}")
print(f"Size category: {analysis['size_category']}")
print(f"Is balanced: {analysis['is_balanced']}")
```

## 🎨 Why MLModelSelect?

1. **Not Just a Wrapper**: While it uses scikit-learn under the hood, MLModelSelect adds unique features like intelligent model comparison
2. **Rapid Prototyping**: Quickly test multiple models with minimal code
3. **Educational**: Great for learning about different ML algorithms
4. **Production-Ready**: Consistent interface makes it easy to swap models in production
5. **Comprehensive**: Covers classification, regression, and clustering

## 📖 More Examples

Check out the `examples/` directory for complete working examples:
- `quickstart.py` - Simple introduction
- `classification_example.py` - Multi-class classification
- `regression_example.py` - Regression with multiple models
- `clustering_example.py` - Clustering comparison

## 🤝 Contributing

Contributions are welcome! This library aims to provide the most comprehensive collection of ready-to-use ML models.

## 📄 License

MIT License

## 🔗 Links

- Repository: https://github.com/gencersarp/MLModelSelect
- Issues: https://github.com/gencersarp/MLModelSelect/issues
