"""Classification models for MLModelSelect."""

import numpy as np
from typing import Optional, Dict, Any
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier as SKGradientBoostingClassifier
from sklearn.svm import SVC as SKSVC
from sklearn.neighbors import KNeighborsClassifier as SKKNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier as SKDecisionTreeClassifier
from sklearn.neural_network import MLPClassifier as SKMLPClassifier

from ..base import BaseModel


class LogisticRegression(BaseModel):
    """Logistic Regression classifier with extensive parameters.
    
    Parameters:
        penalty: Regularization type ('l1', 'l2', 'elasticnet', 'none')
        C: Inverse regularization strength
        solver: Optimization algorithm
        max_iter: Maximum iterations
        **kwargs: Additional sklearn parameters
    """
    
    def __init__(self, penalty='l2', C=1.0, solver='lbfgs', max_iter=100, **kwargs):
        super().__init__(penalty=penalty, C=C, solver=solver, 
                        max_iter=max_iter, **kwargs)
        self.model = SKLogisticRegression(
            penalty=penalty, C=C, solver=solver, max_iter=max_iter, **kwargs
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
        
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'classification',
            'algorithm': 'LogisticRegression',
            'parameters': self.params,
            'fitted': self.is_fitted
        }


class RandomForestClassifier(BaseModel):
    """Random Forest classifier with extensive parameters.
    
    Parameters:
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples in leaf
        max_features: Maximum features to consider
        **kwargs: Additional sklearn parameters
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', **kwargs):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features, **kwargs)
        self.model = SKRandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features, **kwargs
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
        
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'classification',
            'algorithm': 'RandomForestClassifier',
            'parameters': self.params,
            'fitted': self.is_fitted
        }


class GradientBoostingClassifier(BaseModel):
    """Gradient Boosting classifier with extensive parameters.
    
    Parameters:
        n_estimators: Number of boosting stages
        learning_rate: Learning rate
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples in leaf
        subsample: Fraction of samples for fitting
        **kwargs: Additional sklearn parameters
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0, **kwargs):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate,
                        max_depth=max_depth, min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf, subsample=subsample, **kwargs)
        self.model = SKGradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf, subsample=subsample, **kwargs
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingClassifier':
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
        
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'classification',
            'algorithm': 'GradientBoostingClassifier',
            'parameters': self.params,
            'fitted': self.is_fitted
        }


class SVM(BaseModel):
    """Support Vector Machine classifier with extensive parameters.
    
    Parameters:
        C: Regularization parameter
        kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        degree: Degree for poly kernel
        gamma: Kernel coefficient
        **kwargs: Additional sklearn parameters
    """
    
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', **kwargs):
        super().__init__(C=C, kernel=kernel, degree=degree, gamma=gamma, **kwargs)
        self.model = SKSVC(C=C, kernel=kernel, degree=degree, gamma=gamma, 
                          probability=True, **kwargs)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVM':
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
        
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'classification',
            'algorithm': 'SVM',
            'parameters': self.params,
            'fitted': self.is_fitted
        }


class KNN(BaseModel):
    """K-Nearest Neighbors classifier with extensive parameters.
    
    Parameters:
        n_neighbors: Number of neighbors
        weights: Weight function ('uniform', 'distance')
        algorithm: Algorithm to compute neighbors
        leaf_size: Leaf size for tree algorithms
        p: Power parameter for Minkowski metric
        **kwargs: Additional sklearn parameters
    """
    
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto',
                 leaf_size=30, p=2, **kwargs):
        super().__init__(n_neighbors=n_neighbors, weights=weights,
                        algorithm=algorithm, leaf_size=leaf_size, p=p, **kwargs)
        self.model = SKKNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
            leaf_size=leaf_size, p=p, **kwargs
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNN':
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
        
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'classification',
            'algorithm': 'KNN',
            'parameters': self.params,
            'fitted': self.is_fitted
        }


class DecisionTree(BaseModel):
    """Decision Tree classifier with extensive parameters.
    
    Parameters:
        criterion: Split quality measure ('gini', 'entropy')
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples in leaf
        max_features: Maximum features to consider
        **kwargs: Additional sklearn parameters
    """
    
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, **kwargs):
        super().__init__(criterion=criterion, max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features, **kwargs)
        self.model = SKDecisionTreeClassifier(
            criterion=criterion, max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features, **kwargs
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
        
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'classification',
            'algorithm': 'DecisionTree',
            'parameters': self.params,
            'fitted': self.is_fitted
        }


class NeuralNetwork(BaseModel):
    """Multi-Layer Perceptron Neural Network classifier.
    
    Parameters:
        hidden_layer_sizes: Tuple of hidden layer sizes
        activation: Activation function ('relu', 'tanh', 'logistic')
        solver: Optimization solver ('adam', 'sgd', 'lbfgs')
        alpha: L2 regularization parameter
        learning_rate: Learning rate schedule
        max_iter: Maximum iterations
        **kwargs: Additional sklearn parameters
    """
    
    def __init__(self, hidden_layer_sizes=(100,), activation='relu',
                 solver='adam', alpha=0.0001, learning_rate='constant',
                 max_iter=200, **kwargs):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                        activation=activation, solver=solver, alpha=alpha,
                        learning_rate=learning_rate, max_iter=max_iter, **kwargs)
        self.model = SKMLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, activation=activation,
            solver=solver, alpha=alpha, learning_rate=learning_rate,
            max_iter=max_iter, **kwargs
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NeuralNetwork':
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
        
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'classification',
            'algorithm': 'NeuralNetwork',
            'parameters': self.params,
            'fitted': self.is_fitted
        }


__all__ = [
    'LogisticRegression',
    'RandomForestClassifier',
    'GradientBoostingClassifier',
    'SVM',
    'KNN',
    'DecisionTree',
    'NeuralNetwork',
]
