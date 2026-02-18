"""Regression models for MLModelSelect."""

import numpy as np
from typing import Dict, Any
from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.linear_model import Ridge as SKRidge
from sklearn.linear_model import Lasso as SKLasso
from sklearn.linear_model import ElasticNet as SKElasticNet
from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor as SKGradientBoostingRegressor
from sklearn.svm import SVR as SKSVR
from sklearn.neighbors import KNeighborsRegressor as SKKNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor as SKDecisionTreeRegressor

from ..base import BaseModel


class LinearRegression(BaseModel):
    """Linear Regression model with extensive parameters.
    
    Parameters:
        fit_intercept: Whether to calculate intercept
        normalize: Deprecated, use StandardScaler instead
        **kwargs: Additional sklearn parameters
    """
    
    def __init__(self, fit_intercept=True, **kwargs):
        super().__init__(fit_intercept=fit_intercept, **kwargs)
        self.model = SKLinearRegression(fit_intercept=fit_intercept, **kwargs)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'regression',
            'algorithm': 'LinearRegression',
            'parameters': self.params,
            'fitted': self.is_fitted
        }


class Ridge(BaseModel):
    """Ridge Regression (L2 regularization) with extensive parameters.
    
    Parameters:
        alpha: Regularization strength
        fit_intercept: Whether to calculate intercept
        solver: Solver to use
        max_iter: Maximum iterations
        **kwargs: Additional sklearn parameters
    """
    
    def __init__(self, alpha=1.0, fit_intercept=True, solver='auto', 
                 max_iter=None, **kwargs):
        super().__init__(alpha=alpha, fit_intercept=fit_intercept, 
                        solver=solver, max_iter=max_iter, **kwargs)
        self.model = SKRidge(alpha=alpha, fit_intercept=fit_intercept,
                            solver=solver, max_iter=max_iter, **kwargs)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Ridge':
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'regression',
            'algorithm': 'Ridge',
            'parameters': self.params,
            'fitted': self.is_fitted
        }


class Lasso(BaseModel):
    """Lasso Regression (L1 regularization) with extensive parameters.
    
    Parameters:
        alpha: Regularization strength
        fit_intercept: Whether to calculate intercept
        max_iter: Maximum iterations
        tol: Tolerance for optimization
        **kwargs: Additional sklearn parameters
    """
    
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, 
                 tol=0.0001, **kwargs):
        super().__init__(alpha=alpha, fit_intercept=fit_intercept,
                        max_iter=max_iter, tol=tol, **kwargs)
        self.model = SKLasso(alpha=alpha, fit_intercept=fit_intercept,
                            max_iter=max_iter, tol=tol, **kwargs)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Lasso':
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'regression',
            'algorithm': 'Lasso',
            'parameters': self.params,
            'fitted': self.is_fitted
        }


class ElasticNet(BaseModel):
    """ElasticNet (L1 + L2 regularization) with extensive parameters.
    
    Parameters:
        alpha: Regularization strength
        l1_ratio: Mix of L1 and L2 (0=L2, 1=L1)
        fit_intercept: Whether to calculate intercept
        max_iter: Maximum iterations
        tol: Tolerance for optimization
        **kwargs: Additional sklearn parameters
    """
    
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 max_iter=1000, tol=0.0001, **kwargs):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio,
                        fit_intercept=fit_intercept, max_iter=max_iter,
                        tol=tol, **kwargs)
        self.model = SKElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                  fit_intercept=fit_intercept,
                                  max_iter=max_iter, tol=tol, **kwargs)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ElasticNet':
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'regression',
            'algorithm': 'ElasticNet',
            'parameters': self.params,
            'fitted': self.is_fitted
        }


class RandomForestRegressor(BaseModel):
    """Random Forest regressor with extensive parameters.
    
    Parameters:
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples in leaf
        max_features: Maximum features to consider
        **kwargs: Additional sklearn parameters
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=1.0, **kwargs):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features, **kwargs)
        self.model = SKRandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features, **kwargs
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestRegressor':
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'regression',
            'algorithm': 'RandomForestRegressor',
            'parameters': self.params,
            'fitted': self.is_fitted
        }


class GradientBoostingRegressor(BaseModel):
    """Gradient Boosting regressor with extensive parameters.
    
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
        self.model = SKGradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf, subsample=subsample, **kwargs
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingRegressor':
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'regression',
            'algorithm': 'GradientBoostingRegressor',
            'parameters': self.params,
            'fitted': self.is_fitted
        }


class SVR(BaseModel):
    """Support Vector Regression with extensive parameters.
    
    Parameters:
        C: Regularization parameter
        kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        degree: Degree for poly kernel
        gamma: Kernel coefficient
        epsilon: Epsilon in epsilon-SVR model
        **kwargs: Additional sklearn parameters
    """
    
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale',
                 epsilon=0.1, **kwargs):
        super().__init__(C=C, kernel=kernel, degree=degree, gamma=gamma,
                        epsilon=epsilon, **kwargs)
        self.model = SKSVR(C=C, kernel=kernel, degree=degree, gamma=gamma,
                          epsilon=epsilon, **kwargs)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVR':
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'regression',
            'algorithm': 'SVR',
            'parameters': self.params,
            'fitted': self.is_fitted
        }


class KNeighborsRegressor(BaseModel):
    """K-Nearest Neighbors regressor with extensive parameters.
    
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
        self.model = SKKNeighborsRegressor(
            n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
            leaf_size=leaf_size, p=p, **kwargs
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNeighborsRegressor':
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'regression',
            'algorithm': 'KNeighborsRegressor',
            'parameters': self.params,
            'fitted': self.is_fitted
        }


class DecisionTreeRegressor(BaseModel):
    """Decision Tree regressor with extensive parameters.
    
    Parameters:
        criterion: Split quality measure ('squared_error', 'friedman_mse', 'absolute_error')
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples in leaf
        max_features: Maximum features to consider
        **kwargs: Additional sklearn parameters
    """
    
    def __init__(self, criterion='squared_error', max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1, max_features=None, **kwargs):
        super().__init__(criterion=criterion, max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features, **kwargs)
        self.model = SKDecisionTreeRegressor(
            criterion=criterion, max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features, **kwargs
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeRegressor':
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'regression',
            'algorithm': 'DecisionTreeRegressor',
            'parameters': self.params,
            'fitted': self.is_fitted
        }


__all__ = [
    'LinearRegression',
    'Ridge',
    'Lasso',
    'ElasticNet',
    'RandomForestRegressor',
    'GradientBoostingRegressor',
    'SVR',
    'KNeighborsRegressor',
    'DecisionTreeRegressor',
]
