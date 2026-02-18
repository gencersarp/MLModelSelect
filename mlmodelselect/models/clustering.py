"""Clustering models for MLModelSelect."""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.cluster import KMeans as SKKMeans
from sklearn.cluster import DBSCAN as SKDBSCAN
from sklearn.cluster import AgglomerativeClustering as SKAgglomerativeClustering

from ..base import BaseModel


class KMeans(BaseModel):
    """K-Means clustering with extensive parameters.
    
    Parameters:
        n_clusters: Number of clusters
        init: Initialization method ('k-means++', 'random')
        n_init: Number of time the k-means algorithm will run
        max_iter: Maximum number of iterations
        tol: Relative tolerance for convergence
        **kwargs: Additional sklearn parameters
    """
    
    def __init__(self, n_clusters=8, init='k-means++', n_init=10,
                 max_iter=300, tol=0.0001, **kwargs):
        super().__init__(n_clusters=n_clusters, init=init, n_init=n_init,
                        max_iter=max_iter, tol=tol, **kwargs)
        self.model = SKKMeans(n_clusters=n_clusters, init=init, n_init=n_init,
                             max_iter=max_iter, tol=tol, **kwargs)
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'KMeans':
        self.model.fit(X)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit the model and return cluster labels."""
        return self.model.fit_predict(X)
        
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'clustering',
            'algorithm': 'KMeans',
            'parameters': self.params,
            'fitted': self.is_fitted
        }


class DBSCAN(BaseModel):
    """DBSCAN clustering with extensive parameters.
    
    Parameters:
        eps: Maximum distance between samples
        min_samples: Minimum samples in a neighborhood
        metric: Distance metric
        algorithm: Algorithm for computing neighbors
        leaf_size: Leaf size for tree algorithms
        **kwargs: Additional sklearn parameters
    """
    
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean',
                 algorithm='auto', leaf_size=30, **kwargs):
        super().__init__(eps=eps, min_samples=min_samples, metric=metric,
                        algorithm=algorithm, leaf_size=leaf_size, **kwargs)
        self.model = SKDBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                             algorithm=algorithm, leaf_size=leaf_size, **kwargs)
        self.labels_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DBSCAN':
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """DBSCAN doesn't support prediction on new data.
        Returns labels from fit data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if self.labels_ is None:
            raise ValueError("No labels available. Run fit first.")
        return self.labels_
        
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit the model and return cluster labels."""
        return self.model.fit_predict(X)
        
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'clustering',
            'algorithm': 'DBSCAN',
            'parameters': self.params,
            'fitted': self.is_fitted
        }


class AgglomerativeClustering(BaseModel):
    """Agglomerative (Hierarchical) Clustering with extensive parameters.
    
    Parameters:
        n_clusters: Number of clusters
        affinity: Distance metric ('euclidean', 'manhattan', 'cosine')
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
        distance_threshold: Linkage distance threshold above which clusters won't merge
        **kwargs: Additional sklearn parameters
    """
    
    def __init__(self, n_clusters=2, affinity='euclidean', linkage='ward',
                 distance_threshold=None, **kwargs):
        super().__init__(n_clusters=n_clusters, affinity=affinity,
                        linkage=linkage, distance_threshold=distance_threshold, **kwargs)
        self.model = SKAgglomerativeClustering(
            n_clusters=n_clusters, affinity=affinity, linkage=linkage,
            distance_threshold=distance_threshold, **kwargs
        )
        self.labels_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'AgglomerativeClustering':
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Agglomerative clustering doesn't support prediction on new data.
        Returns labels from fit data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if self.labels_ is None:
            raise ValueError("No labels available. Run fit first.")
        return self.labels_
        
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit the model and return cluster labels."""
        return self.model.fit_predict(X)
        
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'clustering',
            'algorithm': 'AgglomerativeClustering',
            'parameters': self.params,
            'fitted': self.is_fitted
        }


__all__ = [
    'KMeans',
    'DBSCAN',
    'AgglomerativeClustering',
]
