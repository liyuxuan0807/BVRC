"""
BVRC-T: TabPFN Regressor for BVRC Algorithm
"""

import numpy as np
from sklearn.decomposition import PCA


class TabPFNRegressorWrapper:
    """
    TabPFN regressor wrapper with unified interface.
    
    Args:
        device: Device for TabPFN ('cuda:0', 'cpu', etc.)
        use_pca: Whether to apply PCA dimensionality reduction
        pca_components: Number of PCA components (TabPFN works best with <500 features)
        random_state: Random seed for reproducibility
    """
    
    def __init__(self, device='cuda:0', use_pca=True, pca_components=400, random_state=42, **kwargs):
        self.device = device
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.random_state = random_state
        self.model = None
        self.pca = None
        self._fitted_pca = False
    
    def _init_model(self):
        """Initialize TabPFNRegressor with compatible parameters."""
        from tabpfn import TabPFNRegressor
        
        candidates = [
            {"device": self.device, "N_ensemble_configurations": 2},
            {"device": self.device, "n_ensemble_configurations": 2},
            {"device": self.device, "N_configurations": 2},
            {"device": self.device},
            {},
        ]
        
        for params in candidates:
            try:
                self.model = TabPFNRegressor(**params)
                return
            except TypeError:
                continue
        
        raise RuntimeError("Failed to initialize TabPFNRegressor. Check installation.")
    
    def fit(self, X, y):
        """
        Fit the TabPFN model.
        
        Args:
            X: Features array [n_samples, n_features]
            y: Labels array [n_samples]
        """
        if self.model is None:
            self._init_model()
        
        y_flat = y.ravel() if hasattr(y, 'ravel') else y
        
        # Apply PCA if needed
        X_transformed = X
        if self.use_pca and X.shape[1] > self.pca_components:
            if not self._fitted_pca:
                self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
                X_transformed = self.pca.fit_transform(X)
                self._fitted_pca = True
            else:
                X_transformed = self.pca.transform(X)
        
        self.model.fit(X_transformed, y_flat)
        return self
    
    def predict(self, X):
        """
        Predict labels for input features.
        
        Args:
            X: Features array [n_samples, n_features]
        
        Returns:
            Predictions array [n_samples]
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Apply PCA if it was used during fitting
        X_transformed = X
        if self._fitted_pca and self.pca is not None:
            X_transformed = self.pca.transform(X)
        
        return self.model.predict(X_transformed)
    
    def clone(self):
        """Create a new instance with the same parameters."""
        new_instance = TabPFNRegressorWrapper(
            device=self.device,
            use_pca=self.use_pca,
            pca_components=self.pca_components,
            random_state=self.random_state
        )
        # Share the fitted PCA if available
        if self._fitted_pca and self.pca is not None:
            new_instance.pca = self.pca
            new_instance._fitted_pca = True
        return new_instance
    
    def reset_pca(self):
        """Reset PCA state for re-fitting."""
        self.pca = None
        self._fitted_pca = False
    
    @staticmethod
    def get_name():
        return "TabPFN"
