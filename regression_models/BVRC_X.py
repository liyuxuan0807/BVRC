"""
BVRC-X: XGBoost Regressor for BVRC Algorithm
"""

import xgboost as xgb


class XGBoostRegressor:
    """
    XGBoost regressor wrapper with unified interface.
    
    Args:
        gpu_id: GPU device ID for XGBoost acceleration
        n_estimators: Number of boosting rounds
        random_state: Random seed for reproducibility
    """
    
    def __init__(self, gpu_id=0, n_estimators=200, random_state=42, **kwargs):
        self.gpu_id = gpu_id
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self._use_gpu = True
    
    def fit(self, X, y):
        """
        Fit the XGBoost model.
        
        Args:
            X: Features array [n_samples, n_features]
            y: Labels array [n_samples]
        """
        y_flat = y.ravel() if hasattr(y, 'ravel') else y
        
        try:
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=self.n_estimators,
                tree_method='gpu_hist',
                device=f'cuda:{self.gpu_id}',
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X, y_flat)
            self._use_gpu = True
        except Exception:
            # Fallback to CPU
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=self.n_estimators,
                tree_method='hist',
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X, y_flat)
            self._use_gpu = False
        
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
        return self.model.predict(X)
    
    def clone(self):
        """Create a new instance with the same parameters."""
        return XGBoostRegressor(
            gpu_id=self.gpu_id,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
    
    @staticmethod
    def get_name():
        return "XGBoost"
