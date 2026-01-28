"""
BVRC Regression Models
Provides different regression backends for the BVRC algorithm.
"""

from .BVRC_X import XGBoostRegressor
from .BVRC_T import TabPFNRegressorWrapper
from .BVRC_D import DiffusionRegressor

__all__ = ['XGBoostRegressor', 'TabPFNRegressorWrapper', 'DiffusionRegressor']


def get_regressor(regressor_type, **kwargs):
    """
    Factory function to create a regressor instance.
    
    Args:
        regressor_type: 'X' (XGBoost), 'T' (TabPFN), or 'D' (Diffusion)
        **kwargs: Additional arguments for the regressor
    
    Returns:
        Regressor instance with fit() and predict() methods
    """
    if regressor_type == 'X':
        return XGBoostRegressor(**kwargs)
    elif regressor_type == 'T':
        return TabPFNRegressorWrapper(**kwargs)
    elif regressor_type == 'D':
        return DiffusionRegressor(**kwargs)
    else:
        raise ValueError(f"Unknown regressor type: {regressor_type}. Choose from 'X', 'T', 'D'")
