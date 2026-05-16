"""
Bank Marketing XGBoost Prediction - Modular ML Engineering Package

Reusable modules for customer subscription prediction:
- data_preprocessing: Data loading, cleaning, and preprocessing
- feature_engineering: Feature creation and validation
- model_training: XGBoost model training with early stopping
- evaluation: Model evaluation metrics and threshold analysis
- business_metrics: Business-focused cost/revenue/profit calculations
- visualization: Plotting and visualization utilities
"""

__version__ = "1.0.0"
__author__ = "Bank Marketing ML Team"

# Import main functions for convenience
from .data_preprocessing import (
    load_and_prepare_data,
    handle_missing_values,
    get_column_types,
    encode_categorical_features,
    split_data,
    scale_features
)

from .feature_engineering import create_features

from .model_training import train_xgboost_model, predict_probabilities, make_predictions

from .evaluation import (
    calculate_metrics,
    get_confusion_matrix,
    evaluate_model,
    analyze_threshold_sensitivity
)

from .business_metrics import (
    analyze_business_impact,
    threshold_business_analysis,
    find_optimal_threshold
)

__all__ = [
    'load_and_prepare_data',
    'handle_missing_values',
    'get_column_types',
    'encode_categorical_features',
    'split_data',
    'scale_features',
    'create_features',
    'train_xgboost_model',
    'predict_probabilities',
    'make_predictions',
    'calculate_metrics',
    'get_confusion_matrix',
    'evaluate_model',
    'analyze_threshold_sensitivity',
    'analyze_business_impact',
    'threshold_business_analysis',
    'find_optimal_threshold'
]
