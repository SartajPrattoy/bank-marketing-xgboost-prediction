"""
Model Training Module

Handles XGBoost model training with early stopping and hyperparameter configuration.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier


# Default hyperparameters
DEFAULT_HYPERPARAMS = {
    'max_depth': 7,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'early_stopping_rounds': 15,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'auc',
    'verbosity': 0
}


def train_xgboost_model(X_train, y_train, X_test, y_test, hyperparams=None):
    """
    Train an XGBoost classifier with early stopping.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features (used for early stopping)
    y_test : pd.Series
        Test target (used for early stopping)
    hyperparams : dict, optional
        Custom hyperparameters (merged with defaults)
    
    Returns:
    --------
    XGBClassifier
        Trained model
    """
    # Use default hyperparameters or merge with custom
    params = DEFAULT_HYPERPARAMS.copy()
    if hyperparams:
        params.update(hyperparams)
    
    # Initialize model
    model = XGBClassifier(**params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    return model


def get_training_history(model):
    """
    Retrieve training history (AUC scores from each boosting round).
    
    Parameters:
    -----------
    model : XGBClassifier
        Trained XGBoost model
    
    Returns:
    --------
    dict
        Dictionary with 'train_auc' and 'test_auc' scores per round
    """
    results = model.evals_result()
    
    return {
        'train_auc': results['validation_0']['auc'],
        'test_auc': results['validation_1']['auc'],
        'best_iteration': model.best_iteration
    }


def get_model_summary(model):
    """
    Get summary statistics about the trained model.
    
    Parameters:
    -----------
    model : XGBClassifier
        Trained XGBoost model
    
    Returns:
    --------
    dict
        Model summary information
    """
    summary = {
        'best_iteration': model.best_iteration,
        'total_trees': model.best_iteration + 1,
        'n_features': model.n_features_in_,
        'max_depth': model.max_depth,
        'learning_rate': model.learning_rate,
        'n_estimators': model.n_estimators
    }
    
    return summary


def get_feature_importance(model, X_train, top_n=20):
    """
    Get feature importance scores from the trained model.
    
    Parameters:
    -----------
    model : XGBClassifier
        Trained XGBoost model
    X_train : pd.DataFrame
        Training features (to get feature names)
    top_n : int, default=20
        Number of top features to return
    
    Returns:
    --------
    pd.DataFrame
        Feature importance dataframe (sorted)
    """
    importance = model.feature_importances_
    feature_names = X_train.columns
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df.head(top_n)


def predict_probabilities(model, X):
    """
    Get prediction probabilities from the trained model.
    
    Parameters:
    -----------
    model : XGBClassifier
        Trained XGBoost model
    X : pd.DataFrame
        Features for prediction
    
    Returns:
    --------
    np.ndarray
        Probability of subscription (column 1)
    """
    y_probs = model.predict_proba(X)[:, 1]
    return y_probs


def make_predictions(model, X, threshold=0.5):
    """
    Make binary predictions using a custom threshold.
    
    Parameters:
    -----------
    model : XGBClassifier
        Trained XGBoost model
    X : pd.DataFrame
        Features for prediction
    threshold : float, default=0.5
        Classification threshold
    
    Returns:
    --------
    np.ndarray
        Binary predictions
    """
    y_probs = predict_probabilities(model, X)
    y_pred = (y_probs >= threshold).astype(int)
    return y_pred
