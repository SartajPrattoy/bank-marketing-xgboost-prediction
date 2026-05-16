"""
Feature Engineering Module

Handles creation of engineered features for the predictive model.
"""

import pandas as pd
import numpy as np


def create_features(df):
    """
    Generate engineered features for customer subscription prediction.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with original features
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with engineered features added
    """
    df = df.copy()
    
    # Feature 1: Contacted Before Flag
    # Indicates whether customer was contacted in a previous campaign
    df['contacted_before'] = (df['pdays'] != -1).astype(int)
    
    # Feature 2: Contact Intensity
    # Ratio of current campaign contacts to previous contact history
    # Captures optimal marketing frequency (engagement without fatigue)
    df['contact_intensity'] = df['campaign'] / (df['previous'] + 1)
    
    return df


def get_feature_descriptions():
    """
    Return descriptions of engineered features.
    
    Returns:
    --------
    dict
        Dictionary mapping feature names to their descriptions
    """
    descriptions = {
        'contacted_before': (
            'Binary flag indicating whether customer was contacted '
            'in a previous campaign (pdays != -1)'
        ),
        'contact_intensity': (
            'Ratio of current campaign contacts to previous contact history. '
            'Captures optimal marketing frequency - engagement without fatigue.'
        )
    }
    return descriptions


def validate_engineered_features(df, engineered_features):
    """
    Validate that engineered features were created successfully.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with engineered features
    engineered_features : list
        List of engineered feature names to validate
    
    Returns:
    --------
    bool
        True if all features exist, False otherwise
    """
    missing_features = [f for f in engineered_features if f not in df.columns]
    
    if missing_features:
        print(f"⚠ Missing engineered features: {missing_features}")
        return False
    
    return True


def get_feature_statistics(df, engineered_features):
    """
    Calculate statistics for engineered features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with engineered features
    engineered_features : list
        List of engineered feature names
    
    Returns:
    --------
    pd.DataFrame
        Statistics summary
    """
    stats = df[engineered_features].describe()
    return stats
