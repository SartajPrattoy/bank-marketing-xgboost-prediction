"""
Data Preprocessing Module

Handles data loading, cleaning, feature preparation, and preprocessing pipelines.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_and_prepare_data(as_frame=True, parser='auto'):
    """
    Load the UCI Bank Marketing dataset from OpenML.
    
    Parameters:
    -----------
    as_frame : bool, default=True
        Whether to return data as a DataFrame
    parser : str, default='auto'
        Parser to use for data loading
    
    Returns:
    --------
    pd.DataFrame
        Loaded and preprocessed dataset
    """
    from sklearn.datasets import fetch_openml
    
    # Load dataset
    data = fetch_openml(name='bank-marketing', version=1, as_frame=as_frame, parser=parser)
    df = data.frame.copy()
    
    # Standardize column names to lowercase
    df.columns = [col.lower() for col in df.columns]
    
    # Define column mapping from OpenML variable names
    column_map = {
        'v1': 'age', 'v2': 'job', 'v3': 'marital', 'v4': 'education',
        'v5': 'default', 'v6': 'balance', 'v7': 'housing', 'v8': 'loan',
        'v9': 'contact', 'v10': 'day', 'v11': 'month', 'v12': 'duration',
        'v13': 'campaign', 'v14': 'pdays', 'v15': 'previous', 'v16': 'poutcome',
        'class': 'target_raw'
    }
    
    # Rename columns
    df = df.rename(columns=column_map)
    
    # Encode target: 1 = subscribed, 0 = did not subscribe
    df['target'] = (df['target_raw'].astype(str) == '1').astype(int)
    df = df.drop(columns=['target_raw'])
    
    return df


def handle_missing_values(df, numeric_cols, categorical_cols):
    """
    Handle missing values in numeric and categorical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numeric_cols : list
        List of numeric column names
    categorical_cols : list
        List of categorical column names
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with missing values handled
    """
    df = df.copy()
    
    # Handle numeric columns: fill with median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Handle categorical columns: fill with mode
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df


def get_column_types(df, exclude_target=True):
    """
    Identify numeric and categorical columns in dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    exclude_target : bool, default=True
        Whether to exclude 'target' column from numeric columns
    
    Returns:
    --------
    tuple
        (numeric_cols, categorical_cols)
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if exclude_target and 'target' in numeric_cols:
        numeric_cols.remove('target')
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return numeric_cols, categorical_cols


def encode_categorical_features(df, categorical_cols):
    """
    Encode categorical features using one-hot encoding.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    categorical_cols : list
        List of categorical column names
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with encoded categorical features
    """
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded


def split_data(X, y, test_size=0.20, stratify=True, random_state=42):
    """
    Split data into training and test sets.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features dataframe
    y : pd.Series
        Target series
    test_size : float, default=0.20
        Proportion of data for test set
    stratify : bool, default=True
        Whether to use stratified split
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y if stratify else None,
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def build_preprocessor(numeric_features, categorical_features=None):
    """
    Build a preprocessing pipeline using ColumnTransformer.
    
    Parameters:
    -----------
    numeric_features : list
        List of numeric feature column names
    categorical_features : list, optional
        List of categorical feature column names (if any)
    
    Returns:
    --------
    ColumnTransformer
        Preprocessing pipeline
    """
    # Numeric feature processing: scaling
    numeric_transformer = Pipeline(
        steps=[('scaler', StandardScaler())]
    )
    
    # Build transformers list
    transformers = [('num', numeric_transformer, numeric_features)]
    
    # Add categorical processing if provided
    if categorical_features and len(categorical_features) > 0:
        # Categorical features are already one-hot encoded, so pass through
        categorical_transformer = 'passthrough'
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    # Create ColumnTransformer
    preprocessor = ColumnTransformer(transformers=transformers)
    
    return preprocessor


def scale_features(X_train, X_test, numeric_features):
    """
    Scale numeric features using StandardScaler.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    numeric_features : list
        List of numeric feature columns to scale
    
    Returns:
    --------
    tuple
        (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled = scaler.transform(X_test[numeric_features])
    
    # Convert to dataframes preserving column names
    X_train_scaled_df = pd.DataFrame(
        X_train_scaled,
        columns=numeric_features,
        index=X_train.index
    )
    
    X_test_scaled_df = pd.DataFrame(
        X_test_scaled,
        columns=numeric_features,
        index=X_test.index
    )
    
    # Combine scaled numeric features with remaining features
    X_train_final = pd.concat([
        X_train_scaled_df,
        X_train.drop(columns=numeric_features)
    ], axis=1)
    
    X_test_final = pd.concat([
        X_test_scaled_df,
        X_test.drop(columns=numeric_features)
    ], axis=1)
    
    return X_train_final, X_test_final, scaler
