"""
Visualization Module

Handles all plotting and visualization functions for EDA, model evaluation, and business analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def set_plotting_style():
    """Configure matplotlib and seaborn for consistent, professional plots."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")


def plot_class_distribution(df, target_column='target'):
    """
    Visualize the distribution of target classes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str, default='target'
        Name of target column
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    subscription_rate = df[target_column].mean() * 100
    class_counts = df[target_column].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=target_column, data=df, ax=ax, palette=['#FF6B6B', '#4ECDC4'])
    ax.set_title("Customer Subscription Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Subscription Status (0=No, 1=Yes)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xticklabels(['Not Subscribed', 'Subscribed'])
    
    # Add value labels on bars
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height,
                f'{int(height):,}',
                ha="center", fontsize=11)
    
    plt.tight_layout()
    return fig


def plot_subscription_by_category(df, category_column='job', target_column='target'):
    """
    Plot subscription rate by categorical variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    category_column : str, default='job'
        Categorical column to group by
    target_column : str, default='target'
        Target column name
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    rates_sorted = df.groupby(category_column, observed=True)[target_column].mean().sort_values(ascending=False)
    sns.barplot(x=rates_sorted.values, y=rates_sorted.index, ax=ax, palette='viridis')
    ax.set_title(f"Subscription Rate by {category_column.capitalize()}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Subscription Rate", fontsize=12)
    ax.set_ylabel(category_column.capitalize(), fontsize=12)
    
    # Add percentage labels
    for i, v in enumerate(rates_sorted.values):
        ax.text(v + 0.005, i, f'{v*100:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_training_history(history_dict):
    """
    Plot model training progress (AUC over boosting rounds).
    
    Parameters:
    -----------
    history_dict : dict
        Dictionary with 'train_auc', 'test_auc', 'best_iteration' keys
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(history_dict['train_auc'], label='Training AUC', linewidth=2)
    ax.plot(history_dict['test_auc'], label='Validation AUC', linewidth=2)
    ax.axvline(history_dict['best_iteration'], color='red', linestyle='--', linewidth=2,
               label=f'Best Iteration: {history_dict["best_iteration"]}')
    
    ax.set_title('XGBoost Training Progress (AUC Score)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Boosting Rounds', fontsize=12)
    ax.set_ylabel('AUC Score', fontsize=12)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Subscribed', 'Subscribed'])
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    ax.set_title('Confusion Matrix - Subscription Prediction', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_roc_curve(model, X_test, y_test, auc_score=None):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    model : sklearn model
        Trained classifier with predict_proba method
    X_test : pd.DataFrame
        Test features
    y_test : array-like
        Test target values
    auc_score : float, optional
        Pre-computed AUC score to display in title
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, linewidth=2)
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=2, label='Random Classifier')
    
    title = 'Receiver Operating Characteristic (ROC) Curve'
    if auc_score is not None:
        title += f'\nAUC-ROC = {auc_score:.4f}'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df, top_n=20):
    """
    Plot feature importance from model.
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        Dataframe with 'feature' and 'importance' columns (sorted)
    top_n : int, default=20
        Number of top features to display
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = importance_df.head(top_n)
    
    sns.barplot(x='importance', y='feature', data=top_features, ax=ax, palette='viridis')
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_threshold_sensitivity(threshold_df, metrics=['Precision', 'Recall']):
    """
    Plot how metrics vary with classification threshold.
    
    Parameters:
    -----------
    threshold_df : pd.DataFrame
        DataFrame with threshold analysis results
    metrics : list, default=['Precision', 'Recall']
        Metrics to plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for metric in metrics:
        ax.plot(threshold_df['Threshold'], threshold_df[metric], 
                marker='o', linewidth=2, label=metric)
    
    ax.set_title('Threshold Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax.set_xlabel('Classification Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticks(threshold_df['Threshold'].unique())
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_business_impact_across_thresholds(business_df, metric='profit'):
    """
    Plot business metrics (profit, ROI, etc.) across thresholds.
    
    Parameters:
    -----------
    business_df : pd.DataFrame
        Business analysis dataframe from threshold_business_analysis()
    metric : str, default='profit'
        Metric to plot ('profit', 'roi', 'conversion_rate', 'cost', 'revenue')
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(business_df['threshold'], business_df[metric], 
            marker='o', linewidth=2.5, markersize=8, color='#2E86AB')
    
    # Highlight optimal threshold (max value)
    optimal_idx = business_df[metric].idxmax()
    optimal_threshold = business_df.loc[optimal_idx, 'threshold']
    optimal_value = business_df.loc[optimal_idx, metric]
    
    ax.plot(optimal_threshold, optimal_value, 'r*', markersize=20, 
            label=f'Optimal: {optimal_threshold} (£{optimal_value:,.0f})')
    
    ax.set_title(f'Business Impact: {metric.capitalize()} vs Threshold', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Classification Threshold', fontsize=12)
    ax.set_ylabel(f'{metric.capitalize()}', fontsize=12)
    ax.set_xticks(business_df['threshold'].unique())
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_profit_comparison(comparison_df):
    """
    Plot profit comparison between baseline and model strategies.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Comparison dataframe from compare_strategies()
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(1)
    width = 0.35
    
    baseline_profit = comparison_df.loc['profit', 'Baseline']
    model_profit = comparison_df.loc['profit', 'Model']
    
    ax.bar(x - width/2, [baseline_profit], width, label='Baseline', color='#FF6B6B')
    ax.bar(x + width/2, [model_profit], width, label='Model', color='#4ECDC4')
    
    ax.set_title('Profit Comparison: Baseline vs Model Strategy', fontsize=14, fontweight='bold')
    ax.set_ylabel('Profit (£)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Profit'])
    ax.legend(fontsize=11)
    
    # Add value labels on bars
    for rect in ax.patches:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                f'£{int(height):,}',
                ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    return fig
