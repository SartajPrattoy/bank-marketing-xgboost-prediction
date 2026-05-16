"""
Model Evaluation Module

Handles evaluation metrics, confusion matrix, ROC analysis, and threshold sensitivity.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
    f1_score
)


def calculate_metrics(y_true, y_pred, y_probs=None):
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    y_probs : array-like, optional
        Predicted probabilities (required for AUC-ROC)
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_probs is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_probs)
    
    return metrics


def get_confusion_matrix(y_true, y_pred):
    """
    Get confusion matrix and its components.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    
    Returns:
    --------
    dict
        Dictionary with confusion matrix components (tn, fp, fn, tp)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'tn': tn,  # True Negatives
        'fp': fp,  # False Positives
        'fn': fn,  # False Negatives
        'tp': tp   # True Positives
    }


def get_roc_curve_data(y_true, y_probs):
    """
    Get ROC curve data.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_probs : array-like
        Predicted probabilities
    
    Returns:
    --------
    tuple
        (fpr, tpr, thresholds, auc)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    
    return fpr, tpr, thresholds, auc


def analyze_threshold_sensitivity(y_true, y_probs, thresholds=None):
    """
    Analyze how metrics vary across different classification thresholds.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_probs : array-like
        Predicted probabilities
    thresholds : list, optional
        List of thresholds to test (default: [0.3, 0.4, 0.5, 0.6, 0.7])
    
    Returns:
    --------
    pd.DataFrame
        Metrics for each threshold
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results.append({
            'Threshold': threshold,
            'Precision': precision,
            'Recall': recall,
            'Accuracy': accuracy,
            'F1': f1
        })
    
    return pd.DataFrame(results)


def generate_classification_report(y_true, y_pred):
    """
    Generate detailed classification report.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    
    Returns:
    --------
    str
        Formatted classification report
    """
    report = classification_report(
        y_true, y_pred,
        target_names=['Not Subscribed', 'Subscribed']
    )
    
    return report


def evaluate_model(y_true, y_pred, y_probs=None):
    """
    Comprehensive model evaluation.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    y_probs : array-like, optional
        Predicted probabilities
    
    Returns:
    --------
    dict
        Complete evaluation results
    """
    results = {
        'metrics': calculate_metrics(y_true, y_pred, y_probs),
        'confusion_matrix': get_confusion_matrix(y_true, y_pred),
        'classification_report': generate_classification_report(y_true, y_pred)
    }
    
    if y_probs is not None:
        results['roc_curve'] = get_roc_curve_data(y_true, y_probs)
        results['threshold_analysis'] = analyze_threshold_sensitivity(y_true, y_probs)
    
    return results


def print_metrics_summary(metrics):
    """
    Print a formatted summary of metrics.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics from calculate_metrics()
    """
    print("=" * 50)
    print("MODEL EVALUATION METRICS")
    print("=" * 50)
    print(f"Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:  {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:     {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1 Score:   {metrics['f1']:.4f}")
    if 'auc_roc' in metrics:
        print(f"AUC-ROC:    {metrics['auc_roc']:.4f}")
    print("=" * 50)
