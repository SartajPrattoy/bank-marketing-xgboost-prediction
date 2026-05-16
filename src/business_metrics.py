"""
Business Metrics Module

Handles business-focused calculations including costs, revenue, profit, and ROI analysis.
This module bridges ML predictions with real business value.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


# Default Business Constants
DEFAULT_COST_PER_CALL = 8  # £ per outbound call
DEFAULT_REVENUE_PER_SUBSCRIPTION = 120  # £ per successful subscription


def calculate_confusion_matrix_components(y_true, y_pred):
    """
    Extract individual confusion matrix components.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    
    Returns:
    --------
    tuple
        (true_negatives, false_positives, false_negatives, true_positives)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn, fp, fn, tp


def calculate_call_costs(false_positives, true_positives, cost_per_call=DEFAULT_COST_PER_CALL):
    """
    Calculate total cost of marketing calls made.
    
    Parameters:
    -----------
    false_positives : int
        Number of false positive predictions (non-subscribers predicted as subscribers)
    true_positives : int
        Number of true positive predictions (actual subscribers predicted correctly)
    cost_per_call : float, default=8
        Cost per outbound call in £
    
    Returns:
    --------
    float
        Total cost of calls made
    """
    total_calls = false_positives + true_positives
    total_cost = total_calls * cost_per_call
    
    return total_cost


def calculate_subscription_revenue(true_positives, revenue_per_subscription=DEFAULT_REVENUE_PER_SUBSCRIPTION):
    """
    Calculate revenue from successful subscriptions.
    
    Parameters:
    -----------
    true_positives : int
        Number of true positive predictions (successful subscriptions)
    revenue_per_subscription : float, default=120
        Revenue per successful subscription in £
    
    Returns:
    --------
    float
        Total revenue from subscriptions
    """
    total_revenue = true_positives * revenue_per_subscription
    
    return total_revenue


def calculate_profit(true_positives, false_positives, 
                     cost_per_call=DEFAULT_COST_PER_CALL,
                     revenue_per_subscription=DEFAULT_REVENUE_PER_SUBSCRIPTION):
    """
    Calculate net profit from marketing campaign.
    
    Parameters:
    -----------
    true_positives : int
        Number of successful subscriptions
    false_positives : int
        Number of unsuccessful calls (wasted marketing spend)
    cost_per_call : float, default=8
        Cost per outbound call in £
    revenue_per_subscription : float, default=120
        Revenue per successful subscription in £
    
    Returns:
    --------
    float
        Net profit (revenue - costs)
    """
    cost = calculate_call_costs(false_positives, true_positives, cost_per_call)
    revenue = calculate_subscription_revenue(true_positives, revenue_per_subscription)
    profit = revenue - cost
    
    return profit


def analyze_business_impact(y_true, y_pred, 
                            cost_per_call=DEFAULT_COST_PER_CALL,
                            revenue_per_subscription=DEFAULT_REVENUE_PER_SUBSCRIPTION):
    """
    Comprehensive business impact analysis for a given threshold.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    cost_per_call : float, default=8
        Cost per outbound call in £
    revenue_per_subscription : float, default=120
        Revenue per successful subscription in £
    
    Returns:
    --------
    dict
        Business metrics including cost, revenue, profit, ROI
    """
    tn, fp, fn, tp = calculate_confusion_matrix_components(y_true, y_pred)
    
    total_calls = fp + tp
    cost = calculate_call_costs(fp, tp, cost_per_call)
    revenue = calculate_subscription_revenue(tp, revenue_per_subscription)
    profit = revenue - cost
    
    # Calculate ROI (return on investment)
    roi = (profit / cost * 100) if cost > 0 else 0
    
    # Calculate conversion rate (precision)
    conversion_rate = (tp / total_calls * 100) if total_calls > 0 else 0
    
    return {
        'total_calls': total_calls,
        'true_positives': tp,
        'false_positives': fp,
        'cost': cost,
        'revenue': revenue,
        'profit': profit,
        'roi': roi,
        'conversion_rate': conversion_rate
    }


def threshold_business_analysis(y_true, y_probs, thresholds=None,
                                cost_per_call=DEFAULT_COST_PER_CALL,
                                revenue_per_subscription=DEFAULT_REVENUE_PER_SUBSCRIPTION):
    """
    Analyze business metrics across different classification thresholds.
    
    This is the KEY function that enables optimal threshold selection based on profit.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_probs : array-like
        Predicted probabilities
    thresholds : list, optional
        List of thresholds to test (default: [0.3, 0.4, 0.5, 0.6, 0.7])
    cost_per_call : float, default=8
        Cost per outbound call in £
    revenue_per_subscription : float, default=120
        Revenue per successful subscription in £
    
    Returns:
    --------
    pd.DataFrame
        Business metrics for each threshold
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        business_metrics = analyze_business_impact(
            y_true, y_pred,
            cost_per_call=cost_per_call,
            revenue_per_subscription=revenue_per_subscription
        )
        
        business_metrics['threshold'] = threshold
        results.append(business_metrics)
    
    df = pd.DataFrame(results)
    
    # Reorder columns for readability
    column_order = ['threshold', 'total_calls', 'true_positives', 'false_positives',
                    'cost', 'revenue', 'profit', 'roi', 'conversion_rate']
    df = df[column_order]
    
    return df


def find_optimal_threshold(y_true, y_probs, thresholds=None,
                           cost_per_call=DEFAULT_COST_PER_CALL,
                           revenue_per_subscription=DEFAULT_REVENUE_PER_SUBSCRIPTION,
                           optimization_metric='profit'):
    """
    Find the optimal threshold that maximizes business value.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_probs : array-like
        Predicted probabilities
    thresholds : list, optional
        List of thresholds to test
    cost_per_call : float, default=8
        Cost per outbound call in £
    revenue_per_subscription : float, default=120
        Revenue per successful subscription in £
    optimization_metric : str, default='profit'
        Metric to optimize: 'profit', 'roi', 'conversion_rate'
    
    Returns:
    --------
    dict
        Optimal threshold and its business metrics
    """
    business_df = threshold_business_analysis(
        y_true, y_probs, thresholds,
        cost_per_call=cost_per_call,
        revenue_per_subscription=revenue_per_subscription
    )
    
    # Find optimal threshold based on specified metric
    optimal_idx = business_df[optimization_metric].idxmax()
    optimal_row = business_df.loc[optimal_idx]
    
    result = {
        'optimal_threshold': optimal_row['threshold'],
        'metrics': optimal_row.to_dict()
    }
    
    return result


def compare_strategies(y_true_baseline, y_true_model, y_pred_baseline, y_pred_model,
                      cost_per_call=DEFAULT_COST_PER_CALL,
                      revenue_per_subscription=DEFAULT_REVENUE_PER_SUBSCRIPTION):
    """
    Compare business metrics between baseline and model-based strategies.
    
    Parameters:
    -----------
    y_true_baseline : array-like
        True values for baseline strategy
    y_true_model : array-like
        True values for model strategy
    y_pred_baseline : array-like
        Predictions for baseline strategy
    y_pred_model : array-like
        Predictions for model strategy
    cost_per_call : float, default=8
        Cost per outbound call in £
    revenue_per_subscription : float, default=120
        Revenue per successful subscription in £
    
    Returns:
    --------
    pd.DataFrame
        Comparison of strategies
    """
    baseline_impact = analyze_business_impact(
        y_true_baseline, y_pred_baseline,
        cost_per_call=cost_per_call,
        revenue_per_subscription=revenue_per_subscription
    )
    
    model_impact = analyze_business_impact(
        y_true_model, y_pred_model,
        cost_per_call=cost_per_call,
        revenue_per_subscription=revenue_per_subscription
    )
    
    comparison = pd.DataFrame({
        'Baseline': baseline_impact,
        'Model': model_impact
    })
    
    # Calculate improvements
    comparison['Improvement'] = comparison['Model'] - comparison['Baseline']
    comparison['% Change'] = (comparison['Improvement'] / comparison['Baseline'].abs() * 100).round(2)
    
    return comparison


def print_business_summary(business_metrics):
    """
    Print a formatted summary of business impact metrics.
    
    Parameters:
    -----------
    business_metrics : dict
        Dictionary from analyze_business_impact()
    """
    print("=" * 70)
    print("BUSINESS IMPACT ANALYSIS")
    print("=" * 70)
    print(f"Total Calls Made:       {business_metrics['total_calls']:,}")
    print(f"Successful Conversions: {business_metrics['true_positives']:,}")
    print(f"Wasted Calls:           {business_metrics['false_positives']:,}")
    print(f"Conversion Rate:        {business_metrics['conversion_rate']:.2f}%")
    print("-" * 70)
    print(f"Total Cost (£):         £{business_metrics['cost']:,.0f}")
    print(f"Total Revenue (£):      £{business_metrics['revenue']:,.0f}")
    print(f"Net Profit (£):         £{business_metrics['profit']:,.0f}")
    print(f"Return on Investment:   {business_metrics['roi']:.1f}%")
    print("=" * 70)
