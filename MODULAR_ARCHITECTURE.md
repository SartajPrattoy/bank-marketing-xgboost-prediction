# Architecture

ML pipeline extracted into reusable modules in `src/`:

- `data_preprocessing.py` - Load, clean, encode, scale
- `feature_engineering.py` - Create features
- `model_training.py` - Train XGBoost
- `evaluation.py` - Metrics and threshold analysis
- `business_metrics.py` - Cost/profit calculations
- `visualization.py` - Plotting functions

## Quick Reference

**Load and preprocess:**
```python
df = load_and_prepare_data()
numeric_cols, categorical_cols = get_column_types(df)
df = handle_missing_values(df, numeric_cols, categorical_cols)
df_encoded = encode_categorical_features(df, categorical_cols)
X_train, X_test, y_train, y_test = split_data(X, y)
X_train_final, X_test_final, _ = scale_features(X_train, X_test, numeric_cols)
```

**Train and evaluate:**
```python
model = train_xgboost_model(X_train, y_train, X_test, y_test)
y_probs = predict_probabilities(model, X_test)
metrics = calculate_metrics(y_test, y_pred, y_probs)
```

**Business analysis:**
```python
business_df = threshold_business_analysis(y_test, y_probs)
optimal = find_optimal_threshold(y_test, y_probs)
```

**Plotting:**
```python
plot_confusion_matrix(y_true, y_pred)
plot_roc_curve(model, X_test, y_test)
plot_business_impact_across_thresholds(business_df)
```

Each module can be imported independently or composed into workflows.
