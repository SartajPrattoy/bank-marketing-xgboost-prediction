# Business Summary

## Objective

Predict likely customer term-deposit subscribers to optimize marketing campaign efficiency and maximize ROI on call costs.

## Key Findings

- XGBoost classifier achieved strong predictive performance on the Bank Marketing dataset
- Threshold optimization significantly improved expected business profit beyond default 0.5 threshold
- Previous customer contact history (pdays, previous) was among the strongest predictive indicators
- SHAP analysis revealed call duration and previous contact outcomes as major drivers of subscription probability
- Contact intensity (engineered feature) captured optimal marketing frequency

## Model Performance

- **Accuracy**: 90.77%
- **Precision**: 93.20%
- **Recall**: 96.59%
- **AUC-ROC**: 0.9352

## Business Impact

**Optimal Threshold**: 0.3

Using the optimized threshold significantly outperforms the default 0.5 threshold:

- **Optimal Profit**: £879,872
- **Default (0.5) Profit**: £859,352
- **Profit Improvement**: £20,520

The optimized threshold reduces unnecessary marketing calls to low-probability customers while improving conversion efficiency. This targeted approach delivers higher ROI on marketing spend through selective outreach to high-probability subscribers.

## Recommendations

1. **Implement Threshold Optimization** - Deploy optimized threshold in production to maximize campaign profit
2. **Prioritize High-Signal Customers** - Focus on customers with longer potential call duration and previous successful interactions
3. **Continuous Monitoring** - Track actual conversion rates and adjust threshold as campaign data evolves
4. **Cost-Benefit Analysis** - Regularly re-evaluate threshold optimization as call costs or subscription revenue changes

## Artifacts

All model outputs, visualizations, and analysis artifacts are included in outputs/:
- **figures/**: Confusion matrix, ROC curve, threshold optimization curves, SHAP explanations
- **reports/**: Classification metrics, threshold analysis, model configuration
