# Model Artifacts

Serialized machine learning artifacts for the customer subscription prediction project are stored here.

| File | Description |
|---|---|
| `xgboost_model.pkl` | Trained XGBoost classification model |
| `preprocessor.pkl` | Serialized preprocessing artifact used before model inference |
| `model_metadata.json` | Model metadata including evaluation summary and training context |
| `threshold_analysis.csv` | Business threshold analysis used for profit optimization |

These files are generated from the notebook pipeline and can be reused for inference, evaluation, or deployment workflows.
