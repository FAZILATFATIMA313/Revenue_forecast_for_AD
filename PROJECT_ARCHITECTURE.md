# AdRevenue Forecast Studio — End-to-End Architecture (Data → Model → Forecast → Causal Insights)

> This document describes how the full project works, including: data loading, feature engineering, model training, probabilistic prediction, reconciliation, campaign allocation, anomaly grounding, and the causal-layer reasoning pipeline. It also explains the output schema used by the Streamlit dashboard.

---

## 1) Repository Components (High Level)

### Key files
- `app.py`
  - Streamlit UI entrypoint.
  - Orchestrates: feature generation → prediction → campaign allocation → anomaly detection → causal insights → charts.
- `src/data_loader.py`
  - Loads historical campaign dataset(s) from `./data/*.csv`.
  - Normalizes columns (spend/revenue/conversions) into a common schema used by feature generation and anomaly detection.
  - Provides validation helpers (e.g., consistency checks).
- `src/generate_features.py`
  - Converts raw daily campaign rows into a *features dataset* used for training and inference.
  - Produces a row per `(request_id, period_days, channel/budget context)` during requests, and rows aligned with `ref_date` for training.
- `src/features.py`
  - Implements the `FeatureEngineer` logic:
    - Spend/revenue aggregates
    - ROAS-derived features
    - Time indicators (month/quarter/holidays if used)
    - Rolling/period-window targets and cap/clip strategies.
- `src/train.py`
  - Trains probabilistic quantile regression heads (LightGBM quantiles).
  - Produces models for:
    - Total revenue quantiles (e.g., `target_total_revenue_q50`)
    - Per-channel revenue quantiles (google/meta/microsoft)
  - Stores a scaler and expected feature list in the `ForecastingModel`.
- `src/predict.py`
