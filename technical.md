# Technical Documentation (technical.md)

## 1) Repository Architecture (High-Level)

This repo implements **probabilistic revenue + blended ROAS forecasting** for digital marketing across:
- **Google Ads**
- **Meta Ads**
- **Microsoft Ads**

The system is organized as a **data → feature engineering → model → prediction → (optional) app** pipeline.

### Main entrypoints
- **Batch/scoring pipeline (submission)**: `run.sh`
  - Calls: `src/generate_features.py` → writes `features.parquet`
  - Calls: `src/predict.py` → writes predictions CSV (`output/predictions.csv` by default)
- **Interactive demo (Streamlit)**: `app.py` (renders forecast + anomalies + methodology)

---

## 2) End-to-End Flow

### A) Batch pipeline (`run.sh`)
1. **Input data ingestion**
   - `src/generate_features.py` calls `src.data_loader.load_all_data(data_dir)`
   - CSVs under `data/` are loaded and normalized.
2. **Feature generation**
   - `src/generate_features.py`:
     - Aggregates to **daily platform totals**
     - Constructs **one feature row per forecast request** (or default request scenarios)
     - Writes `features.parquet`
3. **Model inference**
   - `src/predict.py`:
     - Loads the pickled model from `pickle/model.pkl`
     - Loads `features.parquet`
     - Aligns feature columns to `model.feature_names`
     - Scales with the model’s `StandardScaler`
     - Predicts **quantiles** (P10/P50/P90) for:
       - total revenue (blended across channels)
       - per-channel revenues
     - Computes **blended ROAS quantiles**
     - Writes output CSV with fixed columns

### B) Training (development only)
- `src/train.py`:
  - loads raw CSVs → normalizes them (via `load_all_data`)
  - builds training dataset with `src/features.py::FeatureEngineer`
  - trains multiple **LightGBM quantile regression models**
  - calibrates interval coverage with a simple **coverage factor**
  - serializes model to `pickle/model.pkl`

### C) Streamlit demo (`app.py`)
- User selects planning period + budgets
- App creates a temporary “forecast requests” CSV
- Runs:
  - `src/generate_features.py` (features)
  - `src/predict.py` (predictions)
- Displays:
  - revenue P10/P50/P90 interval chart
  - per-channel revenue breakdown + ROAS-derived metrics
  - anomaly detection tab using `src/anomaly_detector.py`
  - optional LLM insights via Groq

---

## 3) Data Contracts (Schemas & Column Names)

### 3.1 Normalized raw data schema (`src/data_loader.py`)
After loading and cleaning, all CSVs are normalized to:

`STANDARD_COLS`:
- `date`
- `campaign_id`
- `campaign_name`
- `campaign_type`
- `spend`
- `revenue`
- `clicks`
- `impressions`
- `conversions`
- `platform`  (one of: `google`, `meta`, `microsoft`)

This is the schema consumed by feature engineering.

### 3.2 Feature parquet
- `src/generate_features.py` writes a **single parquet file**: `features.parquet`
- Feature columns are designed to be aligned with the trained model’s:
  - `ForecastingModel.feature_names`

**Key principle:** during inference, `src/predict.py` ensures missing model feature columns are created and filled with `0.0`.

### 3.3 Output CSV columns (`src/predict.py`)
`OUTPUT_COLUMNS` is the fixed final contract:

- `request_id`
- `period_days`
- `spend_google`, `spend_meta`, `spend_ms`
- `revenue_p10`, `revenue_p50`, `revenue_p90`
- `blended_roas_p10`, `blended_roas_p50`, `blended_roas_p90`
- `google_revenue_p10/p50/p90`
- `meta_revenue_p10/p50/p90`
- `ms_revenue_p10/p50/p90`

---

## 4) Per-File Technical Details

## `run.sh`
### Flow
- Parses CLI args:
  - `DATA_DIR` (default `./data`)
  - `MODEL_PATH` (default `./pickle/model.pkl`)
  - `OUTPUT_PATH` (default `./output/predictions.csv`)
- Runs:
  1. `python src/generate_features.py --data-dir "$DATA_DIR" --out features.parquet`
  2. `python src/predict.py --features features.parquet --model "$MODEL_PATH" --output "$OUTPUT_PATH"`

### Technology
- Bash
- Python (auto-detection of `python`/`python3` on PATH)

### Architecture notes
- Produces a stable artifact (`features.parquet`) and then consumes it for scoring.

---

## `README.md`
### Purpose
- Documentation of repo goals, pipeline steps, and output columns.

### Architecture notes
- Defines expected submission behavior and optional Streamlit app usage.

---

## `requirements.txt`
### Purpose
- Pinned dependency list for:
  - ML: `lightgbm`, `scikit-learn`, `joblib`, `pyarrow`
  - Analytics/IO: `pandas`, `numpy`
  - App: `streamlit`, `plotly`
  - Optional: `groq`, `python-dotenv`, `holidays`

---

## `src/config.py`
### Flow/Responsibilities
- Centralizes repo constants and paths:
  - `DATA_DIR`, `PICKLE_DIR`, `OUTPUT_DIR`
- Loads optional env vars:
  - `GROQ_API_KEY`, `GROQ_MODEL`
- Provides defaults:
  - `DEFAULT_PERIODS = [30, 60, 90]`
  - `QUANTILES = [0.1, 0.5, 0.9]`
- LightGBM hyperparameters (`LGBM_PARAMS`)
- Anomaly detection thresholds
- `LOG_LEVEL`

### Technology
- `dotenv` optional import
- `pathlib.Path`

---

## `src/logger.py`
### Flow
- `setup_logger(name)` creates a console logger:
  - Level from `LOG_LEVEL`
  - Stream handler to stdout
  - Formatter includes timestamp, level, logger name, and message

### Technology
- Python `logging`

### Architecture notes
- `if not logger.handlers:` prevents duplicate handlers in re-import contexts.

---

## `src/data_loader.py`
### Flow
1. **CSV loading loop**
   - discovers `*.csv` under the provided data directory
2. **Platform detection**
   - `detect_platform(df)` based on CSV column signatures:
     - Google: `segments_date` + `metrics_cost_micros`
     - Microsoft: `timeperiod`
     - Default: Meta
3. **Normalization**
   - `normalize_google`, `normalize_microsoft`, `normalize_meta`
   - Maps platform-specific columns → `STANDARD_COLS`
   - Converts numeric fields using `pd.to_numeric(..., errors='coerce')`
   - Adds `platform` field.
4. **Validation & cleanup**
   - ensures all `STANDARD_COLS` exist
   - clips negative metrics to 0
   - parses `date` to datetime, drops invalid date rows
   - deduplicates by (`campaign_id`, `date`, `platform`)
5. **Concatenation & final summary**
   - merges all platform data into one dataframe

### Technology
- `pandas`, `numpy`
- `glob` for file discovery

### Architecture notes
- The loader is **schema-driven**: feature engineering depends on standardized column names.

---

## `src/features.py` (`FeatureEngineer`)
### Flow (training feature generation)
`FeatureEngineer.create_training_data(df)` builds a **supervised training set**.

1. Aggregate raw daily campaign data → **daily platform totals**
   - `_aggregate_daily_platform`
2. For each period length in `periods` (default from config / passed in):
   - `_create_period_features(daily_platform, period_days)`
   - For each reference date `ref_date`:
     - Feature window: previous `period_days`
     - Target window: next `period_days`
   - Produces row metadata:
     - `ref_date`, `period_days`, plus feature/target start/end markers
   - Computes:
     - sums/means/counts for spend/revenue/clicks/impressions/conversions
     - ROAS per platform inside feature window
     - blended ROAS
     - spend shares
3. Add time features:
   - `_add_time_features`
   - month/quarter/year, weekend indicator, month start/end dummies,
     quarter and month one-hot indicators
   - `is_holiday` uses `holidays.India` + configured `SHOPPING_EVENTS`
   - trend = years since min ref_date
4. Campaign-type features:
   - `_create_campaign_type_features`
   - Normalizes `campaign_type` to a smaller set (search/pmax/video/display/…)
   - For the feature window only, aggregates spend/revenue by campaign type
5. Merge:
   - `_merge_features` merges period features + campaign-type features
6. Targets:
   - `_create_targets`
   - creates:
     - `target_total_revenue`
     - per-platform target revenues: `target_google_revenue`, etc.
     - derived ROAS targets
     - log-transformed target: `target_log_total_revenue` (present but not directly used elsewhere)
7. Cleaning:
   - `_clean_features`
   - fills NaNs, replaces inf, clips extreme spend/revenue (99.5th percentile)
   - removes rows with zero spend across non-campaign-type spend features

### Technology
- `pandas`, `numpy`
- `holidays` for holiday calendars

### Algorithms / Methods
- Rolling window aggregation for supervised learning
- Time-based seasonal features
- Campaign-type normalization and aggregation

### Architecture notes
- Training produces a dataframe used by `src/train.py`.
- Inference uses `src/generate_features.py` (not this training class) and must match feature column naming.

---

## `src/generate_features.py`
### Flow (inference + scenario generation)
Primary function: `generate_features(data_dir, output_path)`

1. Load normalized historical data:
   - `load_all_data(data_dir)`
2. Aggregate daily platform totals:
   - groupby `['date','platform']` summing spend/revenue/clicks/impressions/conversions
   - pivot so each metric has columns like `spend_google`, `revenue_meta`, etc.
   - adds total metrics: `total_spend`, `total_revenue`, etc.
3. Determine forecast requests:
   - `detect_forecast_requests(data_dir)`
   - priority:
     - `forecast_requests.csv`, else `requests.csv`, else `budget_scenarios.csv`
   - else: computes avg daily spend from last 30 days and creates default scenarios for 30/60/90 days.
4. For each request:
   - set `ref_date = last_date + 1 day` (forecast starts tomorrow)
   - compute feature window:
     - last `period_days` days prior to `ref_date`
   - aggregate over the window to create feature fields:
     - `feature_{metric}_{platform}` sums
     - `feature_daily_avg_{...}` means
     - `feature_days_with_{...}` non-zero counts
   - override spend features with the **requested budgets**
     - `feature_spend_google`, `feature_spend_meta`, `feature_spend_microsoft`
   - compute spend shares and ROAS features:
     - `feature_roas_google`, etc.
     - `feature_blended_roas`
   - time features based on `ref_date`:
     - month/quarter/year/day_of_week/week_of_year/is_weekend
     - month start/end heuristics
     - month and quarter dummy indicators
     - simplified `is_holiday = 0`
   - campaign-type features:
     - normalizes campaign type names via `type_mapping`
     - aggregates spend/revenue by campaign type in the feature window
     - creates:
       - `feature_spend_ctype_{ctype}`
       - `feature_revenue_ctype_{ctype}`
       - `feature_roas_ctype_{ctype}`
5. Save:
   - `features_df.to_parquet(output_path, index=False)`

### Technology
- `pandas`, `numpy`
- `pyarrow` indirectly required by parquet writing

### Algorithms / Methods
- Supervised feature-window aggregation (no targets here)
- Scenario detection / default request generation
- Campaign-type normalization mapping to a reduced ontology

### Architecture notes
- Inference depends on consistent feature naming between `generate_features.py` and `train.py`.
- `src/predict.py` compensates for missing columns using `model.feature_names`.

---

## `src/train.py` (`ForecastingModel`)
### Flow
1. Load raw data:
   - `load_all_data(data_dir)`
2. Build training dataset:
   - `FeatureEngineer(periods=[30,60,90]).create_training_data(df)`
3. Choose available targets:
   - `target_total_revenue`, `target_google_revenue`, `target_meta_revenue`, `target_microsoft_revenue`
4. Choose feature columns:
   - `engineer.get_feature_columns(features)`
5. Train:
   - `ForecastingModel.train(features, target_cols, feature_cols)`
   - optional baseline linear regression model for comparison
6. Calibrate prediction intervals:
   - `_calibrate_intervals`:
     - computes empirical coverage of quantile interval `[q10, q90]`
     - if coverage < 0.75, uses factor:
       - `factor = 1.0 + (0.80 - coverage) * 2`
7. Serialize:
   - `model.save()` uses `joblib.dump(model_data, filepath)`

### Model architecture
- `ForecastingModel` holds:
  - `models` dict containing separate models per `(target, quantile)`
  - one `StandardScaler`
  - `feature_names`
  - `calibration_factors` per target

### Algorithms / Methods
- **LightGBM quantile regression**
  - uses `lgb.LGBMRegressor(objective='quantile', alpha=...)`
  - trains on each target separately
  - uses time-series CV:
    - `TimeSeriesSplit(n_splits=5)`
  - computes **pinball loss** per fold (approx eval)
- **Conformal-style calibration (simplified)**
  - adjusts q10/q90 around median using calibration factor

### Technology
- `lightgbm`, `scikit-learn`
- `joblib`

### Architecture notes
- Output quantiles are modeled as separate estimators.
- `predict.py` expects model keys like:
  - `target_total_revenue_q10/q50/q90`
  - `target_google_revenue_q10/q50/q90`, etc.

---

## `src/predict.py`
### Flow (inference)
Function: `predict(features_path, model_path, output_path)`

1. Load features:
   - `pd.read_parquet(features_path)`
2. Load model:
   - `ForecastingModel.load(model_path)`
3. Align feature columns:
   - for every `col in model.feature_names`:
     - if missing in `X`, add `X[col] = 0`
   - select in correct order `X_features = X[model.feature_names]`
4. Scale:
   - `X_scaled = model.scaler.transform(X_features)`
   - re-wrap as dataframe for convenience
5. For each feature row:
   - read budgets and period_days
   - predict quantiles for:
     - total revenue target
     - google/meta/microsoft revenue targets
   - quantile post-processing:
     - clamp revenue to `>= 0`
   - compute blended ROAS:
     - `blended_roas_q = revenue_q / total_spend`
6. Output formatting:
   - enforce dtypes
   - ensure `OUTPUT_COLUMNS` exist
   - writes CSV with `float_format='%.2f'`

### Algorithms / Methods
- Quantile prediction from separate quantile regressors
- Blended ROAS derived metric

### Technology
- `pandas`, `numpy`
- `joblib` via model loader

---

## `src/validate.py` (`ForecastingValidator`)
### Flow
Provides a suite of robustness checks:

- `validate_data_quality`
- `validate_model_robustness` (basic backtesting over last ~180 days)
- `validate_budget_sensitivity` (scenario sanity checks)
- `validate_seasonality` (month-by-month behavior)
- `validate_edge_cases` (zero/very large/negative budgets, period consistency)
- `validate_prediction_consistency`
  - checks quantile monotonicity
  - rough interval width sanity
- `simulate_real_scenario`
  - compares revenue for two budget allocations

### Technology
- `pandas`, `numpy`

### Architecture notes
- Contains helper `_create_simple_features` for producing a minimal feature vector.
- This file is **not used by `run.sh`** but can be used for development validation.

---

## `src/anomaly_detector.py` (`AnomalyDetector`)
### Flow
`AnomalyDetector.detect_all(df)` runs multiple detectors and returns structured results:

Outputs:
- `spend_outliers` (rolling IQR upper/lower bounds)
- `revenue_outliers` (rolling IQR on revenue, only where revenue > 0)
- `roas_outliers` (ROAS extreme above threshold using IQR/median-based bounds)
- `zero_conversion_spend` (spend above median threshold with conversions==0 and revenue==0)
- `sudden_changes` (day-over-day pct change > threshold for spend/revenue/clicks/impressions)
- `campaign_gaps` (missing days > 3 between successive observations)
- `summary`:
  - totals by anomaly type
  - totals by platform
  - severity bucket

### Algorithms / Methods
- Rolling-window **IQR** outlier detection (`ANOMALY_WINDOW_DAYS`)
- ROAS anomaly bound:
  - `upper_bound = max(q75 + 3*IQR, median*10)`
- Sudden changes via percentage change threshold
- Gap detection via `.diff().dt.days`

### Technology
- `pandas`, `numpy`

### Architecture notes
- Designed for UI consumption:
  - Streamlit renders subsets and summary metrics.

---

## `src/analyze_campaigns.py` (`CampaignAnalyzer`)
### Flow
Exploratory analysis utilities for ad campaign datasets:
- computes derived metrics: CTR, conversion rate, ROAS, CPC, CPM
- provides:
  - platform summary
  - campaign-type performance
  - top-performing campaigns
  - time series trends (daily + weekly)
  - distribution/percentiles
  - segment analysis and generated “insights”

### Technology
- `pandas`, `numpy`
- plotting imports exist but the functions primarily return data structures

### Architecture notes
- Standalone analysis tool; not integrated into the scoring pipeline.

---

## `app.py` (Streamlit Demo)
### Flow
Implements a multi-tab dashboard:
- **Forecast tab**:
  - user picks horizon + budgets
  - model loaded from `pickle/model.pkl`
  - builds temp “forecast_requests.csv”
  - runs:
    - `src.generate_features.generate_features`
    - `src.predict.predict`
  - renders:
    - revenue interval bar chart (P10–P90 base, P50 marker)
    - channel-level breakdown (P10/P50/P90 deltas + ROAS captions)
    - blended ROAS gauge
  - optional LLM insights via Groq (guarded by availability + `GROQ_API_KEY`)
- **Data Explorer tab**:
  - filters by platform + campaign_type
  - scatter plot spend vs revenue (or conversions fallback)
- **Anomalies tab**:
  - uses `AnomalyDetector.detect_all(df)`
  - renders charts and tables for outliers + gaps
- **Methodology tab**:
  - shows methodology summary text
  - displays top feature importances (if model loaded and feature importance available)

### Technology
- `streamlit`
- `plotly.graph_objects` / `plotly.express`
- optional `groq`

### Architecture notes
- Uses Streamlit caching:
  - `@st.cache_data` for loaded data
  - `@st.cache_resource` for model

---

## 5) Cross-Cutting Architecture Considerations

### Feature/Model alignment dependency
- Training uses `src/features.py` to generate training matrices.
- Inference uses `src/generate_features.py` to generate feature vectors.
- `src/predict.py` makes inference robust by:
  - ensuring all required `model.feature_names` exist
  - filling missing features with 0

### Quantile model design
- Uses three quantile levels (P10/P50/P90) for each target.
- Each quantile is a separate LightGBM regressor.

### Interval calibration
- Implements a simplified empirical coverage adjustment:
  - measures actual coverage of `[P10, P90]` on calibration data
  - widens/narrows quantile outputs around P50 using a calibration factor

### UI anomaly detection
- Rolling IQR, threshold rules, and gap detection produce a human-readable anomaly summary.
