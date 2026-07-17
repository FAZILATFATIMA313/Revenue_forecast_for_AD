# AdRevenue Forecast Studio — Full Project Working (Data → Features → Model → Prediction → Output)

This document describes the end-to-end architecture and working of the whole project, including:
- data ingestion and normalization
- feature engineering (training vs inference)
- modeling technique (probabilistic quantile regression)
- prediction and output schema
- reconciliation/bootstrap logic (channel → blended totals)
- anomaly detection + causal reasoning layer (Groq optional)
- repository execution entrypoints and contracts

---

## 0) High-level system design (architecture)

The project is an end-to-end pipeline built around a single repeated pattern:

**Data → Normalization → Feature Engineering → Probabilistic Forecasting → Output Contract → (Optional) Insights**

### Primary execution paths
1. **Batch scoring / submission pipeline** (required by `run.sh`)
   - `run.sh`
     1. `src/generate_features.py`
     2. `src/predict.py`
2. **Interactive demo (Streamlit)**
   - `app.py`
     - loads data (via `src.data_loader.load_all_data`)
     - generates a temporary forecast request CSV
     - runs `src/generate_features.generate_features`
     - runs `src.predict.predict`
     - shows anomalies (`src.anomaly_detector`)
     - optionally generates causal insights (`src.causal_layer`, Groq-based if configured)

3. **Training (development only)**
   - `src/train.py`
     - loads data
     - generates supervised training dataset (`src/features.py::FeatureEngineer`)
     - trains multiple LightGBM quantile regressors
     - calibrates intervals
     - saves model artifact (`pickle/model.pkl`)

---

## 1) Data ingestion & normalization (`src/data_loader.py`)

### 1.1 Inputs
At runtime, the system reads **all CSV files** under `data/` (or a provided `--data-dir`):
- `load_all_data(data_dir)` finds `data_dir/*.csv`

### 1.2 Platform detection
`detect_platform(df)` uses CSV column signatures:
- **Google**: `segments_date` + `metrics_cost_micros`
- **Microsoft**: `timeperiod`
- else default to **Meta**

### 1.3 Normalization into a standard schema
All platforms are normalized into the same canonical columns:

**`STANDARD_COLS`**
- `date`
- `campaign_id`
- `campaign_name`
- `campaign_type`
- `spend`
- `revenue`
- `clicks`
- `impressions`
- `conversions`
- `platform`  (`google`, `meta`, `microsoft`)

Normalization details:
- **Google**
  - converts `metrics_cost_micros` → `spend` by dividing by `1_000_000`
  - maps campaign fields and metrics into standard columns
- **Microsoft**
  - maps `timeperiod → date`, `campaignid → campaign_id`, etc.
- **Meta**
  - finds likely date/campaign id/campaign name columns by pattern
  - maps metric patterns (e.g., conversion-related columns)
  - fills missing `clicks/impressions/conversions` with zero

### 1.4 Validation and cleanup
`validate_and_clean(df)`:
- ensures all `STANDARD_COLS` exist (fills missing ones with defaults)
- converts numeric columns via `pd.to_numeric(..., errors='coerce').fillna(0)`
- clips negative spend/revenue/clicks/etc to `0`
- parses `date` to datetime and drops invalid date rows
- removes duplicates by:
  - (`campaign_id`, `date`, `platform`)

### 1.5 Campaign consistency validation (explicit ingestion requirement)
`validate_campaign_consistency(df)` performs:
- missing campaign_id/campaign_type checks
- inconsistent campaign naming across time (per `platform + campaign_id`)
- inconsistent campaign_type across time
- sudden platform reassignment (same `campaign_id` appearing on multiple platforms)
- daily date-range gaps per campaign (within each platform)

This function logs a dedicated section:
- **“?? CAMPAIGN CONSISTENCY VALIDATION”**

The Streamlit Data Explorer tab surfaces these results.

---

## 2) Feature engineering (training vs inference)

Important: **training features** and **inference features** must be consistent in naming with what the model expects. The project uses two separate pipelines:

- Training supervised feature generator:
  - `src/features.py::FeatureEngineer`
- Inference/scoring feature generator:
  - `src/generate_features.py::generate_features`

Both produce feature vectors aligned through the model’s `feature_names` list.

---

## 3) Inference feature generation (`src/generate_features.py`)

### 3.1 Inputs to inference feature generation
- loads normalized historical data via `load_all_data(data_dir)`
- determines forecast requests:
  - prefers `forecast_requests.csv`
  - else `requests.csv`
  - else `budget_scenarios.csv`
  - else generates default scenarios for 30/60/90 using the last 30 days avg spend

A forecast request row contains:
- `request_id`
- `period_days`
- `spend_google`
- `spend_meta`
- `spend_ms`

### 3.2 Aggregation to daily platform totals
It aggregates raw daily campaign rows:
- group by `['date','platform']`
- sum spend/revenue/clicks/impressions/conversions
- pivot to columns like:
  - `spend_google`, `revenue_meta`, etc.

Then adds totals:
- `total_spend`, `total_revenue`, etc.

### 3.3 Construct one feature row per forecast request
For each request:
- set `ref_date = last_date + 1 day` (forecast starts tomorrow)
- compute a **feature window** of length `period_days`:
  - `[ref_date - period_days, ref_date - 1]`
- create a row with:
  - metadata: `request_id`, `period_days`, `ref_date`
  - budget overrides:
    - `feature_spend_google/meta/microsoft`
  - spend totals and shares:
    - `feature_total_spend`
    - `feature_spend_share_{google/meta/microsoft}`
  - ROAS features for each channel based on history window:
    - `feature_roas_{google/meta/microsoft} = feature_revenue_x / feature_spend_x` (0 if spend=0)
  - blended ROAS in the feature window:
    - `feature_blended_roas = feature_total_revenue / feature_total_spend`
  - rolling window aggregates for all metrics:
    - `feature_{metric_platform}` sums across the window
    - `feature_daily_avg_{...}` mean across the window
    - `feature_days_with_{...}` count of non-zero days
  - time features from `ref_date`:
    - `month`, `quarter`, `year`, `day_of_month`, `day_of_week`
    - `week_of_year`, `is_weekend`
    - `is_month_start`, `is_month_end`
    - month/quarter one-hot dummies: `is_month_{m}`, `is_q{q}`
    - `is_holiday` based on `SHOPPING_EVENTS` membership
    - `weekend_ratio` and trend-like numeric features

  - campaign-type features:
    - normalizes campaign types via mapping to reduced ontology:
      - `SEARCH→search`, `PERFORMANCE_MAX→pmax`, `SHOPPING→shopping`, etc.
    - aggregates spend/revenue over the feature window by `campaign_type_normalized`
    - outputs:
      - `feature_spend_ctype_{ctype}`
      - `feature_revenue_ctype_{ctype}`
      - `feature_roas_ctype_{ctype}`

### 3.4 Output artifact
Writes:
- `features.parquet` (or CSV fallback if parquet engine missing)

---

## 4) Training supervised feature engineering (`src/features.py`)

Training uses `FeatureEngineer(periods=[30,60,90]).create_training_data(df)` which builds a supervised dataset.

### 4.1 Training example generation strategy
For each period length `period_days`:
- choose a reference date `ref_date`
- define:
  - **feature window**: previous `period_days` days `[ref_date - period_days, ref_date - 1]`
  - **target window**: future `period_days` days `[ref_date, ref_date + period_days - 1]`

The row includes:
- feature aggregates from the feature window:
  - platform totals (spend/revenue/clicks/etc)
  - channel shares and ROAS derived features
- target sums over the future window:
  - `target_total_revenue`, `target_{plat}_revenue`, etc.
- derived targets:
  - `target_blended_roas = target_total_revenue / target_total_spend`
  - `target_{plat}_roas` similarly
- time features (based on `ref_date`)
- campaign-type features aggregated in the feature window

### 4.2 Holiday and shopping events
- `holidays` is optional. If unavailable, holiday features do not break execution.
- `is_holiday` uses `SHOPPING_EVENTS` mapping in this codebase.

### 4.3 Cleaning
`_clean_features()`:
- replaces inf with NaN and fills NaNs with 0 for numeric columns
- caps extreme values (feature spend/revenue) above 99.5th percentile
- removes rows where spend features indicate no activity

---

## 5) Modeling technique: probabilistic quantile regression (`src/train.py`)

### 5.1 Model architecture
`ForecastingModel` trains multiple independent LightGBM regressors for quantiles.

- Uses a `StandardScaler` for feature scaling.
- Holds:
  - `models`: mapping of keys to trained LightGBM models
    - key pattern: `{target}_q{int(alpha*100)}`
  - `feature_names`: list of columns used for training
  - `calibration_factors`: per target interval widening factor
  - `metadata` stored in the pickled model artifact

### 5.2 Targets trained
Training prepares these targets (if present):
- `target_total_revenue`
- `target_google_revenue`
- `target_meta_revenue`
- `target_microsoft_revenue`

For each target and each quantile in:
- `QUANTILES = [0.1, 0.5, 0.9]`

It trains a LightGBM regressor with:
- `objective='quantile'`
- `alpha={quantile}`

### 5.3 Time-series cross validation
Uses:
- `TimeSeriesSplit(n_splits=5)`

Evaluation metric:
- pinball loss approximation computed on validation predictions.

### 5.4 Interval calibration (simplified)
`_calibrate_intervals()`:
- checks empirical coverage of the raw quantile interval `[q10, q90]`
- if coverage < 0.75, applies an expansion factor:
  - `factor = 1.0 + (0.80 - coverage) * 2`
- stores `calibration_factors[target]`

During prediction, `p10/p90` are expanded around the median `p50` using this factor.

### 5.5 Baseline linear regression (diagnostic)
A `LinearRegression` baseline is trained for total revenue to help sanity-check predictability (stored as `baseline_model`).

### 5.6 Serialization artifact
`model.save()`:
- uses `joblib.dump(model_data, filepath, compress=3)`
- saved to:
  - `pickle/model.pkl` under `PICKLE_DIR`

---

## 6) Prediction and output generation (`src/predict.py`)

### 6.1 Inputs
`predict(features_path, model_path, output_path)`:
- reads features from parquet or CSV
- loads model:
  - `ForecastingModel.load(model_path)`

### 6.2 Feature alignment
Prediction must use exactly the training feature list:
- For each `col in model.feature_names`:
  - if missing in input features:
    - add column filled with `0`
- selects:
  - `X_features = X[model.feature_names]`
- scales:
  - `model.scaler.transform(X_features)`

### 6.3 Output contract
The output CSV contract is fixed to `OUTPUT_COLUMNS`:

- `request_id`, `period_days`
- budgets: `spend_google`, `spend_meta`, `spend_ms`
- blended revenue quantiles:
  - `revenue_p10`, `revenue_p50`, `revenue_p90`
- blended ROAS quantiles:
  - `blended_roas_p10/p50/p90`
- per-channel revenue quantiles:
  - `google_revenue_p10/p50/p90`
  - `meta_revenue_p10/p50/p90`
  - `ms_revenue_p10/p50/p90`

### 6.4 Channel → blended reconciliation bootstrap (key technique)
The project supports a redesigned reconciliation approach:

Function:
- `_bootstrap_reconciled_blended_quantiles(...)`

Goal:
- produce **coherent quantiles** such that blended totals are derived from a **joint bootstrap** of channel revenues.

Core steps:
1. build training-like features from history via `FeatureEngineer(...).create_training_data(df_hist)`
2. filter to the requested `period_days`
3. compute residuals on historical windows for each channel using the trained **q50 heads**:
   - `resid = y_actual - y_pred_q50`
4. for a request:
   - predict channel q50 values (`g_req_p50`, `m_req_p50`, `ms_req_p50`)
5. bootstrap residuals with a **single set of bootstrap indices shared across channels**:
   - `idx = rng.integers(0, n, size=n_boot)`
   - `boot_g = g_req_p50 + g_resid[idx]`, similarly for meta and ms
6. compute quantiles from bootstrap samples:
   - per-channel: quantiles of `boot_g`, `boot_m`, `boot_ms`
   - blended total: quantiles of `boot_total = boot_g + boot_m + boot_ms`

This produces:
- `google_revenue_p10/p50/p90`
- `meta_revenue_p10/p50/p90`
- `ms_revenue_p10/p50/p90`
- `revenue_p10/p50/p90`

Safety:
- values are clipped at `>= 0`

Fallback:
- if reconciliation fails, it falls back to legacy independent quantile heads:
  - predicts `target_total_revenue_q10/q50/q90`
  - predicts each channel quantile independently
- then computes blended ROAS.

### 6.5 Blended ROAS derivation
After revenue quantiles are ready:
- `blended_roas_q = revenue_q / (spend_google + spend_meta + spend_ms)` if spend > 0 else 0

### 6.6 Output formatting and type enforcement
Before saving:
- coerces numeric columns to numeric
- ensures required columns exist
- writes CSV with `float_format='%.2f'`
- ensures `request_id` stays string and `period_days` is int

---

## 7) Campaign-level allocation (Streamlit enhancement)

While not required for submission output CSV, the Streamlit app provides:
- `allocate_campaign_level_from_history(...)`

It:
1. computes historical trailing spend shares per campaign (top N + other)
2. allocates predicted platform revenue quantiles down to campaigns proportionally to those spend shares
3. computes deterministic spend allocation and campaign-level ROAS.

This is a “no retraining” allocation method.

---

## 8) Anomaly detection (`src/anomaly_detector.py`)

The system grounds causal explanations using anomalies computed from historical data.

`AnomalyDetector.detect_all(df)` runs:

1. **Spend outliers**
   - rolling window IQR bounds (window size `ANOMALY_WINDOW_DAYS`)
   - flags:
     - high spend (above upper bound)
     - low spend (below lower bound but > 0)

2. **Revenue outliers**
   - rolling IQR upper bound on revenue

3. **ROAS outliers**
   - computes `roas = revenue / spend`
   - flags ROAS extreme using:
     - `upper_bound = max(q75 + 3*IQR, median*10)`

4. **Zero-conversion spend days**
   - spend above median threshold but conversions==0 and revenue==0

5. **Sudden changes**
   - percentage change > threshold in:
     - spend, revenue, clicks, impressions

6. **Campaign gaps**
   - if daily gap between successive campaign observations > 3 days

It returns:
- detailed tables:
  - `spend_outliers`, `revenue_outliers`, `roas_outliers`, etc.
- and a compact `summary`:
  - `severity` bucket based on total anomalies count
  - `top_issues` derived from which anomaly types are most prevalent
  - `by_type` and `by_platform`

---

## 9) AI causal insights layer (`src/causal_layer.py`)

This layer produces an explanation payload grounded in:
- forecast quantiles (revenue + blended ROAS)
- feature importances from the trained model
- anomaly summary/evidence records
- period-over-period deltas (budget vs recent baseline spend)

### 9.1 What is extracted
- `extract_forecast_quantiles(pred_row)`
  - returns minimal quantiles used for causal attribution:
    - revenue p10/p50/p90
    - blended_roas p10/p50/p90
    - spend inputs for each channel

- `extract_feature_importances(model)`
  - reads LightGBM feature importances from:
    - `model.models["target_total_revenue_q50"]`
  - returns top-N feature names and importance values

- `_summarize_anomalies_for_grounding(anomaly_results)`
  - compresses anomaly tables into evidence records:
    - date/platform/campaign_name/metric deviation fields (up to a max rows per type)
  - includes:
    - `anomalies.summary`
    - `evidence_records` for each anomaly type

### 9.2 Prompting and output schema enforcement
The Groq path calls an LLM with a strict anti-hallucination instruction:
- “use ONLY provided structured JSON payload”
- “Every bullet MUST cite at least one provided signal”
- expects EXACT JSON output with:
  - `causal_attribution_bullets` (3-4 string bullets)
  - `risk_flags` array (risk, severity, evidence)
  - `used_signals` (feature drivers, anomalies, delta keys)
  - `llm_used=true`

### 9.3 Deterministic fallback
If Groq is unavailable:
- `_rule_engine_causal_reasoning(payload)` runs
- still returns the same schema:
  - bullets and risk flags are generated from:
    - anomaly summary `top_issues`
    - feature driver names
    - delta keys
- sets `llm_used=false`

---

## 10) Repository entrypoints & working order

### 10.1 Submission pipeline: `run.sh`
`run.sh` is the official batch entrypoint:
1. sets defaults:
   - `DATA_DIR=./data`
   - `MODEL_PATH=./pickle/model.pkl`
   - `OUTPUT_PATH=./output/predictions.csv`
2. runs:
   - `python src/generate_features.py --data-dir "$DATA_DIR" --out features.csv`
   - `python src/predict.py --features "$FEATURES_PATH" --model "$MODEL_PATH" --output "$OUTPUT_PATH"`

Outputs:
- `output/predictions.csv`

### 10.2 Streamlit demo: `app.py`
When user clicks “Generate Forecast”:
1. creates a temporary `forecast_requests.csv`
2. copies `./data/*.csv` into a temp dir
3. runs:
   - `src.generate_features.generate_features(data_tmp, features_path)`
   - `src.predict.predict(features_path, model.pkl, predictions.csv)`
4. reads the first row of predictions
5. renders:
   - revenue P10/P50/P90 and ROAS
   - channel breakdown
6. optional:
   - anomalies computed from full loaded history
   - causal insights via `src.causal_layer.generate_causal_outputs`

---

## 11) Output correctness and contracts (what consumers rely on)

### 11.1 The submission consumer contract
The system is designed so that `output/predictions.csv` always contains:
- stable columns listed in `src/predict.py::OUTPUT_COLUMNS`
- numeric quantiles for revenue and derived blended ROAS
- budgets repeated for context

### 11.2 Key dependency: feature name alignment
Inference is made robust by:
- `src/predict.py` adding missing model feature columns with `0`
- selecting features in the exact `model.feature_names` order

---

## 12) Summary of algorithms / techniques used

1. **Schema-driven ingestion**
   - auto platform detection + normalization
2. **Training supervised time-window aggregation**
   - features from previous `period_days`
   - targets from next `period_days`
3. **Probabilistic forecasting**
   - LightGBM quantile regression (P10/P50/P90)
4. **Interval calibration**
   - coverage-factor expansion of P10/P90 around P50
5. **Empirical bootstrap reconciliation**
   - joint residual bootstrap using shared indices across channels
   - produces coherent blended totals from channel quantiles
6. **Anomaly grounding**
   - rolling IQR outlier detection + ROAS bounds + gap detection
7. **Causal explanation layer**
   - Groq-based structured causal bullets with grounding constraints
   - deterministic fallback rule-engine

---

## 13) Where to look in code for each stage

- Data ingestion & normalization
  - `src/data_loader.py`
- Training feature engineering
  - `src/features.py`
- Inference feature generation
  - `src/generate_features.py`
- Model training and serialization
  - `src/train.py`
- Inference + output CSV creation and reconciliation
  - `src/predict.py`
- Anomaly detection
  - `src/anomaly_detector.py`
- Causal insights layer (Groq optional)
  - `src/causal_layer.py`
- Execution entrypoint
  - `run.sh`
- Demo UI
  - `app.py`
