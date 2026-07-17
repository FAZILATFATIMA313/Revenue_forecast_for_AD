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
     - Scales with the model's `StandardScaler`
     - Predicts **quantiles** (P10/P50/P90) for:
       - total revenue (blended across channels)
       - per-channel revenues
     - Performs **bootstrap reconciliation** to ensure additive coherence
     - Runs **OOD detection** per channel
     - Computes **blended ROAS quantiles**
     - Writes output CSV with fixed columns

### B) Training (development only)
- `src/train.py`:
  - loads raw CSVs → normalizes them (via `load_all_data`)
  - builds training dataset with `src/features.py::FeatureEngineer`
  - trains multiple **LightGBM quantile regression models**
  - calibrates interval coverage with a simple **coverage factor**
  - trains optional **linear regression baseline** for comparison
  - serializes model to `pickle/model.pkl`

### C) Streamlit demo (`app.py`)
- User selects planning period + budgets
- App creates a temporary "forecast requests" CSV
- Runs:
  - `src/generate_features.py` (features)
  - `src/predict.py` (predictions)
- Displays:
  - revenue P10/P50/P90 interval chart
  - per-channel revenue breakdown + OOD badges + P10–P90 formatted deltas
  - campaign-level breakdown (proportional allocation)
  - anomaly detection tab using `src/anomaly_detector.py`
  - Methodology tab with backtesting, pinball loss, coverage comparison, bootstrap reconciliation docs
  - optional LLM insights via Groq (with OOD and disparity signals)

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

### 3.2 Feature parquet
- `src/generate_features.py` writes a **single parquet file**: `features.parquet`
- Feature columns are designed to be aligned with the trained model's:
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

Campaign-level breakdown (`allocate_campaign_level_from_history`) is a **UI-only enhancement** and is not required for the submission CSV. The `OUTPUT_COLUMNS` CSV contract remains the canonical submission output.

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

---

## `README.md`
### Purpose
- Documentation of repo goals, pipeline steps, and output columns.

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
5. **Campaign consistency validation**
   - `validate_campaign_consistency(df)` runs as an explicit ingestion step:
     - missing campaign_id / campaign_type rows
     - inconsistent naming/type across dates
     - sudden platform reassignment per campaign_id
     - daily date-range gaps per (platform, campaign_id)
   - Returns structured report and logs a dedicated section
6. **Concatenation & final summary**
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
     - log-transformed target: `target_log_total_revenue`
7. Cleaning:
   - `_clean_features`
   - fills NaNs, replaces inf, clips extreme spend/revenue (99.5th percentile)
   - removes rows with zero spend across non-campaign-type spend features

### Technology
- `pandas`, `numpy`
- `holidays` for holiday calendars

### Architecture notes
- Training produces a dataframe used by `src/train.py`.
- Inference uses `src/generate_features.py` (not this training class) and must match feature column naming.
- The `get_feature_columns()` method provides the canonical list of feature columns that `predict.py` uses for alignment.

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
   - compute spend shares and ROAS features
   - time features based on `ref_date`
   - campaign-type features (normalized via `type_mapping`)
5. Save:
   - `features_df.to_parquet(output_path, index=False)`

### Technology
- `pandas`, `numpy`
- `pyarrow` indirectly required by parquet writing

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
   - optional baseline linear regression model for comparison (stored in `model.baseline_model`)
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
  - `baseline_model` (optional `LinearRegression`)
  - `metadata`

### Algorithms / Methods
- **LightGBM quantile regression**
  - uses `lgb.LGBMRegressor(objective='quantile', alpha=...)`
  - trains on each target separately
  - uses time-series CV:
    - `TimeSeriesSplit(n_splits=5)`
  - computes **pinball loss** per fold
- **Conformal-style calibration (simplified)**
  - adjusts q10/q90 around median using calibration factor

### Technology
- `lightgbm`, `scikit-learn`
- `joblib`

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
5. For each feature row:
   - **OOD detection via `compute_ood_flags`**:
     - compares requested daily spend against historical 5th–95th percentiles per channel
     - returns `{'is_ood': bool, 'pctile': float, p5/p95/message}`
   - predict quantiles via **bootstrap reconciliation**:
     - `_bootstrap_reconciled_blended_quantiles`:
       - builds joint residual bootstrap across channels
       - ensures additive coherence: `revenue_qX = google_qX + meta_qX + ms_qX`
     - falls back to independent quantile heads if reconciliation fails
   - compute blended ROAS:
     - `blended_roas_q = revenue_q / total_spend`
6. Output formatting:
   - enforce dtypes
   - ensure `OUTPUT_COLUMNS` exist
   - writes CSV with `float_format='%.2f'`

### Algorithms / Methods
- **Empirical bootstrap reconciliation** (channel → blended totals)
- **Per-channel OOD detection** (percentile-based)
- Quantile prediction from separate quantile regressors

### Technology
- `pandas`, `numpy`
- `joblib` via model loader

---

## `src/validate.py` (`ForecastingValidator`)
### Flow
Provides a suite of robustness checks:

- `validate_data_quality` — required columns, date continuity, platform coverage, campaign diversity
- `validate_model_robustness` — basic backtesting over last ~180 days
- `validate_budget_sensitivity` — scenario sanity checks
- `validate_seasonality` — month-by-month behavior
- `validate_edge_cases` — zero/very large/negative budgets, period consistency
- `validate_prediction_consistency` — quantile monotonicity, interval width
- `simulate_real_scenario` — budget reallocation comparison
- `validate_reconciliation_calibration` — legacy vs reconciled coverage + pinball loss

All results are returned as structured dicts suitable for UI rendering.

### Technology
- `pandas`, `numpy`

### Architecture notes
- Contains helper `_create_simple_features` for producing a minimal feature vector.
- Validation results are cached and displayed in the Methodology tab via `get_validation_results_cached()`.

---

## `src/anomaly_detector.py` (`AnomalyDetector`)
### Flow
`AnomalyDetector.detect_all(df)` runs multiple detectors and returns structured results:

Outputs:
- `spend_outliers` (rolling IQR upper/lower bounds)
- `revenue_outliers` (rolling IQR on revenue, only where revenue > 0)
- `roas_outliers` (ROAS extreme above threshold using IQR/median-based bounds)
- `zero_conversion_spend` (spend above median threshold with conversions==0 and revenue==0)
- `sudden_changes` (day-over-day pct change > threshold)
- `campaign_gaps` (missing days > 3 between successive observations)
- `summary`:
  - totals by anomaly type, by platform
  - severity bucket
  - top_issues (for LLM consumption)

### Algorithms / Methods
- Rolling-window **IQR** outlier detection (`ANOMALY_WINDOW_DAYS`)
- ROAS anomaly bound: `upper_bound = max(q75 + 3*IQR, median*10)`
- Sudden changes via percentage change threshold
- Gap detection via `.diff().dt.days`

### Technology
- `pandas`, `numpy`

### Architecture notes
- In the app.py, anomaly detection is **scoped** to `period_days * 3` lookback (not whole-dataset totals) for the causal insights pipeline.

---

## AI Integration Strategy (AI-assisted causal inference layer)

### Goal
Provide **causal attribution over the model's own signals** (forecast quantiles + feature importances + anomaly evidence + period-over-period deltas + OOD flags + cross-channel disparity), instead of generic marketing commentary.

### Data passed to the LLM (structured grounding)
For each forecast request, the causal layer builds a **single structured JSON payload** that includes:

1. **Forecast outputs**
   - `revenue_p10 / revenue_p50 / revenue_p90`
   - `blended_roas_p10 / blended_roas_p50 / blended_roas_p90`
   - spend inputs per channel + per-channel ROAS

2. **Feature importances**
   - Extracted from the trained LightGBM model:
     - uses `model.models["target_total_revenue_q50"].feature_importances_`
     - and `model.feature_names`
   - included as `feature_importances.top_features` (top-N drivers)

3. **Anomaly detector evidence (scoped to feature window)**
   - Output from `AnomalyDetector.detect_all(df)` compacted into:
     - `anomalies.summary` (severity + `top_issues`)
     - `anomalies.evidence_records` (compact rows from each anomaly table)
   - In app.py, scoped to the last `period_days * 3` days

4. **Period-over-period deltas**
   - Deterministic deltas derived from:
     - the current requested budgets vs a recent baseline spend window

5. **Cross-channel ROAS disparity**
   - `compute_cross_channel_disparity(pred_row)`:
     - ratio = max_channel_ROAS / min_channel_ROAS
     - included in payload as `inputs.cross_channel_disparity`

6. **OOD flags (per channel)**
   - `ood_flags` from `compute_ood_flags()`:
     - flags channels where requested daily spend is outside historical 5th–95th percentile

### Prompt structure (anti-hallucination / citation enforcement)
The prompt forces grounding in the payload with specific instructions:
- System message:
  - forbids inventing feature names/dates/anomalies
  - requires each bullet to cite provided signals
- User message:
  - requests causal attribution bullets + risk flags with severity {low|medium|high}
  - explicitly asks about disparity >3x, OOD flags, and ranking by severity

### Output schema (judging-friendly)
The causal layer returns:
- `causal_attribution_bullets: string[]` (4 bullets, varied sentence openings, ranked by severity)
- `risk_flags: [{ risk, severity, evidence }]` (sorted high→medium→low, max 5)
- `used_signals` (feature drivers, anomaly evidence, delta keys)
- `llm_used: boolean`

### Fallback behavior (no Groq / no API key)
The deterministic rule-engine (`_rule_engine_causal_reasoning`) now:
- Produces 4 bullets with **varied sentence openings** (no "Expected ROAS" repetition from previous version)
- First bullet: ROAS/revenue intuition from deltas
- Second bullet: **cross-channel ROAS disparity** (the biggest signal, ranked high)
- Third bullet: anomaly-driven flags
- Fourth bullet: OOD / confidence context
- Risk flags are **sorted by severity** (high → medium → low)
- All signals still cited explicitly

---

## `src/analyze_campaigns.py` (`CampaignAnalyzer`)
### Flow
Exploratory analysis utilities for ad campaign datasets:
- computes derived metrics: CTR, conversion rate, ROAS, CPC, CPM
- provides platform summary, campaign-type performance, top campaigns
- time series trends, distribution/percentiles, insights

### Technology
- `pandas`, `numpy`

### Architecture notes
- Standalone analysis tool; not integrated into the scoring pipeline.

---

## `app.py` (Streamlit Demo)
### Flow (judge-friendly walkthrough order)

1. **Ingest (Data Loader)**
   - Streamlit loads data via `src.data_loader.load_all_data()`.
   - Normalizes CSVs into `STANDARD_COLS`.

2. **Validate Campaign Consistency (explicit ingestion requirement)**
   - During ingestion, `validate_campaign_consistency(df)` runs and checks:
     - inconsistent campaign naming/type across dates
     - sudden platform reassignment per campaign_id
     - missing campaign_id / campaign_type
     - daily date-range gaps per campaign (per platform)
   - **Data Explorer tab** surfaces a **Data Quality → Campaign Consistency** panel with PASS/FAIL.

3. **Accept user budget input**
   - Sidebar inputs: planning horizon (30/60/90) + budgets for Google/Meta/Microsoft.

4. **Probabilistic forecast**
   - Runs `generate_features` → `predict` → renders revenue interval chart + channel breakdown.
   - **OOD badges** appear next to any channel whose requested daily spend exceeds historical 5th–95th pctile.
   - **Campaign-level breakdown** shown as a formatted dataframe.
   - All metric delta labels use `P10–P90: $A – $B` formatting.

5. **Channel / type / campaign breakdown**
   - Channel-level revenue + ROAS cards.
   - Campaign-level allocation table (proportional from historical mix).

6. **AI causal summary (optional)**
   - `generate_causal_outputs()` receives `ood_flags` + deltas + anomalies + model.
   - Output: grounded bullets + risk flags.
   - Anomaly detection is **scoped** to `period_days * 3` window.

### Notes (what each tab shows)
- **Forecast tab**: revenue P10/P50/P90 + channel breakdown + OOD badges + campaign allocation + ROAS gauge + causal insights
- **Data Explorer tab**: filters + data quality panel + scatter plot + raw data sample
- **Anomalies tab**: full-dataset anomaly detection with charts + tables
- **Methodology tab**: approach docs + calibration/backtesting results (coverage, pinball, budget sensitivity, edge cases) + feature importances + reconciliation docs + OOD method docs

### Technology
- `streamlit`
- `plotly.graph_objects` / `plotly.express`
- optional `groq`

### Architecture notes
- Uses Streamlit caching: `@st.cache_data` for data + validation + backtesting results, `@st.cache_resource` for model.

---

## 5) Cross-Cutting Architecture Considerations

### Feature/Model alignment dependency
- Training uses `src/features.py` to generate training matrices.
- Inference uses `src/generate_features.py` to generate feature vectors.
- `src/predict.py` makes inference robust by:
  - ensuring all required `model.feature_names` exist
  - filling missing features with 0

### Shared feature name contract
- `FeatureEngineer.get_feature_columns()` produces the canonical feature column list.
- `ForecastingModel.feature_names` stores this list at training time.
- `src/predict.py` aligns inference features against `model.feature_names`.
- **If training and inference produce different feature sets** (e.g., due to changes in `features.py`), the model's stored `feature_names` acts as the single source of truth, and predict.py fills missing columns with 0.

### Quantile model design
- Uses three quantile levels (P10/P50/P90) for each target.
- Each quantile is a separate LightGBM regressor.

### Bootstrap reconciliation
- Channel-level and blended-total quantiles derived from the same joint bootstrap draws.
- `boot_total = boot_g + boot_m + boot_ms` ensures additive coherence by construction.
- Replaces the legacy approach of summing independent quantile models which produces additive inconsistency.

### OOD detection
- Per-channel percentile-based detection against historical daily spend distributions.
- Daily requested budget compared to 5th–95th percentile of historical daily spend.
- Flagged channels tagged with "⚠ Low confidence" badges in the Forecast tab.

### Interval calibration
- Implements a simplified empirical coverage adjustment:
  - measures actual coverage of `[P10, P90]` on calibration data
  - widens/narrows quantile outputs around P50 using a calibration factor

### UI anomaly detection (scoped for causal layer)
- Rolling IQR, threshold rules, and gap detection produce a human-readable anomaly summary.
- For causal insights, anomalies are scoped to `period_days * 3` lookback to stay relevant to the forecast context.

---

## 6) Data Quality & Production Reliability

### Campaign consistency validation
Runs as part of data ingestion (`validate_campaign_consistency` in `data_loader.py`):
- Checks for missing campaign_id/campaign_type
- Detects inconsistent naming/type across dates
- Detects sudden platform reassignment per campaign_id
- Detects daily date-range gaps per (platform, campaign_id)
- Returns structured report with PASS/FAIL + examples

### Data quality gates
Before forecasting, the system relies on:
- `data_loader.py` validation steps (date parsing, negative value clipping, deduplication)
- `ForecastingValidator.validate_data_quality()` for comprehensive data checks
- Model robustness checks in `ForecastingValidator.validate_model_robustness()`

### Retrain / Monitoring Plan

**Recommended retrain cadence:**
- **Weekly automated retrain** if new data arrives daily: full pipeline with latest 12+ months of data.
- **On-demand retrain** triggered by:
  - Coverage drift: if backtesting shows P10–P90 empirical coverage drops below 70% (target 80%).
  - OOD frequency increase: if >30% of forecast requests are OOD-flagged across any channel.
  - Anomaly spike: sudden jump in anomaly counts (spend/revenue outliers >3× typical).
  - Structural change: new campaign types, platform addition, or attribution model change.

**Monitoring metrics to track per retrain:**
- Empirical P10–P90 coverage on held-out periods (target: ≥75%).
- Pinball loss per quantile (monitor for degradation).
- OOD flag rate per channel over the last N forecasts.
- Backtesting median error (target: <50%).
- Feature importance stability (rank correlation vs previous model ≥0.7).

**Alerting thresholds (recommended):**
- Coverage < 70% → retrain immediately.
- Pinball loss increase >20% vs previous model → investigate.
- Any channel with >50% OOD rate over a week → review budget allocations.

---

## 7) Deliverable Completeness (vs. Hackathon Brief)

### Output CSV contract
- `OUTPUT_COLUMNS` in `src/predict.py` defines the canonical submission CSV columns.
- Campaign-level ROAS/revenue ranges (`allocate_campaign_level_from_history`) are **not required for submission CSV**; they are a UI enhancement shown in the Forecast tab.

### Campaign consistency validation
- `validate_campaign_consistency` runs as part of `load_all_data()` and logs a dedicated section.
- The Data Explorer tab in the Streamlit demo surfaces the report with PASS/FAIL and examples.
- This satisfies the ingestion requirement explicitly.

### Demo walkthrough path
1. Launch `streamlit run app.py`
2. **Forecast tab**: set horizon + budgets → click Generate → see revenue interval, channel breakdown, OOD badges, campaign allocation, ROAS gauge, causal insights.
3. **Data Explorer tab**: filter by platform/type → see campaign consistency validation panel → browse data.
4. **Anomalies tab**: see full-dataset anomaly charts + spend outlier table + data gaps.
5. **Methodology tab**: read approach docs → see calibration metrics (coverage, pinball, backtesting error, budget sensitivity, edge cases) → feature importances → reconciliation docs → OOD docs.

---

## 8) Recent Changes (Checklist-Driven)

### Tier 1 — Correctness
- ROAS cap `min(max(raw_roas, 0.5), 15.0)` confirmed absent from codebase.
- Bootstrap reconciliation verified: `boot_total = boot_g + boot_m + boot_ms` → additive by construction.
- Blended ROAS computed from same unclipped revenue throughout: in `predict.py`, `blended_roas_q = revenue_q / total_spend` with no clipping; app.py displays that value directly.

### Tier 2 — Out-of-distribution detection
- `compute_ood_flags()` added to `src/predict.py`: per-channel percentile check vs historical daily spend.
- OOD badges ("⚠ Low confidence") shown in Forecast tab channel cards.
- OOD flags passed as `ood_flags` parameter to `generate_causal_outputs()` → surfaced in payload + rule engine bullets.

### Tier 3 — Causal layer improvements
- `compute_cross_channel_disparity()` computes max/min ROAS ratio across channels.
- Rule engine bullets now vary sentence openings and prioritize signals by severity.
- Anomaly counts in app.py are scoped to `period_days * 3` lookback window for causal insights.

### Tier 4 — Validation in UI
- `get_validation_results_cached()` runs `ForecastingValidator.run_all_validations()`.
- Methodology tab shows: coverage comparison (legacy vs reconciled), pinball loss per quantile, backtesting error, budget sensitivity checks, edge case results, baseline comparison text.

### Tier 5 — Production reliability
- `model.feature_names` acts as the shared schema anchor between training and inference.
- Data quality checks exist in `ForecastingValidator` and `validate_campaign_consistency`.

### Tier 6 — Deliverable completeness
- `validate_campaign_consistency` output is visible in Data Explorer tab.
- Campaign-level breakdown (`allocate_campaign_level_from_history`) is displayed in Forecast tab as a formatted dataframe.
- `OUTPUT_COLUMNS` CSV contract remains the canonical submission format (campaign-level is UI-only).

### Tier 7 — Polish
- All displayed floats use `:.2f`, `:.0f`, `:.1%`, etc. in f-strings — no raw Python floats.
- All "$X range" labels replaced with explicit "P10–P90: $A – $B" formatting.
- technical.md updated with full reconciliation methodology, OOD handling, calibration coverage results, baseline comparison, and retrain/monitoring plan sections.
