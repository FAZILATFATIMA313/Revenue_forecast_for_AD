# AdRevenue — Probabilistic Revenue Forecasting

**Probabilistic revenue and ROAS forecasting utility for digital marketing agencies.**

---

## Overview

This repository forecasts **probabilistic revenue and blended ROAS** across **Google Ads**, **Meta Ads**, and **Microsoft Ads** using historical campaign performance data.

## Key Features

- Probabilistic forecasts (P10 / P50 / P90)
- Multi-channel support: Google, Meta, Microsoft
- Multiple horizons: 30 / 60 / 90-day planning periods
- Output includes:
  - Revenue quantiles for each platform and blended
  - Blended ROAS quantiles (P10 / P50 / P90)

---

## Quick Start

### Prerequisites

- Python **3.10+**
- `pip`

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Forecasting (Required by Submission Pipeline)

The entry point is `run.sh` at the repo root.

**Default usage:**

```bash
bash run.sh
```

**Explicit command:**

```bash
bash run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

### What `run.sh` Does

1. Runs feature generation from the contents of `data/`
2. Loads the trained model from `pickle/model.pkl`
3. Writes predictions to the provided output path (CSV)

---

## Repository Structure

```text
.
├── run.sh                    # Entry point (required)
├── requirements.txt          # Pinned dependencies (required)
├── README.md                 # This file
├── data/                     # Input data folder (overwritten at test time)
├── pickle/                   # Trained model artifact(s)
│   └── model.pkl              # Required model file
├── src/                      # Code
│   ├── generate_features.py  # Feature generation pipeline step
│   ├── predict.py            # Prediction pipeline step
│   ├── train.py              # Model training (development only)
│   └── validate.py           # Validation utilities (optional)
└── output/                   # Predictions output (generated)
```

---

## Output Format

The pipeline produces a CSV at `OUTPUT_PATH` (default: `./output/predictions.csv`).

The file is written with the following columns (fixed by `src/predict.py`):

| Column | Description |
|---|---|
| `request_id` | Unique identifier for the forecast request |
| `period_days` | Planning horizon (30 / 60 / 90) |
| `spend_google`, `spend_meta`, `spend_ms` | Budget inputs per platform |
| `revenue_p10`, `revenue_p50`, `revenue_p90` | Blended revenue quantiles |
| `blended_roas_p10`, `blended_roas_p50`, `blended_roas_p90` | Blended ROAS quantiles |
| `google_revenue_p10/p50/p90` | Google Ads revenue quantiles |
| `meta_revenue_p10/p50/p90` | Meta Ads revenue quantiles |
| `ms_revenue_p10/p50/p90` | Microsoft Ads revenue quantiles |

---

## Streamlit Demo App (Optional)

A Streamlit dashboard is included for interactive demos.

**App source:** `src/app.py`

### Start the App

```bash
streamlit run app.py
```

### What the App Provides

**Sidebar controls:**
- Planning horizon selector: **30 / 60 / 90 days**
- Budget inputs for **Google / Meta / Microsoft**
- Optional AI insights toggle (Groq)

**Tabs:**

1. **Forecast** — Generates features in a temporary folder, runs prediction using `pickle/model.pkl`, and renders revenue quantiles + ROAS.
2. **Data Explorer** — Filters and visualizes historical data (platform + campaign type) and shows a raw data preview.
3. **Anomalies** — Uses `AnomalyDetector` to detect spend/revenue anomalies and data gaps.
4. **Methodology** — Shows high-level approach and (if available) top feature importances from the loaded model.

### Groq-Based AI Insights

- The app can optionally generate short insights using Groq.
- If `GROQ_API_KEY` is not configured (via `.env`), it will show an informational message and continue without AI insights.

---

## Methodology (High Level)

### Feature Generation

`src/generate_features.py`:
- Reads historical performance from `data/`
- Aggregates daily data into a platform-level view
- Constructs forecast feature vectors per planning scenario
- Writes features to a parquet file (used by `predict.py`)

### Prediction

`src/predict.py`:
- Loads the pickled forecasting model
- Aligns input features with the model's expected feature list
- Produces quantile predictions (P10 / P50 / P90)
- Computes blended ROAS from predicted revenue and the provided spend budgets
- Writes the CSV fresh on every run

---

## Data Requirements

The pipeline reads whatever exists under `data/` at runtime. It will attempt to detect forecast request rows from files such as:

- `forecast_requests.csv`
- `requests.csv`
- `budget_scenarios.csv`

If none of these exist, it generates default scenarios based on recent historical spend patterns.

---

## Reproducibility & Constraints

- Dependencies are pinned in `requirements.txt`
- The submission pipeline does not require internet access at runtime
- The model is loaded from `pickle/model.pkl` (no retraining during scoring)

---

## Python Version

This repo currently runs on:
- **Python 3.13.7** (as used in this environment)

The code is intended for **Python 3.10+**, as stated above.

---

## Submission Information

- **Run command:** `bash run.sh ./data ./pickle/model.pkl ./output/predictions.csv`
- **Model path:** `./pickle/model.pkl`
- **Output path:** `./output/predictions.csv`
