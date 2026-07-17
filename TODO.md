# TODO - Campaign Consistency Validation (Priority 3)

- [x] Add `validate_campaign_consistency(df)` in `src/data_loader.py`
  - [x] Check missing `campaign_id` / `campaign_type`
  - [x] Check inconsistent `campaign_name` / `campaign_type` across dates
  - [x] Check sudden platform reassignment per `campaign_id`
  - [x] Check date-range gaps per `campaign_id` (per platform)
  - [x] Return structured report and log a clearly named validation section

- [x] Call `validate_campaign_consistency()` from `load_all_data()` so it’s an explicit ingestion step in pipeline logs

- [x] Update Streamlit `Data Explorer` tab in `app.py`
  - [x] Add “Data Quality” panel (“Campaign Consistency” summary)
  - [x] Display key metrics + an expander with top examples
  - [x] Cache the computed report for performance

- [ ] Smoke test
  - [ ] Run Streamlit and confirm panel renders
  - [ ] Verify loader logs include the new “CAMPAIGN CONSISTENCY VALIDATION” section
  - [ ] Backend smoke (blocked): `python src/validate.py --full`
    - [x] Attempted, failed with `ModuleNotFoundError: No module named 'lightgbm'`

---

# TODO - Hierarchical Reconciliation + Real Calibration Reporting (Priority 5)

- [ ] Step 1: Define new model schema (leaf-level channel quantiles; remove blended-total head usage for production)
  - [ ] Update `src/train.py` to train leaf/channel quantile models only (google/meta/ms)
  - [ ] Ensure saved model metadata includes enough info for reconciliation/bootstrap

- [ ] Step 2: Implement empirical bootstrap reconciliation (channel → blended totals) without summing quantiles
  - [x] Update `src/predict.py`:
    - [x] Generate channel-level forecasts (quantiles) from the same joint bootstrap draws
    - [x] Build residual bootstrap from historical periods to derive blended P10/P50/P90
    - [x] Ensure additive coherence by constructing totals from bootstrapped aggregates

- [ ] Step 3: Validation + comparison against legacy independent intervals
  - [ ] Update `src/validate.py` with a new validator:
    - [ ] Rolling-origin backtest
    - [ ] Compare old independent blended-total intervals vs new reconciled/blended intervals
    - [ ] Report empirical P10–P90 coverage and pinball loss for blended totals
  - [ ] Wire validator outputs into a structured dict for UI rendering

- [ ] Step 4: Wire validation metrics into Streamlit Tab 4 (Methodology)
  - [ ] Update `app.py` Tab 4:
    - [ ] Add “Calibration / Backtesting Results” section
    - [ ] Show achieved coverage + pinball loss (old vs new)
    - [ ] Add baseline comparison summary (if available from training/validation)

- [ ] Step 5: Thorough smoke test
  - [ ] Run `python src/train.py` and confirm model saves/loads
  - [ ] Run `python src/generate_features.py --data-dir data --out features.parquet`
  - [ ] Run `python src/predict.py --features features.parquet --model pickle/model.pkl --output output/predictions.csv`
  - [ ] Run Streamlit and navigate:
    - [ ] Forecast tab
    - [ ] Data Explorer tab
    - [ ] Anomalies tab
    - [ ] Methodology tab (ensure backtest metrics render)
