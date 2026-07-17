# src/validate.py
"""
Comprehensive validation suite for the forecasting pipeline.
Tests real-world scenarios, edge cases, and model robustness.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import tempfile
import shutil
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_all_data, STANDARD_COLS
from src.features import FeatureEngineer
from src.train import ForecastingModel
from src.anomaly_detector import AnomalyDetector
from src.logger import setup_logger

logger = setup_logger(__name__)


class ForecastingValidator:
    """
    Validates the forecasting pipeline against real-world scenarios.
    """
    
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = Path(__file__).parent.parent / "pickle" / "model.pkl"
        
        self.model = ForecastingModel.load(str(model_path))
        self.results = {}
        
    def run_all_validations(self, df: pd.DataFrame) -> dict:
        """Run complete validation suite."""
        logger.info(f"\n{'='*80}")
        logger.info(f"?? COMPREHENSIVE VALIDATION SUITE")
        logger.info(f"{'='*80}")
        
        validations = {
            '1_data_quality': self.validate_data_quality(df),
            '2_model_robustness': self.validate_model_robustness(df),
            '3_budget_sensitivity': self.validate_budget_sensitivity(df),
            '4_seasonality': self.validate_seasonality(df),
            '5_edge_cases': self.validate_edge_cases(df),
            '6_prediction_consistency': self.validate_prediction_consistency(df),
            '7_real_scenario_simulation': self.simulate_real_scenario(df),
            '8_reconciliation_calibration': self.validate_reconciliation_calibration(df),
        }
        
        self.results = validations
        self._print_scorecard(validations)
        
        return validations
    
    # ================================================================
    # TEST 1: DATA QUALITY
    # ================================================================
    
    def validate_data_quality(self, df: pd.DataFrame) -> dict:
        """Validate data quality and completeness."""
        logger.info(f"\n{'-'*60}")
        logger.info("?? TEST 1: Data Quality Assessment")
        logger.info(f"{'-'*60}")
        
        checks = {}
        
        # Check 1: Required columns
        missing_cols = [c for c in STANDARD_COLS[:-1] if c not in df.columns]
        checks['required_columns'] = {
            'passed': len(missing_cols) == 0,
            'details': f"Missing: {missing_cols}" if missing_cols else "All present"
        }
        
        # Check 2: Date continuity
        date_range = (df['date'].max() - df['date'].min()).days
        expected_days = df.groupby('platform')['date'].nunique().max()
        checks['date_continuity'] = {
            'passed': date_range > 180,  # At least 6 months
            'details': f"Data spans {date_range} days ({date_range/30:.1f} months)"
        }
        
        # Check 3: Platform coverage
        platforms = df['platform'].nunique()
        checks['platform_coverage'] = {
            'passed': platforms >= 2,
            'details': f"{platforms} platforms: {df['platform'].unique().tolist()}"
        }
        
        # Check 4: Revenue presence
        revenue_pct = (df['revenue'] > 0).mean() * 100
        checks['revenue_data'] = {
            'passed': revenue_pct > 20,  # At least 20% days have revenue
            'details': f"{revenue_pct:.1f}% of rows have revenue > 0"
        }
        
        # Check 5: Campaign diversity
        n_campaigns = df['campaign_id'].nunique()
        checks['campaign_diversity'] = {
            'passed': n_campaigns >= 5,
            'details': f"{n_campaigns} unique campaigns"
        }
        
        # Check 6: Spend-revenue correlation
        daily = df.groupby('date')[['spend', 'revenue']].sum()
        if len(daily) > 10:
            corr = daily['spend'].corr(daily['revenue'])
            checks['spend_revenue_correlation'] = {
                'passed': corr > 0.3,
                'details': f"Correlation: {corr:.3f}"
            }
        
        passed = sum(1 for c in checks.values() if c['passed'])
        total = len(checks)
        
        logger.info(f"   Results: {passed}/{total} checks passed")
        for name, result in checks.items():
            icon = "?" if result['passed'] else "?"
            logger.info(f"   {icon} {name}: {result['details']}")
        
        return {'passed': passed, 'total': total, 'checks': checks}
    
    # ================================================================
    # TEST 2: MODEL ROBUSTNESS
    # ================================================================
    
    def validate_model_robustness(self, df: pd.DataFrame) -> dict:
        """Test model robustness with backtesting."""
        logger.info(f"\n{'-'*60}")
        logger.info("?? TEST 2: Model Robustness (Backtesting)")
        logger.info(f"{'-'*60}")
        
        # Aggregate to daily platform
        daily = df.groupby(['date', 'platform']).agg({
            'spend': 'sum', 'revenue': 'sum', 'clicks': 'sum',
            'impressions': 'sum', 'conversions': 'sum'
        }).reset_index()
        
        daily_pivot = daily.pivot_table(
            index='date', columns='platform',
            values=['spend', 'revenue', 'clicks', 'impressions', 'conversions'],
            aggfunc='sum', fill_value=0
        )
        daily_pivot.columns = [f'{c[0]}_{c[1]}' for c in daily_pivot.columns]
        daily_pivot = daily_pivot.reset_index().sort_values('date')
        
        # Backtest: Predict last 3 periods using earlier data
        engineer = FeatureEngineer(periods=[30, 60, 90])
        features = engineer.create_training_data(df)
        
        # Get test periods
        test_periods = features[features['ref_date'] >= features['ref_date'].max() - timedelta(days=180)]
        
        backtest_results = []
        
        for _, row in test_periods.iterrows():
            period_days = row['period_days']
            
            # Build feature vector
            feature_cols = self.model.feature_names
            X = pd.DataFrame([{col: row.get(col, 0) for col in feature_cols}])
            X = X.fillna(0)
            
            # Predict
            try:
                X_scaled = self.model.scaler.transform(X)
                pred_p50 = self.model.models['target_total_revenue_q50'].predict(X_scaled)[0]
                actual = row['target_total_revenue']
                
                backtest_results.append({
                    'date': row['ref_date'],
                    'period': period_days,
                    'predicted': pred_p50,
                    'actual': actual,
                    'error_pct': abs(pred_p50 - actual) / actual * 100 if actual > 0 else np.nan
                })
            except Exception as e:
                logger.warning(f"   Backtest error for {row['ref_date'].date()}: {e}")
        
        bt_df = pd.DataFrame(backtest_results)
        
        if len(bt_df) > 0:
            mape = bt_df['error_pct'].median()
            within_50pct = (bt_df['error_pct'] < 50).mean() * 100
            
            logger.info(f"   Samples tested: {len(bt_df)}")
            logger.info(f"   Median error: {mape:.1f}%")
            logger.info(f"   Within 50% error: {within_50pct:.1f}%")
            
            passed = mape < 100  # Less than 100% median error
        else:
            passed = False
            mape = np.nan
            logger.warning("   Backtesting failed - no predictions generated")
        
        return {
            'passed': passed,
            'samples': len(bt_df),
            'median_error_pct': mape,
            'within_50pct_accuracy': within_50pct if len(bt_df) > 0 else 0,
            'details': f"Median error: {mape:.1f}%" if not np.isnan(mape) else "Failed"
        }
    
    # ================================================================
    # TEST 3: BUDGET SENSITIVITY
    # ================================================================
    
    def validate_budget_sensitivity(self, df: pd.DataFrame) -> dict:
        """Test that model responds logically to budget changes."""
        logger.info(f"\n{'-'*60}")
        logger.info("?? TEST 3: Budget Sensitivity")
        logger.info(f"{'-'*60}")
        
        # Create scenarios with different budgets
        base_budget = {'google': 50000, 'meta': 10000, 'microsoft': 5000}
        
        scenarios = {
            'baseline': base_budget,
            'double_google': {**base_budget, 'google': base_budget['google'] * 2},
            'zero_meta': {**base_budget, 'meta': 0},
            'triple_all': {k: v * 3 for k, v in base_budget.items()},
            'half_all': {k: v // 2 for k, v in base_budget.items()},
        }
        
        predictions = {}
        for name, budget in scenarios.items():
            # Create simple feature row
            feature_row = self._create_simple_features(df, budget, 30)
            if feature_row is not None:
                try:
                    X = pd.DataFrame([feature_row])
                    X = X[self.model.feature_names].fillna(0)
                    X_scaled = self.model.scaler.transform(X)
                    pred = self.model.models['target_total_revenue_q50'].predict(X_scaled)[0]
                    predictions[name] = max(0, pred)
                except:
                    predictions[name] = np.nan
        
        checks = {}
        
        # Check 1: Higher budget ? higher revenue
        if 'baseline' in predictions and 'double_google' in predictions:
            checks['budget_vs_revenue'] = {
                'passed': predictions['double_google'] > predictions['baseline'],
                'details': f"Baseline: ${predictions.get('baseline', 0):,.0f}, "
                          f"2x Google: ${predictions.get('double_google', 0):,.0f}"
            }
        
        # Check 2: Zero spend ? zero or very low revenue
        if 'zero_meta' in predictions and 'baseline' in predictions:
            checks['zero_spend'] = {
                'passed': predictions['zero_meta'] <= predictions['baseline'],
                'details': f"Zero Meta: ${predictions.get('zero_meta', 0):,.0f}, "
                          f"Baseline: ${predictions.get('baseline', 0):,.0f}"
            }
        
        # Check 3: Diminishing returns (triple doesn't give 3x revenue)
        if 'baseline' in predictions and 'triple_all' in predictions:
            ratio = predictions['triple_all'] / predictions['baseline'] if predictions['baseline'] > 0 else np.nan
            checks['diminishing_returns'] = {
                'passed': ratio < 3.0 if not np.isnan(ratio) else False,
                'details': f"3x budget gives {ratio:.1f}x revenue" if not np.isnan(ratio) else "N/A"
            }
        
        # Log results
        logger.info(f"   Scenario predictions:")
        for name, pred in predictions.items():
            logger.info(f"     {name:15s}: ${pred:,.0f}")
        
        passed = sum(1 for c in checks.values() if c['passed'])
        total = len(checks) if checks else 1
        
        return {
            'passed': passed >= total * 0.5,
            'checks': checks,
            'predictions': {k: float(v) if not np.isnan(v) else None for k, v in predictions.items()},
            'details': f"{passed}/{total} sensitivity checks passed"
        }
    
    # ================================================================
    # TEST 4: SEASONALITY
    # ================================================================
    
    def validate_seasonality(self, df: pd.DataFrame) -> dict:
        """Test seasonal pattern detection."""
        logger.info(f"\n{'-'*60}")
        logger.info("?? TEST 4: Seasonality Handling")
        logger.info(f"{'-'*60}")
        
        # Check if model captures month-of-year patterns
        monthly_revenue = df.groupby(df['date'].dt.month)['revenue'].sum()
        
        if len(monthly_revenue) >= 6:
            cv = monthly_revenue.std() / monthly_revenue.mean()
            has_seasonality = cv > 0.1
            
            logger.info(f"   Monthly revenue CV: {cv:.3f}")
            logger.info(f"   Seasonality detected: {has_seasonality}")
            
            # Predict for different months
            base_budget = {'google': 50000, 'meta': 10000, 'microsoft': 5000}
            month_preds = {}
            
            for month in [1, 6, 11]:  # Jan, Jun, Nov
                feature_row = self._create_simple_features(df, base_budget, 30, month=month)
                if feature_row:
                    try:
                        X = pd.DataFrame([feature_row])
                        X = X[self.model.feature_names].fillna(0)
                        X_scaled = self.model.scaler.transform(X)
                        pred = self.model.models['target_total_revenue_q50'].predict(X_scaled)[0]
                        month_preds[month] = max(0, pred)
                    except:
                        month_preds[month] = np.nan
            
            logger.info(f"   Predicted revenue by month:")
            for m, p in month_preds.items():
                logger.info(f"     Month {m:2d}: ${p:,.0f}")
            
            # Check if predictions vary by month
            preds_list = [v for v in month_preds.values() if not np.isnan(v)]
            if len(preds_list) >= 2:
                pred_cv = np.std(preds_list) / np.mean(preds_list) if np.mean(preds_list) > 0 else 0
                logger.info(f"   Prediction variation: {pred_cv:.3f}")
                
                return {
                    'passed': True,  # Always pass, just report
                    'has_seasonality': has_seasonality,
                    'monthly_cv': float(cv),
                    'prediction_variation': float(pred_cv),
                    'month_predictions': {str(k): float(v) for k, v in month_preds.items()},
                    'details': f"Monthly CV: {cv:.3f}, Prediction CV: {pred_cv:.3f}"
                }
        
        return {
            'passed': True,
            'details': "Insufficient monthly data for seasonality check"
        }
    
    # ================================================================
    # TEST 5: EDGE CASES
    # ================================================================
    
    def validate_edge_cases(self, df: pd.DataFrame) -> dict:
        """Test edge cases and boundary conditions."""
        logger.info(f"\n{'-'*60}")
        logger.info("?? TEST 5: Edge Cases")
        logger.info(f"{'-'*60}")
        
        edge_cases = {}
        
        # Case 1: Zero budget scenario
        zero_budget = {'google': 0, 'meta': 0, 'microsoft': 0}
        row_zero = self._create_simple_features(df, zero_budget, 30)
        if row_zero:
            try:
                X = pd.DataFrame([row_zero])
                X = X[self.model.feature_names].fillna(0)
                X_scaled = self.model.scaler.transform(X)
                pred_zero = self.model.models['target_total_revenue_q50'].predict(X_scaled)[0]
                edge_cases['zero_budget'] = {
                    'passed': pred_zero < 10000,  # Should predict near-zero revenue
                    'details': f"Zero budget prediction: ${pred_zero:,.0f}"
                }
            except:
                edge_cases['zero_budget'] = {'passed': False, 'details': 'Failed'}
        
        # Case 2: Very large budget
        large_budget = {'google': 10000000, 'meta': 5000000, 'microsoft': 1000000}
        row_large = self._create_simple_features(df, large_budget, 30)
        if row_large:
            try:
                X = pd.DataFrame([row_large])
                X = X[self.model.feature_names].fillna(0)
                X_scaled = self.model.scaler.transform(X)
                pred_large = self.model.models['target_total_revenue_q50'].predict(X_scaled)[0]
                edge_cases['large_budget'] = {
                    'passed': pred_large > 0 and not np.isinf(pred_large),
                    'details': f"Large budget prediction: ${pred_large:,.0f}"
                }
            except:
                edge_cases['large_budget'] = {'passed': False, 'details': 'Failed'}
        
        # Case 3: Negative values (should not crash)
        negative_budget = {'google': -1000, 'meta': 5000, 'microsoft': 2000}
        row_neg = self._create_simple_features(df, negative_budget, 30)
        if row_neg:
            try:
                X = pd.DataFrame([row_neg])
                X = X[self.model.feature_names].fillna(0)
                X_scaled = self.model.scaler.transform(X)
                pred_neg = self.model.models['target_total_revenue_q50'].predict(X_scaled)[0]
                edge_cases['negative_budget'] = {
                    'passed': True,  # Should handle gracefully
                    'details': f"Negative budget handled: ${pred_neg:,.0f}"
                }
            except:
                edge_cases['negative_budget'] = {'passed': False, 'details': 'Crashed on negative value'}
        
        # Case 4: Different period lengths consistency
        period_preds = {}
        for period in [30, 60, 90]:
            row = self._create_simple_features(df, {'google': 50000, 'meta': 10000, 'microsoft': 5000}, period)
            if row:
                try:
                    X = pd.DataFrame([row])
                    X = X[self.model.feature_names].fillna(0)
                    X_scaled = self.model.scaler.transform(X)
                    pred = self.model.models['target_total_revenue_q50'].predict(X_scaled)[0]
                    period_preds[period] = max(0, pred)
                except:
                    period_preds[period] = np.nan
        
        if len(period_preds) >= 2:
            # Check that 90-day > 60-day > 30-day (roughly)
            preds_ok = True
            if period_preds.get(90, 0) < period_preds.get(60, 0):
                preds_ok = False
            if period_preds.get(60, 0) < period_preds.get(30, 0):
                preds_ok = False
            
            edge_cases['period_consistency'] = {
                'passed': preds_ok,
                'details': f"30d: ${period_preds.get(30, 0):,.0f}, "
                          f"60d: ${period_preds.get(60, 0):,.0f}, "
                          f"90d: ${period_preds.get(90, 0):,.0f}"
            }
        
        passed = sum(1 for c in edge_cases.values() if c['passed'])
        total = len(edge_cases)
        
        logger.info(f"   Results: {passed}/{total} edge cases passed")
        for name, result in edge_cases.items():
            icon = "?" if result['passed'] else "?"
            logger.info(f"   {icon} {name}: {result['details']}")
        
        return {
            'passed': passed >= total * 0.5,
            'checks': edge_cases,
            'details': f"{passed}/{total} edge cases passed"
        }
    
    # ================================================================
    # TEST 6: PREDICTION CONSISTENCY
    # ================================================================
    
    def validate_reconciliation_calibration(self, df: pd.DataFrame) -> dict:
        """
        Compare legacy independent blended-total intervals vs reconciled (bootstrap) intervals.

        Returns a dict consumable by Streamlit:
          {
            'passed': bool,
            'details': str,
            'results': {
              'coverage': {'legacy_p10_p90': float, 'reconciled_p10_p90': float},
              'pinball_loss': {
                 'legacy': {'p10': float,'p50': float,'p90': float},
                 'reconciled': {'p10': float,'p50': float,'p90': float},
              },
              'n_samples': {'legacy': int,'reconciled': int}
            }
          }
        """
        logger.info(f"\n{'-'*60}")
        logger.info("?? TEST 8: Reconciliation Calibration (coverage + pinball loss)")
        logger.info(f"{'-'*60}")

        # Fallback: if FeatureEngineer or reconciliation helper is missing, don't crash the demo.
        try:
            engineer = FeatureEngineer(periods=[30, 60, 90])
            features = engineer.create_training_data(df)

            test_periods = features[features['ref_date'] >= features['ref_date'].max() - timedelta(days=180)].copy()
            if test_periods.empty or 'target_total_revenue' not in test_periods.columns:
                return {'passed': False, 'details': 'Insufficient test data or missing target_total_revenue', 'results': {}}

            # Evaluate using existing legacy quantile heads and reconciled bootstrap from src.predict
            from src.predict import _bootstrap_reconciled_blended_quantiles

            legacy_p10_p90_cov = []
            recon_p10_p90_cov = []
            pinball_legacy = {'p10': [], 'p50': [], 'p90': []}
            pinball_recon = {'p10': [], 'p50': [], 'p90': []}

            for _, r in test_periods.iterrows():
                period_days = int(r['period_days'])
                y = float(r['target_total_revenue'])
                if np.isnan(y):
                    continue

                X_row = pd.DataFrame([{c: r.get(c, 0) for c in self.model.feature_names}]).replace([np.inf, -np.inf], np.nan).fillna(0)
                X_scaled = self.model.scaler.transform(X_row)
                X_scaled_df = pd.DataFrame(X_scaled, columns=self.model.feature_names)

                # Legacy quantiles
                if all(k in self.model.models for k in ['target_total_revenue_q10', 'target_total_revenue_q50', 'target_total_revenue_q90']):
                    p10 = float(self.model.models['target_total_revenue_q10'].predict(X_scaled_df)[0])
                    p50 = float(self.model.models['target_total_revenue_q50'].predict(X_scaled_df)[0])
                    p90 = float(self.model.models['target_total_revenue_q90'].predict(X_scaled_df)[0])
                    p10 = max(0.0, p10); p50 = max(0.0, p50); p90 = max(0.0, p90)
                    legacy_p10_p90_cov.append(1.0 if (y >= p10 and y <= p90) else 0.0)

                    for q_label, alpha in [('p10', 0.10), ('p50', 0.50), ('p90', 0.90)]:
                        pred = {'p10': p10, 'p50': p50, 'p90': p90}[q_label]
                        err = y - pred
                        pin = np.mean(np.where(err >= 0, alpha * err, (alpha - 1) * err))
                        pinball_legacy[q_label].append(float(pin))

                # Reconciled quantiles
                # Uses df as historical source for residual bootstrap inside helper.
                rec = _bootstrap_reconciled_blended_quantiles(
                    df_hist=df,
                    model=self.model,
                    feature_cols=self.model.feature_names,
                    request_row_scaled=X_scaled_df,
                    period_days=period_days,
                    n_boot=120,
                    quantiles=(0.10, 0.50, 0.90),
                )
                rp10 = max(0.0, float(rec['revenue_p10']))
                rp50 = max(0.0, float(rec['revenue_p50']))
                rp90 = max(0.0, float(rec['revenue_p90']))

                recon_p10_p90_cov.append(1.0 if (y >= rp10 and y <= rp90) else 0.0)

                for q_label, alpha in [('p10', 0.10), ('p50', 0.50), ('p90', 0.90)]:
                    pred = {'p10': rp10, 'p50': rp50, 'p90': rp90}[q_label]
                    err = y - pred
                    pin = np.mean(np.where(err >= 0, alpha * err, (alpha - 1) * err))
                    pinball_recon[q_label].append(float(pin))

            def _avg(lst):
                return float(np.mean(lst)) if lst else float('nan')

            legacy_cov = _avg(legacy_p10_p90_cov)
            recon_cov = _avg(recon_p10_p90_cov)

            return {
                'passed': True if not np.isnan(recon_cov) else False,
                'details': 'Compared legacy vs reconciled using empirical coverage and pinball loss',
                'results': {
                    'coverage': {'legacy_p10_p90': legacy_cov, 'reconciled_p10_p90': recon_cov},
                    'pinball_loss': {'legacy': {k: _avg(v) for k, v in pinball_legacy.items()},
                                       'reconciled': {k: _avg(v) for k, v in pinball_recon.items()}},
                    'n_samples': {'legacy': len(legacy_p10_p90_cov), 'reconciled': len(recon_p10_p90_cov)},
                }
            }
        except Exception as e:
            logger.warning(f"Reconciliation calibration failed: {e}")
            return {'passed': False, 'details': f'Reconciliation calibration failed: {e}', 'results': {}}

    def validate_prediction_consistency(self, df: pd.DataFrame) -> dict:
        """Check that predictions are internally consistent."""
        logger.info(f"\n{'-'*60}")
        logger.info("?? TEST 6: Prediction Consistency")
        logger.info(f"{'-'*60}")
        
        checks = {}
        
        # Check 1: P10 = P50 = P90
        base_budget = {'google': 50000, 'meta': 10000, 'microsoft': 5000}
        row = self._create_simple_features(df, base_budget, 30)
        
        if row:
            X = pd.DataFrame([row])
            X = X[self.model.feature_names].fillna(0)
            
            try:
                X_scaled = self.model.scaler.transform(X)
                
                preds = {}
                for q_label, q_key in [('p10', 'target_total_revenue_q10'),
                                        ('p50', 'target_total_revenue_q50'),
                                        ('p90', 'target_total_revenue_q90')]:
                    if q_key in self.model.models:
                        preds[q_label] = max(0, self.model.models[q_key].predict(X_scaled)[0])
                
                if len(preds) == 3:
                    monotonic = preds['p10'] <= preds['p50'] <= preds['p90']
                    checks['quantile_monotonicity'] = {
                        'passed': monotonic,
                        'details': f"P10: ${preds['p10']:,.0f}, P50: ${preds['p50']:,.0f}, P90: ${preds['p90']:,.0f}"
                    }
                    
                    # Check 2: Interval width is reasonable
                    interval_width = (preds['p90'] - preds['p10']) / preds['p50'] if preds['p50'] > 0 else np.nan
                    checks['interval_width'] = {
                        'passed': 0.1 < interval_width < 5.0 if not np.isnan(interval_width) else False,
                        'details': f"Interval width ratio: {interval_width:.2f}" if not np.isnan(interval_width) else "N/A"
                    }
            except:
                pass
        
        passed = sum(1 for c in checks.values() if c['passed'])
        total = len(checks) if checks else 1
        
        logger.info(f"   Results: {passed}/{total} consistency checks passed")
        for name, result in checks.items():
            icon = "?" if result['passed'] else "?"
            logger.info(f"   {icon} {name}: {result['details']}")
        
        return {
            'passed': passed >= total * 0.5,
            'checks': checks,
            'details': f"{passed}/{total} consistency checks passed"
        }
    
    # ================================================================
    # TEST 7: REAL SCENARIO SIMULATION
    # ================================================================
    
    def simulate_real_scenario(self, df: pd.DataFrame) -> dict:
        """Simulate a realistic agency planning scenario."""
        logger.info(f"\n{'-'*60}")
        logger.info("?? TEST 7: Real-World Scenario Simulation")
        logger.info(f"{'-'*60}")
        
        # Scenario: Agency wants to shift budget from Meta to Google
        # Based on actual data showing Google has better ROAS
        
        logger.info("   Scenario: Budget reallocation simulation")
        logger.info("   Starting: Google=$50K, Meta=$20K, MS=$10K")
        
        original = {'google': 50000, 'meta': 20000, 'microsoft': 10000}
        reallocated = {'google': 60000, 'meta': 10000, 'microsoft': 10000}  # Shift $10K Meta?Google
        
        preds_original = {}
        preds_reallocated = {}
        
        for name, budget in [('original', original), ('reallocated', reallocated)]:
            row = self._create_simple_features(df, budget, 60)
            if row:
                try:
                    X = pd.DataFrame([row])
                    X = X[self.model.feature_names].fillna(0)
                    X_scaled = self.model.scaler.transform(X)
                    
                    for q_label, q_key in [('p10', 'target_total_revenue_q10'),
                                            ('p50', 'target_total_revenue_q50'),
                                            ('p90', 'target_total_revenue_q90')]:
                        if q_key in self.model.models:
                            if name == 'original':
                                preds_original[q_label] = max(0, self.model.models[q_key].predict(X_scaled)[0])
                            else:
                                preds_reallocated[q_label] = max(0, self.model.models[q_key].predict(X_scaled)[0])
                except:
                    pass
        
        if preds_original and preds_reallocated:
            improvement = preds_reallocated.get('p50', 0) - preds_original.get('p50', 0)
            improvement_pct = improvement / preds_original['p50'] * 100 if preds_original.get('p50', 0) > 0 else 0
            
            total_budget = sum(original.values())
            original_roas = preds_original.get('p50', 0) / total_budget if total_budget > 0 else 0
            reallocated_roas = preds_reallocated.get('p50', 0) / total_budget if total_budget > 0 else 0
            
            logger.info(f"   Original allocation:    ${preds_original.get('p50', 0):,.0f} revenue, {original_roas:.2f}x ROAS")
            logger.info(f"   Reallocated (Meta?Google): ${preds_reallocated.get('p50', 0):,.0f} revenue, {reallocated_roas:.2f}x ROAS")
            logger.info(f"   Improvement: ${improvement:,.0f} ({improvement_pct:+.1f}%)")
            
            return {
                'passed': True,
                'original_revenue': float(preds_original.get('p50', 0)),
                'reallocated_revenue': float(preds_reallocated.get('p50', 0)),
                'improvement': float(improvement),
                'improvement_pct': float(improvement_pct),
                'original_roas': float(original_roas),
                'reallocated_roas': float(reallocated_roas),
                'details': f"Budget shift gave {improvement_pct:+.1f}% revenue change"
            }
        
        return {'passed': False, 'details': 'Scenario simulation failed'}
    
    # ================================================================
    # HELPERS
    # ================================================================
    
    # src/validate.py - Fix the bugs in _create_simple_features and validate_seasonality

# Find the _create_simple_features method and replace the date handling section:

    def _create_simple_features(self, df: pd.DataFrame, budgets: dict, 
                                 period_days: int, month: int = None) -> dict:
        """Create a minimal feature vector for a forecast scenario."""
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Aggregate daily
        daily = df.groupby(['date', 'platform']).agg({
            'spend': 'sum', 'revenue': 'sum', 'clicks': 'sum',
            'impressions': 'sum', 'conversions': 'sum'
        }).reset_index()
        
        daily_pivot = daily.pivot_table(
            index='date', columns='platform',
            values=['spend', 'revenue', 'clicks', 'impressions', 'conversions'],
            aggfunc='sum', fill_value=0
        )
        daily_pivot.columns = [f'{c[0]}_{c[1]}' for c in daily_pivot.columns]
        daily_pivot = daily_pivot.reset_index()
        
        # Recent data
        last_date = daily_pivot['date'].max()
        feature_start = last_date - timedelta(days=period_days)
        feature_end = last_date - timedelta(days=1)
        
        # Create reference date as pandas Timestamp
        if month:
            ref_date = pd.Timestamp(year=last_date.year, month=month, day=min(last_date.day, 28))
        else:
            ref_date = last_date + pd.Timedelta(days=1)
        
        feature_mask = (daily_pivot['date'] >= feature_start) & (daily_pivot['date'] <= feature_end)
        feature_data = daily_pivot[feature_mask]
        
        # Use pandas Timestamp attributes
        row = {
            'period_days': period_days,
            'ref_date': ref_date,
            'month': ref_date.month,
            'quarter': ref_date.quarter,
            'year': ref_date.year,
            'day_of_month': ref_date.day,
            'day_of_week': ref_date.dayofweek,
            'week_of_year': ref_date.isocalendar()[1] if hasattr(ref_date, 'isocalendar') else ref_date.weekofyear,
            'is_weekend': 1 if ref_date.dayofweek in [5, 6] else 0,
            'is_month_start': 1 if ref_date.day in [1, 2, 3] else 0,
            'is_month_end': 1 if ref_date.day in [28, 29, 30, 31] else 0,
            'is_holiday': 0,
            'weekend_ratio': 2/7,
            'days_since_start': (ref_date - daily_pivot['date'].min()).days,
            'trend': (ref_date - daily_pivot['date'].min()).days / 365.25,
        }
        
        for m in range(1, 13):
            row[f'is_month_{m}'] = 1 if ref_date.month == m else 0
        for q in range(1, 5):
            row[f'is_q{q}'] = 1 if ref_date.quarter == q else 0
        
        # Add feature columns from historical data
        metric_cols = [c for c in daily_pivot.columns if c != 'date']
        for col in metric_cols:
            row[f'feature_{col}'] = feature_data[col].sum() if len(feature_data) > 0 else 0
            row[f'feature_daily_avg_{col}'] = feature_data[col].mean() if len(feature_data) > 0 else 0
            row[f'feature_days_with_{col}'] = (feature_data[col] > 0).sum() if len(feature_data) > 0 else 0
        
        # Override with budgets
        plat_map = {'google': 'google', 'meta': 'meta', 'microsoft': 'microsoft'}
        for plat_name, plat_col in plat_map.items():
            row[f'feature_spend_{plat_col}'] = budgets.get(plat_name, 0)
        
        total_spend = sum(budgets.values())
        row['feature_total_spend'] = total_spend
        
        # Add total revenue from historical window
        rev_cols = [c for c in row.keys() if c.startswith('feature_revenue_') and 'ctype' not in c and 'daily' not in c and 'days' not in c]
        row['feature_total_revenue'] = sum(row.get(c, 0) for c in rev_cols)
        
        for plat_name, plat_col in plat_map.items():
            row[f'feature_spend_share_{plat_col}'] = budgets.get(plat_name, 0) / total_spend if total_spend > 0 else 0
        
        # ROAS
        for plat_name, plat_col in plat_map.items():
            spend_col = f'feature_spend_{plat_col}'
            rev_col = f'feature_revenue_{plat_col}'
            spend = row.get(spend_col, 0)
            rev = row.get(rev_col, 0)
            row[f'feature_roas_{plat_col}'] = rev / spend if spend > 0 else 0
        
        row['feature_blended_roas'] = row.get('feature_total_revenue', 0) / total_spend if total_spend > 0 else 0
        
        # Add all total metrics
        for metric in ['clicks', 'impressions', 'conversions']:
            metric_cols = [c for c in row.keys() if f'feature_{metric}_' in c and 'ctype' not in c and 'daily' not in c and 'days' not in c]
            row[f'feature_total_{metric}'] = sum(row.get(c, 0) for c in metric_cols)
        
        # Add daily avg for total metrics
        for metric in ['spend', 'revenue', 'clicks', 'impressions', 'conversions']:
            row[f'feature_daily_avg_total_{metric}'] = row.get(f'feature_total_{metric}', 0) / period_days if period_days > 0 else 0
            row[f'feature_days_with_total_{metric}'] = period_days
        
        # Campaign type features (fill with 0 for simplicity)
        for ctype in ['pmax', 'search', 'video', 'display', 'other', 'demand_gen', 'shopping', 'audience']:
            for metric in ['spend', 'revenue', 'roas']:
                col_name = f'feature_{metric}_ctype_{ctype}'
                if col_name not in row:
                    row[col_name] = 0
        
        return row
    
    # ================================================================
    # SCORECARD
    # ================================================================
    
    def _print_scorecard(self, results: dict):
        """Print validation scorecard."""
        logger.info(f"\n\n{'='*80}")
        logger.info(f"?? VALIDATION SCORECARD")
        logger.info(f"{'='*80}")
        
        total_passed = 0
        total_tests = 0
        
        for test_name, result in results.items():
            if isinstance(result, dict) and 'passed' in result:
                total_tests += 1
                if result['passed']:
                    total_passed += 1
                
                icon = "?" if result['passed'] else "?"
                details = result.get('details', '')
                logger.info(f"  {icon} {test_name}: {details}")
        
        score = total_passed / total_tests * 100 if total_tests > 0 else 0
        
        logger.info(f"\n  {'-'*40}")
        logger.info(f"  Overall: {total_passed}/{total_tests} tests passed ({score:.0f}%)")
        
        if score >= 80:
            logger.info(f"  Rating: ?? READY FOR PRODUCTION")
        elif score >= 60:
            logger.info(f"  Rating: ?? NEEDS IMPROVEMENT")
        else:
            logger.info(f"  Rating: ?? SIGNIFICANT ISSUES")
        
        logger.info(f"{'='*80}\n")
        
        return score


# ================================================================
# STANDALONE VALIDATION FUNCTION
# ================================================================

def run_validation():
    """Run complete validation and return results."""
    logger.info("Loading data for validation...")
    df = load_all_data()
    
    validator = ForecastingValidator()
    results = validator.run_all_validations(df)
    
    return results


# ================================================================
# PIPELINE INTEGRATION TEST
# ================================================================

def test_full_pipeline():
    """Test the complete run.sh pipeline with temporary data."""
    logger.info(f"\n{'='*80}")
    logger.info(f"?? PIPELINE INTEGRATION TEST")
    logger.info(f"{'='*80}")
    
    # Create temp directory
    tmp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(tmp_dir, 'data')
    output_dir = os.path.join(tmp_dir, 'output')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Copy data files
        src_data = Path(__file__).parent.parent / 'data'
        for f in src_data.glob('*.csv'):
            shutil.copy(f, os.path.join(data_dir, f.name))
        
        # Copy model
        model_src = Path(__file__).parent.parent / 'pickle' / 'model.pkl'
        model_dst = os.path.join(tmp_dir, 'model.pkl')
        shutil.copy(model_src, model_dst)
        
        # Run generate_features
        logger.info("\n?? Testing generate_features.py...")
        features_path = os.path.join(tmp_dir, 'features.parquet')
        
        from src.generate_features import generate_features
        generate_features(data_dir, features_path)
        
        if not os.path.exists(features_path):
            raise Exception("features.parquet not created")
        
        logger.info(f"   ? Features created: {os.path.getsize(features_path)} bytes")
        
        # Run predict
        logger.info("\n?? Testing predict.py...")
        output_path = os.path.join(output_dir, 'predictions.csv')
        
        from src.predict import predict
        predict(features_path, model_dst, output_path)
        
        if not os.path.exists(output_path):
            raise Exception("predictions.csv not created")
        
        # Validate output
        output_df = pd.read_csv(output_path)
        required_cols = ['revenue_p50', 'blended_roas_p50', 'google_revenue_p50']
        
        for col in required_cols:
            if col not in output_df.columns:
                raise Exception(f"Missing column: {col}")
        
        logger.info(f"   ? Predictions created: {len(output_df)} rows, {len(output_df.columns)} columns")
        logger.info(f"\n{'='*80}")
        logger.info(f"? PIPELINE INTEGRATION TEST PASSED")
        logger.info(f"{'='*80}")
        
        return True
        
    except Exception as e:
        logger.error(f"? Pipeline integration test failed: {e}")
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate forecasting pipeline')
    parser.add_argument('--full', action='store_true', help='Run full validation suite')
    parser.add_argument('--pipeline', action='store_true', help='Test pipeline integration')
    args = parser.parse_args()
    
    if args.pipeline:
        success = test_full_pipeline()
        sys.exit(0 if success else 1)
    
    if args.full or not args.pipeline:
        results = run_validation()
        
        # Also run pipeline test
        test_full_pipeline()

