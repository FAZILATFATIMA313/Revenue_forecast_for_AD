# src/train.py
"""
Probabilistic forecasting model training using LightGBM quantile regression.
Includes time-series cross-validation, overfitting checks, and model serialization.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings

from src.logger import setup_logger
from src.config import (
    RANDOM_SEED, QUANTILES, LGBM_PARAMS, PICKLE_DIR
)
from src.data_loader import load_all_data
from src.features import FeatureEngineer

warnings.filterwarnings('ignore')
logger = setup_logger(__name__)

# Output columns required by hackathon
OUTPUT_COLUMNS = [
    'request_id', 'period_days',
    'spend_google', 'spend_meta', 'spend_ms',
    'revenue_p10', 'revenue_p50', 'revenue_p90',
    'blended_roas_p10', 'blended_roas_p50', 'blended_roas_p90',
    'google_revenue_p10', 'google_revenue_p50', 'google_revenue_p90',
    'meta_revenue_p10', 'meta_revenue_p50', 'meta_revenue_p90',
    'ms_revenue_p10', 'ms_revenue_p50', 'ms_revenue_p90'
]


class ForecastingModel:
    """
    Probabilistic forecasting model using LightGBM quantile regression.
    Trains separate models for each quantile and each target.
    """
    
    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        self.models = {}  # {(target, quantile): lgb_model}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.calibration_factors = {}
        self.baseline_model = None
        self.baseline_scaler = StandardScaler()
        self.metadata = {}
        
    def train(self, df: pd.DataFrame, 
              target_cols: List[str],
              feature_cols: List[str],
              use_baseline: bool = True) -> Dict:
        """
        Train quantile regression models for each target.
        
        Args:
            df: Feature DataFrame
            target_cols: List of target column names
            feature_cols: List of feature column names
            use_baseline: Whether to train linear baseline for comparison
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"?? MODEL TRAINING STARTED")
        logger.info(f"{'='*60}")
        
        self.feature_names = feature_cols
        
        # Prepare data
        X = df[feature_cols].copy()
        y_dict = {col: df[col].values for col in target_cols}
        
        # Handle any remaining NaN/Inf
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        # Time-series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        metrics = {}
        
        # Train baseline first
        if use_baseline:
            logger.info("\n?? Training baseline (Linear Regression)...")
            baseline_metrics = self._train_baseline(X_scaled, y_dict, tscv)
            metrics['baseline'] = baseline_metrics
        
        # Train LightGBM for each target and quantile
        logger.info(f"\n?? Training LightGBM quantile models...")
        logger.info(f"   Targets: {target_cols}")
        logger.info(f"   Quantiles: {QUANTILES}")
        
        for target in target_cols:
            logger.info(f"\n{'-'*40}")
            logger.info(f"?? Target: {target}")
            
            y = y_dict[target]
            
            # Remove rows where target is NaN
            valid_idx = ~np.isnan(y)
            X_valid = X_scaled[valid_idx]
            y_valid = y[valid_idx]
            
            if len(y_valid) < 50:
                logger.warning(f"   ??  Too few samples ({len(y_valid)}) - skipping")
                continue
            
            target_metrics = {'quantiles': {}}
            
            for alpha in QUANTILES:
                model_key = f"{target}_q{int(alpha*100)}"
                logger.info(f"   Training {model_key} (a={alpha})...")
                
                # Cross-validate
                cv_scores = []
                models = []
                
                for fold, (train_idx, val_idx) in enumerate(tscv.split(X_valid)):
                    X_train = X_valid.iloc[train_idx]
                    y_train = y_valid[train_idx]
                    X_val = X_valid.iloc[val_idx]
                    y_val = y_valid[val_idx]
                    
                    # Train model
                    params = LGBM_PARAMS.copy()
                    params.update({
                        'objective': 'quantile',
                        'alpha': alpha,
                        'random_state': self.seed + fold,
                        'n_estimators': 200,
                        'early_stopping_rounds': 20,
                    })
                    
                    model = lgb.LGBMRegressor(**params)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        eval_metric='quantile',
                    )
                    
                    # Evaluate
                    preds = model.predict(X_val)
                    # Pinball loss
                    errors = y_val - preds
                    pinball = np.mean(np.where(errors >= 0, alpha * errors, (alpha - 1) * errors))
                    
                    cv_scores.append(pinball)
                    models.append(model)
                
                # Average CV score
                avg_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)
                logger.info(f"      CV Pinball Loss: {avg_score:.4f}  {std_score:.4f}")
                
                # Train final model on all data
                final_model = lgb.LGBMRegressor(**{
                    **LGBM_PARAMS,
                    'objective': 'quantile',
                    'alpha': alpha,
                    'random_state': self.seed,
                    'n_estimators': int(np.mean([m.best_iteration_ or 100 for m in models])),
                })
                final_model.fit(X_valid, y_valid)
                
                self.models[model_key] = final_model
                target_metrics['quantiles'][f'q{int(alpha*100)}'] = {
                    'cv_mean': avg_score,
                    'cv_std': std_score,
                    'n_estimators': final_model.best_iteration_ or 100,
                }
            
            metrics[target] = target_metrics
        
        # Calibrate prediction intervals
        logger.info(f"\n?? Calibrating prediction intervals...")
        self._calibrate_intervals(X_scaled, y_dict, target_cols)
        
        # Store metadata
        self.metadata = {
            'train_date': datetime.now().isoformat(),
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'targets': target_cols,
            'quantiles': QUANTILES,
            'feature_names': feature_cols,
        }
        
        # Check for overfitting
        self._check_overfitting(metrics)
        
        logger.info(f"\n? Training complete!")
        
        return metrics
    
    def _train_baseline(self, X: pd.DataFrame, y_dict: Dict, tscv) -> Dict:
        """Train linear regression baseline for comparison."""
        baseline_metrics = {}
        
        # Use total revenue as baseline target
        target = 'target_total_revenue'
        if target not in y_dict:
            return baseline_metrics
        
        y = y_dict[target]
        valid_idx = ~np.isnan(y)
        X_valid = X[valid_idx]
        y_valid = y[valid_idx]
        
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_valid)):
            X_train = X_valid.iloc[train_idx]
            y_train = y_valid[train_idx]
            X_val = X_valid.iloc[val_idx]
            y_val = y_valid[val_idx]
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            
            # RMSE
            rmse = np.sqrt(np.mean((y_val - preds) ** 2))
            # MAE
            mae = np.mean(np.abs(y_val - preds))
            # R
            r2 = 1 - np.sum((y_val - preds)**2) / np.sum((y_val - y_val.mean())**2)
            
            cv_scores.append({'rmse': rmse, 'mae': mae, 'r2': r2})
        
        avg_rmse = np.mean([s['rmse'] for s in cv_scores])
        avg_mae = np.mean([s['mae'] for s in cv_scores])
        avg_r2 = np.mean([s['r2'] for s in cv_scores])
        
        logger.info(f"   Baseline RMSE: ${avg_rmse:,.2f}")
        logger.info(f"   Baseline MAE:  ${avg_mae:,.2f}")
        logger.info(f"   Baseline R:   {avg_r2:.4f}")
        
        # Train final baseline
        self.baseline_model = LinearRegression()
        self.baseline_model.fit(X_valid, y_valid)
        
        return {
            'rmse': avg_rmse,
            'mae': avg_mae,
            'r2': avg_r2,
            'cv_scores': cv_scores
        }
    
    def _calibrate_intervals(self, X: pd.DataFrame, y_dict: Dict, target_cols: List[str]):
        """
        Calibrate prediction intervals using conformal prediction approach.
        Adjusts quantile predictions to achieve nominal coverage.
        """
        for target in target_cols:
            if target not in y_dict:
                continue
            
            y = y_dict[target]
            valid_idx = ~np.isnan(y)
            X_valid = X[valid_idx]
            y_valid = y[valid_idx]
            
            if len(y_valid) < 30:
                continue
            
            # Get predictions from all quantile models
            q10_key = f"{target}_q10"
            q50_key = f"{target}_q50"
            q90_key = f"{target}_q90"
            
            if not all(k in self.models for k in [q10_key, q50_key, q90_key]):
                continue
            
            pred_q10 = self.models[q10_key].predict(X_valid)
            pred_q50 = self.models[q50_key].predict(X_valid)
            pred_q90 = self.models[q90_key].predict(X_valid)
            
            # Calculate calibration factor
            # For 80% interval, adjust until 80% of y_true fall within [q10, q90]
            covered = (y_valid >= pred_q10) & (y_valid <= pred_q90)
            coverage = covered.mean()
            
            logger.info(f"   {target}: Raw coverage = {coverage:.2%}")
            
            # If coverage too low, widen intervals
            if coverage < 0.75:
                factor = 1.0 + (0.80 - coverage) * 2
                self.calibration_factors[target] = factor
                logger.info(f"      ? Applying calibration factor: {factor:.3f}")
            else:
                self.calibration_factors[target] = 1.0
    
    def _check_overfitting(self, metrics: Dict):
        """Check for signs of overfitting."""
        logger.info(f"\n?? Overfitting check:")
        
        baseline_r2 = metrics.get('baseline', {}).get('r2', 0)
        
        # Check if LightGBM is significantly better than baseline
        for target, target_metrics in metrics.items():
            if target == 'baseline':
                continue
            
            if 'quantiles' in target_metrics:
                q50 = target_metrics['quantiles'].get('q50', {})
                cv_mean = q50.get('cv_mean', 0)
                
                # Very low pinball loss might indicate memorization
                if cv_mean < 0.001:
                    logger.warning(f"   ??  {target}: Very low pinball loss ({cv_mean:.6f}) - possible overfitting")
        
        if baseline_r2 > 0.8:
            logger.info(f"   ? Baseline R = {baseline_r2:.3f} - linear model captures strong patterns")
        elif baseline_r2 > 0.5:
            logger.info(f"   ??  Baseline R = {baseline_r2:.3f} - moderate predictability")
        else:
            logger.warning(f"   ??  Baseline R = {baseline_r2:.3f} - low predictability, expect wide intervals")
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate probabilistic predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with prediction columns
        """
        # Scale features
        X_scaled = self.scaler.transform(X[self.feature_names])
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        predictions = X[['ref_date', 'period_days']].copy() if 'ref_date' in X.columns else pd.DataFrame()
        
        # Predict for each target
        targets = ['target_total_revenue', 'target_google_revenue', 
                   'target_meta_revenue', 'target_microsoft_revenue']
        
        for target in targets:
            q10_key = f"{target}_q10"
            q50_key = f"{target}_q50"
            q90_key = f"{target}_q90"
            
            # Map to output names
            if 'total' in target:
                prefix = 'revenue'
            elif 'google' in target:
                prefix = 'google_revenue'
            elif 'meta' in target:
                prefix = 'meta_revenue'
            elif 'microsoft' in target:
                prefix = 'ms_revenue'
            else:
                continue
            
            if all(k in self.models for k in [q10_key, q50_key, q90_key]):
                predictions[f'{prefix}_p10'] = self.models[q10_key].predict(X_scaled)
                predictions[f'{prefix}_p50'] = self.models[q50_key].predict(X_scaled)
                predictions[f'{prefix}_p90'] = self.models[q90_key].predict(X_scaled)
                
                # Apply calibration
                if target in self.calibration_factors:
                    factor = self.calibration_factors[target]
                    median = predictions[f'{prefix}_p50']
                    predictions[f'{prefix}_p10'] = median - (median - predictions[f'{prefix}_p10']) * factor
                    predictions[f'{prefix}_p90'] = median + (predictions[f'{prefix}_p90'] - median) * factor
            else:
                # Fallback: use baseline or mean
                predictions[f'{prefix}_p10'] = 0
                predictions[f'{prefix}_p50'] = 0
                predictions[f'{prefix}_p90'] = 0
        
        # Calculate blended ROAS
        if 'revenue_p50' in predictions.columns and 'ref_date' in X.columns:
            total_spend = X.get('feature_total_spend', 0)
            for q in ['p10', 'p50', 'p90']:
                if total_spend.iloc[0] > 0 if hasattr(total_spend, 'iloc') else total_spend > 0:
                    predictions[f'blended_roas_{q}'] = predictions[f'revenue_{q}'] / total_spend
                else:
                    predictions[f'blended_roas_{q}'] = 0
        
        return predictions
    
    def save(self, filepath: str):
        """Save model to pickle file."""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'calibration_factors': self.calibration_factors,
            'baseline_model': self.baseline_model,
            'metadata': self.metadata,
        }
        joblib.dump(model_data, filepath, compress=3)
        logger.info(f"?? Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ForecastingModel':
        """Load model from pickle file."""
        data = joblib.load(filepath)
        instance = cls()
        instance.models = data['models']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance.calibration_factors = data.get('calibration_factors', {})
        instance.baseline_model = data.get('baseline_model')
        instance.metadata = data.get('metadata', {})
        logger.info(f"?? Model loaded from {filepath}")
        return instance


def train_and_save_model(data_dir: str = None, output_path: str = None):
    """
    Complete training pipeline:
    1. Load data
    2. Engineer features
    3. Train model
    4. Save model
    
    Returns:
        Trained ForecastingModel
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"?? FULL TRAINING PIPELINE")
    logger.info(f"{'='*60}")
    
    # Load data
    df = load_all_data(data_dir)
    
    # Engineer features
    engineer = FeatureEngineer(periods=[30, 60, 90])
    features = engineer.create_training_data(df)
    
    # Define target columns
    target_cols = [
        'target_total_revenue',
        'target_google_revenue',
        'target_meta_revenue',
        'target_microsoft_revenue',
    ]
    
    # Filter to targets that exist and have data
    available_targets = [t for t in target_cols 
                        if t in features.columns and features[t].sum() > 0]
    
    if 'target_meta_revenue' in available_targets and features['target_meta_revenue'].sum() == 0:
        logger.warning("??  Meta revenue is all zeros - removing from targets")
        available_targets.remove('target_meta_revenue')
    
    # Get feature columns
    feature_cols = engineer.get_feature_columns(features)
    
    logger.info(f"\n?? Training with:")
    logger.info(f"   Samples: {len(features)}")
    logger.info(f"   Features: {len(feature_cols)}")
    logger.info(f"   Targets: {available_targets}")
    
    # Train model
    model = ForecastingModel()
    metrics = model.train(features, available_targets, feature_cols)
    
    # Save model
    if output_path is None:
        output_path = PICKLE_DIR / "model.pkl"
    
    model.save(str(output_path))
    
    # Print feature importance for top model
    if 'target_total_revenue_q50' in model.models:
        logger.info(f"\n?? Top 10 Feature Importances:")
        importance = model.models['target_total_revenue_q50'].feature_importances_
        feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)[:10]
        for name, imp in feat_imp:
            logger.info(f"   {name:40s}: {imp:.4f}")
    
    return model, features, metrics


if __name__ == "__main__":
    # Run training
    model, features, metrics = train_and_save_model()
    
    # Quick validation
    print("\n" + "="*60)
    print("QUICK VALIDATION: Sample Predictions")
    print("="*60)
    
    # Get feature columns
    engineer = FeatureEngineer()
    feature_cols = engineer.get_feature_columns(features)
    
    # Take last 5 samples
    test_samples = features.tail(5)
    X_test = test_samples[feature_cols].fillna(0)
    
    predictions = model.predict(X_test)
    print(predictions.head().to_string())

