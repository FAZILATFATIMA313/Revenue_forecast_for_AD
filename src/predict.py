# src/predict.py
"""
Prediction script for hackathon submission.
Loads model and features, generates probabilistic forecasts.
"""
import pandas as pd
import numpy as np
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import ForecastingModel
from src.logger import setup_logger

logger = setup_logger(__name__)

# Required output columns
OUTPUT_COLUMNS = [
    'request_id', 'period_days',
    'spend_google', 'spend_meta', 'spend_ms',
    'revenue_p10', 'revenue_p50', 'revenue_p90',
    'blended_roas_p10', 'blended_roas_p50', 'blended_roas_p90',
    'google_revenue_p10', 'google_revenue_p50', 'google_revenue_p90',
    'meta_revenue_p10', 'meta_revenue_p50', 'meta_revenue_p90',
    'ms_revenue_p10', 'ms_revenue_p50', 'ms_revenue_p90'
]


def predict(features_path: str, model_path: str, output_path: str):
    """
    Generate predictions from features and model.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"?? GENERATING PREDICTIONS")
    logger.info(f"{'='*60}")
    
    # Load features
    logger.info(f"?? Loading features from {features_path}")
    features = pd.read_parquet(features_path)
    logger.info(f"   Shape: {features.shape}")
    
    # Load model
    logger.info(f"?? Loading model from {model_path}")
    model = ForecastingModel.load(model_path)
    logger.info(f"   Features expected: {len(model.feature_names)}")
    logger.info(f"   Models trained: {len(model.models)}")
    
    # Align features with training
    X = features.copy()
    
    # Fill missing feature columns
    for col in model.feature_names:
        if col not in X.columns:
            X[col] = 0
            logger.warning(f"   Missing feature: {col} - filling with 0")
    
    # Select only feature columns in correct order
    X_features = X[model.feature_names].fillna(0)
    
    # Scale
    X_scaled = model.scaler.transform(X_features)
    X_scaled_df = pd.DataFrame(X_scaled, columns=model.feature_names)
    
    # Generate predictions
    logger.info("\n?? Generating predictions...")
    predictions = []
    
    for i, (_, row) in enumerate(features.iterrows()):
        x_row = X_scaled_df.iloc[[i]]
        
        # Build prediction with DIRECT values - no nested functions
        try:
            req_id = str(row.get('request_id', f'req_{i}'))
        except:
            req_id = f'req_{i}'
        
        try:
            pd_val = int(float(row.get('period_days', 30)))
        except:
            pd_val = 30
        
        try:
            sg = float(row.get('spend_google', row.get('feature_spend_google', 0)))
        except:
            sg = 0.0
        
        try:
            sm = float(row.get('spend_meta', row.get('feature_spend_meta', 0)))
        except:
            sm = 0.0
        
        try:
            sms = float(row.get('spend_ms', row.get('feature_spend_microsoft', 0)))
        except:
            sms = 0.0
        
        pred = {
            'request_id': req_id,
            'period_days': pd_val,
            'spend_google': sg,
            'spend_meta': sm,
            'spend_ms': sms,
        }
            # Total revenue predictions
        for q_label, q_key in [('p10', 'target_total_revenue_q10'),
                                    ('p50', 'target_total_revenue_q50'),
                                    ('p90', 'target_total_revenue_q90')]:
                if q_key in model.models:
                    val = model.models[q_key].predict(x_row)[0]
                    pred[f'revenue_{q_label}'] = float(max(0, val))
                else:
                    pred[f'revenue_{q_label}'] = 0.0
            
            # Google revenue
        for q_label, q_key in [('p10', 'target_google_revenue_q10'),
                                    ('p50', 'target_google_revenue_q50'),
                                    ('p90', 'target_google_revenue_q90')]:
                if q_key in model.models:
                    val = model.models[q_key].predict(x_row)[0]
                    pred[f'google_revenue_{q_label}'] = float(max(0, val))
                else:
                    pred[f'google_revenue_{q_label}'] = 0.0
            
            # Meta revenue
        for q_label, q_key in [('p10', 'target_meta_revenue_q10'),
                                    ('p50', 'target_meta_revenue_q50'),
                                    ('p90', 'target_meta_revenue_q90')]:
                if q_key in model.models:
                    val = model.models[q_key].predict(x_row)[0]
                    pred[f'meta_revenue_{q_label}'] = float(max(0, val))
                else:
                    pred[f'meta_revenue_{q_label}'] = 0.0
            
            # Microsoft revenue
        for q_label, q_key in [('p10', 'target_microsoft_revenue_q10'),
                                    ('p50', 'target_microsoft_revenue_q50'),
                                    ('p90', 'target_microsoft_revenue_q90')]:
                if q_key in model.models:
                    val = model.models[q_key].predict(x_row)[0]
                    pred[f'ms_revenue_{q_label}'] = float(max(0, val))
                else:
                    pred[f'ms_revenue_{q_label}'] = 0.0
            
            # Blended ROAS
        total_spend = pred['spend_google'] + pred['spend_meta'] + pred['spend_ms']
        for q_label in ['p10', 'p50', 'p90']:
                if total_spend > 0:
                    pred[f'blended_roas_{q_label}'] = float(pred[f'revenue_{q_label}'] / total_spend)
                else:
                    pred[f'blended_roas_{q_label}'] = 0.0
            
            # FINAL SAFETY: Convert ALL values to float/int
        for key in pred:
                if key == 'request_id':
                    pred[key] = str(pred[key])
                elif key == 'period_days':
                    pred[key] = int(pred[key])
                else:
                    pred[key] = float(pred[key])
            
        predictions.append(pred)
    
    # Create output DataFrame
    output_df = pd.DataFrame(predictions)
    
    # FORCE all numeric columns to proper types
    for col in output_df.columns:
        if col == 'request_id':
            output_df[col] = output_df[col].astype(str)
        elif col == 'period_days':
            output_df[col] = output_df[col].astype(int)
        else:
            output_df[col] = pd.to_numeric(output_df[col], errors='coerce').fillna(0.0)

    # Ensure all required columns exist and have stable dtypes
    for col in OUTPUT_COLUMNS:
        if col not in output_df.columns:
            output_df[col] = 0.0

    output_df = output_df[OUTPUT_COLUMNS]

    # Save with clean formatting
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    output_df.to_csv(output_path, index=False, float_format='%.2f')
    
    logger.info(f"\nPredictions saved to {output_path}")
    logger.info(f"   Shape: {output_df.shape}")
    logger.info(f"\nPrediction Summary:")
    logger.info(f"{'='*60}")
    
    for _, row in output_df.iterrows():
        logger.info(f"\n   Request: {row['request_id']} ({row['period_days']} days)")
        logger.info(f"   Budget: Google=${row['spend_google']:,.0f}, "
                        f"Meta=${row['spend_meta']:,.0f}, MS=${row['spend_ms']:,.0f}")
        logger.info(f"   Revenue: ${row['revenue_p10']:,.0f} - ${row['revenue_p50']:,.0f} - ${row['revenue_p90']:,.0f}")
        logger.info(f"   ROAS: {row['blended_roas_p10']:.2f}x - {row['blended_roas_p50']:.2f}x - {row['blended_roas_p90']:.2f}x")
    
    logger.info(f"\n{'='*60}\n")
    
    for col in output_df.columns:
        if col == 'request_id':
            output_df[col] = output_df[col].astype(str).str.replace(' 00:00:00', '').str.strip()
        else:
            output_df[col] = pd.to_numeric(output_df[col], errors='coerce').fillna(0)
    
    # Ensure period_days is int
    output_df['period_days'] = output_df['period_days'].astype(int)
    
    return output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default='features.parquet', help='Features file')
    parser.add_argument('--model', default='./pickle/model.pkl', help='Model path')
    parser.add_argument('--output', default='./output/predictions.csv', help='Output path')
    args = parser.parse_args()
    
    predict(args.features, args.model, args.output)

