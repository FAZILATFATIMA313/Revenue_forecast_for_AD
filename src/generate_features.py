# src/generate_features.py
"""
Feature generation script for hackathon submission.
Reads data from DATA_DIR and creates forecast features.
Handles both training (creates feature matrix) and inference (creates forecast requests).
"""
import pandas as pd
import numpy as np
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from src.data_loader import load_all_data
from src.features import FeatureEngineer
from src.logger import setup_logger


logger = setup_logger(__name__)


def detect_forecast_requests(data_dir: str) -> pd.DataFrame:
    """
    Check if there's a forecast_requests.csv or similar file.
    If not, generate default forecast scenarios.
    
    Returns DataFrame with columns: request_id, period_days, spend_google, spend_meta, spend_ms
    """
    requests_file = None
    for fname in ['forecast_requests.csv', 'requests.csv', 'budget_scenarios.csv']:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            requests_file = fpath
            break
    
    if requests_file:
        logger.info(f"?? Found forecast requests file: {os.path.basename(requests_file)}")
        requests_df = pd.read_csv(requests_file)
        
        # Normalize column names
        col_map = {
            'request_id': 'request_id',
            'period_days': 'period_days',
            'period': 'period_days',
            'days': 'period_days',
            'spend_google': 'spend_google',
            'google_spend': 'spend_google',
            'google_budget': 'spend_google',
            'spend_meta': 'spend_meta',
            'meta_spend': 'spend_meta',
            'meta_budget': 'spend_meta',
            'spend_ms': 'spend_ms',
            'microsoft_spend': 'spend_ms',
            'ms_spend': 'spend_ms',
            'bing_spend': 'spend_ms',
        }
        
        requests_df = requests_df.rename(columns={
            k: v for k, v in col_map.items() if k in requests_df.columns
        })
        
        return requests_df
    
    else:
        logger.info("?? No forecast requests file found - generating default scenarios")
        
        # Load data to get recent spend patterns
        df = load_all_data(data_dir)
        
        # Get last 30 days average spend per platform
        last_date = df['date'].max()
        recent = df[df['date'] >= last_date - timedelta(days=30)]
        
        avg_spend = {}
        for plat in ['google', 'meta', 'microsoft']:
            plat_data = recent[recent['platform'] == plat]
            avg_spend[plat] = plat_data['spend'].sum() / 30 if len(plat_data) > 0 else 0
        
        logger.info(f"   Recent daily avg spend: Google=${avg_spend['google']:,.2f}, "
                   f"Meta=${avg_spend['meta']:,.2f}, MS=${avg_spend['microsoft']:,.2f}")
        
        # Generate scenarios for each period
        scenarios = []
        for period_days in [30, 60, 90]:
            scenarios.append({
                'request_id': f'default_{period_days}d',
                'period_days': period_days,
                'spend_google': avg_spend['google'] * period_days,
                'spend_meta': avg_spend['meta'] * period_days,
                'spend_ms': avg_spend['microsoft'] * period_days,
            })
        
        return pd.DataFrame(scenarios)


def generate_features(data_dir: str, output_path: str):
    """
    Main function: Generate features for forecasting.
    
    Steps:
    1. Load historical data
    2. Detect or create forecast requests
    3. For each request, create feature vector
    4. Save to parquet
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"?? GENERATING FORECAST FEATURES")
    logger.info(f"{'='*60}")
    
    # Load historical data
    df = load_all_data(data_dir)
    
    # Aggregate to daily platform level
    daily = df.groupby(['date', 'platform']).agg({
        'spend': 'sum',
        'revenue': 'sum',
        'clicks': 'sum',
        'impressions': 'sum',
        'conversions': 'sum'
    }).reset_index()
    
    # Pivot
    daily_pivot = daily.pivot_table(
        index='date',
        columns='platform',
        values=['spend', 'revenue', 'clicks', 'impressions', 'conversions'],
        aggfunc='sum',
        fill_value=0
    )
    daily_pivot.columns = [f'{col[0]}_{col[1]}' for col in daily_pivot.columns]
    daily_pivot = daily_pivot.reset_index().sort_values('date')
    
    # Add totals
    for metric in ['spend', 'revenue', 'clicks', 'impressions', 'conversions']:
        plat_cols = [c for c in daily_pivot.columns if c.startswith(metric)]
        daily_pivot[f'total_{metric}'] = daily_pivot[plat_cols].sum(axis=1)
    
    logger.info(f"   Historical data: {len(daily_pivot)} days, "
               f"{daily_pivot['date'].min().date()} to {daily_pivot['date'].max().date()}")
    
    # Get forecast requests
    requests = detect_forecast_requests(data_dir)
    logger.info(f"   Forecast requests: {len(requests)}")
    
    # Engineer feature for each request
    engineer = FeatureEngineer()
    last_date = daily_pivot['date'].max()
    
    all_features = []
    
    for _, req in requests.iterrows():
        period_days = int(req['period_days'])
        ref_date = last_date + timedelta(days=1)  # Forecast from tomorrow
        
        # Feature window: last N days
        feature_start = ref_date - timedelta(days=period_days)
        feature_end = ref_date - timedelta(days=1)
        
        feature_mask = (daily_pivot['date'] >= feature_start) & (daily_pivot['date'] <= feature_end)
        feature_data = daily_pivot[feature_mask]
        
        # Build feature row
        row = {
            'request_id': req.get('request_id', f'req_{len(all_features)}'),
            'period_days': period_days,
            'ref_date': ref_date,
            'spend_google': req['spend_google'],
            'spend_meta': req.get('spend_meta', 0),
            'spend_ms': req.get('spend_ms', 0),
        }
        
        # Aggregate feature window
        metric_cols = [c for c in daily_pivot.columns if c != 'date']
        for col in metric_cols:
            row[f'feature_{col}'] = feature_data[col].sum() if len(feature_data) > 0 else 0
            row[f'feature_daily_avg_{col}'] = feature_data[col].mean() if len(feature_data) > 0 else 0
            row[f'feature_days_with_{col}'] = (feature_data[col] > 0).sum() if len(feature_data) > 0 else 0
        
        # Override with requested budgets
        row['feature_spend_google'] = req['spend_google']
        row['feature_spend_meta'] = req.get('spend_meta', 0)
        row['feature_spend_microsoft'] = req.get('spend_ms', 0)
        row['feature_total_spend'] = row['feature_spend_google'] + row['feature_spend_meta'] + row['feature_spend_microsoft']
        
        # Spend shares
        total = row['feature_total_spend']
        for plat in ['google', 'meta', 'microsoft']:
            row[f'feature_spend_share_{plat}'] = row[f'feature_spend_{plat}'] / total if total > 0 else 0
        
        # ROAS from feature window
        for plat in ['google', 'meta', 'microsoft']:
            spend_col = f'feature_spend_{plat}'
            rev_col = f'feature_revenue_{plat}'
            spend = row.get(spend_col, 0)
            rev = row.get(rev_col, 0)
            row[f'feature_roas_{plat}'] = rev / spend if spend > 0 else 0
        
        total_spend = row.get('feature_total_spend', 0)
        total_rev = row.get('feature_total_revenue', 0)
        row['feature_blended_roas'] = total_rev / total_spend if total_spend > 0 else 0
        
        # Time features
        row['month'] = ref_date.month
        row['quarter'] = ref_date.quarter
        row['year'] = ref_date.year
        row['day_of_month'] = ref_date.day
        row['day_of_week'] = ref_date.weekday()
        row['week_of_year'] = ref_date.isocalendar()[1]
        row['is_weekend'] = 1 if ref_date.weekday() in [5, 6] else 0
        row['is_month_start'] = 1 if ref_date.day in [1, 2, 3] else 0
        row['is_month_end'] = 1 if ref_date.day in [28, 29, 30, 31] else 0
        
        for m in range(1, 13):
            row[f'is_month_{m}'] = 1 if ref_date.month == m else 0
        for q in range(1, 5):
            row[f'is_q{q}'] = 1 if ref_date.quarter == q else 0
        
        row['is_holiday'] = 0  # Simplified
        row['weekend_ratio'] = 2/7
        
        min_date = daily_pivot['date'].min()
        row['days_since_start'] = (ref_date - min_date).days
        row['trend'] = row['days_since_start'] / 365.25
        
        # src/generate_features.py - Replace the campaign type section (around line 180-195)
# Find this block in generate_features() and replace with:

        # Campaign type features - MATCH TRAINING NAMING
        # Normalize campaign type names to match training
        type_mapping = {
            'SEARCH': 'search',
            'Search': 'search',
            'PERFORMANCE_MAX': 'pmax',
            'PerformanceMax': 'pmax',
            'DISPLAY': 'display',
            'Display': 'display',
            'VIDEO': 'video',
            'Video': 'video',
            'DEMAND_GEN': 'demand_gen',
            'SHOPPING': 'shopping',
            'Shopping': 'shopping',
            'Audience': 'audience',
            'unknown': 'other',
        }
        
        df_normalized = df.copy()
        df_normalized['campaign_type_normalized'] = df_normalized['campaign_type'].map(type_mapping).fillna('other')
        
        ctype_data = df_normalized.groupby(['date', 'campaign_type_normalized']).agg({
            'spend': 'sum', 'revenue': 'sum'
        }).reset_index()
        
        ctype_mask = (ctype_data['date'] >= feature_start) & (ctype_data['date'] <= feature_end)
        ctype_window = ctype_data[ctype_mask]
        
        for ctype in ctype_window['campaign_type_normalized'].unique():
            ctype_spend = ctype_window[ctype_window['campaign_type_normalized'] == ctype]['spend'].sum()
            ctype_rev = ctype_window[ctype_window['campaign_type_normalized'] == ctype]['revenue'].sum()
            row[f'feature_spend_ctype_{ctype}'] = ctype_spend
            row[f'feature_revenue_ctype_{ctype}'] = ctype_rev
            row[f'feature_roas_ctype_{ctype}'] = ctype_rev / ctype_spend if ctype_spend > 0 else 0
        all_features.append(row)
    
    # Create DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Fill missing columns with 0
    features_df = features_df.fillna(0)
    
    # Save
    features_df.to_parquet(output_path, index=False)
    logger.info(f"\n? Features saved to {output_path}")
    logger.info(f"   Shape: {features_df.shape}")
    logger.info(f"   Requests: {features_df['request_id'].tolist()}")
    
    return features_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data', help='Data directory')
    parser.add_argument('--out', default='features.parquet', help='Output path')
    args = parser.parse_args()
    
    generate_features(args.data_dir, args.out)

