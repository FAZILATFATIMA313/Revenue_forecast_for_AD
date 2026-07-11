# src/data_loader.py
"""
Flexible data loader that auto-detects platform from CSV structure
and normalizes to a common schema.
"""
import pandas as pd
import numpy as np
import glob
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from src.logger import setup_logger
from src.config import DATA_DIR

logger = setup_logger(__name__)

# Standard column names after normalization
STANDARD_COLS = [
    'date', 'campaign_id', 'campaign_name', 'campaign_type',
    'spend', 'revenue', 'clicks', 'impressions', 'conversions', 'platform'
]

# Platform detection patterns (case-insensitive)
PLATFORM_PATTERNS = {
    'google': {
        'required_cols': ['segments_date', 'metrics_cost_micros'],
        'optional_cols': ['metrics_conversions_value', 'metrics_clicks', 
                         'metrics_impressions', 'metrics_conversions']
    },
    'microsoft': {
        'required_cols': ['timeperiod', 'spend'],
        'optional_cols': ['revenue', 'clicks', 'impressions', 'conversions']
    },
    'meta': {
        'required_cols': ['spend'],
        'optional_cols': ['revenue', 'clicks', 'impressions', 'conversions']
    }
}


def detect_platform(df: pd.DataFrame) -> str:
    """
    Auto-detect platform from column names.
    
    Returns:
        'google', 'microsoft', or 'meta'
    """
    cols = set(df.columns.str.lower().str.replace(' ', '_').str.strip())
    
    # Check Google patterns
    if 'segments_date' in cols and 'metrics_cost_micros' in cols:
        logger.info("?? Detected: Google Ads")
        return 'google'
    
    # Check Microsoft patterns
    if 'timeperiod' in cols:
        logger.info("?? Detected: Microsoft Ads")
        return 'microsoft'
    
    # Default to Meta
    logger.info("?? Detected: Meta Ads (default)")
    return 'meta'


def normalize_google(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Google Ads CSV to standard schema.
    Google uses cost in micros (1,000,000 micros = 1 unit)
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()
    
    mapping = {
        'segments_date': 'date',
        'campaign_id': 'campaign_id',
        'campaign_name': 'campaign_name',
        'campaign_advertising_channel_type': 'campaign_type',
        'metrics_cost_micros': 'spend_raw',
        'metrics_conversions_value': 'revenue',
        'metrics_clicks': 'clicks',
        'metrics_impressions': 'impressions',
        'metrics_conversions': 'conversions'
    }
    
    # Only rename columns that exist
    rename_dict = {k: v for k, v in mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)
    
    # Convert micros to actual currency (divide by 1,000,000)
    if 'spend_raw' in df.columns:
        df['spend'] = pd.to_numeric(df['spend_raw'], errors='coerce').fillna(0) / 1_000_000
        df.drop(columns=['spend_raw'], inplace=True)
    
    df['platform'] = 'google'
    return df


def normalize_microsoft(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Microsoft Ads CSV to standard schema."""
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()
    
    mapping = {
        'timeperiod': 'date',
        'campaignid': 'campaign_id',
        'campaignname': 'campaign_name',
        'campaigntype': 'campaign_type',
        'spend': 'spend',
        'revenue': 'revenue',
        'clicks': 'clicks',
        'impressions': 'impressions',
        'conversions': 'conversions'
    }
    
    rename_dict = {k: v for k, v in mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)
    df['platform'] = 'microsoft'
    return df


def normalize_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Meta Ads CSV to standard schema."""
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()
    
    # Meta commonly uses date_start, campaign_id, campaign_name
    date_cols = [c for c in df.columns if 'date' in c and 'start' in c]
    campaign_id_cols = [c for c in df.columns if 'campaign' in c and 'id' in c]
    campaign_name_cols = [c for c in df.columns if 'campaign' in c and 'name' in c]
    
    mapping = {}
    
    if date_cols:
        mapping[date_cols[0]] = 'date'
    if campaign_id_cols:
        mapping[campaign_id_cols[0]] = 'campaign_id'
    if campaign_name_cols:
        mapping[campaign_name_cols[0]] = 'campaign_name'
    
    # Map common metric columns
    metric_patterns = {
        'spend': 'spend',
        'revenue': 'revenue',
        'purchase_value': 'revenue',
        'clicks': 'clicks',
        'impressions': 'impressions',
        'conversion': 'revenue',
        'conversions': 'conversions',
        'purchase': 'conversions'
    }
    
    for col in df.columns:
        col_lower = col.lower()
        for pattern, target in metric_patterns.items():
            if pattern in col_lower and target not in mapping.values():
                mapping[col] = target
                break
    
    rename_dict = {k: v for k, v in mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)
    df['platform'] = 'meta'
    
    # Add missing columns with zeros
    for col in ['clicks', 'impressions', 'conversions']:
        if col not in df.columns:
            df[col] = 0
    
    return df


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate data types, handle missing values, remove duplicates.
    """
    logger.info("?? Cleaning and validating data...")
    
    # Ensure all expected columns exist
    for col in STANDARD_COLS:
        if col not in df.columns:
            if col in ['spend', 'revenue', 'clicks', 'impressions', 'conversions']:
                df[col] = 0
            else:
                df[col] = 'unknown'
            logger.warning(f"Column '{col}' missing - filling with default")
    
    # Convert numeric columns
    numeric_cols = ['spend', 'revenue', 'clicks', 'impressions', 'conversions']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Handle negative values (set to 0)
    for col in numeric_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            logger.warning(f"?? Found {neg_count} negative values in '{col}' - setting to 0")
            df[col] = df[col].clip(lower=0)
    
    # Parse dates with multiple format attempts
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Drop rows with invalid dates
    invalid_dates = df['date'].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"?? Dropping {invalid_dates} rows with invalid dates")
        df = df.dropna(subset=['date'])
    
    # Remove duplicates (same campaign, same date, same platform)
    before = len(df)
    df = df.drop_duplicates(subset=['campaign_id', 'date', 'platform'], keep='last')
    after = len(df)
    if before > after:
        logger.info(f"Removed {before - after} duplicate rows")
    
    # Sort by date
    df = df.sort_values(['platform', 'campaign_id', 'date']).reset_index(drop=True)
    
    return df[STANDARD_COLS]


def load_single_file(filepath: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load and normalize a single CSV file.
    
    Returns:
        Tuple of (DataFrame or None if failed, status message)
    """
    try:
        logger.info(f"?? Loading: {os.path.basename(filepath)}")
        df = pd.read_csv(filepath, encoding='utf-8-sig', low_memory=False)
        logger.info(f"   Shape: {df.shape}, Columns: {list(df.columns)[:5]}...")
        
        if df.empty:
            return None, "Empty file"
        
        platform = detect_platform(df)
        
        if platform == 'google':
            df = normalize_google(df)
        elif platform == 'microsoft':
            df = normalize_microsoft(df)
        else:
            df = normalize_meta(df)
        
        df = validate_and_clean(df)
        logger.info(f"   ? Normalized: {len(df)} rows, {df['campaign_id'].nunique()} campaigns")
        return df, "success"
        
    except Exception as e:
        logger.error(f"   ? Failed to load {filepath}: {str(e)}")
        return None, str(e)


def load_all_data(data_dir: str = None) -> pd.DataFrame:
    """
    Load all CSV files from data directory and combine into single DataFrame.
    
    Args:
        data_dir: Path to data directory (default: from config)
    
    Returns:
        Combined DataFrame with all platforms
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    data_dir = Path(data_dir)
    logger.info(f"{'='*60}")
    logger.info(f"?? Loading data from: {data_dir}")
    logger.info(f"{'='*60}")
    
    csv_files = sorted(glob.glob(str(data_dir / "*.csv")))
    
    if not csv_files:
        logger.error(f"? No CSV files found in {data_dir}")
        raise FileNotFoundError(f"No CSV files in {data_dir}")
    
    logger.info(f"Found {len(csv_files)} file(s): {[os.path.basename(f) for f in csv_files]}")
    
    all_dfs = []
    stats = {'success': 0, 'failed': 0, 'files': []}
    
    for filepath in csv_files:
        df, status = load_single_file(filepath)
        
        if df is not None:
            all_dfs.append(df)
            stats['success'] += 1
            stats['files'].append({
                'name': os.path.basename(filepath),
                'rows': len(df),
                'campaigns': df['campaign_id'].nunique(),
                'platforms': df['platform'].unique().tolist(),
                'date_range': f"{df['date'].min().date()} to {df['date'].max().date()}"
            })
        else:
            stats['failed'] += 1
    
    if not all_dfs:
        raise ValueError("No valid data loaded from any file!")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(['platform', 'campaign_id', 'date']).reset_index(drop=True)
    
    # Final deduplication across files
    before = len(combined)
    combined = combined.drop_duplicates(subset=['platform', 'campaign_id', 'date'], keep='last')
    after = len(combined)
    if before > after:
        logger.info(f"Removed {before - after} cross-file duplicates")
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"?? DATA LOADING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total rows:         {len(combined):,}")
    logger.info(f"Total campaigns:     {combined['campaign_id'].nunique():,}")
    logger.info(f"Date range:          {combined['date'].min().date()} to {combined['date'].max().date()}")
    logger.info(f"Platforms found:     {combined['platform'].unique().tolist()}")
    logger.info(f"Campaign types:      {combined['campaign_type'].unique().tolist()}")
    
    logger.info(f"\n?? Platform breakdown:")
    for plat in combined['platform'].unique():
        plat_df = combined[combined['platform'] == plat]
        logger.info(f"  {plat.upper():12s}: {len(plat_df):,} rows, "
                    f"{plat_df['campaign_id'].nunique():,} campaigns, "
                    f"${plat_df['spend'].sum():,.2f} spend, "
                    f"${plat_df['revenue'].sum():,.2f} revenue")
    
    # Data quality checks
    zero_spend = (combined['spend'] == 0).sum()
    zero_revenue = (combined['revenue'] == 0).sum()
    if zero_spend > 0:
        logger.info(f"\n??  {zero_spend} rows ({100*zero_spend/len(combined):.1f}%) have zero spend")
    if zero_revenue > 0:
        logger.info(f"??  {zero_revenue} rows ({100*zero_revenue/len(combined):.1f}%) have zero revenue")
    
    return combined


def get_data_summary(df: pd.DataFrame) -> Dict:
    """Generate summary statistics for the loaded data."""
    summary = {
        'total_rows': len(df),
        'total_campaigns': df['campaign_id'].nunique(),
        'date_range': (df['date'].min(), df['date'].max()),
        'platforms': df['platform'].unique().tolist(),
        'campaign_types': df['campaign_type'].unique().tolist(),
        'platform_stats': {}
    }
    
    for plat in df['platform'].unique():
        plat_df = df[df['platform'] == plat]
        summary['platform_stats'][plat] = {
            'rows': len(plat_df),
            'campaigns': plat_df['campaign_id'].nunique(),
            'total_spend': float(plat_df['spend'].sum()),
            'total_revenue': float(plat_df['revenue'].sum()),
            'avg_daily_spend': float(plat_df['spend'].mean()),
            'roas': float(plat_df['revenue'].sum() / plat_df['spend'].sum()) 
                    if plat_df['spend'].sum() > 0 else 0
        }
    
    return summary


if __name__ == "__main__":
    # Test the loader
    try:
        df = load_all_data()
        
        print("\n" + "="*60)
        print("?? SAMPLE DATA (first 5 rows)")
        print("="*60)
        print(df.head().to_string())
        
        print("\n" + "="*60)
        print("?? DATA SUMMARY")
        print("="*60)
        summary = get_data_summary(df)
        for key, value in summary.items():
            if key != 'platform_stats':
                print(f"{key}: {value}")
        
        print("\n" + "="*60)
        print("?? COLUMN INFO")
        print("="*60)
        print(df.dtypes)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

