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


def validate_campaign_consistency(df: pd.DataFrame) -> Dict:
    """
    Explicit campaign consistency validation for ingestion requirement.

    Checks:
      - Campaigns with inconsistent naming/type across dates
      - Campaigns with sudden platform reassignment
      - Missing campaign_id/campaign_type values
      - Date-range gaps per campaign (per platform)
    """
    logger.info(f"\n{'='*60}")
    logger.info("?? CAMPAIGN CONSISTENCY VALIDATION")
    logger.info(f"{'='*60}")

    required_cols = ['date', 'campaign_id', 'campaign_name', 'campaign_type', 'platform']
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        msg = f"Missing required columns for consistency validation: {missing_required}"
        logger.error(f"? {msg}")
        return {
            'passed': False,
            'checks': {},
            'summary': {'error': msg}
        }

    # Defensive normalization for comparisons
    work = df.copy()
    work['campaign_id'] = work['campaign_id'].astype(str)
    work['campaign_name'] = work['campaign_name'].astype(str)
    work['campaign_type'] = work['campaign_type'].astype(str)
    work['platform'] = work['platform'].astype(str).str.strip().str.lower()

    # Consider empty strings / None-like as missing
    def _is_missing_str(s: pd.Series) -> pd.Series:
        s2 = s.astype(str)
        s2 = s2.replace({'none': np.nan, 'nan': np.nan, 'null': np.nan})
        return s2.isna() | (s2.astype(str).str.strip() == '') | (s2.astype(str).str.strip().str.lower() == 'unknown')

    missing_campaign_id = _is_missing_str(work['campaign_id']).sum()
    missing_campaign_type = _is_missing_str(work['campaign_type']).sum()

    # Inconsistent naming/type across dates (per platform + campaign_id)
    grp = work.groupby(['platform', 'campaign_id'])
    inconsistent_name = []
    inconsistent_type = []
    for (platform, campaign_id), g in grp:
        names = sorted({str(x) for x in g['campaign_name'].unique() if str(x).strip() != ''})
        types = sorted({str(x) for x in g['campaign_type'].unique() if str(x).strip() != ''})
        if len(names) > 1:
            inconsistent_name.append({
                'platform': platform,
                'campaign_id': campaign_id,
                'campaign_name_variants': names[:5],
                'num_variants': len(names),
            })
        if len(types) > 1:
            inconsistent_type.append({
                'platform': platform,
                'campaign_id': campaign_id,
                'campaign_type_variants': types[:5],
                'num_variants': len(types),
            })

    # Sudden platform reassignment (per campaign_id across time)
    platform_segments = []
    for campaign_id, g in work.sort_values('date').groupby('campaign_id'):
        platforms = g['platform'].tolist()
        # Count transitions
        transitions = [0]
        for i in range(1, len(platforms)):
            transitions.append(1 if platforms[i] != platforms[i-1] else 0)
        num_transitions = int(sum(transitions))
        distinct_platforms = sorted(set(platforms))
        # Flag if campaign_id appears on more than one platform at any point
        if len(distinct_platforms) > 1:
            # Also compute contiguous segments count
            segment_count = 1
            for i in range(1, len(platforms)):
                if platforms[i] != platforms[i-1]:
                    segment_count += 1
            platform_segments.append({
                'campaign_id': campaign_id,
                'platforms_seen': distinct_platforms,
                'segment_count': segment_count,
                'date_start': g['date'].min().date(),
                'date_end': g['date'].max().date(),
            })

    # Date gaps per campaign (per platform, daily gaps between min..max)
    gap_rows = []
    # Reduce work: operate at per-day presence level
    presence = work[['platform', 'campaign_id', 'date']].drop_duplicates()
    for (platform, campaign_id), g in presence.groupby(['platform', 'campaign_id']):
        g = g.sort_values('date')
        min_d = g['date'].min()
        max_d = g['date'].max()
        if pd.isna(min_d) or pd.isna(max_d):
            continue
        # Build expected day index
        expected_days = pd.date_range(min_d, max_d, freq='D')
        actual_days = set(g['date'].dt.normalize().tolist())
        missing_days = [d for d in expected_days.normalize().tolist() if d not in actual_days]
        if len(missing_days) > 0:
            # Collapse into ranges for compact reporting
            missing_days_sorted = sorted(missing_days)
            ranges = []
            start = missing_days_sorted[0]
            prev = start
            for d in missing_days_sorted[1:]:
                if (d - prev).days == 1:
                    prev = d
                    continue
                ranges.append((start.date(), prev.date()))
                start = d
                prev = d
            ranges.append((start.date(), prev.date()))
            gap_rows.append({
                'platform': platform,
                'campaign_id': campaign_id,
                'missing_day_count': int(len(missing_days)),
                'gap_ranges': [(str(a), str(b)) for a, b in ranges[:5]],
                'date_start': min_d.date(),
                'date_end': max_d.date(),
            })

    # Assemble report
    checks = {
        'missing_campaign_values': {
            'passed': (missing_campaign_id == 0) and (missing_campaign_type == 0),
            'details': f"Missing campaign_id rows: {int(missing_campaign_id)}, missing campaign_type rows: {int(missing_campaign_type)}",
            'examples': []
        },
        'inconsistent_campaign_name_across_dates': {
            'passed': len(inconsistent_name) == 0,
            'details': f"Inconsistent name variants found in {len(inconsistent_name)} (platform,campaign_id) groups",
            'examples': inconsistent_name[:10],
        },
        'inconsistent_campaign_type_across_dates': {
            'passed': len(inconsistent_type) == 0,
            'details': f"Inconsistent campaign_type variants found in {len(inconsistent_type)} (platform,campaign_id) groups",
            'examples': inconsistent_type[:10],
        },
        'sudden_platform_reassignment': {
            'passed': len(platform_segments) == 0,
            'details': f"Campaign IDs reassigned across platforms: {len(platform_segments)}",
            'examples': platform_segments[:10],
        },
        'date_range_gaps_per_campaign': {
            'passed': len(gap_rows) == 0,
            'details': f"Campaigns with daily date gaps (per platform): {len(gap_rows)}",
            'examples': gap_rows[:10],
        },
    }

    passed = all(v.get('passed', False) for v in checks.values())
    summary = {
        'passed': passed,
        'missing_campaign_id_rows': int(missing_campaign_id),
        'missing_campaign_type_rows': int(missing_campaign_type),
        'inconsistent_name_groups': len(inconsistent_name),
        'inconsistent_type_groups': len(inconsistent_type),
        'platform_reassignment_campaign_ids': len(platform_segments),
        'campaigns_with_date_gaps': len(gap_rows),
    }

    # Human-readable logging
    passed_count = sum(1 for v in checks.values() if v.get('passed'))
    total_checks = len(checks)
    logger.info(f"   Results: {passed_count}/{total_checks} checks passed")
    for name, result in checks.items():
        icon = "?" if result['passed'] else "!"
        logger.info(f"   {icon} {name}: {result.get('details','')}")
    logger.info(f"{'='*60}\n")

    return {'passed': passed, 'summary': summary, 'checks': checks}


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

    # Explicit ingestion requirement: campaign consistency validation
    try:
        _ = validate_campaign_consistency(combined)
    except Exception as e:
        logger.error(f"? Campaign consistency validation failed: {e}")

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

