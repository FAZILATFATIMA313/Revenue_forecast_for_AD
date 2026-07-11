# src/features.py

"""

Feature engineering pipeline for e-commerce forecasting.

Transforms daily campaign-level data into planning-period features

suitable for probabilistic forecasting models.

"""

import pandas as pd

import numpy as np

from typing import Dict, List, Tuple, Optional

from datetime import datetime, timedelta

from pathlib import Path

import holidays

from src.logger import setup_logger

from src.config import DEFAULT_PERIODS, RANDOM_SEED

logger = setup_logger(__name__)

# Indian holidays (add more as needed)

INDIAN_HOLIDAYS = holidays.India(years=[2024, 2025, 2026])

# Global shopping events

SHOPPING_EVENTS = {

    # Format: 'YYYY-MM-DD': 'event_name'

    '2024-01-26': 'Republic Day',

    '2024-08-15': 'Independence Day',

    '2024-10-02': 'Gandhi Jayanti',

    '2024-10-31': 'Diwali',

    '2024-11-01': 'Diwali',

    '2024-11-02': 'Diwali',

    '2024-11-29': 'Black Friday',

    '2024-12-02': 'Cyber Monday',

    '2025-01-26': 'Republic Day',

    '2025-08-15': 'Independence Day',

    '2025-10-02': 'Gandhi Jayanti',

    '2025-10-20': 'Diwali',

    '2025-10-21': 'Diwali',

    '2025-11-28': 'Black Friday',

    '2025-12-01': 'Cyber Monday',

    '2026-01-26': 'Republic Day',

    '2026-08-15': 'Independence Day',

    '2026-10-02': 'Gandhi Jayanti',

}

# End-of-month and start-of-month (common salary/spending patterns)

EOM_START_DAYS = [1, 2, 3, 28, 29, 30, 31]

class FeatureEngineer:

    """

    Creates training-ready feature matrices from daily marketing data.

    Key features created:

    - Aggregated spend/revenue per channel per period

    - Log-transformed spend (captures diminishing returns)

    - Time-based features (month, quarter, weekend ratio)

    - Seasonality indicators (holidays, shopping events)

    - Campaign-type level aggregation

    """

    def __init__(self, periods: List[int] = None, seed: int = RANDOM_SEED):

        self.periods = periods or DEFAULT_PERIODS

        self.seed = seed

        np.random.seed(seed)

    def create_training_data(self, df: pd.DataFrame,

                            min_campaign_days: int = 30) -> pd.DataFrame:

        """

        Main method: Create training dataset from daily campaign data.

        Args:

            df: Normalized daily data from data_loader

            min_campaign_days: Minimum days a campaign must have

        Returns:

            DataFrame with one row per (date, period_length) combination

            containing aggregated features and targets

        """

        logger.info(f"\n{'='*60}")

        logger.info(f"?? FEATURE ENGINEERING STARTED")

        logger.info(f"{'='*60}")

        # Step 1: Aggregate to daily platform totals first

        daily_platform = self._aggregate_daily_platform(df)

        # Step 2: Create period-level features

        all_features = []

        for period_days in self.periods:

            logger.info(f"\n?? Creating features for {period_days}-day periods...")

            period_features = self._create_period_features(daily_platform, period_days)

            all_features.append(period_features)

            logger.info(f"   Generated {len(period_features)} training examples")

        # Combine all periods

        feature_df = pd.concat(all_features, ignore_index=True)

        # Step 3: Add time features

        feature_df = self._add_time_features(feature_df)

        # Step 4: Add campaign-type level features

        logger.info(f"\n?? Creating campaign-type features...")

        campaign_type_features = self._create_campaign_type_features(df, feature_df)

        # Step 5: Merge everything

        final_df = self._merge_features(feature_df, campaign_type_features)

        # Step 6: Create target variables

        final_df = self._create_targets(final_df)

        # Step 7: Handle missing/invalid values

        final_df = self._clean_features(final_df)

        # Log feature summary

        self._log_feature_summary(final_df)

        logger.info(f"\n? Feature engineering complete: {len(final_df)} rows, "

                   f"{len(final_df.columns)} columns")

        return final_df

    def _aggregate_daily_platform(self, df: pd.DataFrame) -> pd.DataFrame:

        """

        Aggregate daily data to platform level.

        Sums spend, revenue, clicks, impressions, conversions per platform per day.

        """

        logger.info("\n?? Aggregating daily data by platform...")

        daily = df.groupby(['date', 'platform']).agg({

            'spend': 'sum',

            'revenue': 'sum',

            'clicks': 'sum',

            'impressions': 'sum',

            'conversions': 'sum'

        }).reset_index()

        # Pivot to get one column per platform

        daily_pivot = daily.pivot_table(

            index='date',

            columns='platform',

            values=['spend', 'revenue', 'clicks', 'impressions', 'conversions'],

            aggfunc='sum',

            fill_value=0

        )

        # Flatten column names

        daily_pivot.columns = [f'{col[0]}_{col[1]}' for col in daily_pivot.columns]

        daily_pivot = daily_pivot.reset_index()

        daily_pivot = daily_pivot.sort_values('date')

        # Add total metrics

        for metric in ['spend', 'revenue', 'clicks', 'impressions', 'conversions']:

            plat_cols = [c for c in daily_pivot.columns if c.startswith(metric)]

            daily_pivot[f'total_{metric}'] = daily_pivot[plat_cols].sum(axis=1)

        logger.info(f"   Created {len(daily_pivot)} daily platform rows")

        logger.info(f"   Platforms: {[c.split('_')[1] for c in daily_pivot.columns if c.startswith('spend_')]}")

        return daily_pivot

    def _create_period_features(self, daily: pd.DataFrame,

                                period_days: int) -> pd.DataFrame:

        """

        Create features for a specific planning period length.

        For each date, create a row representing the PREVIOUS period_days as features,

        and the NEXT period_days as the target.

        This simulates: "Given the last 30 days, predict the next 30 days"

        """

        features_list = []

        # We need at least 2period_days of data

        min_date = daily['date'].min() + timedelta(days=period_days * 2)

        max_date = daily['date'].max() - timedelta(days=period_days)

        eligible_dates = daily[(daily['date'] >= min_date) &

                               (daily['date'] <= max_date)]['date']

        for ref_date in eligible_dates:

            # Feature window: [ref_date - period_days, ref_date)

            feature_start = ref_date - timedelta(days=period_days)

            feature_end = ref_date - timedelta(days=1)

            # Target window: [ref_date, ref_date + period_days)

            target_start = ref_date

            target_end = ref_date + timedelta(days=period_days - 1)

            # Extract feature window data

            feature_mask = (daily['date'] >= feature_start) & (daily['date'] <= feature_end)

            feature_data = daily[feature_mask]

            if len(feature_data) < period_days * 0.7:  # At least 70% of days present

                continue

            # Extract target window data

            target_mask = (daily['date'] >= target_start) & (daily['date'] <= target_end)

            target_data = daily[target_mask]

            # Create row

            row = {

                'ref_date': ref_date,

                'period_days': period_days,

                'feature_start': feature_start,

                'feature_end': feature_end,

                'target_start': target_start,

                'target_end': target_end,

            }

            # Feature window aggregations (what we KNOW at prediction time)

            metric_cols = [c for c in daily.columns if c != 'date']

            for col in metric_cols:

                row[f'feature_{col}'] = feature_data[col].sum()

                row[f'feature_daily_avg_{col}'] = feature_data[col].mean()

                row[f'feature_days_with_{col}'] = (feature_data[col] > 0).sum()

            # Target window aggregations (what we want to PREDICT)

            for col in metric_cols:

                row[f'target_{col}'] = target_data[col].sum()

            # Additional derived features

            # ROAS in feature window

            for plat in ['google', 'meta', 'microsoft']:

                spend_col = f'feature_spend_{plat}'

                rev_col = f'feature_revenue_{plat}'

                if spend_col in row and row[spend_col] > 0:

                    row[f'feature_roas_{plat}'] = row[rev_col] / row[spend_col]

                else:

                    row[f'feature_roas_{plat}'] = 0

            # Total feature ROAS

            total_spend = row.get('feature_total_spend', 0)

            total_rev = row.get('feature_total_revenue', 0)

            row['feature_blended_roas'] = total_rev / total_spend if total_spend > 0 else 0

            # Spend ratios (channel mix)

            if total_spend > 0:

                for plat in ['google', 'meta', 'microsoft']:

                    spend_col = f'feature_spend_{plat}'

                    row[f'feature_spend_share_{plat}'] = row.get(spend_col, 0) / total_spend

            features_list.append(row)

        result = pd.DataFrame(features_list)

        result['period_days'] = period_days

        return result

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:

        """

        Add temporal features based on the reference date.

        """

        logger.info("\n?? Adding time-based features...")

        df = df.copy()

        df['ref_date'] = pd.to_datetime(df['ref_date'])

        # Basic time components

        df['month'] = df['ref_date'].dt.month

        df['quarter'] = df['ref_date'].dt.quarter

        df['year'] = df['ref_date'].dt.year

        df['day_of_month'] = df['ref_date'].dt.day

        df['day_of_week'] = df['ref_date'].dt.dayofweek

        df['week_of_year'] = df['ref_date'].dt.isocalendar().week.astype(int)

        # Seasonal indicators

        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        df['is_month_start'] = df['day_of_month'].isin([1, 2, 3]).astype(int)

        df['is_month_end'] = df['day_of_month'].isin([28, 29, 30, 31]).astype(int)

        # Quarter dummies

        for q in [1, 2, 3, 4]:

            df[f'is_q{q}'] = (df['quarter'] == q).astype(int)

        # Month dummies (for seasonality)

        for m in range(1, 13):

            df[f'is_month_{m}'] = (df['month'] == m).astype(int)

        # Holiday features

        df['is_holiday'] = df['ref_date'].apply(

            lambda d: 1 if d.strftime('%Y-%m-%d') in SHOPPING_EVENTS else 0

        )

        # Weekend ratio in the period (approximate)

        df['weekend_ratio'] = 2/7  # Default ~28.6% of days are weekends

        # Trend feature (days since start of data)

        min_date = df['ref_date'].min()

        df['days_since_start'] = (df['ref_date'] - min_date).dt.days

        df['trend'] = df['days_since_start'] / 365.25  # Years since start

        logger.info(f"   Added {df.columns.tolist()[-15:]} time features")

        return df

    def _create_campaign_type_features(self, df: pd.DataFrame,

                                       feature_df: pd.DataFrame) -> pd.DataFrame:

        """

        Create campaign-type level aggregated features.

        Matches campaign types across platforms (SEARCH, PERFORMANCE_MAX, etc.)

        """

        logger.info("\n?? Creating campaign-type features...")

        # Normalize campaign types

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

        }

        df = df.copy()

        df['campaign_type_normalized'] = df['campaign_type'].map(type_mapping).fillna('other')

        # Aggregate by date and normalized campaign type

        type_daily = df.groupby(['date', 'campaign_type_normalized']).agg({

            'spend': 'sum',

            'revenue': 'sum',

            'clicks': 'sum',

            'conversions': 'sum'

        }).reset_index()

        # For each row in feature_df, get the campaign-type spend in the feature window

        type_features_list = []

        for _, row in feature_df.iterrows():

            feat_start = row['feature_start']

            feat_end = row['feature_end']

            type_row = {

                'ref_date': row['ref_date'],

                'period_days': row['period_days']

            }

            # Get data for feature window

            type_mask = (type_daily['date'] >= feat_start) & (type_daily['date'] <= feat_end)

            window_data = type_daily[type_mask]

            # Sum by campaign type

            for ctype in window_data['campaign_type_normalized'].unique():

                ctype_data = window_data[window_data['campaign_type_normalized'] == ctype]

                type_row[f'feature_spend_ctype_{ctype}'] = ctype_data['spend'].sum()

                type_row[f'feature_revenue_ctype_{ctype}'] = ctype_data['revenue'].sum()

                if ctype_data['spend'].sum() > 0:

                    type_row[f'feature_roas_ctype_{ctype}'] = (

                        ctype_data['revenue'].sum() / ctype_data['spend'].sum()

                    )

            type_features_list.append(type_row)

        type_features = pd.DataFrame(type_features_list)

        logger.info(f"   Campaign types found: {sorted(type_daily['campaign_type_normalized'].unique())}")

        return type_features

    def _merge_features(self, feature_df: pd.DataFrame,

                        type_features: pd.DataFrame) -> pd.DataFrame:

        """Merge platform-level and campaign-type features."""

        merged = feature_df.merge(

            type_features,

            on=['ref_date', 'period_days'],

            how='left'

        )

        # Fill missing campaign type features with 0

        ctype_cols = [c for c in merged.columns if 'ctype' in c]

        for col in ctype_cols:

            merged[col] = merged[col].fillna(0)

        return merged

    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:

        """

        Create target variables for forecasting:

        - Total revenue (primary)

        - Per-platform revenue (secondary)

        - Blended ROAS (derived)

        """

        logger.info("\n?? Creating target variables...")

        df = df.copy()

        # Primary target: total revenue

        df['target_total_revenue'] = df['target_total_revenue'].fillna(0)

        # Per-platform revenue targets

        for plat in ['google', 'meta', 'microsoft']:

            col = f'target_revenue_{plat}'

            if col in df.columns:

                df[f'target_{plat}_revenue'] = df[col].fillna(0)

        # ROAS targets (revenue / spend for target period)

        total_spend_target = df['target_total_spend'].fillna(0)

        df['target_blended_roas'] = np.where(

            total_spend_target > 0,

            df['target_total_revenue'] / total_spend_target,

            0

        )

        # Per-platform ROAS targets

        for plat in ['google', 'meta', 'microsoft']:

            spend_col = f'target_spend_{plat}'

            rev_col = f'target_revenue_{plat}'

            if spend_col in df.columns and rev_col in df.columns:

                spend = df[spend_col].fillna(0)

                rev = df[rev_col].fillna(0)

                df[f'target_{plat}_roas'] = np.where(spend > 0, rev / spend, 0)

        # Log-transform targets (for models that benefit from it)

        df['target_log_total_revenue'] = np.log1p(df['target_total_revenue'])

        return df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:

        """

        Handle missing values, infinite values, and outliers in features.

        """

        logger.info("\n?? Cleaning features...")

        df = df.copy()

        # Replace infinities with NaN

        df = df.replace([np.inf, -np.inf], np.nan)

        # Fill NaN in numeric columns with 0

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:

            df[col] = df[col].fillna(0)

        # Cap extreme values at 99.5th percentile for key features

        cap_cols = [c for c in df.columns if 'feature_spend' in c or 'feature_revenue' in c]

        for col in cap_cols:

            if col in df.columns:

                cap = df[col].quantile(0.995)

                extreme_count = (df[col] > cap).sum()

                if extreme_count > 0:

                    df[col] = df[col].clip(upper=cap)

                    logger.info(f"   Capped {extreme_count} values in {col} at {cap:.2f}")

        # Remove rows where all spend features are 0 (no activity period)

        spend_cols = [c for c in df.columns if 'feature_spend' in c and 'ctype' not in c]

        if spend_cols:

            df = df[df[spend_cols].sum(axis=1) > 0]

        return df

    def _log_feature_summary(self, df: pd.DataFrame):

        """Log summary statistics about the feature matrix."""

        logger.info(f"\n{'='*60}")

        logger.info(f"?? FEATURE ENGINEERING SUMMARY")

        logger.info(f"{'='*60}")

        logger.info(f"Total training examples: {len(df):,}")

        logger.info(f"Total features: {len(df.columns):,}")

        # Period distribution

        logger.info(f"\nPeriod distribution:")

        for period in sorted(df['period_days'].unique()):

            count = len(df[df['period_days'] == period])

            logger.info(f"  {period}-day periods: {count:,} examples")

        # Date range

        logger.info(f"\nDate range:")

        logger.info(f"  Ref dates: {df['ref_date'].min().date()} to {df['ref_date'].max().date()}")

        # Target statistics

        logger.info(f"\nTarget variable statistics:")

        logger.info(f"  Total revenue - Mean: ${df['target_total_revenue'].mean():,.2f}, "

                   f"Median: ${df['target_total_revenue'].median():,.2f}")

        logger.info(f"  Blended ROAS - Mean: {df['target_blended_roas'].mean():.2f}x, "

                   f"Median: {df['target_blended_roas'].median():.2f}x")

        # Platform breakdown

        for plat in ['google', 'meta', 'microsoft']:

            rev_col = f'target_{plat}_revenue'

            if rev_col in df.columns:

                logger.info(f"  {plat.capitalize()} revenue - Mean: ${df[rev_col].mean():,.2f}")

        # Feature categories

        feature_categories = {

            'Spend features': [c for c in df.columns if 'feature_spend' in c],

            'Revenue features': [c for c in df.columns if 'feature_revenue' in c and 'target' not in c],

            'ROAS features': [c for c in df.columns if 'feature_roas' in c],

            'Time features': [c for c in df.columns if c.startswith(('month', 'quarter', 'year', 'is_', 'weekend', 'trend'))],

            'Target variables': [c for c in df.columns if c.startswith('target')],

            'Campaign type features': [c for c in df.columns if 'ctype' in c],

        }

        logger.info(f"\nFeature categories:")

        for cat, cols in feature_categories.items():

            logger.info(f"  {cat}: {len(cols)}")

        logger.info(f"{'='*60}\n")

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:

        """

        Return list of feature columns (excluding targets and metadata).

        """

        exclude_patterns = ['target_', 'ref_date', 'feature_start', 'feature_end',

                          'target_start', 'target_end']

        feature_cols = [c for c in df.columns

                       if not any(p in c for p in exclude_patterns)]

        return feature_cols

    def get_target_columns(self, df: pd.DataFrame) -> List[str]:

        """Return list of target columns."""

        return [c for c in df.columns if c.startswith('target_')]

def create_forecast_features(daily_data: pd.DataFrame,

                             future_budgets: Dict[str, float],

                             period_days: int,

                             ref_date: datetime) -> pd.DataFrame:

    """

    Create features for a single forecast scenario.

    Args:

        daily_data: Daily platform-level data

        future_budgets: Dict like {'google': 50000, 'meta': 30000, 'microsoft': 20000}

        period_days: 30, 60, or 90

        ref_date: Forecast reference date

    Returns:

        Single-row DataFrame with features ready for prediction

    """

    feature_start = ref_date - timedelta(days=period_days)

    feature_end = ref_date - timedelta(days=1)

    # Extract feature window data

    feature_mask = (daily_data['date'] >= feature_start) & (daily_data['date'] <= feature_end)

    feature_data = daily_data[feature_mask]

    row = {

        'ref_date': ref_date,

        'period_days': period_days,

    }

    # Aggregate feature window

    metric_cols = [c for c in daily_data.columns if c != 'date']

    for col in metric_cols:

        row[f'feature_{col}'] = feature_data[col].sum() if len(feature_data) > 0 else 0

        row[f'feature_daily_avg_{col}'] = feature_data[col].mean() if len(feature_data) > 0 else 0

    # Override spend features with user-provided budgets

    for plat, budget in future_budgets.items():

        row[f'feature_spend_{plat}'] = budget

    # Recalculate total spend and share

    total_budget = sum(future_budgets.values())

    row['feature_total_spend'] = total_budget

    for plat, budget in future_budgets.items():

        row[f'feature_spend_share_{plat}'] = budget / total_budget if total_budget > 0 else 0

    # Time features

    row['month'] = ref_date.month

    row['quarter'] = ref_date.quarter

    row['year'] = ref_date.year

    row['day_of_week'] = ref_date.weekday()

    row['is_weekend'] = 1 if ref_date.weekday() in [5, 6] else 0

    row['is_month_start'] = 1 if ref_date.day in [1, 2, 3] else 0

    row['is_month_end'] = 1 if ref_date.day in [28, 29, 30, 31] else 0

    row['is_holiday'] = 1 if ref_date.strftime('%Y-%m-%d') in SHOPPING_EVENTS else 0

    row['weekend_ratio'] = 2/7

    for m in range(1, 13):

        row[f'is_month_{m}'] = 1 if ref_date.month == m else 0

    for q in range(1, 5):

        row[f'is_q{q}'] = 1 if ref_date.quarter == q else 0

    row['trend'] = 2.5  # Approximate years from 2024 start to mid-2026

    result = pd.DataFrame([row])

    # Ensure all columns that training data had

    # (prediction script will handle alignment)

    return result

if __name__ == "__main__":

    from src.data_loader import load_all_data

    # Load data

    df = load_all_data()

    # Create features

    engineer = FeatureEngineer(periods=[30, 60, 90])

    features = engineer.create_training_data(df)

    # Show sample

    print("\n" + "="*60)

    print("SAMPLE TRAINING DATA")

    print("="*60)

    # Show key columns

    key_cols = ['ref_date', 'period_days',

                'feature_total_spend', 'feature_total_revenue',

                'target_total_revenue', 'target_blended_roas'] + \
    [c for c in features.columns if 'feature_spend_share' in c]

    available_cols = [c for c in key_cols if c in features.columns]

    print(features[available_cols].head(10).to_string())

    print("\n" + "="*60)

    print("FEATURE COLUMNS")

    print("="*60)

    feat_cols = engineer.get_feature_columns(features)

    for i, col in enumerate(feat_cols, 1):

        print(f"  {i:3d}. {col}")

    print("\n" + "="*60)

    print("TARGET COLUMNS")

    print("="*60)

    target_cols = engineer.get_target_columns(features)

    for i, col in enumerate(target_cols, 1):

        print(f"  {i:3d}. {col}")

