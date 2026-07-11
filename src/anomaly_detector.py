# src/anomaly_detector.py
"""
Anomaly detection for e-commerce marketing data.
Detects outliers in spend, revenue, ROAS, and campaign behavior.
All results are logged to terminal and returned as structured data for UI.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
from src.logger import setup_logger
from src.config import ANOMALY_IQR_MULTIPLIER, ANOMALY_WINDOW_DAYS

warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = setup_logger(__name__)


class AnomalyDetector:
    """
    Multi-method anomaly detection for marketing data.
    
    Methods used:
    1. IQR-based (rolling window) - spend & revenue
    2. ROAS anomalies - extreme efficiency values
    3. Zero-spend detection - campaigns with spend but no conversions
    4. Percentage change - sudden spikes/drops
    5. Campaign gaps - missing data periods
    """
    
    def __init__(self, iqr_multiplier: float = ANOMALY_IQR_MULTIPLIER,
                 zscore_threshold: float = 3.0,
                 pct_change_threshold: float = 200.0):
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold
        self.pct_change_threshold = pct_change_threshold
        self.anomalies = {}
        self.stats = {}
        
    def detect_all(self, df: pd.DataFrame) -> Dict:
        """
        Run all anomaly detection methods.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"?? ANOMALY DETECTION STARTED")
        logger.info(f"{'='*60}")
        
        results = {
            'spend_outliers': self.detect_spend_outliers(df),
            'revenue_outliers': self.detect_revenue_outliers(df),
            'roas_outliers': self.detect_roas_outliers(df),
            'zero_conversion_spend': self.detect_zero_conversion_spend(df),
            'sudden_changes': self.detect_sudden_changes(df),
            'campaign_gaps': self.detect_campaign_gaps(df),
            'summary': {}
        }
        
        results['summary'] = self._generate_summary(results)
        self._print_summary(results['summary'])
        
        return results
    
    def _safe_deviation(self, value: float, bound: float) -> float:
        """Calculate percentage deviation safely, avoiding divide-by-zero."""
        if pd.isna(bound) or bound == 0:
            return np.nan
        return ((value - bound) / bound) * 100
    
    def detect_spend_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect spend outliers using rolling IQR method."""
        logger.info("\n?? Checking for spend anomalies...")
        
        outliers_list = []
        
        for (platform, campaign_id), group in df.groupby(['platform', 'campaign_id']):
            group = group.sort_values('date').set_index('date')
            
            if len(group) < 7:
                continue
            
            # Rolling statistics
            rolling_median = group['spend'].rolling(f'{ANOMALY_WINDOW_DAYS}D', min_periods=7).median()
            rolling_q75 = group['spend'].rolling(f'{ANOMALY_WINDOW_DAYS}D', min_periods=7).quantile(0.75)
            rolling_q25 = group['spend'].rolling(f'{ANOMALY_WINDOW_DAYS}D', min_periods=7).quantile(0.25)
            rolling_iqr = rolling_q75 - rolling_q25
            
            upper_bound = rolling_q75 + self.iqr_multiplier * rolling_iqr
            lower_bound = (rolling_q25 - self.iqr_multiplier * rolling_iqr).clip(lower=0)
            
            # Flag outliers (avoid zero-division in bounds)
            is_outlier_high = group['spend'] > upper_bound
            is_outlier_low = (group['spend'] < lower_bound) & (group['spend'] > 0)
            
            # Process high outliers
            high_dates = group.index[is_outlier_high]
            for d in high_dates:
                ub = upper_bound.loc[d] if d in upper_bound.index else np.nan
                outliers_list.append({
                    'date': d,
                    'platform': platform,
                    'campaign_id': campaign_id,
                    'campaign_name': group.loc[d, 'campaign_name'],
                    'campaign_type': group.loc[d, 'campaign_type'],
                    'metric': 'spend',
                    'type': 'high',
                    'value': group.loc[d, 'spend'],
                    'bound': ub,
                    'deviation_pct': self._safe_deviation(group.loc[d, 'spend'], ub)
                })
            
            # Process low outliers
            low_dates = group.index[is_outlier_low]
            for d in low_dates:
                lb = lower_bound.loc[d] if d in lower_bound.index else np.nan
                outliers_list.append({
                    'date': d,
                    'platform': platform,
                    'campaign_id': campaign_id,
                    'campaign_name': group.loc[d, 'campaign_name'],
                    'campaign_type': group.loc[d, 'campaign_type'],
                    'metric': 'spend',
                    'type': 'low',
                    'value': group.loc[d, 'spend'],
                    'bound': lb,
                    'deviation_pct': self._safe_deviation(lb, group.loc[d, 'spend'])
                })
        
        result_df = pd.DataFrame(outliers_list)
        
        if len(result_df) > 0:
            logger.info(f"   ??  Found {len(result_df)} spend anomalies")
            logger.info(f"      High outliers: {(result_df['type']=='high').sum()}")
            logger.info(f"      Low outliers:  {(result_df['type']=='low').sum()}")
        else:
            logger.info(f"   ? No spend anomalies detected")
        
        return result_df
    
    def detect_revenue_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect revenue outliers using IQR method."""
        logger.info("\n?? Checking for revenue anomalies...")
        
        outliers_list = []
        df_with_revenue = df[df['revenue'] > 0].copy()
        
        if len(df_with_revenue) == 0:
            logger.info("   ??  No revenue data to analyze")
            return pd.DataFrame()
        
        for (platform, campaign_id), group in df_with_revenue.groupby(['platform', 'campaign_id']):
            group = group.sort_values('date').set_index('date')
            
            if len(group) < 7:
                continue
            
            rolling_q75 = group['revenue'].rolling(f'{ANOMALY_WINDOW_DAYS}D', min_periods=7).quantile(0.75)
            rolling_q25 = group['revenue'].rolling(f'{ANOMALY_WINDOW_DAYS}D', min_periods=7).quantile(0.25)
            rolling_iqr = rolling_q75 - rolling_q25
            
            upper_bound = rolling_q75 + self.iqr_multiplier * rolling_iqr
            
            is_outlier = group['revenue'] > upper_bound
            
            for d in group.index[is_outlier]:
                ub = upper_bound.loc[d] if d in upper_bound.index else np.nan
                outliers_list.append({
                    'date': d,
                    'platform': platform,
                    'campaign_id': campaign_id,
                    'campaign_name': group.loc[d, 'campaign_name'],
                    'campaign_type': group.loc[d, 'campaign_type'],
                    'metric': 'revenue',
                    'type': 'high',
                    'value': group.loc[d, 'revenue'],
                    'bound': ub,
                    'deviation_pct': self._safe_deviation(group.loc[d, 'revenue'], ub)
                })
        
        result_df = pd.DataFrame(outliers_list)
        
        if len(result_df) > 0:
            logger.info(f"   ??  Found {len(result_df)} revenue anomalies")
        else:
            logger.info(f"   ? No revenue anomalies detected")
        
        return result_df
    
    def detect_roas_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect ROAS anomalies."""
        logger.info("\n?? Checking for ROAS anomalies...")
        
        df_with_spend = df[df['spend'] > 0].copy()
        
        if len(df_with_spend) == 0:
            logger.info("   ??  No spend data for ROAS calculation")
            return pd.DataFrame()
        
        df_with_spend['roas'] = df_with_spend['revenue'] / df_with_spend['spend']
        
        outliers_list = []
        
        for (platform, campaign_id), group in df_with_spend.groupby(['platform', 'campaign_id']):
            group = group[group['roas'] > 0].sort_values('date').set_index('date')
            
            if len(group) < 7:
                continue
            
            median_roas = group['roas'].median()
            q75_roas = group['roas'].quantile(0.75)
            q25_roas = group['roas'].quantile(0.25)
            iqr_roas = q75_roas - q25_roas
            
            # Use 3IQR or 10 median, whichever is higher
            upper_bound = max(q75_roas + 3 * iqr_roas, median_roas * 10)
            
            high_roas = group[group['roas'] > upper_bound]
            
            for d in high_roas.index:
                outliers_list.append({
                    'date': d,
                    'platform': platform,
                    'campaign_id': campaign_id,
                    'campaign_name': group.loc[d, 'campaign_name'],
                    'campaign_type': group.loc[d, 'campaign_type'],
                    'metric': 'roas',
                    'type': 'high',
                    'value': group.loc[d, 'roas'],
                    'median_roas': median_roas,
                    'deviation_pct': self._safe_deviation(group.loc[d, 'roas'], median_roas)
                })
        
        result_df = pd.DataFrame(outliers_list)
        
        if len(result_df) > 0:
            logger.info(f"   ??  Found {len(result_df)} ROAS anomalies")
        else:
            logger.info(f"   ? No ROAS anomalies detected")
        
        return result_df
    
    def detect_zero_conversion_spend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect days with spend but zero conversions."""
        logger.info("\n?? Checking for zero-conversion spend days...")
        
        spend_threshold = df['spend'].median() * 0.5
        
        zero_conv = df[(df['spend'] > spend_threshold) & 
                       (df['conversions'] == 0) & 
                       (df['revenue'] == 0)].copy()
        
        if len(zero_conv) > 0:
            logger.info(f"   ??  Found {len(zero_conv)} zero-conversion spend days")
            worst = zero_conv.nlargest(5, 'spend')[['date', 'platform', 'campaign_name', 'spend']]
            for _, row in worst.iterrows():
                logger.info(f"      {row['date'].date()} | {row['platform']:10s} | "
                          f"{str(row['campaign_name'])[:40]:40s} | ${row['spend']:,.2f}")
        else:
            logger.info(f"   ? No zero-conversion spend days found")
        
        return zero_conv
    
    def detect_sudden_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect sudden day-over-day changes."""
        logger.info("\n?? Checking for sudden metric changes...")
        
        sudden_changes = []
        
        for (platform, campaign_id), group in df.groupby(['platform', 'campaign_id']):
            group = group.sort_values('date').reset_index(drop=True)
            
            if len(group) < 3:
                continue
            
            for metric in ['spend', 'revenue', 'clicks', 'impressions']:
                # Day-over-day percentage change
                pct_change = group[metric].pct_change() * 100
                
                for i in range(1, len(group)):
                    if abs(pct_change.iloc[i]) > self.pct_change_threshold:
                        prev_val = group.iloc[i-1][metric]
                        curr_val = group.iloc[i][metric]
                        
                        if prev_val > 0 and curr_val > 0:
                            sudden_changes.append({
                                'date': group.iloc[i]['date'],
                                'platform': platform,
                                'campaign_id': campaign_id,
                                'campaign_name': group.iloc[i]['campaign_name'],
                                'campaign_type': group.iloc[i]['campaign_type'],
                                'metric': metric,
                                'type': 'sudden_change',
                                'previous_value': prev_val,
                                'current_value': curr_val,
                                'change_pct': pct_change.iloc[i]
                            })
        
        result_df = pd.DataFrame(sudden_changes)
        
        if len(result_df) > 0:
            logger.info(f"   ??  Found {len(result_df)} sudden changes")
            for metric in ['spend', 'revenue', 'clicks', 'impressions']:
                count = len(result_df[result_df['metric'] == metric])
                if count > 0:
                    logger.info(f"      {metric}: {count}")
        else:
            logger.info(f"   ? No sudden changes detected")
        
        return result_df
    
    def detect_campaign_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect campaigns with data gaps."""
        logger.info("\n?? Checking for campaign data gaps...")
        
        gaps_list = []
        
        for (platform, campaign_id), group in df.groupby(['platform', 'campaign_id']):
            group = group.sort_values('date').reset_index(drop=True)
            
            if len(group) < 2:
                continue
            
            # Calculate date differences
            date_diff = group['date'].diff().dt.days
            
            for i in range(1, len(group)):
                gap_days = date_diff.iloc[i]
                if gap_days > 3:
                    gaps_list.append({
                        'platform': platform,
                        'campaign_id': campaign_id,
                        'campaign_name': group.iloc[i]['campaign_name'],
                        'campaign_type': group.iloc[i]['campaign_type'],
                        'gap_start': group.iloc[i-1]['date'],
                        'gap_end': group.iloc[i]['date'],
                        'gap_days': int(gap_days),
                        'spend_before_gap': group.iloc[i-1]['spend'],
                        'spend_after_gap': group.iloc[i]['spend']
                    })
        
        result_df = pd.DataFrame(gaps_list)
        
        if len(result_df) > 0:
            logger.info(f"   ??  Found {len(result_df)} data gaps (>3 days)")
            for _, gap in result_df.head(5).iterrows():
                name = str(gap['campaign_name'])[:30]
                logger.info(f"      {gap['platform']:10s} | {name:30s} | "
                          f"Gap: {gap['gap_start'].date()} ? {gap['gap_end'].date()} "
                          f"({gap['gap_days']} days)")
        else:
            logger.info(f"   ? No significant data gaps found")
        
        return result_df
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate summary statistics."""
        summary = {
            'total_anomalies': 0,
            'by_type': {},
            'by_platform': {},
            'severity': 'low',
            'top_issues': []
        }
        
        for anomaly_type, df in results.items():
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                count = len(df)
                summary['total_anomalies'] += count
                summary['by_type'][anomaly_type] = count
                
                if 'platform' in df.columns:
                    for plat in df['platform'].unique():
                        if plat not in summary['by_platform']:
                            summary['by_platform'][plat] = 0
                        summary['by_platform'][plat] += len(df[df['platform'] == plat])
        
        # Determine severity (adjusted for larger datasets)
        if summary['total_anomalies'] == 0:
            summary['severity'] = 'none'
        elif summary['total_anomalies'] < 100:
            summary['severity'] = 'low'
        elif summary['total_anomalies'] < 500:
            summary['severity'] = 'medium'
        else:
            summary['severity'] = 'high'
        
        issue_map = {
            'spend_outliers': "unusual spend days - may affect budget forecasting",
            'revenue_outliers': "unusual revenue spikes - verify conversion tracking",
            'roas_outliers': "abnormal ROAS readings - possible attribution issues",
            'zero_conversion_spend': "days with spend but zero conversions - check tracking",
            'sudden_changes': "sudden metric changes - campaign modifications or data errors?",
            'campaign_gaps': "data gaps found - incomplete historical data"
        }
        
        for atype, count in sorted(summary['by_type'].items(), key=lambda x: x[1], reverse=True):
            if atype in issue_map:
                summary['top_issues'].append(f"{count} {issue_map[atype]}")
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """Print formatted summary."""
        logger.info(f"\n{'='*60}")
        logger.info(f"?? ANOMALY DETECTION SUMMARY")
        logger.info(f"{'='*60}")
        
        severity_emoji = {'none': '?', 'low': '??', 'medium': '??', 'high': '??'}
        emoji = severity_emoji.get(summary['severity'], '?')
        
        logger.info(f"Severity: {emoji} {summary['severity'].upper()}")
        logger.info(f"Total anomalies found: {summary['total_anomalies']}")
        
        if summary['total_anomalies'] > 0:
            logger.info(f"\nBreakdown by type:")
            for atype, count in summary['by_type'].items():
                logger.info(f"   {atype}: {count}")
            
            logger.info(f"\nBreakdown by platform:")
            for plat, count in summary['by_platform'].items():
                logger.info(f"   {plat}: {count}")
            
            logger.info(f"\n?? Top issues for LLM analysis:")
            for i, issue in enumerate(summary['top_issues'][:3], 1):
                logger.info(f"  {i}. {issue}")
        
        logger.info(f"{'='*60}\n")


def get_anomaly_report_for_llm(anomaly_results: Dict) -> str:
    """Generate LLM-formatted anomaly summary."""
    summary = anomaly_results.get('summary', {})
    
    if summary.get('total_anomalies', 0) == 0:
        return "? No anomalies detected in the historical data."
    
    report = f"?? Anomaly Detection Report:\n"
    report += f"Severity: {summary.get('severity', 'unknown').upper()}\n"
    report += f"Total anomalies: {summary.get('total_anomalies', 0)}\n\n"
    
    report += "Top issues:\n"
    for issue in summary.get('top_issues', [])[:5]:
        report += f" {issue}\n"
    
    report += "\nPlatform breakdown:\n"
    for plat, count in summary.get('by_platform', {}).items():
        report += f" {plat.upper()}: {count} anomalies\n"
    
    return report


if __name__ == "__main__":
    from src.data_loader import load_all_data
    
    # Load data
    df = load_all_data()
    
    # Run anomaly detection
    detector = AnomalyDetector()
    results = detector.detect_all(df)
    
    # Show samples
    print("\n" + "="*60)
    print("SAMPLE ANOMALIES (for Streamlit display)")
    print("="*60)
    
    for key, df_result in results.items():
        if isinstance(df_result, pd.DataFrame) and len(df_result) > 0:
            print(f"\n{key.upper()}: {len(df_result)} rows")
            print(df_result.head(3).to_string())
    
    # LLM report
    print("\n" + "="*60)
    print("LLM-FORMATTED REPORT")
    print("="*60)
    print(get_anomaly_report_for_llm(results))

