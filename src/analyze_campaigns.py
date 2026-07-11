"""
Campaign Performance Analysis Script
Analyzes ad campaign data across Google, Meta, and Microsoft platforms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_all_data
from src.logger import setup_logger

logger = setup_logger(__name__)

class CampaignAnalyzer:
    def __init__(self, df):
        """Initialize with loaded dataframe"""
        self.df = df
        self.clean_data()
        
    def clean_data(self):
        """Prepare data for analysis"""
        # Standardize campaign types
        type_mapping = {
            'SEARCH': 'Search',
            'Search': 'Search',
            'PERFORMANCE_MAX': 'Performance Max',
            'PerformanceMax': 'Performance Max',
            'DISPLAY': 'Display',
            'VIDEO': 'Video',
            'DEMAND_GEN': 'Demand Gen',
            'SHOPPING': 'Shopping',
            'Shopping': 'Shopping',
            'unknown': 'Other',
            'Audience': 'Audience'
        }
        self.df['campaign_type_std'] = self.df['campaign_type'].map(type_mapping).fillna('Other')
        
        # Calculate derived metrics
        self.df['ctr'] = (self.df['clicks'] / self.df['impressions'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
        self.df['conversion_rate'] = (self.df['conversions'] / self.df['clicks'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
        self.df['roas'] = (self.df['revenue'] / self.df['spend']).replace([np.inf, -np.inf], 0).fillna(0)
        self.df['cpc'] = (self.df['spend'] / self.df['clicks']).replace([np.inf, -np.inf], 0).fillna(0)
        self.df['cpm'] = (self.df['spend'] / self.df['impressions'] * 1000).replace([np.inf, -np.inf], 0).fillna(0)
        
        # Filter out rows with zero spend for performance metrics
        self.df_active = self.df[self.df['spend'] > 0].copy()
        
    def platform_summary(self):
        """Summary statistics by platform"""
        logger.info("\n" + "="*60)
        logger.info("?? PLATFORM PERFORMANCE SUMMARY")
        logger.info("="*60)
        
        # Filter active campaigns
        active_df = self.df_active[self.df_active['spend'] > 0]
        
        summary = active_df.groupby('platform').agg({
            'campaign_id': 'nunique',
            'spend': 'sum',
            'revenue': 'sum',
            'clicks': 'sum',
            'impressions': 'sum',
            'conversions': 'sum',
            'ctr': 'mean',
            'conversion_rate': 'mean',
            'roas': 'mean',
            'cpc': 'mean'
        }).round(2)
        
        summary['roi'] = ((summary['revenue'] - summary['spend']) / summary['spend'] * 100).round(2)
        summary['avg_daily_spend'] = (summary['spend'] / active_df.groupby('platform')['date'].nunique()).round(2)
        
        # Format currency columns
        for col in ['spend', 'revenue', 'cpc']:
            summary[col] = summary[col].apply(lambda x: f"${x:,.2f}")
        
        # Format percentage columns
        for col in ['ctr', 'conversion_rate', 'roi']:
            summary[col] = summary[col].apply(lambda x: f"{x:.2f}%")
        
        print(summary)
        return summary
    
    def campaign_type_performance(self):
        """Performance by campaign type"""
        logger.info("\n" + "="*60)
        logger.info("?? CAMPAIGN TYPE PERFORMANCE")
        logger.info("="*60)
        
        active_df = self.df_active[self.df_active['spend'] > 0]
        
        type_summary = active_df.groupby('campaign_type_std').agg({
            'campaign_id': 'nunique',
            'spend': 'sum',
            'revenue': 'sum',
            'clicks': 'sum',
            'impressions': 'sum',
            'conversions': 'sum',
            'roas': 'mean',
            'ctr': 'mean'
        }).round(2)
        
        type_summary['roi'] = ((type_summary['revenue'] - type_summary['spend']) / type_summary['spend'] * 100).round(2)
        type_summary = type_summary.sort_values('spend', ascending=False)
        
        # Format
        for col in ['spend', 'revenue']:
            type_summary[col] = type_summary[col].apply(lambda x: f"${x:,.2f}")
        
        for col in ['roas', 'ctr']:
            type_summary[col] = type_summary[col].apply(lambda x: f"{x:.2f}")
        
        print(type_summary)
        return type_summary
    
    def top_performing_campaigns(self, n=10):
        """Top performing campaigns by revenue"""
        logger.info(f"\n{'='*60}")
        logger.info(f"?? TOP {n} CAMPAIGNS BY REVENUE")
        logger.info("="*60)
        
        active_df = self.df_active[self.df_active['spend'] > 0]
        
        top_campaigns = active_df.groupby(['campaign_id', 'campaign_name', 'platform', 'campaign_type_std']).agg({
            'spend': 'sum',
            'revenue': 'sum',
            'clicks': 'sum',
            'impressions': 'sum',
            'conversions': 'sum',
            'roas': 'mean'
        }).round(2)
        
        top_campaigns['roi'] = ((top_campaigns['revenue'] - top_campaigns['spend']) / top_campaigns['spend'] * 100).round(2)
        top_campaigns = top_campaigns.sort_values('revenue', ascending=False).head(n)
        
        # Format
        top_campaigns['spend'] = top_campaigns['spend'].apply(lambda x: f"${x:,.2f}")
        top_campaigns['revenue'] = top_campaigns['revenue'].apply(lambda x: f"${x:,.2f}")
        top_campaigns['roas'] = top_campaigns['roas'].apply(lambda x: f"{x:.2f}x")
        top_campaigns['roi'] = top_campaigns['roi'].apply(lambda x: f"{x:.2f}%")
        
        print(top_campaigns)
        return top_campaigns
    
    def time_series_trends(self):
        """Analyze daily trends"""
        logger.info("\n" + "="*60)
        logger.info("?? TIME SERIES TRENDS")
        logger.info("="*60)
        
        active_df = self.df_active[self.df_active['spend'] > 0]
        
        # Daily aggregations
        daily = active_df.groupby('date').agg({
            'spend': 'sum',
            'revenue': 'sum',
            'clicks': 'sum',
            'impressions': 'sum'
        }).reset_index()
        
        daily['roas'] = (daily['revenue'] / daily['spend']).round(2)
        daily['ctr'] = (daily['clicks'] / daily['impressions'] * 100).round(2)
        
        # Weekly aggregations
        daily['week'] = daily['date'].dt.isocalendar().week
        daily['year'] = daily['date'].dt.year
        weekly = daily.groupby(['year', 'week']).agg({
            'spend': 'sum',
            'revenue': 'sum',
            'clicks': 'sum',
            'impressions': 'sum'
        }).reset_index()
        
        weekly['roas'] = (weekly['revenue'] / weekly['spend']).round(2)
        weekly['week_start'] = weekly.apply(lambda x: f"W{x['week']}-{x['year']}", axis=1)
        
        # Calculate trends
        spend_trend = ((daily['spend'].iloc[-30:].mean() - daily['spend'].iloc[:30].mean()) / daily['spend'].iloc[:30].mean() * 100).round(2)
        revenue_trend = ((daily['revenue'].iloc[-30:].mean() - daily['revenue'].iloc[:30].mean()) / daily['revenue'].iloc[:30].mean() * 100).round(2)
        
        logger.info(f"?? Spend trend (last 30 days vs first 30 days): {spend_trend}%")
        logger.info(f"?? Revenue trend (last 30 days vs first 30 days): {revenue_trend}%")
        logger.info(f"?? Average daily spend: ${daily['spend'].mean():,.2f}")
        logger.info(f"?? Average daily revenue: ${daily['revenue'].mean():,.2f}")
        logger.info(f"?? Overall ROAS: {(daily['revenue'].sum() / daily['spend'].sum()):.2f}x")
        
        return daily, weekly
    
    def performance_metrics_distribution(self):
        """Distribution of key performance metrics"""
        logger.info("\n" + "="*60)
        logger.info("?? PERFORMANCE METRICS DISTRIBUTION")
        logger.info("="*60)
        
        active_df = self.df_active[self.df_active['spend'] > 0]
        
        # Campaign-level metrics
        campaign_metrics = active_df.groupby('campaign_id').agg({
            'spend': 'sum',
            'revenue': 'sum',
            'clicks': 'sum',
            'impressions': 'sum',
            'conversions': 'sum'
        }).round(2)
        
        campaign_metrics['roas'] = (campaign_metrics['revenue'] / campaign_metrics['spend']).round(2)
        campaign_metrics['roi'] = ((campaign_metrics['revenue'] - campaign_metrics['spend']) / campaign_metrics['spend'] * 100).round(2)
        campaign_metrics['cpc'] = (campaign_metrics['spend'] / campaign_metrics['clicks']).round(2)
        campaign_metrics['conversion_rate'] = (campaign_metrics['conversions'] / campaign_metrics['clicks'] * 100).round(2)
        
        # Replace infinities
        campaign_metrics = campaign_metrics.replace([np.inf, -np.inf], 0)
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90]
        stats = {}
        for col in ['spend', 'revenue', 'roas', 'roi', 'cpc', 'conversion_rate']:
            if col in campaign_metrics.columns:
                stats[col] = campaign_metrics[col].quantile([p/100 for p in percentiles]).round(2)
        
        stats_df = pd.DataFrame(stats)
        stats_df.index = [f"{p}th" for p in percentiles]
        
        print("\n?? Campaign-level metrics percentiles:")
        print(stats_df)
        
        # Count profitable campaigns
        profitable = campaign_metrics[campaign_metrics['revenue'] > campaign_metrics['spend']]
        break_even = campaign_metrics[campaign_metrics['revenue'] == campaign_metrics['spend']]
        loss_making = campaign_metrics[campaign_metrics['revenue'] < campaign_metrics['spend']]
        
        logger.info(f"\n?? Campaign profitability:")
        logger.info(f"   Profitable: {len(profitable)} ({len(profitable)/len(campaign_metrics)*100:.1f}%)")
        logger.info(f"   Break-even: {len(break_even)} ({len(break_even)/len(campaign_metrics)*100:.1f}%)")
        logger.info(f"   Loss-making: {len(loss_making)} ({len(loss_making)/len(campaign_metrics)*100:.1f}%)")
        
        return campaign_metrics
    
    def segment_analysis(self):
        """Analyze performance by various segments"""
        logger.info("\n" + "="*60)
        logger.info("?? SEGMENT ANALYSIS")
        logger.info("="*60)
        
        active_df = self.df_active[self.df_active['spend'] > 0]
        
        # Platform  Campaign Type
        segment_performance = active_df.groupby(['platform', 'campaign_type_std']).agg({
            'spend': 'sum',
            'revenue': 'sum',
            'clicks': 'sum',
            'impressions': 'sum',
            'conversions': 'sum',
            'campaign_id': 'nunique'
        }).round(2)
        
        segment_performance['roas'] = (segment_performance['revenue'] / segment_performance['spend']).round(2)
        segment_performance['roi'] = ((segment_performance['revenue'] - segment_performance['spend']) / segment_performance['spend'] * 100).round(2)
        segment_performance = segment_performance.sort_values('spend', ascending=False)
        
        # Format
        segment_display = segment_performance.copy()
        for col in ['spend', 'revenue']:
            segment_display[col] = segment_display[col].apply(lambda x: f"${x:,.2f}")
        
        print("\n?? Platform  Campaign Type Performance:")
        print(segment_display)
        
        return segment_performance
    
    def generate_insights(self):
        """Generate actionable insights"""
        logger.info("\n" + "="*60)
        logger.info("?? KEY INSIGHTS & RECOMMENDATIONS")
        logger.info("="*60)
        
        active_df = self.df_active[self.df_active['spend'] > 0]
        
        # Calculate key metrics
        total_spend = active_df['spend'].sum()
        total_revenue = active_df['revenue'].sum()
        total_roas = total_revenue / total_spend if total_spend > 0 else 0
        
        # Best performing platform
        platform_perf = active_df.groupby('platform').agg({
            'spend': 'sum',
            'revenue': 'sum'
        })
        platform_perf['roas'] = platform_perf['revenue'] / platform_perf['spend']
        best_platform = platform_perf['roas'].idxmax()
        
        # Best performing campaign type
        type_perf = active_df.groupby('campaign_type_std').agg({
            'spend': 'sum',
            'revenue': 'sum'
        })
        type_perf['roas'] = type_perf['revenue'] / type_perf['spend']
        best_type = type_perf['roas'].idxmax()
        
        # Worst performing
        worst_type = type_perf['roas'].idxmin()
        
        # High spend low ROAS
        campaign_metrics = active_df.groupby(['campaign_id', 'campaign_name', 'platform']).agg({
            'spend': 'sum',
            'revenue': 'sum'
        })
        campaign_metrics['roas'] = campaign_metrics['revenue'] / campaign_metrics['spend']
        high_spend_low_roas = campaign_metrics[
            (campaign_metrics['spend'] > campaign_metrics['spend'].quantile(0.75)) & 
            (campaign_metrics['roas'] < campaign_metrics['roas'].quantile(0.25))
        ].sort_values('spend', ascending=False)
        
        # Generate recommendations
        insights = []
        
        insights.append(f"? Overall ROAS: {total_roas:.2f}x (Revenue: ${total_revenue:,.2f} from ${total_spend:,.2f} spend)")
        insights.append(f"? Best performing platform: {best_platform.upper()} with {platform_perf.loc[best_platform, 'roas']:.2f}x ROAS")
        insights.append(f"? Best performing campaign type: {best_type} with {type_perf.loc[best_type, 'roas']:.2f}x ROAS")
        insights.append(f"??  Worst performing campaign type: {worst_type} with {type_perf.loc[worst_type, 'roas']:.2f}x ROAS")
        
        if len(high_spend_low_roas) > 0:
            insights.append(f"\n??  HIGH SPEND, LOW ROAS CAMPAIGNS TO REVIEW:")
            for idx, row in high_spend_low_roas.head(5).iterrows():
                insights.append(f"   - {row['campaign_name']} ({row['platform']}): ${row['spend']:,.2f} spend, {row['roas']:.2f}x ROAS")
        
        # Meta revenue issue
        meta_data = active_df[active_df['platform'] == 'meta']
        if len(meta_data) > 0 and meta_data['revenue'].sum() == 0:
            insights.append(f"\n?? CRITICAL: Meta/Facebook campaigns show $0 revenue - check conversion tracking!")
        
        # Print insights
        for insight in insights:
            logger.info(insight)
        
        return insights
    
    def run_full_analysis(self):
        """Run all analysis methods"""
        logger.info("\n" + "?? STARTING COMPREHENSIVE CAMPAIGN ANALYSIS")
        
        # Run all analyses
        platform_summary = self.platform_summary()
        campaign_type = self.campaign_type_performance()
        top_campaigns = self.top_performing_campaigns(10)
        daily, weekly = self.time_series_trends()
        metrics_dist = self.performance_metrics_distribution()
        segment = self.segment_analysis()
        insights = self.generate_insights()
        
        logger.info("\n" + "="*60)
        logger.info("? ANALYSIS COMPLETE")
        logger.info("="*60)
        
        return {
            'platform_summary': platform_summary,
            'campaign_type': campaign_type,
            'top_campaigns': top_campaigns,
            'daily_trends': daily,
            'weekly_trends': weekly,
            'metrics_distribution': metrics_dist,
            'segment_analysis': segment,
            'insights': insights
        }

def main():
    """Main execution function"""
    logger.info("?? Campaign Performance Analysis")
    logger.info("="*60)
    
    # Load data
    df = load_all_data()
    
    if df is None or df.empty:
        logger.error("No data loaded. Exiting analysis.")
        return
    
    # Initialize analyzer
    analyzer = CampaignAnalyzer(df)
    
    # Run full analysis
    results = analyzer.run_full_analysis()
    
    # Optional: Save results to CSV
    save_results = input("\n?? Save analysis results to CSV? (y/n): ").lower().strip()
    if save_results == 'y':
        for name, data in results.items():
            if isinstance(data, pd.DataFrame):
                filename = f"analysis_{name}.csv"
                data.to_csv(filename)
                logger.info(f"? Saved {filename}")
    
    return results

if __name__ == "__main__":
    main()

