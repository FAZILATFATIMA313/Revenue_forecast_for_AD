"""AdRevenue Forecast Studio - Streamlit Demo Dashboard
Probabilistic Revenue Forecasting with AI-Assisted Insights
"""

import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.anomaly_detector import AnomalyDetector
from src.config import GROQ_API_KEY, GROQ_MODEL, PICKLE_DIR
from src.data_loader import load_all_data
from src.logger import setup_logger
from src.predict import predict
from src.train import ForecastingModel

# Optional: Groq for LLM insights
try:
    from groq import Groq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

logger = setup_logger(__name__)


st.set_page_config(
    page_title="AdRevenue Forecast Studio - Revenue Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e0e0e0;
    }
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 10px 0;

        /* Force visible text on light backgrounds */
        color: #000 !important;
    }

    .insight-box * {
        color: #000 !important;
        -webkit-text-fill-color: #000 !important;
    }
    
    /* ===== TAB FONT SIZE ===== */
    /* [data-baseweb="tab"] IS the button element itself - target it directly,
       not "button[data-baseweb='tab']" or "[role='tablist'] button" which
       don't reliably match current Streamlit markup. */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 24px !important;
        font-weight: 600 !important;
        padding: 14px 28px !important;
        height: auto !important;
        white-space: nowrap !important;
    }
    .stTabs [data-baseweb="tab"] p {
        font-size: 24px !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
    }

    /* NOTE: Plotly chart font sizes are intentionally NOT set here.
       Plotly renders text as SVG and computes layout (margins, tick
       spacing, legend box size) from the font size given in the Python
       layout config. CSS overrides only resize the glyphs visually and
       get reverted on any redraw (zoom, resize, tab switch). Chart fonts
       are now set via style_chart() below instead - see chart calls. */
</style>
""",
    unsafe_allow_html=True,
)
def style_chart(fig, title_size: int = 20, axis_size: int = 16, legend_size: int = 16, base_size: int = 14):
    """Apply consistent font sizing to a Plotly figure via its layout config.

    This is the reliable way to size Plotly text - CSS can't do it because
    Plotly computes layout (margins, tick spacing) from these values itself.
    """
    fig.update_layout(
        font=dict(size=base_size),
        title_font=dict(size=title_size, family="Arial, sans-serif"),
        legend=dict(font=dict(size=legend_size)),
    )
    fig.update_xaxes(tickfont=dict(size=axis_size))
    fig.update_yaxes(tickfont=dict(size=axis_size))
    return fig


# ============================================================
# CACHE FUNCTIONS
# ============================================================


@st.cache_data(ttl=3600)
def load_data_cached() -> pd.DataFrame:
    return load_all_data()


@st.cache_resource
def load_model_cached() -> ForecastingModel | None:
    model_path = PICKLE_DIR / "model.pkl"
    if model_path.exists():
        return ForecastingModel.load(str(model_path))
    return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=80)
    st.markdown("## Controls")
    st.markdown("---")

    st.markdown("### Planning Period")
    period_days = st.selectbox(
        "Forecast Horizon",
        options=[30, 60, 90],
        index=1,
        help="Number of days to forecast",
    )

    st.markdown("---")
    st.markdown("### Budget Allocation")

    try:
        df = load_data_cached()
        recent = df[df["date"] >= df["date"].max() - timedelta(days=30)]
        default_google = recent[recent["platform"] == "google"]["spend"].sum() / 30 * period_days
        default_meta = recent[recent["platform"] == "meta"]["spend"].sum() / 30 * period_days
        default_ms = (
            recent[recent["platform"] == "microsoft"]["spend"].sum() / 30 * period_days
        )
    except Exception:
        default_google, default_meta, default_ms = 50000, 10000, 5000

    google_budget = st.number_input(
        "Google Ads Budget ($)",

        min_value=0,
        value=int(default_google),
        step=5000,
        format="%d",
    )
    meta_budget = st.number_input(
        "Meta Ads Budget ($)",
        min_value=0,
        value=int(default_meta),
        step=5000,
        format="%d",
    )
    ms_budget = st.number_input(
        "Microsoft Ads Budget ($)",
        min_value=0,
        value=int(default_ms),
        step=1000,
        format="%d",
    )

    total_budget = google_budget + meta_budget + ms_budget
    st.metric("Total Budget", f"${total_budget:,.0f}")

    if total_budget > 0:
        st.markdown("#### Budget Split")
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=["Google", "Meta", "Microsoft"],
                    values=[google_budget, meta_budget, ms_budget],
                    hole=0.4,
                    marker_colors=["#4285F4", "#1877F2", "#00A4EF"],
                )
            ]
        )
        fig_pie.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(style_chart(fig_pie), width='stretch')

    st.markdown("---")

    forecast_button = st.button(
        "Generate Forecast",
        type="primary",
        width='stretch'  ,
    )

    enable_llm = st.checkbox("Enable AI Insights (Groq)", value=True)


# ============================================================
# MAIN CONTENT
# ============================================================

st.markdown('<h1 class="main-header">AdRevenue Forecast Studio</h1>', unsafe_allow_html=True)
st.markdown("### Probabilistic Revenue Forecasting for Ad Campaigns")
st.markdown("---")

(tab1, tab2, tab3, tab4) = st.tabs(["Forecast", "Data Explorer", "Anomalies", "Methodology"])


# ============================================================
# TAB 1: FORECAST
# ============================================================

with tab1:
    if forecast_button:
        st.spinner("Generating probabilistic forecast...")

        model = load_model_cached()
        if model is None:
            st.error("Model not found! Run `python src/train.py` first.")
            st.stop()

        # Prepare temp request data + feature generation
        from src.generate_features import generate_features  # local import to keep app startup fast

        tmp_dir = tempfile.mkdtemp()
        try:
            # Create request CSV in temp dir (generate_features expects a CSV)
            requests_df = pd.DataFrame(
                [
                    {
                        "request_id": "streamlit_forecast",
                        "period_days": period_days,
                        "spend_google": google_budget,
                        "spend_meta": meta_budget,
                        "spend_ms": ms_budget,
                    }
                ]
            )
            requests_path = os.path.join(tmp_dir, "forecast_requests.csv")
            requests_df.to_csv(requests_path, index=False)

            # Copy data to temp
            data_tmp = os.path.join(tmp_dir, "./data")
            os.makedirs(data_tmp, exist_ok=True)
          
            for f in os.listdir("./data"):
                if f.endswith(".csv"):
                    import shutil

                    shutil.copy(os.path.join("./data", f), os.path.join(data_tmp, f))
            import shutil

            shutil.copy(requests_path, os.path.join(data_tmp, "forecast_requests.csv"))

            # Generate features + predict
            features_path = os.path.join(tmp_dir, "features.parquet")
            generate_features(data_tmp, features_path)

            predict(features_path, str(PICKLE_DIR / "model.pkl"), os.path.join(tmp_dir, "predictions.csv"))
            predictions = pd.read_csv(os.path.join(tmp_dir, "predictions.csv"))

            pred = predictions.iloc[0]
            MIN_REALISTIC_BUDGET = 5000
            
            for prefix in ['revenue', 'google_revenue', 'meta_revenue', 'ms_revenue']:
                for q in ['p10', 'p50', 'p90']:
                    col = f'{prefix}_{q}'
                    if col in pred:
                        pred[col] = max(0, float(pred[col]))
            
            # Clip ROAS to realistic e-commerce range (0.5x - 15x)
            total_spend = google_budget + meta_budget + ms_budget
            for q in ['p10', 'p50', 'p90']:
                roas_col = f'blended_roas_{q}'
                if roas_col in pred and total_spend > 0:
                    raw_roas = pred[f'revenue_{q}'] / total_spend
                    pred[roas_col] = min(max(raw_roas, 0.5), 15.0)
                    
        finally:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

        st.success(f"Forecast generated for {period_days}-day period!")

        st.markdown("## Revenue Forecast")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Conservative (P10)")
            st.markdown(f"## ${pred['revenue_p10']:,.0f}")
            if total_budget > 0:
                st.caption(f"ROAS: {pred['blended_roas_p10']:.2f}x")

        with col2:
            st.markdown("### Expected (P50)")
            st.markdown(f"## ${pred['revenue_p50']:,.0f}")
            if total_budget > 0:
                st.caption(f"ROAS: {pred['blended_roas_p50']:.2f}x")

        with col3:
            st.markdown("### Optimistic (P90)")
            st.markdown(f"## ${pred['revenue_p90']:,.0f}")
            if total_budget > 0:
                st.caption(f"ROAS: {pred['blended_roas_p90']:.2f}x")

        st.markdown("---")
        st.markdown("### Revenue Prediction Range")

        fig_rev = go.Figure()
        fig_rev.add_trace(
            go.Bar(
                x=["Revenue"],
                y=[pred["revenue_p90"] - pred["revenue_p10"]],
                base=pred["revenue_p10"],
                name="80% Prediction Interval",
                marker_color="rgba(102, 126, 234, 0.5)",
                text=f"${pred['revenue_p10']:,.0f} - ${pred['revenue_p90']:,.0f}",
                textposition="inside",
            )
        )
        fig_rev.add_trace(
            go.Scatter(
                x=["Revenue"],
                y=[pred["revenue_p50"]],
                mode="markers",
                name="Median (P50)",
                marker=dict(color="#764ba2", size=20, symbol="diamond"),
            )
        )
        fig_rev.update_layout(
            height=300,
            title="Total Revenue Forecast with 80% Confidence Interval",
            yaxis_title="Revenue ($)",
            showlegend=True,
        )
        st.plotly_chart(style_chart(fig_rev), width='stretch')

        st.markdown("---")
        st.markdown("### Channel-Level Breakdown")

        col_g, col_m, col_ms = st.columns(3)
        with col_g:
            st.markdown("#### Google Ads")
            st.metric(
                "Expected Revenue",
                f"${pred['google_revenue_p50']:,.0f}",
                delta=f"${pred['google_revenue_p90'] - pred['google_revenue_p10']:,.0f} range",
            )
            if google_budget > 0:
                st.caption(f"ROAS: {pred['google_revenue_p50']/google_budget:.2f}x")

        with col_m:
            st.markdown("#### Meta Ads")
            st.metric(
                "Expected Revenue",
                f"${pred['meta_revenue_p50']:,.0f}",
                delta=f"${pred['meta_revenue_p90'] - pred['meta_revenue_p10']:,.0f} range",
            )
            if meta_budget > 0:
                st.caption(f"ROAS: {pred['meta_revenue_p50']/meta_budget:.2f}x")

        with col_ms:
            st.markdown("#### Microsoft Ads")
            st.metric(
                "Expected Revenue",
                f"${pred['ms_revenue_p50']:,.0f}",
                delta=f"${pred['ms_revenue_p90'] - pred['ms_revenue_p10']:,.0f} range",
            )
            if ms_budget > 0:
                st.caption(f"ROAS: {pred['ms_revenue_p50']/ms_budget:.2f}x")

        # Channel comparison
        fig_channel = go.Figure()
        channels = ["Google", "Meta", "Microsoft"]
        p50s = [pred["google_revenue_p50"], pred["meta_revenue_p50"], pred["ms_revenue_p50"]]
        p10s = [pred["google_revenue_p10"], pred["meta_revenue_p10"], pred["ms_revenue_p10"]]
        p90s = [pred["google_revenue_p90"], pred["meta_revenue_p90"], pred["ms_revenue_p90"]]

        fig_channel.add_trace(
            go.Bar(
                x=channels,
                y=[p90 - p10 for p90, p10 in zip(p90s, p10s)],
                base=p10s,
                name="80% Interval",
                marker_color=["#4285F4", "#1877F2", "#00A4EF"],
                opacity=0.6,
            )
        )
        fig_channel.add_trace(
            go.Scatter(
                x=channels,
                y=p50s,
                mode="markers",
                name="Median",
                marker=dict(color="#764ba2", size=15, symbol="diamond"),
            )
        )
        fig_channel.update_layout(
            height=350,
            title="Revenue by Channel (with Prediction Intervals)",
            yaxis_title="Revenue ($)",
        )
        st.plotly_chart(style_chart(fig_channel), width='stretch')

        st.markdown("---")
        st.markdown("### Blended ROAS Forecast")

        col_roas1, col_roas2 = st.columns([1, 2])
        with col_roas1:
            fig_roas = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=float(pred["blended_roas_p50"]),
                    delta={"reference": 3.0, "increasing": {"color": "green"}},
                    title={"text": "Expected ROAS"},
                    domain={"x": [0, 1], "y": [0, 1]},
                    gauge={
                        "axis": {"range": [0, 15]},
                        "bar": {"color": "#667eea"},
                        "steps": [
                            {"range": [0, 1], "color": "#ff4444"},
                            {"range": [1, 3], "color": "#ffaa00"},
                            {"range": [3, 6], "color": "#00cc44"},
                            {"range": [6, 15], "color": "#00aa44"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 1.0,
                        },
                    },
                )
            )
            fig_roas.update_layout(height=300)
            st.plotly_chart(style_chart(fig_roas), width='stretch')

        with col_roas2:
            st.markdown(
                f"""
<div class="metric-card">
  <h4>ROAS Interpretation</h4>
  <p><b>Conservative:</b> {pred['blended_roas_p10']:.2f}x</p>
  <p><b>Expected:</b> {pred['blended_roas_p50']:.2f}x</p>
  <p><b>Optimistic:</b> {pred['blended_roas_p90']:.2f}x</p>
  <hr>
  <p><b>For every $1 spent:</b></p>
  <p> Conservative return: ${pred['blended_roas_p10']:.2f}</p>
  <p> Expected return: ${pred['blended_roas_p50']:.2f}</p>
  <p> Optimistic return: ${pred['blended_roas_p90']:.2f}</p>
</div>
""",
                unsafe_allow_html=True,
            )

        # AI Insights
        if enable_llm:
            if GROQ_AVAILABLE and GROQ_API_KEY:
                st.markdown("---")
                st.markdown("### AI-Generated Insights")

                with st.spinner("Generating AI insights with Groq..."):
                    try:
                        client = Groq(api_key=GROQ_API_KEY)
                        context = f"""
Forecast Scenario:
- Period: {period_days} days
- Total Budget: ${total_budget:,.0f}
- Google Budget: ${google_budget:,.0f}
- Meta Budget: ${meta_budget:,.0f}
- Microsoft Budget: ${ms_budget:,.0f}

Forecast Results:
- Revenue (P10/P50/P90): ${pred['revenue_p10']:,.0f} / ${pred['revenue_p50']:,.0f} / ${pred['revenue_p90']:,.0f}
- Blended ROAS (P10/P50/P90): {pred['blended_roas_p10']:.2f}x / {pred['blended_roas_p50']:.2f}x / {pred['blended_roas_p90']:.2f}x
"""

                        response = client.chat.completions.create(
                            model=GROQ_MODEL,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are an expert e-commerce marketing analyst. Provide concise, actionable insights about forecast results. Keep response under 200 words.",
                                },
                                {
                                    "role": "user",
                                    "content": "Analyze this forecast and provide: (1) key insight, (2) one risk, (3) one actionable recommendation.\n\n" + context,
                                },
                            ],
                            max_tokens=300,
                            temperature=0.7,
                        )

                        insight_text = response.choices[0].message.content
                        st.markdown(
                            f"""
<div class="insight-box">
  {insight_text}
</div>
""",
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.warning(f"AI insights unavailable: {e}")
            else:
                st.info("Install `groq` package and set GROQ_API_KEY in .env for AI insights")

    else:
        st.markdown(
            """
<div class="metric-card">
  <h3>Welcome to AdRevenue Forecast Studio</h3>
  <p>This demo showcases probabilistic revenue forecasting for e-commerce marketing.</p>
  <p><b>To get started:</b></p>
  <ol>
    <li>Set your planning period (30/60/90 days)</li>
    <li>Adjust budget allocations in the sidebar</li>
    <li>Click <b>"Generate Forecast"</b></li>
  </ol>
  <p><i>Enable AI Insights to get Groq-powered analysis of your forecast.</i></p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### Historical Data Overview")
        try:
            df = load_data_cached()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Campaigns", df["campaign_id"].nunique())
            with col3:
                start_date = df["date"].min().date()
                end_date = df["date"].max().date()
                st.write(f"**Date Range:** {start_date} - {end_date}")
            with col4:
                st.metric("Platforms", len(df["platform"].unique()))

            daily_platform = df.groupby(["date", "platform"])["revenue"].sum().reset_index()
            daily_pivot = daily_platform.pivot(index="date", columns="platform", values="revenue").fillna(0)

            fig_trend = px.area(
                daily_pivot,
                title="Daily Revenue by Platform",
                labels={"value": "Revenue ($)", "date": "Date"},
                color_discrete_map={"google": "#4285F4", "meta": "#1877F2", "microsoft": "#00A4EF"},
            )
            st.plotly_chart(style_chart(fig_trend), width='stretch')
        except Exception as e:
            st.warning(f"Data preview unavailable: {e}")


# ============================================================
# TAB 2: DATA EXPLORER
# ============================================================

with tab2:
    st.markdown("## Data Explorer")
    try:
        df = load_data_cached()

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            selected_platforms = st.multiselect(
                "Platform",
                options=df["platform"].unique(),
                default=df["platform"].unique(),
            )
        with col_f2:
            selected_types = st.multiselect(
                "Campaign Type",
                options=df["campaign_type"].unique(),
                default=sorted(df["campaign_type"].unique()),
            )

        # Normalize for robust filtering (case/whitespace/aliases)
        df["platform"] = df["platform"].astype(str).str.strip().str.lower()
        df["campaign_type"] = df["campaign_type"].astype(str).str.strip().str.upper()

        def _norm_platform(v: str) -> str:
            v = str(v).strip().lower()
            aliases = {
                "google": "google",
                "meta": "meta",
                "microsoft": "microsoft",
                "ms": "microsoft",
                "microsoft_ads": "microsoft",
            }
            return aliases.get(v, v)

        selected_platforms_norm = [_norm_platform(p) for p in selected_platforms]
        selected_types_norm = [str(t).strip().upper() for t in selected_types]

        filtered = df[
            (df["platform"].isin(selected_platforms_norm))
            & (df["campaign_type"].isin(selected_types_norm))
        ]

        st.markdown("### Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Filtered Rows", f"{len(filtered):,}")
        with col2:
            st.metric("Total Spend", f"${filtered['spend'].sum():,.0f}")
        with col3:
            st.metric("Total Revenue", f"${filtered['revenue'].sum():,.0f}")
        with col4:
            roas = filtered["revenue"].sum() / filtered["spend"].sum() if filtered["spend"].sum() > 0 else 0
            st.metric("Overall ROAS", f"{roas:.2f}x")

        # Meta (and some other CSVs) may not have a usable revenue value column.
        # Loader normalizes the metric as `conversions` (plural).
        revenue_total = float(filtered["revenue"].sum()) if "revenue" in filtered.columns else 0.0

        if revenue_total > 0 and "revenue" in filtered.columns:
            y_col = "revenue"
        elif "conversions" in filtered.columns:
            y_col = "conversions"
        else:
            # Defensive fallback: pick whichever exists among known target columns, else error.
            candidate_cols = [c for c in ["revenue", "conversions"] if c in filtered.columns]
            if not candidate_cols:
                raise KeyError("Neither 'revenue' nor 'conversions' columns exist in the loaded data.")
            y_col = candidate_cols[0]

        chart_title = (
            "Daily Campaign Performance (Revenue)"
            if y_col == "revenue"
            else "Daily Campaign Performance (Conversions)"
        )

        st.markdown(f"### Spend vs {y_col.capitalize()} by Platform")
        fig_scatter = px.scatter(
            filtered,
            x="spend",
            y=y_col,
            color="platform",
            hover_data=["campaign_name", "date"],
            title=chart_title,
            opacity=0.6,
            color_discrete_map={"google": "#4285F4", "meta": "#1877F2", "microsoft": "#00A4EF"},
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(style_chart(fig_scatter), width='stretch')

        st.markdown("### Raw Data Sample")
        st.dataframe(
            filtered.head(100)[
                [
                    "date",
                    "platform",
                    "campaign_name",
                    "campaign_type",
                    "spend",
                    "revenue",
                    "clicks",
                    "conversions",
                ]
            ],
            width='stretch',
        )
    except Exception as e:
        st.error(f"Error loading data: {e}")


# ============================================================
# TAB 3: ANOMALIES
# ============================================================

with tab3:
    st.markdown("## Anomaly Detection")
    try:
        df = load_data_cached()
        detector = AnomalyDetector()
        results = detector.detect_all(df)

        summary = results["summary"]
        severity_colors = {"none": "green", "low": "yellow", "medium": "orange", "high": "red"}
        color = severity_colors.get(summary["severity"], "gray")

        st.markdown(
            f"""
<div class="metric-card">
  <h3>Data Health: <span style="color:{color}">{summary['severity'].upper()}</span></h3>
  <p>Total anomalies detected: <b>{summary['total_anomalies']}</b></p>
</div>
""",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### By Type")
            by_type = summary.get("by_type", {})
            if by_type:
                fig_type = px.bar(
                    x=list(by_type.keys()),
                    y=list(by_type.values()),
                    title="Anomalies by Type",
                    labels={"x": "Type", "y": "Count"},
                    color=list(by_type.values()),
                    color_continuous_scale="Reds",
                )
                fig_type.update_layout(height=300)
                st.plotly_chart(style_chart(fig_type), width='stretch')

        with col2:
            st.markdown("#### By Platform")
            by_platform = summary.get("by_platform", {})
            if by_platform:
                fig_plat = px.pie(
                    names=list(by_platform.keys()),
                    values=list(by_platform.values()),
                    title="Anomalies by Platform",
                    color_discrete_map={"google": "#4285F4", "meta": "#1877F2", "microsoft": "#00A4EF"},
                )
                fig_plat.update_layout(height=300)
                st.plotly_chart(style_chart(fig_plat), width='stretch')

        st.markdown("---")
        st.markdown("### Spend Anomalies")
        spend_outliers = results.get("spend_outliers", pd.DataFrame())
        if len(spend_outliers) > 0:
            recent_outliers = spend_outliers[
                spend_outliers["date"] >= spend_outliers["date"].max() - timedelta(days=90)
            ]
            if len(recent_outliers) > 0:
                fig_spend = px.scatter(
                    recent_outliers,
                    x="date",
                    y="value",
                    color="platform",
                    hover_data=["campaign_name", "type", "deviation_pct"],
                    title="Recent Spend Anomalies (Last 90 Days)",
                    color_discrete_map={"google": "#4285F4", "meta": "#1877F2", "microsoft": "#00A4EF"},
                )
                st.plotly_chart(style_chart(fig_spend), width='stretch')

            st.dataframe(
                spend_outliers.head(20)[["date", "platform", "campaign_name", "type", "value", "deviation_pct"]],
                width='stretch',
            )
        else:
            st.success("No spend anomalies detected")

        st.markdown("---")
        st.markdown("### Data Gaps")
        gaps = results.get("campaign_gaps", pd.DataFrame())
        if len(gaps) > 0:
            st.dataframe(gaps[["platform", "campaign_name", "gap_start", "gap_end", "gap_days"]], width='stretch')
        else:
            st.success("No data gaps detected")

    except Exception as e:
        st.error(f"Error in anomaly detection: {e}")


# ============================================================
# TAB 4: METHODOLOGY
# ============================================================

with tab4:
    st.markdown("## Methodology")
    st.markdown(
        """
<div class="metric-card">
  <h3>Forecasting Approach</h3>

  <h4>1. Data Aggregation</h4>
  <p>Daily campaign-level data is aggregated to platform-level totals and then rolled up into planning periods (30/60/90 days).</p>

  <h4>2. Feature Engineering</h4>
  <p>Features include spend/revenue aggregates, ROAS metrics, time indicators (month/quarter/holidays), and campaign-type aggregates.</p>

  <h4>3. Model</h4>
  <p>LightGBM quantile regression trained for P10/P50/P90 targets to produce probabilistic intervals.</p>

  <h4>4. Interval Calibration</h4>
  <p>Conformal-style calibration adjusts interval width for better coverage.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Feature Importance")

    model = load_model_cached()
    if model and "target_total_revenue_q50" in model.models:
        importance = model.models["target_total_revenue_q50"].feature_importances_
        feat_names = model.feature_names

        top_idx = np.argsort(importance)[-15:][::-1]
        top_feats = [feat_names[i] for i in top_idx]
        top_imps = [importance[i] for i in top_idx]

        fig_imp = px.bar(
            x=top_imps,
            y=top_feats,
            orientation="h",
            title="Top 15 Feature Importances (Total Revenue Model)",
            labels={"x": "Importance", "y": "Feature"},
            color=top_imps,
            color_continuous_scale="Purples",
        )
        fig_imp.update_layout(height=450)
        st.plotly_chart(style_chart(fig_imp), width='stretch')
    else:
        st.info("Train the model first to see feature importances")

