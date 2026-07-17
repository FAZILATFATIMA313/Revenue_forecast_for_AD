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
from src.causal_layer import generate_causal_outputs
from src.config import GROQ_API_KEY, GROQ_MODEL, PICKLE_DIR
from src.data_loader import load_all_data
from src.logger import setup_logger
from src.predict import predict, compute_ood_flags
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

    .ood-badge {
        display: inline-block;
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffc107;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 6px;
        vertical-align: middle;
    }
    
    /* ===== TAB FONT SIZE ===== */
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
</style>
""",
    unsafe_allow_html=True,
)


def style_chart(fig, title_size: int = 20, axis_size: int = 16, legend_size: int = 16, base_size: int = 14):
    """Apply consistent font sizing to a Plotly figure via its layout config."""
    fig.update_layout(
        font=dict(size=base_size),
        title_font=dict(size=title_size, family="Arial, sans-serif"),
        legend=dict(font=dict(size=legend_size)),
    )
    fig.update_xaxes(tickfont=dict(size=axis_size))
    fig.update_yaxes(tickfont=dict(size=axis_size))
    return fig


def fmt_float(val: float, decimals: int = 2) -> str:
    """Round a float to a fixed number of decimal places, returning a string."""
    if val == 0 or val != val:  # catches NaN
        return "0" if decimals == 0 else f"0.{'0'*decimals}"
    return f"{val:.{decimals}f}"


# ============================================================
# CACHE FUNCTIONS
# ============================================================


@st.cache_data(ttl=3600)
def load_data_cached() -> pd.DataFrame:
    return load_all_data()


@st.cache_data(ttl=3600)
def campaign_consistency_report_cached() -> dict:
    """
    Explicit ingestion requirement:
    make campaign consistency validation visible + demoable.
    """
    from src.data_loader import validate_campaign_consistency

    df = load_data_cached()
    return validate_campaign_consistency(df)


@st.cache_resource
def load_model_cached() -> ForecastingModel | None:
    model_path = PICKLE_DIR / "model.pkl"
    if model_path.exists():
        return ForecastingModel.load(str(model_path))
    return None


@st.cache_data(ttl=3600)
def get_validation_results_cached() -> dict:
    """Run validation suite and cache results for Methodology tab."""
    try:
        from src.validate import ForecastingValidator

        df = load_data_cached()
        model = load_model_cached()
        if model is None:
            return {"error": "Model not trained"}
        validator = ForecastingValidator()
        # Override model with loaded instance
        validator.model = model
        results = validator.run_all_validations(df)
        return results
    except Exception as e:
        return {"error": str(e)}


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

    # Pre-compute historical daily spend percentiles for OOD hints
    ood_hints = {"google": "", "meta": "", "ms": ""}
    try:
        df_all = load_data_cached()
        for plat_norm, plat_key in [("google", "google"), ("meta", "meta"), ("microsoft", "ms")]:
            plat_daily = df_all[df_all["platform"] == plat_norm].groupby("date")["spend"].sum().dropna()
            if len(plat_daily) >= 10:
                p5 = float(np.percentile(plat_daily.values, 5))
                p95 = float(np.percentile(plat_daily.values, 95))
                ood_hints[plat_key] = f"Typical daily range: ${p5:,.0f}-${p95:,.0f}"
    except Exception:
        pass

    google_budget = st.number_input(
        "Google Ads Budget ($)",

        min_value=0,
        value=int(default_google),
        step=5000,
        format="%d",
        help=ood_hints["google"],
    )
    meta_budget = st.number_input(
        "Meta Ads Budget ($)",
        min_value=0,
        value=int(default_meta),
        step=5000,
        format="%d",
        help=ood_hints["meta"],
    )
    ms_budget = st.number_input(
        "Microsoft Ads Budget ($)",
        min_value=0,
        value=int(default_ms),
        step=1000,
        format="%d",
        help=ood_hints["ms"],
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

    # Disable forecast button when total budget is zero — prevents pointless 2-minute wait
    budget_too_low = total_budget <= 0
    if budget_too_low:
        st.warning("Enter a budget above $0 to run a forecast.")

    forecast_button = st.button(
        "Generate Forecast",
        type="primary",
        disabled=budget_too_low,
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
        model = load_model_cached()
        if model is None:
            st.error("Model not found! Run `python src/train.py` first.")
            st.stop()

        # Multi-stage status display — shows progress during the 2-3 minute forecast
        status_placeholder = st.empty()

        # Prepare temp request data + feature generation
        from src.generate_features import generate_features

        tmp_dir = tempfile.mkdtemp()
        pred = None
        ood_flags = {}
        campaign_alloc = pd.DataFrame()

        with st.status("Generating probabilistic forecast...", expanded=True) as forecast_status:
            forecast_status.write("Preparing data...")

            try:
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

                forecast_status.write("Generating features...")
                features_path = os.path.join(tmp_dir, "features.parquet")
                generate_features(data_tmp, features_path)

                forecast_status.write("Running prediction model...")
                predict(features_path, str(PICKLE_DIR / "model.pkl"), os.path.join(tmp_dir, "predictions.csv"))
                predictions = pd.read_csv(os.path.join(tmp_dir, "predictions.csv"))

                pred = predictions.iloc[0]

                # OOD flags
                forecast_status.write("Checking confidence...")
                budgets_for_ood = {"google": float(google_budget), "meta": float(meta_budget), "ms": float(ms_budget)}
                df_hist = load_data_cached()
                ood_flags = compute_ood_flags(
                    df_hist, budgets_for_ood, period_days,
                    pred_row=pred.to_dict() if hasattr(pred, "to_dict") else dict(pred),
                )

                # Campaign-level breakdown
                from src.predict import allocate_campaign_level_from_history
                try:
                    budgets = {
                        "google": float(google_budget),
                        "meta": float(meta_budget),
                        "ms": float(ms_budget),
                    }
                    campaign_alloc = allocate_campaign_level_from_history(
                        df_hist=df_hist, pred_row=pred, period_days=period_days,
                        budgets=budgets, top_n=12, share_lookback_days=period_days,
                    )
                except Exception:
                    campaign_alloc = pd.DataFrame()

                for prefix in ['revenue', 'google_revenue', 'meta_revenue', 'ms_revenue']:
                    for q in ['p10', 'p50', 'p90']:
                        col = f'{prefix}_{q}'
                        if col in pred:
                            pred[col] = max(0, float(pred[col]))

                forecast_status.write("Forecast complete!")
                # Let status stay visible briefly, then collapse
                import time; time.sleep(0.5)

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
                text=f"P10-P90: ${pred['revenue_p10']:,.0f} - ${pred['revenue_p90']:,.0f}",
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

        def _channel_header(name: str, ch_key: str) -> str:
            flag = ood_flags.get(ch_key, {})
            if isinstance(flag, dict) and flag.get('is_ood'):
                return f"**{name}** <span class='ood-badge'>[!] Low confidence</span>"
            return f"**{name}**"

        col_g, col_m, col_ms = st.columns(3)
        with col_g:
            st.markdown(f"#### {_channel_header('Google Ads', 'google')}", unsafe_allow_html=True)
            st.metric(
                "Expected Revenue",
                f"${pred['google_revenue_p50']:,.0f}",
            )
            st.caption(f"P10–P90 range: {pred['google_revenue_p10']:,.0f} to {pred['google_revenue_p90']:,.0f} USD")
            if google_budget > 0:
                st.caption(f"ROAS: {pred['google_revenue_p50']/google_budget:.2f}x")

        with col_m:
            st.markdown(f"#### {_channel_header('Meta Ads', 'meta')}", unsafe_allow_html=True)
            st.metric(
                "Expected Revenue",
                f"${pred['meta_revenue_p50']:,.0f}",
            )
            st.caption(f"P10–P90 range: {pred['meta_revenue_p10']:,.0f} to {pred['meta_revenue_p90']:,.0f} USD")
            if meta_budget > 0:
                st.caption(f"ROAS: {pred['meta_revenue_p50']/meta_budget:.2f}x")

        with col_ms:
            st.markdown(f"#### {_channel_header('Microsoft Ads', 'ms')}", unsafe_allow_html=True)
            st.metric(
                "Expected Revenue",
                f"${pred['ms_revenue_p50']:,.0f}",
            )
            st.caption(f"P10–P90 range: {pred['ms_revenue_p10']:,.0f} to {pred['ms_revenue_p90']:,.0f} USD")
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
            # Dynamic gauge axis: scale to ~1.5x the P90 value, floor at 15
            roas_p90 = float(pred["blended_roas_p90"])
            gauge_max = max(15.0, roas_p90 * 1.5)
            fig_roas = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=float(pred["blended_roas_p50"]),
                    delta={"reference": 3.0, "increasing": {"color": "green"}},
                    title="Expected ROAS",
                    domain={"x": [0, 1], "y": [0, 1]},
                    gauge={
                        "axis": {"range": [0, gauge_max]},
                        "bar": {"color": "#667eea"},
                        "steps": [
                            {"range": [0, min(1, gauge_max)], "color": "#ff4444"},
                            {"range": [min(1, gauge_max), min(3, gauge_max)], "color": "#ffaa00"},
                            {"range": [min(3, gauge_max), min(6, gauge_max)], "color": "#00cc44"},
                            {"range": [min(6, gauge_max), gauge_max], "color": "#00aa44"},
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
            # Don't pass through style_chart — it sets title_font on figures
            # without a layout title, which creates a phantom title with
            # undefined text that Plotly renders as literal "undefined".
            st.plotly_chart(fig_roas, width='stretch')

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

        # ---- Download forecast as CSV ----
        st.markdown("---")
        forecast_csv = pd.DataFrame([{
            "period_days": period_days,
            "spend_google": pred["spend_google"],
            "spend_meta": pred["spend_meta"],
            "spend_ms": pred["spend_ms"],
            "revenue_p10": pred["revenue_p10"],
            "revenue_p50": pred["revenue_p50"],
            "revenue_p90": pred["revenue_p90"],
            "blended_roas_p10": pred["blended_roas_p10"],
            "blended_roas_p50": pred["blended_roas_p50"],
            "blended_roas_p90": pred["blended_roas_p90"],
            "google_revenue_p10": pred["google_revenue_p10"],
            "google_revenue_p50": pred["google_revenue_p50"],
            "google_revenue_p90": pred["google_revenue_p90"],
            "meta_revenue_p10": pred["meta_revenue_p10"],
            "meta_revenue_p50": pred["meta_revenue_p50"],
            "meta_revenue_p90": pred["meta_revenue_p90"],
            "ms_revenue_p10": pred["ms_revenue_p10"],
            "ms_revenue_p50": pred["ms_revenue_p50"],
            "ms_revenue_p90": pred["ms_revenue_p90"],
        }])
        csv_bytes = forecast_csv.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download this forecast as CSV",
            data=csv_bytes,
            file_name=f"forecast_{period_days}d.csv",
            mime="text/csv",
        )

        # Campaign-level breakdown (if available)
        if campaign_alloc is not None and len(campaign_alloc) > 0:
            st.markdown("---")
            st.markdown("### Campaign-Level Breakdown")

            st.markdown("Top campaigns by channel (estimated from historical spend mix):")
            display_cols = ["platform", "campaign_name", "campaign_type",
                            "revenue_p50", "spend_allocated", "roas_p50"]
            available_display = [c for c in display_cols if c in campaign_alloc.columns]
            st.dataframe(
                campaign_alloc[available_display].head(20).style.format({
                    "revenue_p50": "${:,.0f}",
                    "spend_allocated": "${:,.0f}",
                    "roas_p50": "{:.2f}x",
                }),
                width='stretch',
            )
            st.caption("Note: Campaign-level allocation is proportional based on historical spend shares, not independently modeled.")

        # AI-Assisted Causal Insights (first-class pipeline step)
        if enable_llm:
            st.markdown("---")
            st.markdown("### AI-Assisted Causal Insights")

            try:
                # Structured grounding signals: anomalies + model feature importances + period-over-period deltas
                df_hist = load_data_cached()
                detector = AnomalyDetector()

                # Scope anomaly detection to the relevant feature window + recent lookback
                # (period_days * 2 captures the feature window + buffer)
                scope_start = df_hist["date"].max() - timedelta(days=period_days * 3)
                df_scoped = df_hist[df_hist["date"] >= scope_start].copy()
                anomaly_results = detector.detect_all(df_scoped)

                # Deterministic period-over-period deltas (budgets vs recent baseline daily spend)
                last_30 = df_hist[df_hist["date"] >= df_hist["date"].max() - timedelta(days=30)]
                prior_daily = {
                    "google": float(last_30[last_30["platform"] == "google"]["spend"].sum() / 30)
                    if len(last_30[last_30["platform"] == "google"]) > 0
                    else 0.0,
                    "meta": float(last_30[last_30["platform"] == "meta"]["spend"].sum() / 30)
                    if len(last_30[last_30["platform"] == "meta"]) > 0
                    else 0.0,
                    "ms": float(last_30[last_30["platform"] == "microsoft"]["spend"].sum() / 30)
                    if len(last_30[last_30["platform"] == "microsoft"]) > 0
                    else 0.0,
                }

                this_daily = {
                    "google": float(google_budget) / max(1, int(period_days)),
                    "meta": float(meta_budget) / max(1, int(period_days)),
                    "ms": float(ms_budget) / max(1, int(period_days)),
                }

                deltas = {}
                for k in ["google", "meta", "ms"]:
                    prior = prior_daily.get(k, 0.0)
                    curr = this_daily.get(k, 0.0)
                    if prior and prior > 0:
                        delta_pct = (curr - prior) / prior * 100.0
                        deltas[f"spend_{k}"] = {"delta_pct": float(delta_pct)}
                    else:
                        deltas[f"spend_{k}"] = {"delta_pct": 0.0}

                with st.spinner("Generating grounded causal attribution + risk flags..."):
                    causal = generate_causal_outputs(
                        period_days=period_days,
                        pred_row=pred.to_dict() if hasattr(pred, "to_dict") else dict(pred),
                        model=model,
                        anomaly_results=anomaly_results,
                        period_over_period_deltas=deltas,
                        ood_flags=ood_flags,
                    )

                bullets = causal.get("causal_attribution_bullets", []) or []
                risk_flags = causal.get("risk_flags", []) or []

                st.markdown(
                    f"""
<div class="insight-box">
  <h4>Why the model expects this move</h4>
  <ul>
    {''.join([f"<li>{b}</li>" for b in bullets])}
  </ul>
  <hr/>
  <h4>Operational risk flags</h4>
  <ul>
    {''.join([f"<li><b>{r.get('severity','')}&nbsp;</b> {r.get('risk','')} — {r.get('evidence','')}</li>" for r in risk_flags])}
  </ul>
</div>
""",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                logger.warning("Causal insights unavailable", exc_info=True)
                st.warning("AI insights are temporarily unavailable — showing forecast without causal explanation.")

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

        # When forecast is not running yet, skip forecast-only sections.
        st.stop()

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
        daily_pivot = daily_platform.pivot(
            index="date", columns="platform", values="revenue"
        ).fillna(0)

        fig_trend = px.area(
            daily_pivot,
            title="Daily Revenue by Platform",
            labels={"value": "Revenue ($)", "date": "Date"},
            color_discrete_map={"google": "#4285F4", "meta": "#1877F2", "microsoft": "#00A4EF"},
        )
        st.plotly_chart(style_chart(fig_trend), width="stretch")
    except Exception as e:
        logger.warning("Data preview unavailable", exc_info=True)
        st.warning("Historical data preview is temporarily unavailable — check the Data Explorer tab for details.")


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

        # ============================================================
        # Data Quality / Campaign Consistency (explicit ingestion step)
        # ============================================================
        st.markdown("---")
        st.markdown("## Data Quality")

        qc = None
        try:
            qc = campaign_consistency_report_cached()
        except Exception as e:
            logger.warning("Campaign consistency validation unavailable", exc_info=True)
            st.warning("Data quality report is temporarily unavailable — the data may still be valid.")

        if isinstance(qc, dict) and "summary" in qc:
            s = qc["summary"] or {}
            passed = bool(s.get("passed", False))

            status_color = "green" if passed else "red"
            st.markdown(
                f"""
<div class="metric-card">
  <h4>Campaign Consistency: <span style="color:{status_color}">{'PASS' if passed else 'FAIL'}</span></h4>
  <p><b>Missing campaign_id rows:</b> {s.get('missing_campaign_id_rows', 0):,}</p>
  <p><b>Missing campaign_type rows:</b> {s.get('missing_campaign_type_rows', 0):,}</p>
  <p><b>Inconsistent name groups:</b> {s.get('inconsistent_name_groups', 0):,}</p>
  <p><b>Inconsistent type groups:</b> {s.get('inconsistent_type_groups', 0):,}</p>
  <p><b>Platform reassignment campaign IDs:</b> {s.get('platform_reassignment_campaign_ids', 0):,}</p>
  <p><b>Campaigns with date gaps:</b> {s.get('campaigns_with_date_gaps', 0):,}</p>
</div>
""",
                unsafe_allow_html=True,
            )

            with st.expander("Details & examples (from validation checks)"):
                checks = qc.get("checks", {}) if isinstance(qc.get("checks", {}), dict) else {}
                if not checks:
                    st.info("No check details available.")
                else:
                    # Render top examples for each check
                    for check_name, check_result in checks.items():
                        passed_flag = bool(check_result.get("passed", False))
                        icon = "[PASS]" if passed_flag else "[WARN]"
                        st.markdown(f"### {icon} {check_name}")
                        st.write(check_result.get("details", ""))
                        examples = check_result.get("examples", []) or []
                        if examples:
                            st.json(examples[:10])
                        else:
                            st.caption("No examples")
        else:
            st.info("Run-time report not available yet.")

        revenue_total = float(filtered["revenue"].sum()) if "revenue" in filtered.columns else 0.0

        if revenue_total > 0 and "revenue" in filtered.columns:
            y_col = "revenue"
        elif "conversions" in filtered.columns:
            y_col = "conversions"
        else:
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
        logger.warning("Error loading data in Data Explorer tab", exc_info=True)
        st.error("Unable to load campaign data — check that the data files are present in the `./data` directory.")


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
        logger.warning("Error in anomaly detection", exc_info=True)
        st.error("Anomaly detection is temporarily unavailable — underlying data may be incomplete.")


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
  <p>Empirical bootstrap reconciliation is used to ensure additive coherence: blended total quantiles are derived from the same joint bootstrap draws as the channel-level quantiles, so P50 = google_P50 + meta_P50 + ms_P50 by construction.</p>

  <h4>5. OOD Detection</h4>
  <p>Per-channel out-of-distribution detection compares requested daily spend against historical 5th-95th percentiles. Channel cards flagged with "[!] Low confidence" indicate extrapolation risk.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Calibration & Backtesting Results")

    try:
        val_results = get_validation_results_cached()
        if isinstance(val_results, dict) and "error" not in val_results:
            # Coverage comparison
            rec_cal = val_results.get("8_reconciliation_calibration", {})
            if isinstance(rec_cal, dict) and "results" in rec_cal:
                r_res = rec_cal["results"]
                cov = r_res.get("coverage", {})
                pb = r_res.get("pinball_loss", {})
                n_samp = r_res.get("n_samples", {})

                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.markdown("#### Empirical P10–P90 Coverage")
                    legacy_cov = cov.get("legacy_p10_p90", float('nan'))
                    recon_cov = cov.get("reconciled_p10_p90", float('nan'))
                    st.metric("Legacy (independent heads)", f"{legacy_cov:.1%}" if not np.isnan(legacy_cov) else "N/A")
                    st.metric("Reconciled (bootstrap)", f"{recon_cov:.1%}" if not np.isnan(recon_cov) else "N/A",
                              delta=f"{recon_cov - legacy_cov:+.1%}" if not (np.isnan(recon_cov) or np.isnan(legacy_cov)) else None)
                    st.caption(f"Samples: {n_samp.get('legacy', 0)} legacy, {n_samp.get('reconciled', 0)} reconciled")

                with col_c2:
                    st.markdown("#### Pinball Loss (lower is better)")
                    leg_pb = pb.get("legacy", {})
                    rec_pb = pb.get("reconciled", {})
                    for q_label in ["p10", "p50", "p90"]:
                        lv = leg_pb.get(q_label, float('nan'))
                        rv = rec_pb.get(q_label, float('nan'))
                        better = rv < lv if not (np.isnan(rv) or np.isnan(lv)) else None
                        delta_str = f"{rv - lv:+.4f}" if not (np.isnan(rv) or np.isnan(lv)) else None
                        st.metric(
                            f"{q_label.upper()}",
                            f"{rv:.4f}" if not np.isnan(rv) else "N/A",
                            delta=delta_str,
                            delta_color="normal" if better else "inverse",
                        )

                st.markdown("---")

            # Model robustness (backtesting error)
            rob = val_results.get("2_model_robustness", {})
            if isinstance(rob, dict):
                st.markdown("#### Backtesting (Rolling-Origin)")
                st.metric("Samples Tested", rob.get("samples", 0))
                st.metric("Median Error", f"{rob.get('median_error_pct', 0):.1f}%" if not np.isnan(rob.get('median_error_pct', float('nan'))) else "N/A")
                st.metric("Within 50% Accuracy", f"{rob.get('within_50pct_accuracy', 0):.1f}%")

            st.markdown("---")

            # Budget sensitivity checks
            bs = val_results.get("3_budget_sensitivity", {})
            if isinstance(bs, dict) and "checks" in bs:
                ch = bs["checks"]
                st.markdown("#### Budget Sensitivity Checks")
                for cname, cresult in ch.items():
                    icon = "[PASS]" if cresult.get("passed") else "[WARN]"
                    st.write(f"{icon} **{cname}**: {cresult.get('details', '')}")

            st.markdown("---")

            ec = val_results.get("5_edge_cases", {})
            if isinstance(ec, dict) and "checks" in ec:
                st.markdown("#### Edge Case Handling")
                for cname, cresult in ec["checks"].items():
                    icon = "[PASS]" if cresult.get("passed") else "[WARN]"
                    st.write(f"{icon} **{cname}**: {cresult.get('details', '')}")

            st.markdown("---")

            # Baseline comparison
            bs_metrics = val_results.get("2_model_robustness", {})
            if bs_metrics.get("median_error_pct", 0) and not np.isnan(bs_metrics.get("median_error_pct", float('nan'))):
                st.markdown("#### Baseline Model Comparison")
                st.markdown(
                    "The LightGBM quantile model is compared against a linear regression baseline during training. "
                    "The LightGBM model captures non-linear channel interactions and saturation effects that the linear model cannot represent. "
                    f"Backtesting median error: {bs_metrics.get('median_error_pct', 0):.1f}%."
                )
        else:
            err_msg = str(val_results.get('error', ''))
            if 'not trained' in err_msg.lower():
                st.info("Validation results require a trained model — run `python src/train.py` first.")
            else:
                logger.warning(f"Validation results unavailable: {err_msg}")
                st.info("Calibration and backtesting results will appear here once the validation suite completes.")

    except Exception as e:
        logger.warning("Backtesting results not available", exc_info=True)
        st.info("Calibration data will appear here once the model is trained and validated — no action needed.")

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

    st.markdown("---")
    st.markdown("### Reconciliation Methodology")

    st.markdown(
        """
<div class="metric-card">
  <h4>Bootstrap Reconciliation</h4>
  <p>Channel-level and blended-total revenue quantiles are derived from the same set of joint bootstrap residual draws:</p>
  <ol>
    <li>Compute q50 (median) predictions for each channel's revenue model from the feature window.</li>
    <li>Generate residuals = actual - predicted q50 from historical training periods with matching period length.</li>
    <li>Draw N=300 bootstrap indices (shared across all three channels).</li>
    <li>Build bootstrapped revenue draws: <code>boot_g = g_q50 + g_resid[i]</code> (and similarly for meta, ms).</li>
    <li>Compute blended total as <code>boot_total = boot_g + boot_m + boot_ms</code>.</li>
    <li>Take P10/P50/P90 of each distribution → additive by construction.</li>
  </ol>
  <p>This avoids the additive inconsistency that occurs when summing independently-trained quantile models.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### OOD Detection Method")

    st.markdown(
        """
<div class="metric-card">
  <h4>Out-of-Distribution Detection</h4>
  <p>For each forecast channel, the requested daily spend is compared against the historical daily spend distribution:</p>
  <ul>
    <li><b>Flagged</b> if daily spend is below the 5th percentile or above the 95th percentile.</li>
    <li>A warning badge appears next to the channel name in the Forecast tab.</li>
    <li>OOD flags are passed into the causal layer as an explicit signal for the explanation engine.</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )
