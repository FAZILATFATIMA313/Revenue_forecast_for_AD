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
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import ForecastingModel
from src.logger import setup_logger
from src.data_loader import load_all_data
from src.features import FeatureEngineer
from src.config import KNOWN_CTYPE_BUCKETS

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


MAX_REASONABLE_ROAS = 25.0


def _compute_channel_roas(pred_row: dict, ch_key: str, spend_key: str) -> float:
    rev = float(pred_row.get(f'{ch_key}_revenue_p50', 0.0))
    sp = float(pred_row.get(spend_key, 1.0))
    return rev / sp if sp > 0 else 0.0


def compute_ood_flags(
    df_hist: pd.DataFrame,
    budgets: Dict[str, float],
    period_days: int,
    pred_row: dict | None = None,
) -> Dict[str, dict]:
    if df_hist is None or df_hist.empty:
        return {p: {'is_ood': False, 'message': 'no historical data'} for p in ['google', 'meta', 'ms']}

    df = df_hist.copy()
    df['platform'] = df['platform'].astype(str).str.strip().str.lower()
    daily_requested = {k: v / max(period_days, 1) for k, v in budgets.items()}
    platform_map = {'google': 'google', 'meta': 'meta', 'microsoft': 'ms'}
    ood = {}

    for plat_norm, plat_key in platform_map.items():
        daily_budget = daily_requested.get(plat_key, 0.0)
        plat_daily = df[df['platform'] == plat_norm].groupby('date')['spend'].sum().dropna()
        plat_values = plat_daily.values
        if len(plat_values) >= 10:
            p5 = float(np.percentile(plat_values, 5))
            p95 = float(np.percentile(plat_values, 95))
            pctile = float(np.mean(plat_values < daily_budget))
            spend_ood = daily_budget < p5 or daily_budget > p95
        else:
            p5, p95, pctile, spend_ood = 0.0, 0.0, 0.5, False

        roas_ood = False
        roas_value = None
        if pred_row is not None:
            spend_key = f'spend_{plat_key}'
            rev_prefix = {'google': 'google', 'meta': 'meta', 'ms': 'ms'}[plat_key]
            roas_value = _compute_channel_roas(pred_row, rev_prefix, spend_key)
            roas_ood = roas_value > MAX_REASONABLE_ROAS

        is_ood = spend_ood or roas_ood
        messages = []
        if spend_ood and len(plat_values) >= 10:
            direction = 'above' if daily_budget > p95 else 'below'
            messages.append(f"outside historical spend range ({direction} {pctile:.0%} percentile, "
                           f"historical 5th–95th: ${p5:,.0f}–${p95:,.0f}/day)")
        if roas_ood and roas_value is not None:
            messages.append(f"channel ROAS ({roas_value:.1f}x) exceeds {MAX_REASONABLE_ROAS:.0f}x ceiling — "
                           f"forecast may be implausible")

        msg = '; '.join(messages)
        if msg:
            msg = f"[!] Low confidence — {msg}"

        ood[plat_key] = {
            'is_ood': is_ood, 'daily_requested': daily_budget,
            'p5': p5, 'p95': p95, 'pctile': pctile,
            'roas_value': roas_value, 'roas_ood': roas_ood, 'message': msg,
        }
    return ood


def allocate_campaign_level_from_history(
    df_hist: pd.DataFrame,
    pred_row: pd.Series | dict,
    period_days: int,
    budgets: dict,
    top_n: int = 12,
    share_lookback_days: int | None = None,
) -> pd.DataFrame:
    if isinstance(pred_row, dict):
        pred = pred_row
    else:
        pred = pred_row.to_dict()

    share_lookback_days = int(share_lookback_days or period_days)
    for k in ["google", "meta", "ms"]:
        if k not in budgets:
            budgets[k] = 0.0

    if df_hist.empty:
        return pd.DataFrame()

    df = df_hist.copy()
    df["platform"] = df["platform"].astype(str).str.strip().str.lower()
    df["campaign_type"] = df["campaign_type"].astype(str)
    max_date = pd.to_datetime(df["date"]).max()
    start_date = max_date - timedelta(days=share_lookback_days - 1)
    df_window = df[pd.to_datetime(df["date"]) >= start_date].copy()

    platform_map = {"google": "google", "meta": "meta", "microsoft": "ms", "ms": "ms"}
    df_window["platform_norm"] = df_window["platform"].map(platform_map).fillna(df_window["platform"])

    group_cols = ["platform_norm", "campaign_id", "campaign_name", "campaign_type"]
    for c in ["campaign_id", "campaign_name"]:
        if c not in df_window.columns:
            df_window[c] = "unknown"

    spend_by_camp = (
        df_window.groupby(group_cols, dropna=False)["spend"]
        .sum().reset_index().rename(columns={"spend": "hist_spend"})
    )

    def _alloc_for_platform(pname: str) -> pd.DataFrame:
        plat_spend_budget = float(budgets.get(pname, 0.0))
        if plat_spend_budget <= 0:
            platform_rows = spend_by_camp[spend_by_camp["platform_norm"] == pname].copy()
            if platform_rows.empty:
                return pd.DataFrame()
            platform_rows = platform_rows.nlargest(top_n, "hist_spend")
            other_spend = float(spend_by_camp[
                (spend_by_camp["platform_norm"] == pname)
                & ~spend_by_camp["campaign_id"].isin(platform_rows["campaign_id"])
            ]["hist_spend"].sum())
            platform_rows["campaign_bucket"] = "top"
            platform_rows["hist_spend_share"] = 0.0
            platform_rows = platform_rows.head(top_n).copy()
            if other_spend > 0 or len(platform_rows) > 0:
                other_row = {"platform_norm": pname, "campaign_id": "other", "campaign_name": "Other",
                             "campaign_type": "other", "hist_spend": other_spend,
                             "campaign_bucket": "other", "hist_spend_share": 0.0}
                platform_rows = pd.concat([platform_rows, pd.DataFrame([other_row])], ignore_index=True)
            for q in ["p10", "p50", "p90"]:
                platform_rows[f"revenue_{q}"] = 0.0
                platform_rows[f"spend_{q}"] = 0.0
                platform_rows[f"roas_{q}"] = 0.0
            return platform_rows

        plat_pred_rev = {q: float(pred.get(
            f"{'google' if pname=='google' else 'meta' if pname=='meta' else 'ms'}_revenue_{q}", 0.0))
            for q in ["p10", "p50", "p90"]}
        plat_rows = spend_by_camp[spend_by_camp["platform_norm"] == pname].copy()
        if plat_rows.empty:
            return pd.DataFrame()
        plat_rows = plat_rows.sort_values("hist_spend", ascending=False)
        top = plat_rows.head(top_n).copy()
        other = plat_rows.iloc[top_n:]
        other_hist_spend = float(other["hist_spend"].sum()) if len(other) > 0 else 0.0
        denom = float(plat_rows["hist_spend"].sum())
        if denom <= 0:
            top["hist_spend_share"] = 1.0 / max(1, len(top))
        else:
            top["hist_spend_share"] = top["hist_spend"] / denom
        top["campaign_bucket"] = "top"
        out = top[["platform_norm", "campaign_id", "campaign_name", "campaign_type",
                    "hist_spend", "hist_spend_share", "campaign_bucket"]].copy()
        if other_hist_spend > 0:
            other_share = other_hist_spend / denom if denom > 0 else 0.0
            other_row = pd.DataFrame([{"platform_norm": pname, "campaign_id": "other",
                                        "campaign_name": "Other", "campaign_type": "other",
                                        "hist_spend": other_hist_spend, "hist_spend_share": other_share,
                                        "campaign_bucket": "other"}])
            out = pd.concat([out, other_row], ignore_index=True)
        out_spend = out["hist_spend_share"].astype(float) * plat_spend_budget
        out["spend_allocated"] = out_spend
        for q in ["p10", "p50", "p90"]:
            out[f"revenue_{q}"] = out["hist_spend_share"].astype(float) * plat_pred_rev[q]
            out[f"spend_{q}"] = out["spend_allocated"].astype(float)
            out[f"roas_{q}"] = np.where(out[f"spend_{q}"] > 0, out[f"revenue_{q}"] / out[f"spend_{q}"], 0.0)
        return out

    google_df = _alloc_for_platform("google")
    meta_df = _alloc_for_platform("meta")
    ms_df = _alloc_for_platform("ms")
    all_alloc = pd.concat([google_df, meta_df, ms_df], ignore_index=True) if any(
        [len(x) > 0 for x in [google_df, meta_df, ms_df]]) else pd.DataFrame()
    if all_alloc.empty:
        return all_alloc
    all_alloc = all_alloc.rename(columns={"platform_norm": "platform"})
    return all_alloc


# Fixed seed for reproducibility — same rng used for all bootstrap draws
BOOTSTRAP_RNG = np.random.default_rng(42)


def _prepare_bootstrap_pools(
    df_hist: pd.DataFrame,
    model: ForecastingModel,
    feature_cols: list[str],
) -> dict:
    """
    Pre-compute the bootstrap residual pools shared across all forecast rows.
    Called ONCE per predict() call.
    """
    engineer = FeatureEngineer(periods=[30, 60, 90])
    feats_full = engineer.create_training_data(df_hist)

    pools = {}
    for period_days in [30, 60, 90]:
        feats = feats_full[feats_full["period_days"] == period_days].copy()
        if feats.empty:
            pools[period_days] = None
            continue

        required = ["target_google_revenue", "target_meta_revenue", "target_microsoft_revenue"]
        if not all(c in feats.columns for c in required):
            pools[period_days] = None
            continue

        X_hist = feats[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_hist_scaled = model.scaler.transform(X_hist)
        X_hist_scaled_df = pd.DataFrame(X_hist_scaled, columns=feature_cols, index=feats.index)

        g_actual = feats["target_google_revenue"].values.astype(float)
        m_actual = feats["target_meta_revenue"].values.astype(float)
        ms_actual = feats["target_microsoft_revenue"].values.astype(float)

        if not all(k in model.models for k in
                   ["target_google_revenue_q50", "target_meta_revenue_q50", "target_microsoft_revenue_q50"]):
            pools[period_days] = None
            continue

        g_pred = model.models["target_google_revenue_q50"].predict(X_hist_scaled_df)
        m_pred = model.models["target_meta_revenue_q50"].predict(X_hist_scaled_df)
        ms_pred = model.models["target_microsoft_revenue_q50"].predict(X_hist_scaled_df)

        pools[period_days] = {
            'g_resid': g_actual - g_pred,
            'm_resid': m_actual - m_pred,
            'ms_resid': ms_actual - ms_pred,
            'n': len(feats),
        }
        logger.info(f"   Bootstrap pool for {period_days}d: {len(feats)} samples")
    return pools


def _bootstrap_reconciled_from_pool(
    model: ForecastingModel,
    request_row_scaled: pd.DataFrame,
    period_days: int,
    pool: dict | None,
    n_boot: int = 300,
    quantiles: tuple = (0.10, 0.50, 0.90),
) -> dict | None:
    if pool is None or pool['n'] < 20:
        return None

    n = pool['n']
    g_resid = pool['g_resid']
    m_resid = pool['m_resid']
    ms_resid = pool['ms_resid']

    g_req_p50 = float(model.models["target_google_revenue_q50"].predict(request_row_scaled)[0])
    m_req_p50 = float(model.models["target_meta_revenue_q50"].predict(request_row_scaled)[0])
    ms_req_p50 = float(model.models["target_microsoft_revenue_q50"].predict(request_row_scaled)[0])

    # Fixed seed ensures reproducibility across runs
    idx = BOOTSTRAP_RNG.integers(0, n, size=n_boot)

    boot_g = np.clip(g_req_p50 + g_resid[idx], 0.0, None)
    boot_m = np.clip(m_req_p50 + m_resid[idx], 0.0, None)
    boot_ms = np.clip(ms_req_p50 + ms_resid[idx], 0.0, None)
    boot_total = boot_g + boot_m + boot_ms

    q10, q50, q90 = quantiles
    return {
        "google_revenue_p10": float(np.quantile(boot_g, q10)),
        "google_revenue_p50": float(np.quantile(boot_g, q50)),
        "google_revenue_p90": float(np.quantile(boot_g, q90)),
        "meta_revenue_p10": float(np.quantile(boot_m, q10)),
        "meta_revenue_p50": float(np.quantile(boot_m, q50)),
        "meta_revenue_p90": float(np.quantile(boot_m, q90)),
        "ms_revenue_p10": float(np.quantile(boot_ms, q10)),
        "ms_revenue_p50": float(np.quantile(boot_ms, q50)),
        "ms_revenue_p90": float(np.quantile(boot_ms, q90)),
        "revenue_p10": float(np.quantile(boot_total, q10)),
        "revenue_p50": float(np.quantile(boot_total, q50)),
        "revenue_p90": float(np.quantile(boot_total, q90)),
    }


def predict(features_path: str, model_path: str, output_path: str, data_dir: str | None = None):
    """
    Generate predictions from features and model.
    Bootstrap residual pools are computed ONCE and shared across all request rows.
    The data directory for historical data is resolved relative to the features_path
    (not an absolute config path), so it works on any machine via run.sh args.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"?? GENERATING PREDICTIONS")
    logger.info(f"{'='*60}")

    logger.info(f"?? Loading features from {features_path}")
    if str(features_path).endswith(".parquet"):
        features = pd.read_parquet(features_path)
    else:
        features = pd.read_csv(features_path)
    logger.info(f"   Shape: {features.shape}")

    logger.info(f"?? Loading model from {model_path}")
    model = ForecastingModel.load(model_path)
    logger.info(f"   Features expected: {len(model.feature_names)}")
    logger.info(f"   Models trained: {len(model.models)}")

    X = features.copy()
    for col in model.feature_names:
        if col not in X.columns:
            X[col] = 0
            logger.warning(f"   Missing feature: {col} - filling with 0")

    X_features = X[model.feature_names].fillna(0)
    X_scaled = model.scaler.transform(X_features)
    X_scaled_df = pd.DataFrame(X_scaled, columns=model.feature_names)

    # ---- Resolve data directory ----
    # Priority: 1) explicit --data-dir arg, 2) ../data relative to features file,
    # 3) ./data relative to cwd.
    # Critical for submission: the grading machine will supply a --data-dir arg
    # from run.sh, so none of the fallbacks should ever trigger on the grader.
    if data_dir is not None:
        hist_data_dir = data_dir
    else:
        features_dir = os.path.dirname(os.path.abspath(features_path))
        data_dir_candidate = os.path.join(features_dir, "..", "data")
        if os.path.isdir(data_dir_candidate):
            hist_data_dir = data_dir_candidate
        else:
            hist_data_dir = os.path.join(os.getcwd(), "data")

    # ---- Load historical data ONCE for both bootstrap pool + OOD ----
    df_hist_full = None
    try:
        df_hist_full = load_all_data(hist_data_dir)
    except Exception as e:
        logger.warning(f"   Historical data loading failed: {e}")

    # ---- ONE-TIME bootstrap pool preparation ----
    bootstrap_pools = {}
    if df_hist_full is not None and 'target_google_revenue_q50' in model.models:
        try:
            logger.info("\n?? Pre-computing bootstrap residual pools (once per run)...")
            bootstrap_pools = _prepare_bootstrap_pools(df_hist_full, model, model.feature_names)
        except Exception as e:
            logger.warning(f"   Bootstrap pool preparation failed, falling back to legacy quantile heads: {e}")

    logger.info("\n?? Generating predictions...")
    predictions: list[dict] = []

    for i, (_, row) in enumerate(features.iterrows()):
        x_row = X_scaled_df.iloc[[i]]

        req_id = str(row.get('request_id', f'req_{i}'))
        pd_val = int(float(row.get('period_days', 30)))
        sg = float(row.get('spend_google', row.get('feature_spend_google', 0)))
        sm = float(row.get('spend_meta', row.get('feature_spend_meta', 0)))
        sms = float(row.get('spend_ms', row.get('feature_spend_microsoft', 0)))

        pred = {
            'request_id': req_id, 'period_days': pd_val,
            'spend_google': sg, 'spend_meta': sm, 'spend_ms': sms,
        }

        # OOD detection
        ood_flags = {}
        try:
            budgets_for_ood = {'google': sg, 'meta': sm, 'ms': sms}
            ood_flags = compute_ood_flags(df_hist_full, budgets_for_ood, pd_val) if df_hist_full is not None else {}
        except Exception as e:
            logger.warning(f"   OOD detection failed: {e}")
            ood_flags = {p: {'is_ood': False, 'message': 'OOD check unavailable'} for p in ['google', 'meta', 'ms']}
        pred['ood_flags'] = ood_flags

        # Reconciled quantiles via pre-computed bootstrap pool
        period_days_req = pd_val
        reconciled_ok = False
        if 'target_google_revenue_q50' in model.models and period_days_req in bootstrap_pools:
            try:
                pool = bootstrap_pools.get(period_days_req)
                rec = _bootstrap_reconciled_from_pool(model, x_row, period_days_req, pool, n_boot=300)
                if rec is not None:
                    pred["revenue_p10"] = float(max(0.0, rec["revenue_p10"]))
                    pred["revenue_p50"] = float(max(0.0, rec["revenue_p50"]))
                    pred["revenue_p90"] = float(max(0.0, rec["revenue_p90"]))
                    for q_label in ["p10", "p50", "p90"]:
                        pred[f"google_revenue_{q_label}"] = float(max(0.0, rec[f"google_revenue_{q_label}"]))
                        pred[f"meta_revenue_{q_label}"] = float(max(0.0, rec[f"meta_revenue_{q_label}"]))
                        pred[f"ms_revenue_{q_label}"] = float(max(0.0, rec[f"ms_revenue_{q_label}"]))
                    reconciled_ok = True
            except Exception as e:
                logger.warning(f"   Bootstrap reconciliation failed for row {i}: {e}")

        if not reconciled_ok:
            for q_label, q_key in [
                ('p10', 'target_total_revenue_q10'), ('p50', 'target_total_revenue_q50'), ('p90', 'target_total_revenue_q90'),
            ]:
                val = model.models[q_key].predict(x_row)[0] if q_key in model.models else 0.0
                pred[f'revenue_{q_label}'] = float(max(0, val))
            for q_label, q_key in [
                ('p10', 'target_google_revenue_q10'), ('p50', 'target_google_revenue_q50'), ('p90', 'target_google_revenue_q90'),
            ]:
                val = model.models[q_key].predict(x_row)[0] if q_key in model.models else 0.0
                pred[f'google_revenue_{q_label}'] = float(max(0, val))
            for q_label, q_key in [
                ('p10', 'target_meta_revenue_q10'), ('p50', 'target_meta_revenue_q50'), ('p90', 'target_meta_revenue_q90'),
            ]:
                val = model.models[q_key].predict(x_row)[0] if q_key in model.models else 0.0
                pred[f'meta_revenue_{q_label}'] = float(max(0, val))
            for q_label, q_key in [
                ('p10', 'target_microsoft_revenue_q10'), ('p50', 'target_microsoft_revenue_q50'), ('p90', 'target_microsoft_revenue_q90'),
            ]:
                val = model.models[q_key].predict(x_row)[0] if q_key in model.models else 0.0
                pred[f'ms_revenue_{q_label}'] = float(max(0, val))

        total_spend = pred['spend_google'] + pred['spend_meta'] + pred['spend_ms']
        for q_label in ['p10', 'p50', 'p90']:
            pred[f'blended_roas_{q_label}'] = float(pred[f'revenue_{q_label}'] / total_spend) if total_spend > 0 else 0.0

        for key in list(pred.keys()):
            if key == 'request_id':
                pred[key] = str(pred[key])
            elif key == 'period_days':
                pred[key] = int(pred[key])
            elif key == 'ood_flags':
                continue
            else:
                pred[key] = float(pred[key])

        predictions.append(pred)

    output_df = pd.DataFrame(predictions)
    for col in output_df.columns:
        if col == 'request_id':
            output_df[col] = output_df[col].astype(str)
        elif col == 'period_days':
            output_df[col] = output_df[col].astype(int)
        elif col == 'ood_flags':
            continue
        else:
            output_df[col] = pd.to_numeric(output_df[col], errors='coerce').fillna(0.0)

    for col in OUTPUT_COLUMNS:
        if col not in output_df.columns:
            output_df[col] = 0.0

    output_df = output_df[OUTPUT_COLUMNS]
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    output_df.to_csv(output_path, index=False, float_format='%.2f')

    logger.info(f"\nPredictions saved to {output_path}")
    logger.info(f"   Shape: {output_df.shape}")
    for _, row in output_df.iterrows():
        logger.info(f"\n   Request: {row['request_id']} ({row['period_days']} days)")
        logger.info(f"   Budget: Google=${row['spend_google']:,.0f}, Meta=${row['spend_meta']:,.0f}, MS=${row['spend_ms']:,.0f}")
        logger.info(f"   Revenue: ${row['revenue_p10']:,.0f} - ${row['revenue_p50']:,.0f} - ${row['revenue_p90']:,.0f}")
        logger.info(f"   ROAS: {row['blended_roas_p10']:.2f}x - {row['blended_roas_p50']:.2f}x - {row['blended_roas_p90']:.2f}x")
    logger.info(f"\n{'='*60}\n")

    return output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default='features.parquet', help='Features file')
    parser.add_argument('--model', default='./pickle/model.pkl', help='Model path')
    parser.add_argument('--output', default='./output/predictions.csv', help='Output path')
    parser.add_argument('--data-dir', default=None, help='Data directory (overrides path derivation from features)')
    args = parser.parse_args()
    predict(args.features, args.model, args.output, data_dir=args.data_dir)
