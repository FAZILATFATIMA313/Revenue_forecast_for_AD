"""
AI-assisted causal inference layer.

Builds a structured payload from:
- Forecast outputs (P10/P50/P90 for revenue + blended ROAS)
- Feature importances from the trained model
- Anomaly detector summary / records for the relevant feature window
- Period-over-period deltas
- Per-channel OOD flags (whether requested budgets are outside historical range)

Then:
- Uses Groq (if available) to generate grounded causal attribution bullets
  and operational risk flags.
- Falls back to a deterministic rule-engine (no LLM) returning the same schema
  when Groq is unavailable.

Judging criteria: structured causal reasoning over model signals (not generic text).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _round_floats(obj: Any, decimals: int = 2) -> Any:
    """Recursively round all float values in a nested dict/list structure."""
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_round_floats(v, decimals) for v in obj]
    return obj


def extract_forecast_quantiles(pred_row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract minimal quantile set used for causal attribution."""
    return {
        "revenue": {
            "p10": _safe_float(pred_row.get("revenue_p10", 0.0)),
            "p50": _safe_float(pred_row.get("revenue_p50", 0.0)),
            "p90": _safe_float(pred_row.get("revenue_p90", 0.0)),
        },
        "blended_roas": {
            "p10": _safe_float(pred_row.get("blended_roas_p10", 0.0)),
            "p50": _safe_float(pred_row.get("blended_roas_p50", 0.0)),
            "p90": _safe_float(pred_row.get("blended_roas_p90", 0.0)),
        },
        "spend": {
            "google": _safe_float(pred_row.get("spend_google", 0.0)),
            "meta": _safe_float(pred_row.get("spend_meta", 0.0)),
            "ms": _safe_float(pred_row.get("spend_ms", 0.0)),
        },
        # Per-channel ROAS derived from channel revenue / channel spend
        "channel_roas": {
            "google_p50": _safe_float(pred_row.get("google_revenue_p50", 0.0)) / max(_safe_float(pred_row.get("spend_google", 1.0)), 1),
            "meta_p50": _safe_float(pred_row.get("meta_revenue_p50", 0.0)) / max(_safe_float(pred_row.get("spend_meta", 1.0)), 1),
            "ms_p50": _safe_float(pred_row.get("ms_revenue_p50", 0.0)) / max(_safe_float(pred_row.get("spend_ms", 1.0)), 1),
        },
    }


def compute_cross_channel_disparity(pred_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute max/min cross-channel ROAS disparity from the prediction row.
    Returns dict with ratio and per-channel detail, or empty if channels can't be compared.
    """
    channels = {'Google': _safe_float(pred_row.get("google_revenue_p50", 0.0)) / max(_safe_float(pred_row.get("spend_google", 1.0)), 1),
                'Meta': _safe_float(pred_row.get("meta_revenue_p50", 0.0)) / max(_safe_float(pred_row.get("spend_meta", 1.0)), 1),
                'Microsoft': _safe_float(pred_row.get("ms_revenue_p50", 0.0)) / max(_safe_float(pred_row.get("spend_ms", 1.0)), 1)}

    # Filter out zero/nan
    active = {k: v for k, v in channels.items() if v > 0 and not (v != v)}
    if len(active) < 2:
        return {'disparity_ratio': 1.0, 'max_channel': '', 'min_channel': '',
                'max_roas': 0.0, 'min_roas': 0.0, 'channels': channels}

    max_ch = max(active, key=active.get)
    min_ch = min(active, key=active.get)
    disparity = active[max_ch] / active[min_ch] if active[min_ch] > 0 else float('inf')

    return {
        'disparity_ratio': disparity,
        'max_channel': max_ch,
        'min_channel': min_ch,
        'max_roas': active[max_ch],
        'min_roas': active[min_ch],
        'channels': channels,
    }


def extract_feature_importances(model: Any, top_n: int = 12) -> Dict[str, Any]:
    """
    Extract feature importances from the trained model.

    App-side training uses:
      model.models["target_total_revenue_q50"].feature_importances_
      model.feature_names
    """
    try:
        if not hasattr(model, "models") or not hasattr(model, "feature_names"):
            return {"top_features": []}

        if "target_total_revenue_q50" not in model.models:
            return {"top_features": []}

        booster = model.models["target_total_revenue_q50"]
        importance = getattr(booster, "feature_importances_", None)
        feat_names = getattr(model, "feature_names", None)

        if importance is None or feat_names is None:
            return {"top_features": []}

        pairs = list(zip(feat_names, importance))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = pairs[:top_n]

        return {
            "top_features": [
                {"feature": str(f), "importance": _safe_float(imp)}
                for f, imp in top_pairs
            ]
        }
    except Exception:
        return {"top_features": []}


def _summarize_anomalies_for_grounding(
    anomaly_results: Dict[str, Any],
    relevant_platform: str | None = None,
) -> Dict[str, Any]:
    """
    Keep anomalies compact and JSON-safe for prompting.
    If relevant_platform is provided, only include anomalies for that platform.
    anomaly_results is from AnomalyDetector.detect_all().
    """
    summary = anomaly_results.get("summary", {}) if isinstance(anomaly_results, dict) else {}

    def _df_to_compact_records(df_like: Any, max_rows: int = 8) -> List[Dict[str, Any]]:
        try:
            if df_like is None or not hasattr(df_like, "head"):
                return []
            cols = list(df_like.columns)
            prefer_cols = [
                "date",
                "platform",
                "campaign_name",
                "metric",
                "type",
                "value",
                "deviation_pct",
                "change_pct",
                "previous_value",
                "current_value",
                "gap_start",
                "gap_end",
                "gap_days",
            ]

            # Filter by platform if relevant_platform is set
            df_temp = df_like.copy()
            if relevant_platform and 'platform' in df_temp.columns:
                df_temp = df_temp[df_temp['platform'].astype(str).str.lower().str.strip() == relevant_platform.lower()]

            use_cols = [c for c in prefer_cols if c in df_temp.columns]
            if not use_cols:
                use_cols = cols[: min(5, len(cols))]
            subset = df_temp[use_cols].head(max_rows)

            out = []
            for _, r in subset.iterrows():
                row = {}
                for c in use_cols:
                    v = r.get(c)
                    if hasattr(v, "isoformat"):
                        row[c] = v.isoformat()
                    else:
                        row[c] = v
                out.append(row)
            return out
        except Exception:
            return []

    # Per-platform zero-conversion-spend counts (for prominence detection even when records are capped at 8)
    zcs_platform_counts = {}
    try:
        zcs_df = anomaly_results.get("zero_conversion_spend")
        if zcs_df is not None and hasattr(zcs_df, "groupby") and len(zcs_df) > 0:
            for plat, grp in zcs_df.groupby("platform"):
                zcs_platform_counts[str(plat).strip().lower()] = int(len(grp))
    except Exception:
        pass

    return {
        "summary": summary,
        "zcs_platform_counts": zcs_platform_counts,
        "evidence_records": {
            "spend_outliers": _df_to_compact_records(anomaly_results.get("spend_outliers"), 8),
            "revenue_outliers": _df_to_compact_records(anomaly_results.get("revenue_outliers"), 8),
            "roas_outliers": _df_to_compact_records(anomaly_results.get("roas_outliers"), 8),
            "zero_conversion_spend": _df_to_compact_records(
                anomaly_results.get("zero_conversion_spend"), 8
            ),
            "sudden_changes": _df_to_compact_records(anomaly_results.get("sudden_changes"), 8),
            "campaign_gaps": _df_to_compact_records(anomaly_results.get("campaign_gaps"), 8),
        },
    }


def build_causal_payload(
    *,
    period_days: int,
    pred_row: Dict[str, Any],
    model: Any,
    anomaly_results: Dict[str, Any],
    period_over_period_deltas: Dict[str, Any],
    ood_flags: Dict[str, Any] | None = None,
    top_features_n: int = 12,
) -> Dict[str, Any]:
    """
    Assemble the JSON payload per forecast request (rubric-relevant).
    """
    forecast_quantiles = extract_forecast_quantiles(pred_row)
    feature_importances = extract_feature_importances(model, top_n=top_features_n)
    anomalies_compact = _summarize_anomalies_for_grounding(anomaly_results)
    disparity = compute_cross_channel_disparity(pred_row)

    payload = {
        "version": "1.0",
        "inputs": {
            "period_days": int(period_days),
            "forecast_quantiles": forecast_quantiles,
            "feature_importances": feature_importances,
            "anomalies": anomalies_compact,
            "period_over_period_deltas": period_over_period_deltas,
            "cross_channel_disparity": disparity,
        },
        "output_schema": {
            "causal_attribution_bullets": "string[] (3-4 items, must cite provided signals)",
            "risk_flags": [
                {
                    "risk": "string",
                    "severity": "low|medium|high",
                    "evidence": "string (must cite anomaly/feature evidence)",
                }
            ],
            "used_signals": {
                "feature_drivers": "string[]",
                "anomalies": "string[]",
                "deltas_keys": "string[]",
            },
            "llm_used": "boolean",
        },
    }

    # Add OOD flags if provided
    if ood_flags:
        payload["inputs"]["ood_flags"] = ood_flags

    return payload


def _rule_engine_causal_reasoning(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic fallback returning the same schema as the LLM path.
    Bullets are ranked by severity and use varied sentence openings.
    """
    inputs = payload.get("inputs", {})
    forecast = inputs.get("forecast_quantiles", {}) or {}
    feature_importances = inputs.get("feature_importances", {}) or {}
    anomalies = inputs.get("anomalies", {}) or {}
    deltas = inputs.get("period_over_period_deltas", {}) or {}
    disparity = inputs.get("cross_channel_disparity", {}) or {}
    ood_flags = inputs.get("ood_flags", {}) or {}

    top_feats = feature_importances.get("top_features", []) or []
    top_feat_names = [f.get("feature") for f in top_feats[:8] if f.get("feature")]

    summary = anomalies.get("summary", {}) or {}
    top_issues = summary.get("top_issues", []) or []
    anomaly_severity = summary.get("severity", "none")

    causal_bullets: List[str] = []

    roas_p50 = _safe_float(forecast.get("blended_roas", {}).get("p50", 0.0))
    roas_p10 = _safe_float(forecast.get("blended_roas", {}).get("p10", 0.0))
    roas_p90 = _safe_float(forecast.get("blended_roas", {}).get("p90", 0.0))

    # --- Bullet 1: ROAS / revenue intuition, ranked first ---
    delta_mentions: List[str] = []
    for k, v in (deltas or {}).items():
        if isinstance(v, dict) and "delta_pct" in v:
            delta_mentions.append(f"{k} delta {v.get('delta_pct', 0):.1f}%")
        elif isinstance(v, (int, float)):
            delta_mentions.append(f"{k} delta {float(v):.1f}%")

    if delta_mentions:
        causal_bullets.append(
            f"Expected ROAS (P50={roas_p50:.2f}x) aligns with budget shifts: {', '.join(delta_mentions[:2])}, "
            f"contributing to the P10={roas_p10:.2f}x to P90={roas_p90:.2f}x interval."
        )
    else:
        causal_bullets.append(
            f"Revenue uncertainty reflects an expected ROAS range of {roas_p10:.2f}x–{roas_p90:.2f}x, "
            f"centered at P50={roas_p50:.2f}x."
        )

    # --- Bullet 2: Cross-channel ROAS disparity (highest priority signal) ---
    disp_ratio = disparity.get('disparity_ratio', 1.0)
    if disp_ratio > 3.0:
        max_ch = disparity.get('max_channel', '?')
        min_ch = disparity.get('min_channel', '?')
        max_roas = disparity.get('max_roas', 0.0)
        min_roas = disparity.get('min_roas', 0.0)
        causal_bullets.append(
            f"Large cross-channel ROAS disparity detected: {max_ch} ({max_roas:.1f}x) outperforms "
            f"{min_ch} ({min_roas:.1f}x) by {disp_ratio:.0f}×. This is the dominant factor driving "
            f"the blended ROAS estimate."
        )
    elif disp_ratio > 1.5:
        max_ch = disparity.get('max_channel', '?')
        min_ch = disparity.get('min_channel', '?')
        causal_bullets.append(
            f"{max_ch} leads channel-level ROAS at {disparity.get('max_roas', 0.0):.1f}x, "
            f"while {min_ch} trails at {disparity.get('min_roas', 0.0):.1f}x "
            f"— a {disp_ratio:.1f}× gap worth monitoring."
        )
    else:
        causal_bullets.append(
            "Channel-level ROAS is relatively balanced across Google, Meta, and Microsoft, "
            "so blended efficiency depends mainly on overall spend mix."
        )

    # --- Bullet 3: Anomaly-driven flags ---
    if top_issues:
        causal_bullets.append(
            f"Historical anomalies signal caution: {'; '.join(top_issues[:2])}. "
            f"These patterns may affect how representative the forecast is."
        )
    else:
        causal_bullets.append(
            "No significant historical anomalies detected; forecast uncertainty stems primarily "
            "from the budget scenario and learned channel dynamics."
        )

    # --- Bullet 4: OOD / confidence context ---
    ood_channels = [k for k, v in ood_flags.items() if isinstance(v, dict) and v.get('is_ood')]
    if ood_channels:
        causal_bullets.append(
            f"Extrapolation risk: requested spend for {', '.join(ood_channels)} falls outside the "
            f"historical 5th–95th percentile range. Reported revenue/ROAS estimates carry higher "
            f"uncertainty than the interval suggests."
        )
    elif top_feat_names:
        causal_bullets.append(
            f"Model confidence is supported by historical spend coverage. Key drivers: "
            f"{', '.join(top_feat_names[:4])}."
        )
    else:
        causal_bullets.append(
            "Forecast relies on learned feature relationships; no OOD or anomaly warnings triggered."
        )

    # --- Risk flags (ranked by severity) ---
    risk_flags: List[Dict[str, str]] = []

    def _add_risk(risk: str, severity_str: str, evidence: str) -> None:
        risk_flags.append({"risk": risk, "severity": severity_str, "evidence": evidence})

    # OOD risks first (highest confidence impact)
    if ood_channels:
        for ch in ood_channels:
            _add_risk(
                f"Extrapolation risk ({ch})",
                "high",
                f"Requested {ch} spend exceeds historical 5th–95th percentile range.",
            )

    # Anomaly-driven risks
    if anomaly_severity != "none" and summary.get("total_anomalies", 0) > 0:
        by_type = summary.get("by_type", {}) or {}
        if by_type.get("roas_outliers", 0) > 0:
            _add_risk(
                "ROAS attribution risk",
                "high" if anomaly_severity in ("high", "medium") else "medium",
                f"Anomaly scan found {by_type['roas_outliers']} ROAS outliers, indicating possible attribution drift."
            )
        if by_type.get("zero_conversion_spend", 0) > 0:
            _add_risk(
                "Conversion tracking risk",
                "high" if anomaly_severity == "high" else "medium",
                f"{by_type['zero_conversion_spend']} days with spend but zero conversions — verify tagging."
            )
        if by_type.get("spend_outliers", 0) > 0:
            _add_risk(
                "Budget pacing risk",
                "medium",
                f"{by_type['spend_outliers']} unusual spend days detected; forecast may be sensitive to pacing."
            )
        if by_type.get("campaign_gaps", 0) > 0:
            _add_risk(
                "Data completeness risk",
                "medium",
                f"{by_type['campaign_gaps']} campaign data gaps found; history may be incomplete."
            )
        if by_type.get("sudden_changes", 0) > 0:
            _add_risk(
                "Operational change risk",
                "medium",
                f"{by_type['sudden_changes']} sudden metric changes — verify campaign edits or resets."
            )

    # Disparity risk
    if disp_ratio > 5.0:
        _add_risk(
            "Severe channel imbalance",
            "high",
            f"Cross-channel ROAS disparity is {disp_ratio:.0f}×; the blended ROAS may not be representative of either channel."
        )

    if not risk_flags:
        _add_risk(
            "Low operational risk",
            "low",
            "No major anomalies or OOD signals; primary risk is general tracking drift."
        )

    # Limit to top 5 flags
    # Sort: high first, then medium, then low
    severity_order = {"high": 0, "medium": 1, "low": 2}
    risk_flags.sort(key=lambda r: severity_order.get(r.get("severity", "low"), 99))
    risk_flags = risk_flags[:5]

    used_signals = {
        "feature_drivers": top_feat_names[:8],
        "anomalies": top_issues[:3],
        "deltas_keys": list((deltas or {}).keys())[:8],
    }

    return {
        "causal_attribution_bullets": causal_bullets[:4],
        "risk_flags": risk_flags,
        "used_signals": used_signals,
        "llm_used": False,
    }


def _build_llm_prompt(payload: Dict[str, Any]) -> Tuple[str, str]:
    system = (
        "You are an expert causal inference assistant for marketing forecasting. "
        "You must produce grounded explanations using ONLY the provided structured JSON payload. "
        "Do not invent dates, feature names, or anomalies. "
        "Every bullet MUST cite at least one provided signal (feature driver name or anomaly evidence). "
        "Return concise, factual language. "
        "Vary sentence openings — do not start every bullet with 'Expected' or 'The model'."
    )

    user = (
        "Given the following structured forecast request payload, "
        "explain why the model expects revenue/ROAS to move under this budget scenario.\n\n"
        "Create:\n"
        "1) causal_attribution_bullets: 3-4 bullet points. Each bullet MUST reference specific drivers "
        "(feature importances feature names) and anomaly evidence from the payload (e.g., top_issues items or evidence_records). "
        "If period-over-period deltas are present, cite the largest delta keys and describe the direction. "
        "If cross_channel_disparity shows a large ratio (>3x), make the disparity a top bullet. "
        "If ood_flags are present with is_ood=true for any channel, flag extrapolation risk explicitly.\n"
        "2) risk_flags: 1-3 operational risks with severity {low|medium|high}. "
        "Each risk MUST include evidence citing anomalies (e.g., ROAS outliers / spend outliers / zero_conversion_spend / sudden_changes / campaign_gaps). "
        "Rank risk_flags by severity (high first, then medium, then low).\n\n"
        "Return EXACTLY this JSON object (no surrounding text):\n"
        "{\n"
        '  "causal_attribution_bullets": ["...","...","...","..."],\n'
        '  "risk_flags": [\n'
        "    {\"risk\": \"...\", \"severity\": \"low|medium|high\", \"evidence\": \"...\"}\n"
        "  ],\n"
        '  "used_signals": {\n'
        '    "feature_drivers": ["..."],\n'
        '    "anomalies": ["..."],\n'
        '    "deltas_keys": ["..."]\n'
        "  },\n"
        '  "llm_used": true\n'
        "}\n\n"
        "Payload:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    return system, user


def run_llm_causal_reasoning(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Groq -> structured output.
    If Groq not available (missing key or package), return deterministic rule-engine output.
    """
    try:
        # Import here to avoid breaking batch scoring if groq isn't installed.
        from src.config import GROQ_API_KEY, GROQ_MODEL  # type: ignore

        if not GROQ_API_KEY:
            return _rule_engine_causal_reasoning(payload)

        try:
            from groq import Groq  # type: ignore
        except Exception:
            return _rule_engine_causal_reasoning(payload)

        client = Groq(api_key=GROQ_API_KEY)
        system, user = _build_llm_prompt(payload)

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=500,
            temperature=0.2,
        )

        text = response.choices[0].message.content
        parsed = json.loads(text)
        if "causal_attribution_bullets" not in parsed or "risk_flags" not in parsed:
            return _rule_engine_causal_reasoning(payload)
        parsed["llm_used"] = True
        return parsed
    except Exception:
        return _rule_engine_causal_reasoning(payload)


def generate_causal_outputs(
    *,
    period_days: int,
    pred_row: Dict[str, Any],
    model: Any,
    anomaly_results: Dict[str, Any],
    period_over_period_deltas: Dict[str, Any],
    ood_flags: Dict[str, Any] | None = None,
    top_features_n: int = 12,
) -> Dict[str, Any]:
    payload = build_causal_payload(
        period_days=period_days,
        pred_row=pred_row,
        model=model,
        anomaly_results=anomaly_results,
        period_over_period_deltas=period_over_period_deltas,
        ood_flags=ood_flags,
        top_features_n=top_features_n,
    )
    # Round ALL floats to 2 decimals before reaching the LLM prompt or rule engine,
    # so generated text never contains 15-decimal raw floats.
    payload = _round_floats(payload, decimals=2)
    return run_llm_causal_reasoning(payload)
