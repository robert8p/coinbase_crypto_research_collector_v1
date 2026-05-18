from __future__ import annotations

import json
import time
import traceback
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .live_shadow import LiveShadowService
from .rule_backtests import RuleBacktestService
from .schemas import LiveScanRequest
from .settings import Settings
from .storage import StorageManager


MAX_ADAPTIVE_REPLAY_VARIANTS = 24
MAX_ADAPTIVE_REPLAY_SECONDS = 25.0
MAX_ADAPTIVE_HISTORICAL_ROWS = 120_000
RELAXATION_MIN_PASS_RATIO = 0.5
MAX_ORTHOGONAL_DISCOVERY_SECONDS = 25.0
MAX_ORTHOGONAL_HISTORICAL_ROWS = 120_000
MAX_ORTHOGONAL_CANDIDATES = 48
ORTHOGONAL_MIN_BASELINE_LIFT = 0.002
ORTHOGONAL_MIN_TOUCH_LIFT = 0.03
ORTHOGONAL_MAX_PRODUCT_SHARE = 0.35


def _select_per_product_latest(feature_df: pd.DataFrame, freshness_hours: int = 2) -> tuple[pd.DataFrame, list[str], pd.Timestamp | None]:
    if feature_df.empty:
        return feature_df.copy(), [], None
    out = feature_df.copy()
    out["_ts_utc"] = pd.to_datetime(out["ts"], utc=True)
    global_latest = out["_ts_utc"].max()
    idx = out.groupby("product_id")["_ts_utc"].idxmax()
    snapshot = out.loc[idx].copy()
    stale_products: list[str] = []
    if global_latest is not None:
        freshness = pd.Timedelta(hours=max(int(freshness_hours), 0))
        fresh_mask = (global_latest - snapshot["_ts_utc"]) <= freshness
        stale_products = snapshot.loc[~fresh_mask, "product_id"].astype(str).tolist()
        snapshot = snapshot.loc[fresh_mask].copy()
    snapshot = snapshot.drop(columns=["_ts_utc"], errors="ignore").reset_index(drop=True)
    return snapshot, stale_products, global_latest


class LiveScannerService:
    def __init__(self, settings: Settings, storage: StorageManager, live_shadow: LiveShadowService, rule_service: RuleBacktestService):
        self.settings = settings
        self.storage = storage
        self.live_shadow = live_shadow
        self.rule_service = rule_service

    def latest_manifest(self) -> dict[str, Any]:
        return self.storage.read_latest_live_scan_manifest()

    def _select_live_rules(self, request: LiveScanRequest) -> list[dict[str, Any]]:
        library = self.rule_service._combined_library().get("candidate_rules", [])
        direct_rules = [
            rule for rule in library
            if rule.get("rule_kind", "direct_rule") == "direct_rule" and bool(rule.get("live_eligible", False))
        ]
        if request.selection_mode == "all":
            if not direct_rules:
                raise ValueError("No live-eligible direct rules are available. Mark one or more direct rules as live eligible on the Rules tab.")
            return direct_rules
        selected = set(request.rule_ids)
        if not selected:
            raise ValueError("Select at least one live-eligible direct rule or use selection_mode='all'.")
        rules = [rule for rule in direct_rules if rule.get("merged_rule_id") in selected]
        if not rules:
            raise ValueError("No selected live-eligible direct rules were found in the rule library.")
        return rules

    def _safe_float(self, value: Any) -> float | None:
        try:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None
            if pd.isna(value):
                return None
            return float(value)
        except Exception:
            return None

    def _historical_rule_stats(self) -> dict[str, dict[str, Any]]:
        manifest = self.storage.read_latest_live_shadow_manifest()
        rows = manifest.get("summary_rows", []) if isinstance(manifest, dict) else []
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            rule_id = row.get("merged_rule_id")
            if rule_id:
                out[str(rule_id)] = row
        return out

    def _build_scan_tables(self, signal_rows: pd.DataFrame, latest_ts: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
        if signal_rows.empty:
            empty_shortlist = pd.DataFrame(columns=[
                "scan_ts", "product_id", "base_asset", "quote_asset", "matched_rule_count", "matched_rule_ids",
                "matched_rule_names", "matched_rule_sources", "matched_primary_horizons", "best_priority",
                "ranking_score", "cb_close", "cb_ret_1", "cb_ret_6", "cb_ret_24", "ca_ret_1",
                "cs_coinbase_vs_coinapi_return_diff", "cs_coinbase_vs_coinapi_close_diff", "data_quality_note",
            ])
            return empty_shortlist, pd.DataFrame(columns=["scan_ts"])

        hist = self._historical_rule_stats()
        rule_hits = signal_rows.copy()
        rule_hits["scan_ts"] = pd.Timestamp(latest_ts).isoformat()
        rule_hits["historical_h1_mean_forward_return"] = rule_hits["merged_rule_id"].map(lambda rid: self._safe_float(hist.get(rid, {}).get("h1_mean_forward_return")))
        rule_hits["historical_h4_mean_forward_return"] = rule_hits["merged_rule_id"].map(lambda rid: self._safe_float(hist.get(rid, {}).get("h4_mean_forward_return")))
        rule_hits["historical_h24_mean_forward_return"] = rule_hits["merged_rule_id"].map(lambda rid: self._safe_float(hist.get(rid, {}).get("h24_mean_forward_return")))
        rule_hits["historical_signals"] = rule_hits["merged_rule_id"].map(lambda rid: hist.get(rid, {}).get("signals"))
        rule_hits["historical_largest_product_share"] = rule_hits["merged_rule_id"].map(lambda rid: self._safe_float(hist.get(rid, {}).get("largest_product_share")))
        rule_hits["data_quality_note"] = rule_hits.apply(
            lambda row: "missing_coinapi_context" if pd.isna(row.get("ca_ret_1")) and "CoinAPI" in str(row.get("source_attribution", "")) else "ok",
            axis=1,
        )
        def _historical_hint_for(row: pd.Series) -> float:
            primary = str(row.get("recommended_primary_horizon") or "h4").lower()
            candidates = {
                "h1": ["historical_h1_mean_forward_return", "historical_h4_mean_forward_return", "historical_h24_mean_forward_return"],
                "h4": ["historical_h4_mean_forward_return", "historical_h1_mean_forward_return", "historical_h24_mean_forward_return"],
                "h24": ["historical_h24_mean_forward_return", "historical_h4_mean_forward_return", "historical_h1_mean_forward_return"],
            }.get(primary, ["historical_h4_mean_forward_return", "historical_h1_mean_forward_return", "historical_h24_mean_forward_return"])
            for col in candidates:
                v = row.get(col)
                if v is not None and not pd.isna(v):
                    return float(v)
            return 0.0

        rule_hits["historical_hint"] = rule_hits.apply(_historical_hint_for, axis=1)
        priority_series = pd.to_numeric(rule_hits["priority"], errors="coerce") if "priority" in rule_hits.columns else pd.Series(999.0, index=rule_hits.index)
        rule_hits["priority_component"] = 100 - priority_series.fillna(999).clip(lower=0, upper=999)
        rule_hits["historical_component"] = pd.to_numeric(rule_hits["historical_hint"], errors="coerce").fillna(0).mul(1000)
        rule_hits["ranking_component"] = rule_hits["priority_component"] + rule_hits["historical_component"]

        grouped_rows: list[dict[str, Any]] = []
        for (product_id, scan_ts), group in rule_hits.groupby(["product_id", "scan_ts"], dropna=False):
            priorities = pd.to_numeric(group.get("priority"), errors="coerce")
            best_priority = int(priorities.min()) if priorities.notna().any() else None
            matched_ids = [str(x) for x in group["merged_rule_id"].dropna().astype(str).tolist()]
            matched_names = [str(x) for x in group["rule_name"].dropna().astype(str).tolist()]
            source_tags: list[str] = []
            for value in group.get("source_attribution", pd.Series(dtype=str)).fillna(""):
                for token in str(value).split("|"):
                    token = token.strip()
                    if token and token not in source_tags:
                        source_tags.append(token)
            horizons: list[str] = []
            for value in group.get("recommended_primary_horizon", pd.Series(dtype=str)).fillna(""):
                token = str(value).strip()
                if token and token not in horizons:
                    horizons.append(token)
            quality_notes = [str(x) for x in group.get("data_quality_note", pd.Series(dtype=str)).fillna("").tolist() if x and x != "ok"]
            quality_penalty = 25 if quality_notes else 0
            match_count_component = len(group) * 1000
            tiebreak_component = float(group["ranking_component"].sum())
            ranking_score = float(match_count_component + tiebreak_component - quality_penalty)
            first = group.iloc[0]
            grouped_rows.append({
                "scan_ts": scan_ts,
                "product_id": product_id,
                "base_asset": first.get("base_asset"),
                "quote_asset": first.get("quote_asset"),
                "matched_rule_count": int(len(group)),
                "matched_rule_ids": "|".join(matched_ids),
                "matched_rule_names": "|".join(matched_names),
                "matched_rule_sources": "|".join(source_tags),
                "matched_primary_horizons": "|".join(horizons),
                "best_priority": best_priority,
                "ranking_score": ranking_score,
                "cb_close": self._safe_float(first.get("cb_close")),
                "cb_ret_1": self._safe_float(first.get("cb_ret_1")),
                "cb_ret_6": self._safe_float(first.get("cb_ret_6")),
                "cb_ret_24": self._safe_float(first.get("cb_ret_24")),
                "ca_ret_1": self._safe_float(first.get("ca_ret_1")),
                "cs_coinbase_vs_coinapi_return_diff": self._safe_float(first.get("cs_coinbase_vs_coinapi_return_diff")),
                "cs_coinbase_vs_coinapi_close_diff": self._safe_float(first.get("cs_coinbase_vs_coinapi_close_diff")),
                "historical_hint_mean": self._safe_float(group["historical_hint"].mean()),
                "data_quality_note": "|".join(dict.fromkeys(quality_notes)) if quality_notes else "ok",
            })
        shortlist = pd.DataFrame(grouped_rows)
        if not shortlist.empty:
            shortlist = shortlist.sort_values(["ranking_score", "matched_rule_count", "best_priority", "product_id"], ascending=[False, False, True, True]).reset_index(drop=True)
            shortlist.insert(0, "scanner_rank", np.arange(1, len(shortlist) + 1))
        return shortlist, rule_hits.sort_values(["product_id", "priority", "merged_rule_id"]).reset_index(drop=True)


    def _condition_diagnostics(self, prepared: pd.DataFrame, conditions: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
        diag = pd.DataFrame(index=prepared.index)
        resolved_conditions: list[dict[str, Any]] = []
        missing: list[str] = []
        if not conditions:
            diag["__passes__"] = True
            return diag, resolved_conditions, missing
        pass_cols: list[str] = []
        distance_cols: list[str] = []
        failure_cols: list[str] = []
        for idx, condition in enumerate(conditions, start=1):
            cond_mask, metadata = self.rule_service._condition_mask(prepared, condition)
            resolved_conditions.append(metadata)
            label = str(metadata.get("resolved_field") or metadata.get("field") or f"condition_{idx}")
            pass_col = f"pass_{idx}"
            dist_col = f"distance_{idx}"
            fail_col = f"failure_{idx}"
            if cond_mask is None:
                missing.append(str(condition.get("field")))
                diag[pass_col] = False
                diag[dist_col] = np.inf
                diag[fail_col] = f"missing:{label}"
            else:
                passed = cond_mask.fillna(False).astype(bool)
                diag[pass_col] = passed
                distance = self._condition_distance(prepared, metadata)
                diag[dist_col] = distance.where(~passed, 0.0)
                diag[fail_col] = np.where(passed, "", self._failure_label(prepared, metadata))
            pass_cols.append(pass_col)
            distance_cols.append(dist_col)
            failure_cols.append(fail_col)
        diag["condition_count"] = len(pass_cols)
        diag["passed_conditions"] = diag[pass_cols].sum(axis=1) if pass_cols else 0
        diag["condition_pass_ratio"] = diag["passed_conditions"] / max(len(pass_cols), 1)
        diag["distance_to_trigger"] = diag[distance_cols].replace([np.inf, -np.inf], np.nan).sum(axis=1, min_count=1).fillna(999999.0) if distance_cols else 0.0
        diag["failed_conditions"] = diag[failure_cols].apply(lambda row: " | ".join([str(x) for x in row.tolist() if str(x)]), axis=1) if failure_cols else ""
        diag["__passes__"] = diag["passed_conditions"].eq(len(pass_cols)) if pass_cols else True
        return diag, resolved_conditions, missing

    def _condition_distance(self, df: pd.DataFrame, metadata: dict[str, Any]) -> pd.Series:
        field = metadata.get("resolved_field")
        if field not in df.columns:
            return pd.Series(np.inf, index=df.index)
        series = pd.to_numeric(df[field], errors="coerce")
        logic = metadata.get("logic")
        if logic == "between":
            lower = metadata.get("lower_bound")
            upper = metadata.get("upper_bound")
            lower_gap = pd.to_numeric(pd.Series(lower, index=df.index), errors="coerce") - series
            upper_gap = series - pd.to_numeric(pd.Series(upper, index=df.index), errors="coerce")
            raw_gap = pd.concat([lower_gap, upper_gap], axis=1).max(axis=1).clip(lower=0)
            scale = max(abs(float(lower or 0)), abs(float(upper or 0)), 1.0)
            return (raw_gap / scale).fillna(999999.0)
        threshold = metadata.get("threshold", metadata.get("value"))
        try:
            threshold_f = float(threshold)
        except Exception:
            return pd.Series(999999.0, index=df.index)
        if logic in {">", ">="}:
            raw_gap = (threshold_f - series).clip(lower=0)
        elif logic in {"<", "<="}:
            raw_gap = (series - threshold_f).clip(lower=0)
        elif logic == "==":
            raw_gap = (series - threshold_f).abs()
        elif logic == "!=":
            raw_gap = pd.Series(0.0, index=df.index)
        else:
            raw_gap = pd.Series(999999.0, index=df.index)
        scale = max(abs(threshold_f), 1.0)
        return (raw_gap / scale).fillna(999999.0)

    def _failure_label(self, df: pd.DataFrame, metadata: dict[str, Any]) -> pd.Series:
        field = metadata.get("resolved_field") or metadata.get("field")
        if field not in df.columns:
            return pd.Series(f"missing:{field}", index=df.index)
        actual = df[field]
        logic = metadata.get("logic")
        if logic == "between":
            target = f"between {metadata.get('lower_bound')} and {metadata.get('upper_bound')}"
        else:
            target_value = metadata.get("threshold", metadata.get("value"))
            target = f"{logic} {target_value}"
        return actual.map(lambda v: f"{field} needed {target}, current {v}")

    def _build_near_match_tables(self, snapshot: pd.DataFrame, rules: list[dict[str, Any]], latest_ts: pd.Timestamp, rule_hits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        prepared = self.rule_service._prepare_frame(snapshot)
        rows: list[dict[str, Any]] = []
        coverage: list[dict[str, Any]] = []
        signal_log = self.live_shadow._existing_signal_log()
        signal_log = signal_log.copy() if isinstance(signal_log, pd.DataFrame) else pd.DataFrame()
        if not signal_log.empty and "signal_ts" in signal_log.columns:
            signal_log["signal_ts"] = pd.to_datetime(signal_log["signal_ts"], utc=True, errors="coerce")
        current_hits_by_rule = rule_hits["merged_rule_id"].astype(str).value_counts().to_dict() if not rule_hits.empty and "merged_rule_id" in rule_hits.columns else {}
        for rule in rules:
            rule_near_count = 0
            for instance in self.rule_service._resolve_rule_variants(rule):
                conditions = instance.get("conditions", [])
                diag, resolved_conditions, missing = self._condition_diagnostics(prepared, conditions)
                if missing:
                    coverage.append({
                        "merged_rule_id": instance["merged_rule_id"],
                        "rule_instance_id": instance["instance_id"],
                        "rule_name": instance.get("name"),
                        "current_full_matches": 0,
                        "near_match_rows": 0,
                        "best_condition_pass_ratio": None,
                        "best_distance_to_trigger": None,
                        "missing_features": "|".join(sorted(set(missing))),
                    })
                    continue
                candidate = prepared.join(diag)
                full_matches = int(candidate["__passes__"].sum()) if "__passes__" in candidate.columns else 0
                near = candidate.loc[~candidate["__passes__"].fillna(False)].copy()
                near_count_for_instance = 0
                if not near.empty:
                    near["priority"] = instance.get("priority") if instance.get("priority") is not None else 999
                    near["candidate_score"] = (near["condition_pass_ratio"].astype(float) * 1000.0) - (near["distance_to_trigger"].astype(float).clip(upper=999999) * 25.0) + (100.0 - pd.to_numeric(near["priority"], errors="coerce").fillna(999).clip(lower=0, upper=999))
                    near = near.sort_values(["condition_pass_ratio", "distance_to_trigger", "candidate_score", "product_id"], ascending=[False, True, False, True]).head(10)
                    near_count_for_instance = int(len(near))
                    rule_near_count += near_count_for_instance
                    for _, row in near.iterrows():
                        rows.append({
                            "scan_ts": pd.Timestamp(latest_ts).isoformat(),
                            "product_id": row.get("product_id"),
                            "base_asset": row.get("base_asset"),
                            "quote_asset": row.get("quote_asset"),
                            "merged_rule_id": instance["merged_rule_id"],
                            "rule_instance_id": instance["instance_id"],
                            "rule_name": instance.get("name"),
                            "priority": instance.get("priority"),
                            "condition_count": int(row.get("condition_count", 0) or 0),
                            "passed_conditions": int(row.get("passed_conditions", 0) or 0),
                            "condition_pass_ratio": float(row.get("condition_pass_ratio", 0.0) or 0.0),
                            "distance_to_trigger": float(row.get("distance_to_trigger", 999999.0) or 999999.0),
                            "candidate_score": float(row.get("candidate_score", 0.0) or 0.0),
                            "failed_conditions": row.get("failed_conditions", ""),
                            "cb_close": self._safe_float(row.get("cb_close")),
                            "cb_ret_1": self._safe_float(row.get("cb_ret_1")),
                            "cb_ret_6": self._safe_float(row.get("cb_ret_6")),
                            "cb_ret_24": self._safe_float(row.get("cb_ret_24")),
                            "ca_ret_1": self._safe_float(row.get("ca_ret_1")),
                            "cs_coinbase_vs_coinapi_return_diff": self._safe_float(row.get("cs_coinbase_vs_coinapi_return_diff")),
                        })
                coverage.append({
                    "merged_rule_id": instance["merged_rule_id"],
                    "rule_instance_id": instance["instance_id"],
                    "rule_name": instance.get("name"),
                    "current_full_matches": int(full_matches),
                    "near_match_rows": near_count_for_instance,
                    "best_condition_pass_ratio": float(candidate["condition_pass_ratio"].max()) if not candidate.empty and "condition_pass_ratio" in candidate.columns else None,
                    "best_distance_to_trigger": float(candidate["distance_to_trigger"].min()) if not candidate.empty and "distance_to_trigger" in candidate.columns else None,
                    "missing_features": "",
                })
        near_df = pd.DataFrame(rows)
        if not near_df.empty:
            near_df = near_df.sort_values(["candidate_score", "condition_pass_ratio", "distance_to_trigger"], ascending=[False, False, True]).reset_index(drop=True)
            near_df.insert(0, "near_match_rank", np.arange(1, len(near_df) + 1))
        else:
            near_df = pd.DataFrame(columns=["near_match_rank", "scan_ts", "product_id", "merged_rule_id", "rule_instance_id", "condition_pass_ratio", "distance_to_trigger", "candidate_score", "failed_conditions"])
        coverage_df = pd.DataFrame(coverage)
        if not coverage_df.empty:
            if not signal_log.empty and "merged_rule_id" in signal_log.columns:
                since = pd.Timestamp(latest_ts) - pd.Timedelta(days=7)
                recent = signal_log.loc[pd.to_datetime(signal_log.get("signal_ts"), utc=True, errors="coerce") >= since].copy()
                seven = recent["merged_rule_id"].astype(str).value_counts().to_dict() if not recent.empty else {}
                last = signal_log.dropna(subset=["signal_ts"]).sort_values("signal_ts").groupby("merged_rule_id")["signal_ts"].last().astype(str).to_dict()
            else:
                seven, last = {}, {}
            coverage_df["current_rule_hits"] = coverage_df["merged_rule_id"].astype(str).map(lambda x: int(current_hits_by_rule.get(x, 0)))
            coverage_df["live_hit_count_7d"] = coverage_df["merged_rule_id"].astype(str).map(lambda x: int(seven.get(x, 0)))
            coverage_df["estimated_hourly_hit_rate_7d"] = coverage_df["live_hit_count_7d"] / 168.0
            coverage_df["last_live_signal_ts"] = coverage_df["merged_rule_id"].astype(str).map(lambda x: last.get(x))
            coverage_df = coverage_df.sort_values(["current_full_matches", "best_condition_pass_ratio", "best_distance_to_trigger", "merged_rule_id"], ascending=[False, False, True, True]).reset_index(drop=True)
        return near_df, coverage_df

    def _empty_adaptive_replay_tables(self, status: str = "no_adaptive_replay") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        replay_cols = [
            "status", "scan_ts", "merged_rule_id", "rule_instance_id", "variant_id", "variant_kind",
            "target_horizon", "support_rows", "support_delta_vs_original", "coverage_multiplier_vs_original",
            "distinct_products", "largest_product_share", "mean_forward_return", "median_forward_return",
            "mean_max_up_pct", "touch_rate", "baseline_mean_forward_return", "original_mean_forward_return",
            "mean_forward_return_delta_vs_original", "live_current_promoted_count", "recommendation", "reason",
        ]
        relaxation_cols = [
            "status", "scan_ts", "merged_rule_id", "rule_instance_id", "condition_index", "condition_field",
            "condition_logic", "relaxation_kind", "original_threshold", "relaxed_threshold", "original_condition",
            "relaxed_condition", "current_failed_condition_count", "current_failure_share", "historical_condition_pass_rate",
            "live_current_promoted_count", "variant_id", "recommendation", "reason",
        ]
        frontier_cols = [
            "status", "scan_ts", "merged_rule_id", "rule_instance_id", "variant_id", "variant_kind",
            "target_horizon", "support_rows", "coverage_multiplier_vs_original", "mean_forward_return",
            "mean_forward_return_delta_vs_original", "touch_rate", "touch_rate_delta_vs_original",
            "live_current_promoted_count", "live_usability_score", "recommendation", "reason",
        ]
        return (
            pd.DataFrame(columns=replay_cols).assign(status=status).iloc[0:0],
            pd.DataFrame(columns=relaxation_cols).assign(status=status).iloc[0:0],
            pd.DataFrame(columns=frontier_cols).assign(status=status).iloc[0:0],
        )

    def _adaptive_status_row(self, status: str, latest_ts: pd.Timestamp, reason: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        replay, relaxation, frontier = self._empty_adaptive_replay_tables(status)
        scan_ts = pd.Timestamp(latest_ts).isoformat()
        common = {
            "status": status,
            "scan_ts": scan_ts,
            "merged_rule_id": None,
            "rule_instance_id": None,
            "variant_id": None,
            "recommendation": "not_evaluated",
            "reason": reason,
        }
        replay = pd.concat([replay, pd.DataFrame([{**common, "variant_kind": None, "target_horizon": None}])], ignore_index=True)
        relaxation = pd.concat([relaxation, pd.DataFrame([{
            **common,
            "condition_index": None,
            "condition_field": None,
            "condition_logic": None,
            "relaxation_kind": None,
            "original_threshold": None,
            "relaxed_threshold": None,
            "original_condition": None,
            "relaxed_condition": None,
            "current_failed_condition_count": None,
            "current_failure_share": None,
            "historical_condition_pass_rate": None,
            "live_current_promoted_count": 0,
        }])], ignore_index=True)
        frontier = pd.concat([frontier, pd.DataFrame([{**common, "variant_kind": None, "target_horizon": None}])], ignore_index=True)
        return replay, relaxation, frontier

    def _append_adaptive_status_note(
        self,
        replay_df: pd.DataFrame,
        relaxation_df: pd.DataFrame,
        frontier_df: pd.DataFrame,
        status: str,
        latest_ts: pd.Timestamp,
        reason: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        note_replay, note_relaxation, note_frontier = self._adaptive_status_row(status, latest_ts, reason)
        return (
            pd.concat([replay_df, note_replay], ignore_index=True),
            pd.concat([relaxation_df, note_relaxation], ignore_index=True),
            pd.concat([frontier_df, note_frontier], ignore_index=True),
        )

    def _primary_horizon_for_instance(self, instance: dict[str, Any]) -> str:
        for candidate in [
            instance.get("recommended_primary_horizon"),
            *((instance.get("target_horizons") or [])),
            instance.get("secondary_monitor_horizon"),
            "h4",
        ]:
            normalized = self.rule_service._normalize_horizon_token(candidate)
            if normalized in {"h1", "h4", "h24"}:
                return normalized
        return "h4"

    def _safe_ratio(self, numerator: float | int | None, denominator: float | int | None) -> float | None:
        try:
            if denominator in (None, 0):
                return None
            if pd.isna(denominator):
                return None
            if numerator is None or pd.isna(numerator):
                return None
            return float(numerator) / float(denominator)
        except Exception:
            return None

    def _product_concentration_metrics(self, subset: pd.DataFrame) -> tuple[int, float | None]:
        if subset.empty or "product_id" not in subset.columns:
            return 0, None
        counts = subset["product_id"].value_counts(dropna=False)
        largest = float(counts.iloc[0] / len(subset)) if len(counts) else None
        return int(counts.size), largest

    def _historical_metrics_for_mask(self, df: pd.DataFrame, mask: pd.Series | None, horizon: str) -> dict[str, Any]:
        return_col, max_up_col, touch_col = self.rule_service._target_columns(horizon)
        if mask is None or df.empty:
            subset = df.iloc[0:0].copy()
        else:
            aligned = mask.reindex(df.index).fillna(False).astype(bool)
            subset = df.loc[aligned].copy()
        if return_col in subset.columns:
            subset = subset.loc[pd.to_numeric(subset[return_col], errors="coerce").notna()].copy()
        distinct, largest = self._product_concentration_metrics(subset)
        support = int(len(subset))
        touch_rate = None
        if touch_col and touch_col in subset.columns and support:
            touch_rate = float(pd.to_numeric(subset[touch_col], errors="coerce").mean())
        return {
            "support_rows": support,
            "distinct_products": distinct,
            "largest_product_share": largest,
            "mean_forward_return": float(pd.to_numeric(subset[return_col], errors="coerce").mean()) if support and return_col in subset.columns else None,
            "median_forward_return": float(pd.to_numeric(subset[return_col], errors="coerce").median()) if support and return_col in subset.columns else None,
            "mean_max_up_pct": float(pd.to_numeric(subset[max_up_col], errors="coerce").mean()) if support and max_up_col in subset.columns else None,
            "touch_rate": touch_rate,
        }

    def _global_baseline_metrics(self, df: pd.DataFrame, horizon: str) -> dict[str, Any]:
        if df.empty:
            return self._historical_metrics_for_mask(df, None, horizon)
        return self._historical_metrics_for_mask(df, pd.Series(True, index=df.index), horizon)

    def _threshold_from_metadata(self, metadata: dict[str, Any]) -> float | None:
        for key in ("threshold", "value"):
            value = metadata.get(key)
            try:
                if value is not None and not pd.isna(value):
                    return float(value)
            except Exception:
                continue
        return None

    def _jsonable_condition(self, condition: dict[str, Any]) -> str:
        try:
            return json.dumps(condition, sort_keys=True, default=str)
        except Exception:
            return str(condition)

    def _relaxation_plans_for_condition(
        self,
        condition: dict[str, Any],
        metadata: dict[str, Any],
        failed_values: pd.Series,
    ) -> list[dict[str, Any]]:
        logic = str(metadata.get("logic") or condition.get("logic") or "").strip()
        plans: list[dict[str, Any]] = []
        base = dict(condition)
        numeric_failed = pd.to_numeric(failed_values, errors="coerce").dropna()

        def add_plan(kind: str, updated_condition: dict[str, Any], original_threshold: Any, relaxed_threshold: Any, reason: str) -> None:
            plans.append({
                "relaxation_kind": kind,
                "condition": updated_condition,
                "original_threshold": original_threshold,
                "relaxed_threshold": relaxed_threshold,
                "reason": reason,
            })

        if logic in {"in_bottom_quantile", "in_top_quantile"}:
            try:
                q = float(condition.get("quantile", metadata.get("quantile", 0.2)))
            except Exception:
                return plans
            modest_q = min(0.50, max(q, q * 1.25 + 0.025))
            bridge_q = min(0.75, max(modest_q, q * 1.50 + 0.05))
            if modest_q > q:
                c = dict(base)
                c["logic"] = logic
                c["quantile"] = round(float(modest_q), 6)
                add_plan("modest_quantile_relax", c, q, modest_q, "Widen empirical quantile gate modestly.")
            if bridge_q > modest_q:
                c = dict(base)
                c["logic"] = logic
                c["quantile"] = round(float(bridge_q), 6)
                add_plan("bridge_quantile_relax", c, q, bridge_q, "Widen empirical quantile gate enough to test a larger shadow candidate set.")
            return plans

        if logic == "between":
            try:
                lower = float(metadata.get("lower_bound"))
                upper = float(metadata.get("upper_bound"))
            except Exception:
                return plans
            width = max(abs(upper - lower), abs(upper), abs(lower), 1.0)
            delta = max(width * 0.15, 0.002)
            c = dict(base)
            c["logic"] = "between"
            c["lower_bound"] = lower - delta
            c["upper_bound"] = upper + delta
            add_plan("modest_band_widen", c, f"{lower}..{upper}", f"{c['lower_bound']}..{c['upper_bound']}", "Expand both sides of the accepted band modestly.")
            if not numeric_failed.empty:
                bridge_lower = min(lower, float(numeric_failed.quantile(0.10)))
                bridge_upper = max(upper, float(numeric_failed.quantile(0.90)))
                if bridge_lower < lower or bridge_upper > upper:
                    c2 = dict(base)
                    c2["logic"] = "between"
                    c2["lower_bound"] = bridge_lower
                    c2["upper_bound"] = bridge_upper
                    add_plan("bridge_band_widen", c2, f"{lower}..{upper}", f"{bridge_lower}..{bridge_upper}", "Expand the band to include the central range of current near misses.")
            return plans

        threshold = self._threshold_from_metadata(metadata)
        if threshold is None or logic not in {">", ">=", "<", "<="}:
            return plans

        delta = max(abs(threshold) * 0.15, 0.002)
        if logic in {"<", "<="}:
            modest = threshold + delta
            c = dict(base)
            c["logic"] = "<=" if logic == "<" else logic
            c["value"] = float(modest)
            c.pop("threshold_type", None)
            c.pop("quantile", None)
            add_plan("modest_threshold_relax", c, threshold, modest, "Move upper-bound threshold modestly toward current near misses.")
            if not numeric_failed.empty:
                bridge = float(numeric_failed.quantile(0.25))
                if bridge > modest:
                    c2 = dict(base)
                    c2["logic"] = "<=" if logic == "<" else logic
                    c2["value"] = bridge
                    c2.pop("threshold_type", None)
                    c2.pop("quantile", None)
                    add_plan("bridge_nearest_threshold", c2, threshold, bridge, "Move upper-bound threshold enough to admit the nearest quartile of current failed near misses.")
        elif logic in {">", ">="}:
            modest = threshold - delta
            c = dict(base)
            c["logic"] = ">=" if logic == ">" else logic
            c["value"] = float(modest)
            c.pop("threshold_type", None)
            c.pop("quantile", None)
            add_plan("modest_threshold_relax", c, threshold, modest, "Move lower-bound threshold modestly toward current near misses.")
            if not numeric_failed.empty:
                bridge = float(numeric_failed.quantile(0.75))
                if bridge < modest:
                    c2 = dict(base)
                    c2["logic"] = ">=" if logic == ">" else logic
                    c2["value"] = bridge
                    c2.pop("threshold_type", None)
                    c2.pop("quantile", None)
                    add_plan("bridge_nearest_threshold", c2, threshold, bridge, "Move lower-bound threshold enough to admit the nearest quartile of current failed near misses.")
        return plans

    def _recommend_relaxation(
        self,
        original_metrics: dict[str, Any],
        relaxed_metrics: dict[str, Any],
        baseline_metrics: dict[str, Any],
        live_current_promoted_count: int,
    ) -> tuple[str, str, float]:
        support = int(relaxed_metrics.get("support_rows") or 0)
        original_support = int(original_metrics.get("support_rows") or 0)
        relaxed_mean = relaxed_metrics.get("mean_forward_return")
        original_mean = original_metrics.get("mean_forward_return")
        baseline_mean = baseline_metrics.get("mean_forward_return")
        relaxed_touch = relaxed_metrics.get("touch_rate")
        original_touch = original_metrics.get("touch_rate")
        coverage_multiplier = self._safe_ratio(support, max(original_support, 1)) or 0.0
        quality_delta = None if relaxed_mean is None or original_mean is None else float(relaxed_mean) - float(original_mean)
        baseline_lift = None if relaxed_mean is None or baseline_mean is None else float(relaxed_mean) - float(baseline_mean)
        touch_delta = None if relaxed_touch is None or original_touch is None else float(relaxed_touch) - float(original_touch)
        live_score = (coverage_multiplier * 10.0) + (live_current_promoted_count * 2.0)
        if relaxed_mean is not None:
            live_score += float(relaxed_mean) * 1000.0
        if touch_delta is not None:
            live_score += float(touch_delta) * 20.0
        if support < 20:
            return "reject_fragile", "Historical support below 20 rows after relaxation.", float(live_score)
        if original_mean is not None and quality_delta is not None and quality_delta < -0.005:
            return "reject_quality_drop", "Relaxation materially reduces mean forward return versus the original rule.", float(live_score)
        if baseline_lift is not None and baseline_lift < 0:
            return "watchlist_below_baseline", "Relaxed variant is below the global historical baseline for the target horizon.", float(live_score)
        if support > original_support and live_current_promoted_count > 0:
            return "promote_to_shadow_candidate", "Relaxation increases historical coverage and would promote at least one current near match; validate in live shadow before considering live eligibility.", float(live_score)
        if support > original_support:
            return "watchlist_historical_only", "Relaxation increases historical coverage but does not currently promote live near matches.", float(live_score)
        return "no_coverage_gain", "Relaxation does not increase historical support.", float(live_score)

    def _build_adaptive_replay_tables(
        self,
        snapshot: pd.DataFrame,
        rules: list[dict[str, Any]],
        latest_ts: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        started = time.monotonic()
        deadline = started + MAX_ADAPTIVE_REPLAY_SECONDS
        historical = self.storage.read_frame("feature_table")
        if historical.empty:
            return self._adaptive_status_row("no_historical_feature_table", latest_ts, "No historical feature table exists yet; adaptive replay was skipped.")
        historical_rows_available = int(len(historical))
        historical_rows_used = historical_rows_available
        historical_sample_note = "full_historical_feature_table"
        if historical_rows_available > MAX_ADAPTIVE_HISTORICAL_ROWS:
            historical_sample_note = f"bounded_recent_sample_{MAX_ADAPTIVE_HISTORICAL_ROWS}_of_{historical_rows_available}"
            if "ts" in historical.columns:
                historical = historical.copy()
                historical["_ts_for_adaptive_sample"] = pd.to_datetime(historical["ts"], utc=True, errors="coerce")
                historical = historical.sort_values("_ts_for_adaptive_sample").tail(MAX_ADAPTIVE_HISTORICAL_ROWS).drop(columns=["_ts_for_adaptive_sample"], errors="ignore")
            else:
                historical = historical.tail(MAX_ADAPTIVE_HISTORICAL_ROWS).copy()
            historical_rows_used = int(len(historical))
        prepared_current = self.rule_service._prepare_frame(snapshot)
        prepared_hist = self.rule_service._prepare_frame(historical)
        if prepared_hist.empty:
            return self._adaptive_status_row("no_historical_feature_table", latest_ts, "Historical feature table exists but could not be prepared for adaptive replay.")

        relaxation_rows: list[dict[str, Any]] = []
        replay_rows: list[dict[str, Any]] = []
        frontier_rows: list[dict[str, Any]] = []
        variants_seen = 0
        time_budget_exhausted = False

        def budget_exhausted() -> bool:
            return time.monotonic() >= deadline

        for rule in rules:
            if budget_exhausted():
                time_budget_exhausted = True
                break
            for instance in self.rule_service._resolve_rule_variants(rule):
                if budget_exhausted():
                    time_budget_exhausted = True
                    break
                conditions = instance.get("conditions", [])
                if not conditions:
                    continue
                horizon = self._primary_horizon_for_instance(instance)
                original_mask, hist_resolved, missing_hist = self.rule_service._build_rule_mask(prepared_hist, conditions)
                if missing_hist:
                    relaxation_rows.append({
                        "status": "missing_historical_features",
                        "scan_ts": pd.Timestamp(latest_ts).isoformat(),
                        "merged_rule_id": instance["merged_rule_id"],
                        "rule_instance_id": instance["instance_id"],
                        "condition_index": None,
                        "condition_field": "|".join(sorted(set(missing_hist))),
                        "condition_logic": None,
                        "relaxation_kind": None,
                        "original_threshold": None,
                        "relaxed_threshold": None,
                        "original_condition": None,
                        "relaxed_condition": None,
                        "current_failed_condition_count": None,
                        "current_failure_share": None,
                        "historical_condition_pass_rate": None,
                        "live_current_promoted_count": 0,
                        "variant_id": None,
                        "recommendation": "cannot_replay",
                        "reason": "One or more rule features are missing from the historical feature table.",
                    })
                    continue
                original_metrics = self._historical_metrics_for_mask(prepared_hist, original_mask, horizon)
                baseline_metrics = self._global_baseline_metrics(prepared_hist, horizon)
                diag_current, current_resolved, missing_current = self._condition_diagnostics(prepared_current, conditions)
                if missing_current:
                    continue
                current_candidate = prepared_current.join(diag_current)
                current_full_mask = current_candidate["__passes__"].fillna(False).astype(bool)
                condition_count = max(int(current_candidate["condition_count"].max()) if "condition_count" in current_candidate.columns and not current_candidate.empty else len(conditions), 1)
                min_pass_ratio = 0.0 if condition_count <= 1 else max(RELAXATION_MIN_PASS_RATIO, (condition_count - 1) / condition_count - 1e-9)

                for idx, condition in enumerate(conditions, start=1):
                    if budget_exhausted():
                        time_budget_exhausted = True
                        break
                    if variants_seen >= MAX_ADAPTIVE_REPLAY_VARIANTS:
                        break
                    pass_col = f"pass_{idx}"
                    metadata = current_resolved[idx - 1] if idx - 1 < len(current_resolved) else {}
                    field = metadata.get("resolved_field") or metadata.get("field") or condition.get("field")
                    if pass_col not in current_candidate.columns:
                        continue
                    failed_current = current_candidate.loc[
                        (~current_candidate[pass_col].fillna(False).astype(bool))
                        & (~current_full_mask)
                        & (pd.to_numeric(current_candidate.get("condition_pass_ratio"), errors="coerce") >= min_pass_ratio)
                    ].copy()
                    if failed_current.empty:
                        continue
                    failed_values = failed_current[field] if field in failed_current.columns else pd.Series(dtype=float)
                    current_failed_count = int(len(failed_current))
                    current_failure_share = self._safe_ratio(current_failed_count, max(int((~current_full_mask).sum()), 1))
                    hist_pass_rate = None
                    if idx - 1 < len(hist_resolved):
                        hist_mask_single, _ = self.rule_service._condition_mask(prepared_hist, conditions[idx - 1])
                        if hist_mask_single is not None:
                            hist_pass_rate = float(hist_mask_single.fillna(False).astype(bool).mean())
                    plans = self._relaxation_plans_for_condition(condition, metadata, failed_values)
                    for plan in plans:
                        if budget_exhausted():
                            time_budget_exhausted = True
                            break
                        if variants_seen >= MAX_ADAPTIVE_REPLAY_VARIANTS:
                            break
                        variants_seen += 1
                        relaxed_conditions = [dict(c) for c in conditions]
                        relaxed_conditions[idx - 1] = plan["condition"]
                        relaxed_mask, _, missing_relaxed = self.rule_service._build_rule_mask(prepared_hist, relaxed_conditions)
                        if missing_relaxed:
                            continue
                        current_relaxed_mask, _, _ = self.rule_service._build_rule_mask(prepared_current, relaxed_conditions)
                        current_promoted = 0
                        if current_relaxed_mask is not None:
                            current_promoted = int((current_relaxed_mask.fillna(False).astype(bool) & ~current_full_mask).sum())
                        relaxed_metrics = self._historical_metrics_for_mask(prepared_hist, relaxed_mask, horizon)
                        recommendation, reason, live_score = self._recommend_relaxation(original_metrics, relaxed_metrics, baseline_metrics, current_promoted)
                        support_delta = int(relaxed_metrics.get("support_rows") or 0) - int(original_metrics.get("support_rows") or 0)
                        coverage_multiplier = self._safe_ratio(relaxed_metrics.get("support_rows"), max(int(original_metrics.get("support_rows") or 0), 1))
                        mean_delta = None
                        if relaxed_metrics.get("mean_forward_return") is not None and original_metrics.get("mean_forward_return") is not None:
                            mean_delta = float(relaxed_metrics["mean_forward_return"]) - float(original_metrics["mean_forward_return"])
                        touch_delta = None
                        if relaxed_metrics.get("touch_rate") is not None and original_metrics.get("touch_rate") is not None:
                            touch_delta = float(relaxed_metrics["touch_rate"]) - float(original_metrics["touch_rate"])
                        variant_id = f"{instance['instance_id']}::relax_c{idx}_{plan['relaxation_kind']}"
                        relaxation_rows.append({
                            "status": "ok",
                            "scan_ts": pd.Timestamp(latest_ts).isoformat(),
                            "merged_rule_id": instance["merged_rule_id"],
                            "rule_instance_id": instance["instance_id"],
                            "condition_index": idx,
                            "condition_field": field,
                            "condition_logic": metadata.get("logic") or condition.get("logic"),
                            "relaxation_kind": plan["relaxation_kind"],
                            "original_threshold": plan["original_threshold"],
                            "relaxed_threshold": plan["relaxed_threshold"],
                            "original_condition": self._jsonable_condition(condition),
                            "relaxed_condition": self._jsonable_condition(plan["condition"]),
                            "current_failed_condition_count": current_failed_count,
                            "current_failure_share": current_failure_share,
                            "historical_condition_pass_rate": hist_pass_rate,
                            "live_current_promoted_count": current_promoted,
                            "variant_id": variant_id,
                            "recommendation": recommendation,
                            "reason": reason,
                        })
                        replay_rows.append({
                            "status": "ok",
                            "scan_ts": pd.Timestamp(latest_ts).isoformat(),
                            "merged_rule_id": instance["merged_rule_id"],
                            "rule_instance_id": instance["instance_id"],
                            "variant_id": variant_id,
                            "variant_kind": plan["relaxation_kind"],
                            "target_horizon": horizon,
                            "support_rows": relaxed_metrics.get("support_rows"),
                            "support_delta_vs_original": support_delta,
                            "coverage_multiplier_vs_original": coverage_multiplier,
                            "distinct_products": relaxed_metrics.get("distinct_products"),
                            "largest_product_share": relaxed_metrics.get("largest_product_share"),
                            "mean_forward_return": relaxed_metrics.get("mean_forward_return"),
                            "median_forward_return": relaxed_metrics.get("median_forward_return"),
                            "mean_max_up_pct": relaxed_metrics.get("mean_max_up_pct"),
                            "touch_rate": relaxed_metrics.get("touch_rate"),
                            "baseline_mean_forward_return": baseline_metrics.get("mean_forward_return"),
                            "original_mean_forward_return": original_metrics.get("mean_forward_return"),
                            "mean_forward_return_delta_vs_original": mean_delta,
                            "live_current_promoted_count": current_promoted,
                            "recommendation": recommendation,
                            "reason": reason,
                        })
                        frontier_rows.append({
                            "status": "ok",
                            "scan_ts": pd.Timestamp(latest_ts).isoformat(),
                            "merged_rule_id": instance["merged_rule_id"],
                            "rule_instance_id": instance["instance_id"],
                            "variant_id": variant_id,
                            "variant_kind": plan["relaxation_kind"],
                            "target_horizon": horizon,
                            "support_rows": relaxed_metrics.get("support_rows"),
                            "coverage_multiplier_vs_original": coverage_multiplier,
                            "mean_forward_return": relaxed_metrics.get("mean_forward_return"),
                            "mean_forward_return_delta_vs_original": mean_delta,
                            "touch_rate": relaxed_metrics.get("touch_rate"),
                            "touch_rate_delta_vs_original": touch_delta,
                            "live_current_promoted_count": current_promoted,
                            "live_usability_score": live_score,
                            "recommendation": recommendation,
                            "reason": reason,
                        })
                if variants_seen >= MAX_ADAPTIVE_REPLAY_VARIANTS:
                    break

        replay_df = pd.DataFrame(replay_rows)
        relaxation_df = pd.DataFrame(relaxation_rows)
        frontier_df = pd.DataFrame(frontier_rows)
        if replay_df.empty and relaxation_df.empty and frontier_df.empty:
            return self._adaptive_status_row("no_current_near_match_relaxations", latest_ts, "No current near-match condition relaxations were available to replay.")
        for df in (replay_df, relaxation_df, frontier_df):
            if not df.empty:
                df["historical_rows_available"] = historical_rows_available
                df["historical_rows_used"] = historical_rows_used
                df["historical_sample_note"] = historical_sample_note
                df["adaptive_replay_seconds_budget"] = MAX_ADAPTIVE_REPLAY_SECONDS
                df["adaptive_replay_variants_evaluated"] = variants_seen
        if time_budget_exhausted:
            replay_df, relaxation_df, frontier_df = self._append_adaptive_status_note(
                replay_df,
                relaxation_df,
                frontier_df,
                "adaptive_replay_time_budget_exhausted",
                latest_ts,
                "Adaptive replay returned bounded partial results because the live scan time budget was reached.",
            )
        if not replay_df.empty:
            replay_df = replay_df.sort_values(["live_current_promoted_count", "support_delta_vs_original", "mean_forward_return_delta_vs_original"], ascending=[False, False, False], na_position="last").reset_index(drop=True)
        if not relaxation_df.empty:
            relaxation_df = relaxation_df.sort_values(["live_current_promoted_count", "current_failed_condition_count"], ascending=[False, False], na_position="last").reset_index(drop=True)
        if not frontier_df.empty:
            frontier_df = frontier_df.sort_values(["live_usability_score", "live_current_promoted_count", "coverage_multiplier_vs_original"], ascending=[False, False, False], na_position="last").reset_index(drop=True)
        return replay_df, relaxation_df, frontier_df


    def _empty_orthogonal_discovery_tables(self, status: str = "no_orthogonal_discovery") -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        discovery_cols = [
            "status", "scan_ts", "candidate_id", "candidate_name", "candidate_family", "candidate_kind",
            "target_horizon", "conditions_json", "anchor_field", "source_families", "orthogonality_score",
            "orthogonality_reason", "support_rows", "distinct_products", "largest_product_share",
            "mean_forward_return", "median_forward_return", "mean_max_up_pct", "touch_rate",
            "baseline_mean_forward_return", "baseline_touch_rate", "baseline_lift", "touch_lift",
            "live_current_matches", "promotion_score", "recommendation", "reason",
            "historical_rows_available", "historical_rows_used", "historical_sample_note",
        ]
        gate_cols = [
            "status", "scan_ts", "candidate_id", "candidate_name", "support_gate", "baseline_lift_gate",
            "touch_gate", "concentration_gate", "diversification_gate", "orthogonality_gate", "live_potential_gate",
            "promotion_gate_status", "recommendation", "reason", "support_rows", "distinct_products",
            "largest_product_share", "baseline_lift", "touch_lift", "live_current_matches",
        ]
        payload = {
            "artifact_type": "orthogonal_rule_candidates",
            "schema_version": "1.0",
            "status": status,
            "generated_candidates": [],
            "note": "No orthogonal discovery has been run for this scan.",
        }
        return (
            pd.DataFrame(columns=discovery_cols).assign(status=status).iloc[0:0],
            pd.DataFrame(columns=gate_cols).assign(status=status).iloc[0:0],
            payload,
        )

    def _orthogonal_status_tables(self, status: str, latest_ts: pd.Timestamp, reason: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        discovery, gate, payload = self._empty_orthogonal_discovery_tables(status)
        scan_ts = pd.Timestamp(latest_ts).isoformat()
        row = {
            "status": status,
            "scan_ts": scan_ts,
            "candidate_id": None,
            "candidate_name": None,
            "candidate_family": None,
            "candidate_kind": None,
            "target_horizon": None,
            "conditions_json": None,
            "anchor_field": None,
            "source_families": None,
            "orthogonality_score": None,
            "orthogonality_reason": None,
            "support_rows": None,
            "distinct_products": None,
            "largest_product_share": None,
            "mean_forward_return": None,
            "median_forward_return": None,
            "mean_max_up_pct": None,
            "touch_rate": None,
            "baseline_mean_forward_return": None,
            "baseline_touch_rate": None,
            "baseline_lift": None,
            "touch_lift": None,
            "live_current_matches": 0,
            "promotion_score": None,
            "recommendation": "not_evaluated",
            "reason": reason,
        }
        discovery = pd.concat([discovery, pd.DataFrame([row])], ignore_index=True)
        gate = pd.concat([gate, pd.DataFrame([{
            "status": status,
            "scan_ts": scan_ts,
            "candidate_id": None,
            "candidate_name": None,
            "support_gate": False,
            "baseline_lift_gate": False,
            "touch_gate": False,
            "concentration_gate": False,
            "diversification_gate": False,
            "orthogonality_gate": False,
            "live_potential_gate": False,
            "promotion_gate_status": "not_evaluated",
            "recommendation": "not_evaluated",
            "reason": reason,
            "support_rows": None,
            "distinct_products": None,
            "largest_product_share": None,
            "baseline_lift": None,
            "touch_lift": None,
            "live_current_matches": 0,
        }])], ignore_index=True)
        payload.update({"status": status, "note": reason, "generated_at_scan_ts": scan_ts})
        return discovery, gate, payload

    def _feature_family(self, field: str) -> str:
        field = str(field or "")
        if field.startswith("cs_"):
            return "cross_source"
        if field.startswith("ca_"):
            if "volume" in field or "liquidity" in field:
                return "coinapi_liquidity"
            if "ret" in field or "dist" in field or "breakout" in field:
                return "coinapi_price_action"
            return "coinapi_context"
        if field.startswith("cx_"):
            return "market_context"
        if field.startswith("cb_"):
            if "volume" in field or "dollar_volume" in field:
                return "coinbase_liquidity"
            if "range" in field or "location" in field or "intrabar" in field:
                return "coinbase_bar_shape"
            if "rel_to" in field or "btc" in field or "eth" in field:
                return "coinbase_relative_strength"
            if "ret" in field or "ema" in field or "sma" in field or "breakout" in field:
                return "coinbase_price_action"
            return "coinbase_context"
        return "other"

    def _existing_live_rule_fields(self, rules: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
        fields: set[str] = set()
        families: set[str] = set()
        for rule in rules:
            for instance in self.rule_service._resolve_rule_variants(rule):
                for condition in instance.get("conditions", []) or []:
                    raw = condition.get("field") or condition.get("feature")
                    if raw:
                        resolved = self.rule_service._field_name(str(raw))
                        fields.add(resolved)
                        families.add(self._feature_family(resolved))
        return fields, families

    def _candidate_feature_columns(self, df: pd.DataFrame) -> list[str]:
        preferred = [
            "cs_coinbase_vs_coinapi_return_diff", "cs_coinbase_vs_coinapi_close_diff", "cs_cross_source_divergence_flag",
            "ca_ret_1", "ca_ret_6", "ca_ret_24", "ca_rel_volume_short", "ca_volume_zscore_short",
            "cb_rel_volume_short", "cb_volume_zscore_short", "cb_close_location_in_bar", "cb_intrabar_range_pct",
            "cb_ret_1", "cb_ret_3", "cb_ret_6", "cb_ret_12", "cb_ret_24", "cb_sma_5_dist", "cb_sma_10_dist",
            "cb_sma_20_dist", "cb_ema_12_dist", "cb_ema_26_dist", "cb_breakout_distance_short",
            "cb_rel_to_btc_ret_6", "cb_rel_to_eth_ret_6", "cx_btc_regime_flag", "cx_eth_regime_flag",
        ]
        exclude_prefixes = ("future_", "touched_")
        exclude_exact = {"ts", "product_id", "base_asset", "quote_asset", "coinapi_symbol_id", "feature_version"}
        out: list[str] = []
        for col in preferred + sorted([c for c in df.columns if str(c).startswith(("cb_", "ca_", "cs_", "cx_"))]):
            if col in out or col not in df.columns or col in exclude_exact or str(col).startswith(exclude_prefixes):
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            valid = series.dropna()
            if len(valid) < max(20, min(200, int(len(df) * 0.01))):
                continue
            if valid.nunique(dropna=True) <= 1:
                continue
            out.append(col)
            if len(out) >= 32:
                break
        return out

    def _orthogonal_threshold_specs_for_field(self, hist: pd.DataFrame, field: str) -> list[dict[str, Any]]:
        series = pd.to_numeric(hist[field], errors="coerce").dropna()
        if series.empty:
            return []
        uniq = sorted(series.unique().tolist())
        specs: list[dict[str, Any]] = []
        if len(uniq) <= 3 and set(float(x) for x in uniq).issubset({0.0, 1.0}):
            if 1.0 in uniq:
                specs.append({"logic": "==", "value": 1.0, "kind": "binary_flag_true", "quantile": None})
            return specs
        name = str(field)
        if "range" in name or "volatility" in name or "volume" in name or "dollar_volume" in name or "trades" in name:
            quantiles = [(0.85, ">=", "top_quantile"), (0.15, "<=", "bottom_quantile")]
        elif "ret" in name or "dist" in name or "diff" in name or "location" in name or "breakout" in name:
            quantiles = [(0.90, ">=", "top_quantile"), (0.10, "<=", "bottom_quantile")]
        else:
            quantiles = [(0.90, ">=", "top_quantile"), (0.10, "<=", "bottom_quantile")]
        for q, logic, kind in quantiles:
            try:
                value = float(series.quantile(q))
            except Exception:
                continue
            if pd.isna(value):
                continue
            specs.append({"logic": logic, "value": value, "kind": kind, "quantile": q})
        return specs

    def _orthogonality_score(self, fields: set[str], families: set[str], existing_fields: set[str], existing_families: set[str]) -> tuple[float, str]:
        if not fields:
            return 0.0, "No candidate fields resolved."
        new_fields = fields - existing_fields
        new_families = families - existing_families
        field_score = len(new_fields) / max(len(fields), 1)
        family_score = len(new_families) / max(len(families), 1)
        score = 0.65 * field_score + 0.35 * family_score
        if new_fields and new_families:
            reason = f"Uses new fields ({', '.join(sorted(new_fields)[:4])}) and new source families ({', '.join(sorted(new_families)[:4])})."
        elif new_fields:
            reason = f"Uses new fields ({', '.join(sorted(new_fields)[:4])}) but overlaps current source families."
        else:
            reason = "Candidate mostly reuses fields from the current live rule set."
        return float(score), reason

    def _candidate_rule_payload(self, row: dict[str, Any], live_eligible: bool = False) -> dict[str, Any]:
        try:
            conditions = json.loads(row.get("conditions_json") or "[]")
        except Exception:
            conditions = []
        return {
            "merged_rule_id": row.get("candidate_id"),
            "name": row.get("candidate_name"),
            "family_id": row.get("candidate_family"),
            "rule_kind": "direct_rule",
            "live_eligible": bool(live_eligible),
            "live_candidate_recommended": False,
            "live_candidate_reason": "v1.9.0 generated candidates require manual review and live-shadow validation before eligibility.",
            "recommended_primary_horizon": row.get("target_horizon") or "h4",
            "target_horizons": [row.get("target_horizon") or "h4"],
            "source_attribution": ["orthogonal_discovery_v1_9_0"],
            "why_interesting": row.get("reason"),
            "exact_definition": {"all_conditions": conditions},
        }

    def _orthogonal_candidate_templates(
        self,
        prepared_hist: pd.DataFrame,
        prepared_current: pd.DataFrame,
        existing_fields: set[str],
        existing_families: set[str],
    ) -> list[dict[str, Any]]:
        templates: list[dict[str, Any]] = []
        seen: set[str] = set()
        liquidity_condition: dict[str, Any] | None = None
        if "cb_dollar_volume_proxy" in prepared_hist.columns:
            liq = pd.to_numeric(prepared_hist["cb_dollar_volume_proxy"], errors="coerce").dropna()
            if not liq.empty:
                threshold = max(100000.0, float(liq.quantile(0.35)))
                liquidity_condition = {"field": "cb_dollar_volume_proxy", "logic": ">=", "value": threshold}
        for field in self._candidate_feature_columns(prepared_hist):
            family = self._feature_family(field)
            for spec in self._orthogonal_threshold_specs_for_field(prepared_hist, field):
                conditions = [{"field": field, "logic": spec["logic"], "value": spec["value"]}]
                # Add a light liquidity sanity gate unless the candidate is itself a liquidity rule.
                if liquidity_condition and field != "cb_dollar_volume_proxy" and family not in {"coinbase_liquidity", "coinapi_liquidity"}:
                    conditions.append(dict(liquidity_condition))
                fields = {self.rule_service._field_name(str(c.get("field"))) for c in conditions if c.get("field")}
                families = {self._feature_family(f) for f in fields}
                orth_score, orth_reason = self._orthogonality_score(fields, families, existing_fields, existing_families)
                direction = "high" if spec["logic"] in {">", ">=", "=="} else "low"
                candidate_id = f"ORTHO_{field}_{direction}_{str(spec.get('quantile') or spec['kind']).replace('.', '_')}".upper()
                candidate_id = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in candidate_id)[:120]
                key = json.dumps(conditions, sort_keys=True, default=str)
                if key in seen:
                    continue
                seen.add(key)
                templates.append({
                    "candidate_id": candidate_id,
                    "candidate_name": f"Orthogonal {family}: {field} {spec['logic']} {spec['value']:.6g}",
                    "candidate_family": family,
                    "candidate_kind": spec["kind"],
                    "target_horizon": "h4",
                    "conditions": conditions,
                    "anchor_field": field,
                    "source_families": "|".join(sorted(families)),
                    "orthogonality_score": orth_score,
                    "orthogonality_reason": orth_reason,
                })
        # Prefer candidates with new source families/fields first, then keep the search bounded.
        templates = sorted(templates, key=lambda x: (x["orthogonality_score"], x["candidate_family"], x["candidate_id"]), reverse=True)
        return templates[:MAX_ORTHOGONAL_CANDIDATES]

    def _gate_orthogonal_candidate(
        self,
        row: dict[str, Any],
        historical_rows_used: int,
        historical_product_count: int,
    ) -> tuple[dict[str, Any], str, str, str]:
        support = int(row.get("support_rows") or 0)
        distinct = int(row.get("distinct_products") or 0)
        largest_share = row.get("largest_product_share")
        baseline_lift = row.get("baseline_lift")
        touch_lift = row.get("touch_lift")
        orth_score = float(row.get("orthogonality_score") or 0.0)
        live_matches = int(row.get("live_current_matches") or 0)
        min_support = max(20, min(80, int(max(historical_rows_used, 1) * 0.0005)))
        min_distinct = max(3, min(12, int(max(historical_product_count, 1) * 0.05)))
        gates = {
            "support_gate": support >= min_support,
            "baseline_lift_gate": baseline_lift is not None and float(baseline_lift) >= ORTHOGONAL_MIN_BASELINE_LIFT,
            "touch_gate": touch_lift is None or float(touch_lift) >= ORTHOGONAL_MIN_TOUCH_LIFT,
            "concentration_gate": largest_share is None or float(largest_share) <= ORTHOGONAL_MAX_PRODUCT_SHARE,
            "diversification_gate": distinct >= min_distinct,
            "orthogonality_gate": orth_score >= 0.55,
            "live_potential_gate": live_matches > 0,
        }
        historical_ok = all(gates[k] for k in ["support_gate", "baseline_lift_gate", "touch_gate", "concentration_gate", "diversification_gate", "orthogonality_gate"])
        if historical_ok and gates["live_potential_gate"]:
            return gates, "shadow_candidate_ready", "promote_to_shadow_candidate", "Candidate passes strict historical gates and currently matches at least one live product; validate in live shadow before marking live eligible."
        if historical_ok:
            return gates, "historical_watchlist_ready", "watchlist_historical_only", "Candidate passes strict historical gates but has no current live matches; keep as historical watchlist candidate."
        failed = [k for k, v in gates.items() if not v and k != "live_potential_gate"]
        reason_map = {
            "support_gate": "insufficient historical support",
            "baseline_lift_gate": "insufficient lift over baseline",
            "touch_gate": "insufficient touch-rate lift",
            "concentration_gate": "too concentrated in a few products",
            "diversification_gate": "too few distinct products",
            "orthogonality_gate": "not sufficiently orthogonal to current live rule fields/families",
        }
        reasons = ", ".join(reason_map.get(k, k) for k in failed[:3]) or "failed strict promotion gates"
        return gates, "rejected_by_gate", "reject_promotion_gate", f"Rejected by promotion gates: {reasons}."

    def _build_orthogonal_discovery_tables(
        self,
        snapshot: pd.DataFrame,
        rules: list[dict[str, Any]],
        latest_ts: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        started = time.monotonic()
        deadline = started + MAX_ORTHOGONAL_DISCOVERY_SECONDS
        historical = self.storage.read_frame("feature_table")
        if historical.empty:
            return self._orthogonal_status_tables("no_historical_feature_table", latest_ts, "No historical feature table exists yet; orthogonal discovery was skipped.")
        historical_rows_available = int(len(historical))
        historical_rows_used = historical_rows_available
        historical_sample_note = "full_historical_feature_table"
        if historical_rows_available > MAX_ORTHOGONAL_HISTORICAL_ROWS:
            historical_sample_note = f"bounded_recent_sample_{MAX_ORTHOGONAL_HISTORICAL_ROWS}_of_{historical_rows_available}"
            if "ts" in historical.columns:
                historical = historical.copy()
                historical["_ts_for_orthogonal_sample"] = pd.to_datetime(historical["ts"], utc=True, errors="coerce")
                historical = historical.sort_values("_ts_for_orthogonal_sample").tail(MAX_ORTHOGONAL_HISTORICAL_ROWS).drop(columns=["_ts_for_orthogonal_sample"], errors="ignore")
            else:
                historical = historical.tail(MAX_ORTHOGONAL_HISTORICAL_ROWS).copy()
            historical_rows_used = int(len(historical))
        prepared_hist = self.rule_service._prepare_frame(historical)
        prepared_current = self.rule_service._prepare_frame(snapshot)
        baseline = self._global_baseline_metrics(prepared_hist, "h4")
        baseline_touch = baseline.get("touch_rate")
        existing_fields, existing_families = self._existing_live_rule_fields(rules)
        templates = self._orthogonal_candidate_templates(prepared_hist, prepared_current, existing_fields, existing_families)
        if not templates:
            return self._orthogonal_status_tables("no_candidate_templates", latest_ts, "No orthogonal feature templates were available for this feature table.")
        historical_product_count = int(prepared_hist["product_id"].nunique()) if "product_id" in prepared_hist.columns else 0
        rows: list[dict[str, Any]] = []
        gate_rows: list[dict[str, Any]] = []
        time_budget_exhausted = False
        for template in templates:
            if time.monotonic() >= deadline:
                time_budget_exhausted = True
                break
            mask, _, missing = self.rule_service._build_rule_mask(prepared_hist, template["conditions"])
            if missing or mask is None:
                continue
            current_mask, _, current_missing = self.rule_service._build_rule_mask(prepared_current, template["conditions"])
            live_matches = 0 if current_mask is None or current_missing else int(current_mask.fillna(False).astype(bool).sum())
            metrics = self._historical_metrics_for_mask(prepared_hist, mask, "h4")
            mean = metrics.get("mean_forward_return")
            baseline_mean = baseline.get("mean_forward_return")
            touch = metrics.get("touch_rate")
            baseline_lift = None if mean is None or baseline_mean is None else float(mean) - float(baseline_mean)
            touch_lift = None if touch is None or baseline_touch is None else float(touch) - float(baseline_touch)
            score = 0.0
            if baseline_lift is not None:
                score += float(baseline_lift) * 10000.0
            if touch_lift is not None:
                score += float(touch_lift) * 250.0
            score += np.log1p(float(metrics.get("support_rows") or 0)) * 5.0
            score += float(live_matches) * 10.0
            score += float(template["orthogonality_score"]) * 25.0
            if metrics.get("largest_product_share") is not None:
                score -= float(metrics["largest_product_share"]) * 20.0
            row = {
                "status": "ok",
                "scan_ts": pd.Timestamp(latest_ts).isoformat(),
                "candidate_id": template["candidate_id"],
                "candidate_name": template["candidate_name"],
                "candidate_family": template["candidate_family"],
                "candidate_kind": template["candidate_kind"],
                "target_horizon": "h4",
                "conditions_json": json.dumps(template["conditions"], sort_keys=True, default=str),
                "anchor_field": template["anchor_field"],
                "source_families": template["source_families"],
                "orthogonality_score": template["orthogonality_score"],
                "orthogonality_reason": template["orthogonality_reason"],
                "support_rows": metrics.get("support_rows"),
                "distinct_products": metrics.get("distinct_products"),
                "largest_product_share": metrics.get("largest_product_share"),
                "mean_forward_return": metrics.get("mean_forward_return"),
                "median_forward_return": metrics.get("median_forward_return"),
                "mean_max_up_pct": metrics.get("mean_max_up_pct"),
                "touch_rate": metrics.get("touch_rate"),
                "baseline_mean_forward_return": baseline_mean,
                "baseline_touch_rate": baseline_touch,
                "baseline_lift": baseline_lift,
                "touch_lift": touch_lift,
                "live_current_matches": live_matches,
                "promotion_score": float(score),
                "historical_rows_available": historical_rows_available,
                "historical_rows_used": historical_rows_used,
                "historical_sample_note": historical_sample_note,
            }
            gates, gate_status, recommendation, reason = self._gate_orthogonal_candidate(row, historical_rows_used, historical_product_count)
            row["recommendation"] = recommendation
            row["reason"] = reason
            row["promotion_gate_status"] = gate_status
            rows.append(row)
            gate_rows.append({
                "status": "ok",
                "scan_ts": row["scan_ts"],
                "candidate_id": row["candidate_id"],
                "candidate_name": row["candidate_name"],
                **gates,
                "promotion_gate_status": gate_status,
                "recommendation": recommendation,
                "reason": reason,
                "support_rows": row["support_rows"],
                "distinct_products": row["distinct_products"],
                "largest_product_share": row["largest_product_share"],
                "baseline_lift": row["baseline_lift"],
                "touch_lift": row["touch_lift"],
                "live_current_matches": row["live_current_matches"],
            })
        discovery_df = pd.DataFrame(rows)
        gate_df = pd.DataFrame(gate_rows)
        if discovery_df.empty:
            return self._orthogonal_status_tables("no_candidates_passed_evaluation", latest_ts, "Orthogonal templates were generated but none could be evaluated against the historical feature table.")
        discovery_df = discovery_df.sort_values(["promotion_score", "baseline_lift", "support_rows"], ascending=[False, False, False], na_position="last").reset_index(drop=True)
        gate_df = gate_df.set_index("candidate_id").loc[discovery_df["candidate_id"].tolist()].reset_index() if not gate_df.empty else gate_df
        if time_budget_exhausted:
            discovery_df["time_budget_exhausted"] = True
            gate_df["time_budget_exhausted"] = True
        reviewable = discovery_df.loc[discovery_df["promotion_gate_status"].isin(["shadow_candidate_ready", "historical_watchlist_ready"])].copy()
        generated_candidates = [self._candidate_rule_payload(row.to_dict(), live_eligible=False) for _, row in reviewable.head(20).iterrows()]
        payload = {
            "artifact_type": "orthogonal_rule_candidates",
            "schema_version": "1.0",
            "status": "ok",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scan_ts": pd.Timestamp(latest_ts).isoformat(),
            "app_version": self.settings.app_version,
            "candidate_count": int(len(discovery_df)),
            "reviewable_candidate_count": int(len(reviewable)),
            "shadow_candidate_ready_count": int((discovery_df["promotion_gate_status"] == "shadow_candidate_ready").sum()),
            "historical_watchlist_ready_count": int((discovery_df["promotion_gate_status"] == "historical_watchlist_ready").sum()),
            "generated_candidates": generated_candidates,
            "note": "Generated candidates are evidence artifacts only. They are not auto-promoted to live eligibility; upload/apply manually only after review and live-shadow validation.",
        }
        return discovery_df, gate_df, payload

    def run_cycle(self, request: LiveScanRequest, run_id_override: str | None = None) -> dict[str, Any]:
        step = "live_scan_cycle"
        run_id = run_id_override or self.storage.make_run_id()
        as_of = self.live_shadow._parse_as_of(request.as_of_time_iso)
        lookback_hours = int(request.lookback_hours or self.settings.live_scan_lookback_hours)
        max_products = int(request.max_products or self.settings.live_scan_max_products)
        self.storage.update_status(step, "running", message="Running live scanner", run_id=run_id, phase="initializing")
        try:
            rules = self._select_live_rules(request)
            refresh_refs = request.refresh_references if request.refresh_references is not None else self.settings.live_shadow_auto_refresh_references
            products, mapping = self.live_shadow._ensure_reference_tables(refresh_refs)
            start = as_of - timedelta(hours=lookback_hours)
            end = as_of
            eligible, cb, ca, quotes = self.live_shadow._pull_live_bars(products, mapping, start, end, max_products=max_products, step=step)
            self.storage.update_status(step, "running", message="Computing latest snapshot", phase="computing_features", coinbase_rows=int(len(cb)), coinapi_rows=int(len(ca)), quote_rows=int(len(quotes)))
            feature_df = self.live_shadow._compute_feature_table(products=eligible, cb=cb, ca=ca, quotes=quotes)
            if feature_df.empty:
                raise RuntimeError("Live scanner snapshot is empty.")
            snapshot, stale_products, latest_ts = _select_per_product_latest(feature_df, freshness_hours=self.settings.live_scan_max_staleness_hours)
            if latest_ts is None or snapshot.empty:
                raise RuntimeError("Live scanner snapshot is empty after freshness filtering.")
            self.storage.update_status(step, "running", message="Evaluating live-eligible rules", phase="evaluating_rules", latest_signal_ts=latest_ts.isoformat(), snapshot_rows=int(len(snapshot)), stale_products=stale_products)
            signal_rows, skipped = self.live_shadow._evaluate_snapshot(snapshot, rules, run_id=run_id, decision_time=as_of)
            shortlist, rule_hits = self._build_scan_tables(signal_rows, latest_ts)
            near_matches, coverage_summary = self._build_near_match_tables(snapshot, rules, latest_ts, rule_hits)
            self.storage.update_status(
                step,
                "running",
                message="Running bounded adaptive near-match replay",
                phase="adaptive_replay",
                latest_signal_ts=latest_ts.isoformat(),
                snapshot_rows=int(len(snapshot)),
                shortlist_rows=int(len(shortlist)),
                rule_hits=int(len(rule_hits)),
                near_match_rows=int(len(near_matches)),
                rules_evaluated=int(len(rules)),
                adaptive_replay_seconds_budget=MAX_ADAPTIVE_REPLAY_SECONDS,
                adaptive_replay_variant_cap=MAX_ADAPTIVE_REPLAY_VARIANTS,
            )
            adaptive_warning = None
            try:
                near_match_replay, relaxation_candidates, coverage_frontier = self._build_adaptive_replay_tables(snapshot, rules, latest_ts)
            except Exception as adaptive_exc:
                adaptive_warning = str(adaptive_exc)
                near_match_replay, relaxation_candidates, coverage_frontier = self._adaptive_status_row(
                    "adaptive_replay_failed",
                    latest_ts,
                    f"Adaptive replay failed, but the core live scan completed: {adaptive_exc}",
                )

            self.storage.update_status(
                step,
                "running",
                message="Running bounded orthogonal candidate discovery",
                phase="orthogonal_discovery",
                latest_signal_ts=latest_ts.isoformat(),
                snapshot_rows=int(len(snapshot)),
                shortlist_rows=int(len(shortlist)),
                rule_hits=int(len(rule_hits)),
                near_match_rows=int(len(near_matches)),
                near_match_replay_rows=int(len(near_match_replay)),
                relaxation_candidate_rows=int(len(relaxation_candidates)),
                coverage_frontier_rows=int(len(coverage_frontier)),
                adaptive_replay_warning=adaptive_warning,
                orthogonal_discovery_seconds_budget=MAX_ORTHOGONAL_DISCOVERY_SECONDS,
                orthogonal_candidate_cap=MAX_ORTHOGONAL_CANDIDATES,
            )
            orthogonal_warning = None
            try:
                orthogonal_discovery, promotion_gates, orthogonal_rules_payload = self._build_orthogonal_discovery_tables(snapshot, rules, latest_ts)
            except Exception as orthogonal_exc:
                orthogonal_warning = str(orthogonal_exc)
                orthogonal_discovery, promotion_gates, orthogonal_rules_payload = self._orthogonal_status_tables(
                    "orthogonal_discovery_failed",
                    latest_ts,
                    f"Orthogonal discovery failed, but the core live scan completed: {orthogonal_exc}",
                )

            self.storage.update_status(
                step,
                "running",
                message="Writing live scan artifacts",
                phase="writing_artifacts",
                latest_signal_ts=latest_ts.isoformat(),
                snapshot_rows=int(len(snapshot)),
                shortlist_rows=int(len(shortlist)),
                rule_hits=int(len(rule_hits)),
                near_match_rows=int(len(near_matches)),
                near_match_replay_rows=int(len(near_match_replay)),
                relaxation_candidate_rows=int(len(relaxation_candidates)),
                coverage_frontier_rows=int(len(coverage_frontier)),
                orthogonal_candidate_rows=int(len(orthogonal_discovery)),
                promotion_gate_rows=int(len(promotion_gates)),
                adaptive_replay_warning=adaptive_warning,
                orthogonal_discovery_warning=orthogonal_warning,
            )

            shortlist_path = self.storage.write_csv(shortlist, f"live_scan_results__{run_id}", compress=False)
            hits_path = self.storage.write_csv(rule_hits, f"live_scan_rule_hits__{run_id}", compress=False)
            near_path = self.storage.write_csv(near_matches, f"live_scan_near_matches__{run_id}", compress=False)
            coverage_path = self.storage.write_csv(coverage_summary, f"rule_coverage_summary__{run_id}", compress=False)
            near_replay_path = self.storage.write_csv(near_match_replay, f"near_match_replay_summary__{run_id}", compress=False)
            relaxation_path = self.storage.write_csv(relaxation_candidates, f"rule_relaxation_candidates__{run_id}", compress=False)
            frontier_path = self.storage.write_csv(coverage_frontier, f"coverage_quality_frontier__{run_id}", compress=False)
            orthogonal_path = self.storage.write_csv(orthogonal_discovery, f"orthogonal_candidate_discovery__{run_id}", compress=False)
            promotion_gate_path = self.storage.write_csv(promotion_gates, f"promotion_gate_summary__{run_id}", compress=False)
            orthogonal_rules_path = self.storage.export_path(f"orthogonal_rule_candidates__{run_id}", ".json")
            self.storage.write_json(orthogonal_rules_payload, orthogonal_rules_path)
            summary_df = shortlist[["scanner_rank", "product_id", "matched_rule_count", "matched_rule_ids", "best_priority", "ranking_score", "matched_primary_horizons", "data_quality_note"]].copy() if not shortlist.empty else pd.DataFrame(columns=["scanner_rank", "product_id", "matched_rule_count", "matched_rule_ids", "best_priority", "ranking_score", "matched_primary_horizons", "data_quality_note"])
            summary_path = self.storage.write_csv(summary_df, f"live_scan_summary__{run_id}", compress=False)
            manifest = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "app": self.settings.app_name,
                "version": self.settings.app_version,
                "request": {
                    "lookback_hours": lookback_hours,
                    "max_products": max_products,
                    "max_staleness_hours": int(self.settings.live_scan_max_staleness_hours),
                    "selection_mode": request.selection_mode,
                    "rule_ids": request.rule_ids,
                    "refresh_references": bool(refresh_refs),
                    "as_of_time_iso": as_of.isoformat(),
                },
                "summary": {
                    "latest_signal_ts": latest_ts.isoformat(),
                    "snapshot_rows": int(len(snapshot)),
                    "rule_hits": int(len(rule_hits)),
                    "shortlist_rows": int(len(shortlist)),
                    "rules_evaluated": int(len(rules)),
                    "coinbase_rows": int(len(cb)),
                    "coinapi_rows": int(len(ca)),
                    "stale_products": stale_products,
                    "stale_product_count": int(len(stale_products)),
                    "top_match_product_id": shortlist.iloc[0]["product_id"] if not shortlist.empty else None,
                    "top_match_rule_count": int(shortlist.iloc[0]["matched_rule_count"]) if not shortlist.empty else 0,
                    "near_match_rows": int(len(near_matches)),
                    "coverage_rule_rows": int(len(coverage_summary)),
                    "near_match_replay_rows": int(len(near_match_replay)),
                    "relaxation_candidate_rows": int(len(relaxation_candidates)),
                    "coverage_frontier_rows": int(len(coverage_frontier)),
                    "best_near_match_product_id": near_matches.iloc[0]["product_id"] if not near_matches.empty else None,
                    "best_near_match_rule_id": near_matches.iloc[0]["merged_rule_id"] if not near_matches.empty else None,
                    "best_condition_pass_ratio": float(near_matches.iloc[0]["condition_pass_ratio"]) if not near_matches.empty else None,
                    "best_relaxation_rule_id": coverage_frontier.iloc[0]["merged_rule_id"] if not coverage_frontier.empty else None,
                    "best_relaxation_variant_id": coverage_frontier.iloc[0]["variant_id"] if not coverage_frontier.empty else None,
                    "best_relaxation_recommendation": coverage_frontier.iloc[0]["recommendation"] if not coverage_frontier.empty else None,
                    "orthogonal_candidate_rows": int(len(orthogonal_discovery)),
                    "promotion_gate_rows": int(len(promotion_gates)),
                    "shadow_candidate_ready_count": int((orthogonal_discovery.get("promotion_gate_status") == "shadow_candidate_ready").sum()) if not orthogonal_discovery.empty and "promotion_gate_status" in orthogonal_discovery.columns else 0,
                    "historical_watchlist_ready_count": int((orthogonal_discovery.get("promotion_gate_status") == "historical_watchlist_ready").sum()) if not orthogonal_discovery.empty and "promotion_gate_status" in orthogonal_discovery.columns else 0,
                    "best_orthogonal_candidate_id": orthogonal_discovery.iloc[0]["candidate_id"] if not orthogonal_discovery.empty and "candidate_id" in orthogonal_discovery.columns else None,
                    "best_orthogonal_recommendation": orthogonal_discovery.iloc[0]["recommendation"] if not orthogonal_discovery.empty and "recommendation" in orthogonal_discovery.columns else None,
                    "adaptive_replay_warning": adaptive_warning,
                    "orthogonal_discovery_warning": orthogonal_warning,
                    "skipped_rules": skipped,
                },
                "preview": shortlist.head(50).to_dict(orient="records"),
                "near_match_preview": near_matches.head(50).to_dict(orient="records"),
                "coverage_preview": coverage_summary.head(50).to_dict(orient="records"),
                "near_match_replay_preview": near_match_replay.head(50).to_dict(orient="records"),
                "relaxation_candidate_preview": relaxation_candidates.head(50).to_dict(orient="records"),
                "coverage_quality_frontier_preview": coverage_frontier.head(50).to_dict(orient="records"),
                "orthogonal_candidate_preview": orthogonal_discovery.head(50).to_dict(orient="records"),
                "promotion_gate_preview": promotion_gates.head(50).to_dict(orient="records"),
                "orthogonal_rule_candidates_preview": orthogonal_rules_payload.get("generated_candidates", [])[:20] if isinstance(orthogonal_rules_payload, dict) else [],
            }
            manifest_path = self.storage.export_path(f"live_scan_manifest__{run_id}", ".json")
            self.storage.write_json(manifest, manifest_path)
            pack_path = self.storage.export_path(f"live_scan_pack__{run_id}", ".zip")
            with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for path in [shortlist_path, hits_path, near_path, coverage_path, near_replay_path, relaxation_path, frontier_path, orthogonal_path, promotion_gate_path, orthogonal_rules_path, summary_path, manifest_path]:
                    zf.write(path, arcname=Path(path).name)
            artifacts = [self.storage.file_info(path) for path in [shortlist_path, hits_path, near_path, coverage_path, near_replay_path, relaxation_path, frontier_path, orthogonal_path, promotion_gate_path, orthogonal_rules_path, summary_path, manifest_path, pack_path]]
            manifest["artifacts"] = artifacts
            self.storage.write_latest_live_scan_manifest(manifest)
            self.storage.update_status(step, "completed", message="Live scanner completed", run_id=run_id, latest_signal_ts=latest_ts.isoformat(), shortlist_rows=int(len(shortlist)), rule_hits=int(len(rule_hits)), near_match_rows=int(len(near_matches)), near_match_replay_rows=int(len(near_match_replay)), relaxation_candidate_rows=int(len(relaxation_candidates)), coverage_frontier_rows=int(len(coverage_frontier)), orthogonal_candidate_rows=int(len(orthogonal_discovery)), promotion_gate_rows=int(len(promotion_gates)), rules_evaluated=int(len(rules)), adaptive_replay_warning=adaptive_warning, orthogonal_discovery_warning=orthogonal_warning, pack_artifact=pack_path.name)
            return manifest
        except Exception as exc:
            self.storage.update_status(step, "failed", error=str(exc), traceback=traceback.format_exc())
            raise
