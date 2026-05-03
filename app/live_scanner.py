from __future__ import annotations

import json
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

            shortlist_path = self.storage.write_csv(shortlist, f"live_scan_results__{run_id}", compress=False)
            hits_path = self.storage.write_csv(rule_hits, f"live_scan_rule_hits__{run_id}", compress=False)
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
                    "skipped_rules": skipped,
                },
                "preview": shortlist.head(50).to_dict(orient="records"),
            }
            manifest_path = self.storage.export_path(f"live_scan_manifest__{run_id}", ".json")
            self.storage.write_json(manifest, manifest_path)
            pack_path = self.storage.export_path(f"live_scan_pack__{run_id}", ".zip")
            with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for path in [shortlist_path, hits_path, summary_path, manifest_path]:
                    zf.write(path, arcname=Path(path).name)
            artifacts = [self.storage.file_info(path) for path in [shortlist_path, hits_path, summary_path, manifest_path, pack_path]]
            manifest["artifacts"] = artifacts
            self.storage.write_latest_live_scan_manifest(manifest)
            self.storage.update_status(step, "completed", message="Live scanner completed", run_id=run_id, latest_signal_ts=latest_ts.isoformat(), shortlist_rows=int(len(shortlist)), rule_hits=int(len(rule_hits)), rules_evaluated=int(len(rules)), pack_artifact=pack_path.name)
            return manifest
        except Exception as exc:
            self.storage.update_status(step, "failed", error=str(exc), traceback=traceback.format_exc())
            raise
