from __future__ import annotations

import json
import traceback
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .clients import CoinbaseAdvancedClient, CoinAPIClient
from .features import (
    FEATURE_VERSION,
    add_context_features,
    add_family_features,
    add_relative_features,
)
from .pipeline import ResearchPipeline
from .rule_backtests import RuleBacktestService
from .schemas import LiveShadowRequest
from .settings import Settings
from .storage import StorageManager


class LiveShadowService:
    def __init__(self, settings: Settings, storage: StorageManager, pipeline: ResearchPipeline, rule_service: RuleBacktestService):
        self.settings = settings
        self.storage = storage
        self.pipeline = pipeline
        self.rule_service = rule_service
        self.coinbase = CoinbaseAdvancedClient(settings)
        self.coinapi = CoinAPIClient(settings)

    @property
    def signal_log_name(self) -> str:
        return "live_signal_log"

    @property
    def outcomes_name(self) -> str:
        return "live_signal_outcomes"

    def latest_manifest(self) -> dict[str, Any]:
        return self.storage.read_latest_live_shadow_manifest()

    def _parse_as_of(self, value: str | None) -> datetime:
        if value:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    def _ensure_reference_tables(self, refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        products = self.storage.read_frame("coinbase_products")
        mapping = self.storage.read_frame("coinapi_symbol_mapping")
        if refresh or products.empty:
            self.pipeline.refresh_universe()
            products = self.storage.read_frame("coinbase_products")
        if refresh or mapping.empty:
            self.pipeline.refresh_mappings()
            mapping = self.storage.read_frame("coinapi_symbol_mapping")
        if products.empty or mapping.empty:
            raise RuntimeError("Universe and mappings are required before live shadow validation can run.")
        return products, mapping

    def _pull_live_bars(
        self,
        products: pd.DataFrame,
        mapping: pd.DataFrame,
        start: datetime,
        end: datetime,
        max_products: int,
        step: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        eligible = products.loc[products["eligibility_flag"]].copy()
        if "approximate_quote_24h_volume" in eligible.columns:
            eligible["_sort_volume"] = pd.to_numeric(eligible["approximate_quote_24h_volume"], errors="coerce").fillna(0)
            eligible = eligible.sort_values("_sort_volume", ascending=False)
        eligible = eligible.head(max_products).reset_index(drop=True)
        if not mapping.empty:
            mapping = mapping.sort_values(
                ["coinbase_product_id", "mapping_confidence", "mapping_status", "coinapi_symbol_id"],
                ascending=[True, False, True, True],
                na_position="last",
            )
            mapping = mapping.drop_duplicates(subset=["coinbase_product_id"], keep="first").reset_index(drop=True)
        mapping_lookup = mapping.set_index("coinbase_product_id").to_dict("index") if not mapping.empty else {}

        cb_frames: list[pd.DataFrame] = []
        ca_frames: list[pd.DataFrame] = []
        quote_frames: list[pd.DataFrame] = []
        total = len(eligible)
        for idx, (_, product) in enumerate(eligible.iterrows(), start=1):
            product_id = product["product_id"]
            self.storage.update_status(step, "running", progress=f"{idx}/{total}", current_product=product_id, phase="pulling_live_bars")
            cb = self.coinbase.get_candles(product_id, start=start, end=end, granularity=self.settings.preferred_bar_granularity)
            if not cb.empty:
                cb_frames.append(cb)
            mapped = mapping_lookup.get(product_id, {})
            if mapped.get("coinapi_symbol_id"):
                ca = self.coinapi.get_ohlcv(mapped["coinapi_symbol_id"], start=start, end=end, period_id=self.settings.coinapi_period_id)
                if not ca.empty:
                    if "coinbase_product_id" not in ca.columns:
                        ca["coinbase_product_id"] = product_id
                    ca_frames.append(ca)
                if self.settings.enable_coinapi_quotes:
                    quotes = self.coinapi.get_quote_history(mapped["coinapi_symbol_id"], start=start, end=end)
                    if not quotes.empty:
                        quotes["coinbase_product_id"] = product_id
                        quote_frames.append(quotes)
        cb_df = pd.concat(cb_frames, ignore_index=True) if cb_frames else pd.DataFrame()
        ca_df = pd.concat(ca_frames, ignore_index=True) if ca_frames else pd.DataFrame()
        quotes_df = pd.concat(quote_frames, ignore_index=True) if quote_frames else pd.DataFrame()
        return eligible, cb_df, ca_df, quotes_df

    def _compute_feature_table(
        self,
        products: pd.DataFrame,
        cb: pd.DataFrame,
        ca: pd.DataFrame,
        quotes: pd.DataFrame,
    ) -> pd.DataFrame:
        if cb.empty:
            raise RuntimeError("No Coinbase bars were pulled for live shadow validation.")
        cb = cb.copy()
        cb["ts"] = pd.to_datetime(cb["ts"], utc=True)
        cb = cb.sort_values(["product_id", "ts"])
        cb_features = add_family_features(cb[["product_id", "ts", "open", "high", "low", "close", "volume"]], "cb", include_ema=True)
        base_meta = products[[
            "product_id",
            "base_asset",
            "quote_asset",
            "base_min_size",
            "quote_min_size",
            "price_increment",
            "quote_increment",
        ]].drop_duplicates()
        feature_df = cb_features.merge(base_meta, on="product_id", how="left")

        if not ca.empty:
            ca = ca.copy()
            ca["ts"] = pd.to_datetime(ca["ts"], utc=True)
            ca = ca.rename(columns={"coinbase_product_id": "product_id"})
            ca = ca.sort_values(["product_id", "ts"])
            ca_features = add_family_features(ca[["product_id", "ts", "open", "high", "low", "close", "volume"]], "ca", include_ema=False)
            ca_features = ca_features.rename(columns={"close": "ca_close"})
            ca_features = ca_features[[c for c in ca_features.columns if c.startswith("ca_") or c in ["product_id", "ts"]]]
            ca_meta = ca[["product_id", "ts", "coinapi_symbol_id", "trades_count"]].drop_duplicates(["product_id", "ts"])
            feature_df = feature_df.merge(ca_features, on=["product_id", "ts"], how="left")
            feature_df = feature_df.merge(ca_meta, on=["product_id", "ts"], how="left")
        else:
            feature_df["coinapi_symbol_id"] = None
            feature_df["trades_count"] = np.nan

        if not quotes.empty:
            quotes = quotes.copy()
            quotes["ts"] = pd.to_datetime(quotes["ts"], utc=True)
            spread = quotes.groupby(["coinbase_product_id", "ts"], as_index=False).agg(
                ask_price=("ask_price", "mean"), bid_price=("bid_price", "mean")
            )
            spread["ca_quote_spread_proxy"] = (spread["ask_price"] - spread["bid_price"]) / ((spread["ask_price"] + spread["bid_price"]) / 2)
            feature_df = feature_df.merge(
                spread[["coinbase_product_id", "ts", "ca_quote_spread_proxy"]].rename(columns={"coinbase_product_id": "product_id"}),
                on=["product_id", "ts"],
                how="left",
            )
        else:
            feature_df["ca_quote_spread_proxy"] = np.nan

        if "ca_dollar_volume_proxy" in feature_df.columns:
            feature_df["ca_liquidity_bucket"] = pd.cut(
                feature_df["ca_dollar_volume_proxy"],
                bins=[-np.inf, 1e4, 1e5, 1e6, np.inf],
                labels=["micro", "small", "medium", "large"],
            ).astype("string")
        else:
            feature_df["ca_liquidity_bucket"] = pd.Series(pd.array([None] * len(feature_df), dtype="string"))

        feature_df = add_relative_features(feature_df, "cb", self.settings.benchmark_assets)
        if any(col.startswith("ca_") for col in feature_df.columns):
            feature_df = add_relative_features(feature_df, "ca", self.settings.benchmark_assets)
        else:
            for asset in self.settings.benchmark_assets:
                feature_df[f"ca_rel_to_{asset.lower()}_ret_1"] = np.nan
                feature_df[f"ca_rel_to_{asset.lower()}_ret_6"] = np.nan

        feature_df["cs_coinbase_vs_coinapi_close_diff"] = (feature_df["close"] - feature_df.get("ca_close", np.nan)) / feature_df["close"]
        feature_df["cs_coinbase_vs_coinapi_return_diff"] = feature_df["cb_ret_1"] - feature_df.get("ca_ret_1", np.nan)
        both_present = feature_df["cs_coinbase_vs_coinapi_close_diff"].notna() & feature_df["cs_coinbase_vs_coinapi_return_diff"].notna()
        divergence = (
            (feature_df["cs_coinbase_vs_coinapi_close_diff"].abs() >= self.settings.divergence_threshold)
            | (feature_df["cs_coinbase_vs_coinapi_return_diff"].abs() >= self.settings.divergence_threshold)
        )
        feature_df["cs_cross_source_divergence_state"] = np.where(both_present, divergence.astype(int), -1)
        feature_df["cs_cross_source_divergence_flag"] = np.where(both_present, divergence.astype(int), 0)
        feature_df["ex_min_size_constraint_flag"] = (
            feature_df["base_min_size"].fillna(0) * feature_df["close"].fillna(0) > self.settings.min_notional_reference_usd
        ).astype(int)
        feature_df["ex_increment_constraint_flag"] = (
            feature_df["price_increment"].fillna(0) / feature_df["close"].replace(0, np.nan) > 0.001
        ).fillna(False).astype(int)
        min_notional = feature_df["base_min_size"].fillna(0) * feature_df["close"].fillna(0)
        feature_df["ex_notional_bucket"] = pd.cut(
            min_notional,
            bins=[-np.inf, 1, 10, 50, np.inf],
            labels=["tiny", "low", "medium", "high"],
        ).astype("string")
        if any(col.startswith("ca_") for col in feature_df.columns):
            feature_df = add_context_features(feature_df)
        else:
            feature_df["cx_btc_regime_flag"] = 0
            feature_df["cx_eth_regime_flag"] = 0
        feature_df = feature_df.rename(columns={"close": "cb_close"})
        feature_df["feature_version"] = FEATURE_VERSION
        feature_df = feature_df.sort_values(["product_id", "ts"]).reset_index(drop=True)
        keep_columns = [
            "product_id",
            "base_asset",
            "quote_asset",
            "ts",
            "feature_version",
            "coinapi_symbol_id",
            "trades_count",
            *[c for c in feature_df.columns if c.startswith(("cb_", "ca_", "cx_", "cs_", "ex_"))],
        ]
        keep_columns = list(dict.fromkeys([c for c in keep_columns if c in feature_df.columns]))
        return feature_df[keep_columns]

    def _select_rules(self, request: LiveShadowRequest) -> list[dict[str, Any]]:
        library = self.rule_service._combined_library().get("candidate_rules", [])
        direct_rules = [rule for rule in library if rule.get("rule_kind", "direct_rule") == "direct_rule"]
        if request.selection_mode == "all":
            return direct_rules
        selected = set(request.rule_ids)
        if not selected:
            raise ValueError("Select at least one direct rule or use selection_mode='all'.")
        rules = [rule for rule in direct_rules if rule.get("merged_rule_id") in selected]
        if not rules:
            raise ValueError("No selected direct rules were found in the rule library.")
        return rules

    def _evaluate_snapshot(self, snapshot: pd.DataFrame, rules: list[dict[str, Any]], run_id: str, decision_time: datetime) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        prepared = self.rule_service._prepare_frame(snapshot)
        signal_rows: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for rule in rules:
            instances = self.rule_service._resolve_rule_variants(rule)
            for instance in instances:
                mask, resolved_conditions, missing = self.rule_service._build_rule_mask(prepared, instance.get("conditions", []))
                if mask is None:
                    skipped.append({
                        "rule_instance_id": instance["instance_id"],
                        "merged_rule_id": instance["merged_rule_id"],
                        "reason": f"missing_features:{','.join(sorted(set(missing)))}",
                    })
                    continue
                matched = prepared.loc[mask].copy()
                if matched.empty:
                    continue
                for _, row in matched.iterrows():
                    signal_id = f"{instance['instance_id']}::{row['product_id']}::{pd.Timestamp(row['ts']).isoformat()}"
                    payload = {
                        "signal_id": signal_id,
                        "run_id": run_id,
                        "decision_time": decision_time.isoformat(),
                        "signal_ts": pd.Timestamp(row["ts"]).isoformat(),
                        "product_id": row.get("product_id"),
                        "base_asset": row.get("base_asset"),
                        "quote_asset": row.get("quote_asset"),
                        "rule_instance_id": instance["instance_id"],
                        "merged_rule_id": instance["merged_rule_id"],
                        "rule_name": instance.get("name"),
                        "variant_id": instance.get("variant_id"),
                        "family_id": instance.get("family_id"),
                        "priority": instance.get("priority"),
                        "source_library": instance.get("source_library", "builtin"),
                        "source_attribution": "|".join(instance.get("source_attribution", [])),
                        "recommended_primary_horizon": instance.get("recommended_primary_horizon"),
                        "secondary_monitor_horizon": instance.get("secondary_monitor_horizon"),
                        "target_horizons": "|".join(instance.get("target_horizons", [])),
                        "resolved_conditions": json.dumps(resolved_conditions),
                        "signal_price": row.get("cb_close"),
                    }
                    for col, value in row.items():
                        if col in payload:
                            continue
                        payload[col] = value if not isinstance(value, pd.Timestamp) else value.isoformat()
                    signal_rows.append(payload)
        return pd.DataFrame(signal_rows), skipped

    def _existing_signal_log(self) -> pd.DataFrame:
        df = self.storage.read_frame(self.signal_log_name)
        if not df.empty and "signal_ts" in df.columns:
            df["signal_ts"] = pd.to_datetime(df["signal_ts"], utc=True)
        return df

    def _existing_outcomes(self) -> pd.DataFrame:
        df = self.storage.read_frame(self.outcomes_name)
        if not df.empty and "signal_ts" in df.columns:
            df["signal_ts"] = pd.to_datetime(df["signal_ts"], utc=True)
        return df

    def _append_signals(self, existing: pd.DataFrame, new_signals: pd.DataFrame) -> pd.DataFrame:
        base_columns = list(dict.fromkeys([*existing.columns.tolist(), *new_signals.columns.tolist(), 'signal_id', 'signal_ts', 'product_id', 'rule_instance_id', 'merged_rule_id', 'rule_name']))
        if existing.empty:
            combined = new_signals.copy()
        else:
            combined = pd.concat([existing, new_signals], ignore_index=True)
        if combined.empty:
            return pd.DataFrame(columns=base_columns)
        if "signal_ts" in combined.columns:
            combined["signal_ts"] = pd.to_datetime(combined["signal_ts"], utc=True)
        combined = combined.sort_values(["signal_ts", "product_id", "rule_instance_id"]).drop_duplicates(subset=["signal_id"], keep="last").reset_index(drop=True)
        return combined

    def _resolve_outcomes(self, signal_log: pd.DataFrame, existing_outcomes: pd.DataFrame, cb: pd.DataFrame, latest_available_ts: pd.Timestamp, resolved_at: datetime) -> tuple[pd.DataFrame, int]:
        cb = cb.copy()
        if existing_outcomes.empty:
            existing_outcomes = pd.DataFrame(columns=['signal_id','signal_ts','product_id','rule_instance_id','merged_rule_id','rule_name','signal_price','future_close_return_h1','future_close_return_h4','future_close_return_h24','future_max_up_pct_h1','future_max_up_pct_h4','future_max_up_pct_h24','touched_up_1pct_h4','touched_up_2pct_h24','resolved_at','fully_resolved'])
        if cb.empty or signal_log.empty:
            return existing_outcomes, 0
        cb["ts"] = pd.to_datetime(cb["ts"], utc=True)
        cb_groups = {pid: frame.sort_values("ts").reset_index(drop=True) for pid, frame in cb.groupby("product_id")}
        existing_map = existing_outcomes.set_index("signal_id").to_dict("index") if not existing_outcomes.empty else {}
        resolved_count = 0
        rows: list[dict[str, Any]] = []
        def _coerce_utc_timestamp(value: Any) -> pd.Timestamp:
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                return ts.tz_localize("UTC")
            return ts.tz_convert("UTC")

        for _, signal in signal_log.iterrows():
            signal_id = signal["signal_id"]
            row = existing_map.get(signal_id, {"signal_id": signal_id, "signal_ts": signal["signal_ts"], "product_id": signal["product_id"], "rule_instance_id": signal["rule_instance_id"], "merged_rule_id": signal["merged_rule_id"], "rule_name": signal.get("rule_name")})
            row.setdefault("signal_ts", signal["signal_ts"])
            row.setdefault("product_id", signal["product_id"])
            row.setdefault("rule_instance_id", signal["rule_instance_id"])
            row.setdefault("merged_rule_id", signal["merged_rule_id"])
            row.setdefault("rule_name", signal.get("rule_name"))
            row.setdefault("signal_price", signal.get("signal_price"))
            bars = cb_groups.get(signal["product_id"])
            if bars is None or bars.empty:
                rows.append(row)
                continue
            signal_ts = _coerce_utc_timestamp(signal["signal_ts"])
            entry_price = float(signal.get("signal_price") or np.nan)
            if not np.isfinite(entry_price) or entry_price <= 0:
                rows.append(row)
                continue
            for horizon in (1, 4, 24):
                close_col = f"future_close_return_h{horizon}"
                max_col = f"future_max_up_pct_h{horizon}"
                touch_col = {1: None, 4: "touched_up_1pct_h4", 24: "touched_up_2pct_h24"}[horizon]
                target_ts = signal_ts + timedelta(hours=horizon)
                if latest_available_ts < target_ts:
                    continue
                target_rows = bars.loc[bars["ts"] == target_ts]
                future_rows = bars.loc[(bars["ts"] > signal_ts) & (bars["ts"] <= target_ts)]
                if target_rows.empty or future_rows.empty:
                    continue
                row[close_col] = float(target_rows.iloc[0]["close"] / entry_price - 1)
                row[max_col] = float(future_rows["high"].max() / entry_price - 1)
                if touch_col:
                    threshold = 0.01 if horizon == 4 else 0.02
                    row[touch_col] = float(row[max_col] >= threshold)
            before = existing_map.get(signal_id, {})
            metric_keys = ["future_close_return_h1", "future_close_return_h4", "future_close_return_h24", "future_max_up_pct_h1", "future_max_up_pct_h4", "future_max_up_pct_h24"]

            def _changed(key: str) -> bool:
                if key not in before:
                    return key in row and not pd.isna(row.get(key))
                before_value = before.get(key)
                row_value = row.get(key)
                if pd.isna(before_value) and pd.isna(row_value):
                    return False
                return before_value != row_value

            if any(_changed(key) for key in metric_keys):
                row["resolved_at"] = resolved_at.isoformat()
                resolved_count += 1
            row["fully_resolved"] = pd.notna(row.get("future_close_return_h24"))
            rows.append(row)
        outcomes = pd.DataFrame(rows)
        if not outcomes.empty and "signal_ts" in outcomes.columns:
            outcomes["signal_ts"] = pd.to_datetime(outcomes["signal_ts"], utc=True)
        return outcomes, resolved_count

    def _build_summary(self, signal_log: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
        if signal_log.empty:
            return pd.DataFrame(columns=["merged_rule_id", "rule_name", "signals", "pending_signals", "distinct_products", "largest_product_share", "h1_mean_forward_return", "h4_mean_forward_return", "h24_mean_forward_return"])
        if outcomes.empty:
            outcomes = pd.DataFrame(columns=["signal_id", "product_id", "rule_instance_id", "merged_rule_id", "rule_name", "signal_ts", "fully_resolved"])
        merged = signal_log.merge(outcomes, on=["signal_id", "product_id", "rule_instance_id", "merged_rule_id", "rule_name", "signal_ts"], how="left", suffixes=("", "_out"))
        rows: list[dict[str, Any]] = []
        for (rule_id, rule_name), group in merged.groupby(["merged_rule_id", "rule_name"], dropna=False):
            counts = group["product_id"].value_counts(dropna=False)
            rows.append({
                "merged_rule_id": rule_id,
                "rule_name": rule_name,
                "signals": int(len(group)),
                "pending_signals": int(group["fully_resolved"].fillna(False).eq(False).sum()),
                "distinct_products": int(group["product_id"].nunique(dropna=False)),
                "largest_product_share": float(counts.iloc[0] / len(group)) if len(counts) else None,
                "h1_mean_forward_return": float(group["future_close_return_h1"].mean()) if "future_close_return_h1" in group.columns else None,
                "h4_mean_forward_return": float(group["future_close_return_h4"].mean()) if "future_close_return_h4" in group.columns else None,
                "h24_mean_forward_return": float(group["future_close_return_h24"].mean()) if "future_close_return_h24" in group.columns else None,
                "h1_mean_max_up_pct": float(group["future_max_up_pct_h1"].mean()) if "future_max_up_pct_h1" in group.columns else None,
                "h4_mean_max_up_pct": float(group["future_max_up_pct_h4"].mean()) if "future_max_up_pct_h4" in group.columns else None,
                "h24_mean_max_up_pct": float(group["future_max_up_pct_h24"].mean()) if "future_max_up_pct_h24" in group.columns else None,
                "h4_touch_rate": float(group["touched_up_1pct_h4"].mean()) if "touched_up_1pct_h4" in group.columns else None,
                "h24_touch_rate": float(group["touched_up_2pct_h24"].mean()) if "touched_up_2pct_h24" in group.columns else None,
            })
        return pd.DataFrame(rows).sort_values(["signals", "merged_rule_id"], ascending=[False, True])

    def run_cycle(self, request: LiveShadowRequest, run_id_override: str | None = None) -> dict[str, Any]:
        step = "live_shadow_cycle"
        run_id = run_id_override or self.storage.make_run_id()
        as_of = self._parse_as_of(request.as_of_time_iso)
        lookback_hours = int(request.lookback_hours or self.settings.live_shadow_lookback_hours)
        max_products = int(request.max_products or self.settings.live_shadow_max_products)
        self.storage.update_status(step, "running", message="Running live shadow validation cycle", run_id=run_id, phase="initializing")
        try:
            rules = self._select_rules(request)
            products, mapping = self._ensure_reference_tables(request.refresh_references if request.refresh_references is not None else self.settings.live_shadow_auto_refresh_references)
            start = as_of - timedelta(hours=lookback_hours)
            end = as_of
            eligible, cb, ca, quotes = self._pull_live_bars(products, mapping, start, end, max_products=max_products, step=step)
            self.storage.update_status(step, "running", message="Computing live feature snapshot", phase="computing_features", coinbase_rows=int(len(cb)), coinapi_rows=int(len(ca)), quote_rows=int(len(quotes)))
            feature_df = self._compute_feature_table(products=eligible, cb=cb, ca=ca, quotes=quotes)
            if feature_df.empty:
                raise RuntimeError("Live feature snapshot is empty.")
            latest_ts = pd.to_datetime(feature_df["ts"], utc=True).max()
            snapshot = feature_df.loc[pd.to_datetime(feature_df["ts"], utc=True) == latest_ts].copy()
            self.storage.update_status(step, "running", message="Evaluating live rules", phase="evaluating_rules", latest_signal_ts=latest_ts.isoformat(), snapshot_rows=int(len(snapshot)))
            new_signals, skipped = self._evaluate_snapshot(snapshot, rules, run_id=run_id, decision_time=as_of)
            signal_log = self._append_signals(self._existing_signal_log(), new_signals)
            outcomes, resolved_updates = self._resolve_outcomes(signal_log, self._existing_outcomes(), cb=cb, latest_available_ts=pd.Timestamp(latest_ts), resolved_at=as_of)
            self.storage.write_frame(signal_log, self.signal_log_name)
            self.storage.write_frame(outcomes, self.outcomes_name)
            summary_df = self._build_summary(signal_log, outcomes)

            signal_path = self.storage.write_csv(signal_log, f"live_signal_log__{run_id}", compress=False)
            outcome_path = self.storage.write_csv(outcomes, f"live_signal_outcomes__{run_id}", compress=False)
            summary_path = self.storage.write_csv(summary_df, f"live_validation_summary__{run_id}", compress=False)
            pending = signal_log.merge(outcomes[["signal_id", "fully_resolved"]] if not outcomes.empty else pd.DataFrame(columns=["signal_id", "fully_resolved"]), on="signal_id", how="left") if not signal_log.empty else pd.DataFrame(columns=signal_log.columns.tolist() + ['fully_resolved'])
            pending = pending.loc[pending["fully_resolved"].fillna(False).eq(False)].copy()
            pending_path = self.storage.write_csv(pending, f"live_pending_signals__{run_id}", compress=False)
            manifest = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "app": self.settings.app_name,
                "version": self.settings.app_version,
                "request": {
                    "lookback_hours": lookback_hours,
                    "max_products": max_products,
                    "selection_mode": request.selection_mode,
                    "rule_ids": request.rule_ids,
                    "refresh_references": bool(request.refresh_references),
                    "as_of_time_iso": as_of.isoformat(),
                },
                "summary": {
                    "latest_signal_ts": latest_ts.isoformat(),
                    "snapshot_rows": int(len(snapshot)),
                    "signals_created_this_cycle": int(len(new_signals)),
                    "signals_total": int(len(signal_log)),
                    "pending_signals": int(len(pending)),
                    "resolved_updates_this_cycle": int(resolved_updates),
                    "resolved_signals_total": int(outcomes.get("fully_resolved", pd.Series(dtype=bool)).fillna(False).sum()) if not outcomes.empty else 0,
                    "rules_evaluated": int(len(rules)),
                    "skipped_rules": skipped,
                    "coinbase_rows": int(len(cb)),
                    "coinapi_rows": int(len(ca)),
                },
                "summary_rows": summary_df.head(50).to_dict(orient="records"),
            }
            manifest_path = self.storage.export_path(f"live_validation_manifest__{run_id}", ".json")
            self.storage.write_json(manifest, manifest_path)
            pack_path = self.storage.export_path(f"live_validation_pack__{run_id}", ".zip")
            with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for path in [signal_path, outcome_path, summary_path, pending_path, manifest_path]:
                    zf.write(path, arcname=Path(path).name)
            artifacts = [self.storage.file_info(path) for path in [signal_path, outcome_path, summary_path, pending_path, manifest_path, pack_path]]
            manifest["artifacts"] = artifacts
            self.storage.write_latest_live_shadow_manifest(manifest)
            self.storage.update_status(step, "completed", message="Live shadow validation cycle completed", run_id=run_id, latest_signal_ts=latest_ts.isoformat(), signals_created_this_cycle=int(len(new_signals)), pending_signals=int(len(pending)), resolved_updates_this_cycle=int(resolved_updates), rules_evaluated=int(len(rules)), pack_artifact=pack_path.name)
            return manifest
        except Exception as exc:
            self.storage.update_status(step, "failed", error=str(exc), traceback=traceback.format_exc())
            raise
