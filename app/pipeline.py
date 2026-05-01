from __future__ import annotations

import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from .clients import CoinbaseAdvancedClient, CoinAPIClient
from .features import (
    FEATURE_VERSION,
    add_context_features,
    add_family_features,
    add_future_outcomes,
    add_relative_features,
    build_provenance_dictionary,
)
from .settings import Settings
from .storage import StorageManager


class ResearchPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.storage = StorageManager(settings)
        self.coinbase = CoinbaseAdvancedClient(settings)
        self.coinapi = CoinAPIClient(settings)

    def refresh_universe(self) -> dict[str, Any]:
        step = "universe_refresh"
        self.storage.update_status(step, "running", message="Refreshing Coinbase universe")
        try:
            products = pd.DataFrame(self.coinbase.list_products())
            if products.empty:
                raise RuntimeError("No Coinbase products returned.")
            source_rows = int(len(products))
            if "product_id" in products.columns:
                products = products.drop_duplicates(subset=["product_id"], keep="first").copy()
            deduped_rows = int(len(products))
            products["quote_asset"] = products["quote_currency_id"].fillna(products["quote_display_symbol"])
            products["base_asset"] = products["base_currency_id"].fillna(products["base_display_symbol"])
            products["eligibility_flag"] = True
            products["inclusion_reason"] = "eligible_spot_pair"
            products["exclusion_reason"] = None
            products.loc[products["product_type"] != "SPOT", ["eligibility_flag", "exclusion_reason"]] = [False, "not_spot"]
            products.loc[~products["quote_asset"].isin(self.settings.quote_currencies), ["eligibility_flag", "exclusion_reason"]] = [False, "quote_filtered"]
            if "is_disabled" in products.columns:
                products.loc[products["is_disabled"].fillna(False), ["eligibility_flag", "exclusion_reason"]] = [False, "is_disabled"]
            if self.settings.strict_coinbase_tradability_filters:
                for col in ["trading_disabled", "view_only"]:
                    if col in products.columns:
                        products.loc[products[col].fillna(False), ["eligibility_flag", "exclusion_reason"]] = [False, col]
            if "approximate_quote_24h_volume" not in products.columns:
                products["approximate_quote_24h_volume"] = products.get("volume_24h")
            for col in [
                "price",
                "volume_24h",
                "approximate_quote_24h_volume",
                "base_increment",
                "quote_increment",
                "base_min_size",
                "quote_min_size",
                "base_max_size",
                "quote_max_size",
                "price_increment",
            ]:
                if col in products.columns:
                    products[col] = pd.to_numeric(products[col], errors="coerce")
            eligible = products.loc[products["eligibility_flag"]].copy()
            eligible = eligible.sort_values("approximate_quote_24h_volume", ascending=False)
            if self.settings.top_n_by_volume > 0:
                eligible = eligible.head(self.settings.top_n_by_volume)
            if self.settings.max_universe_size > 0:
                eligible = eligible.head(self.settings.max_universe_size)
            master = products.copy()
            excluded_mask = ~master["product_id"].isin(eligible["product_id"])
            master.loc[excluded_mask, "eligibility_flag"] = False
            master.loc[excluded_mask & master["exclusion_reason"].isna(), "exclusion_reason"] = "top_n_filter"
            columns = [
                "product_id",
                "base_asset",
                "quote_asset",
                "product_type",
                "status",
                "is_disabled",
                "trading_disabled",
                "view_only",
                "cancel_only",
                "limit_only",
                "post_only",
                "auction_mode",
                "price",
                "price_increment",
                "quote_increment",
                "base_increment",
                "base_min_size",
                "quote_min_size",
                "base_max_size",
                "quote_max_size",
                "display_name",
                "product_venue",
                "new_at",
                "volume_24h",
                "approximate_quote_24h_volume",
                "eligibility_flag",
                "inclusion_reason",
                "exclusion_reason",
            ]
            master = master[[c for c in columns if c in master.columns]].copy()
            self.storage.write_frame(master, "coinbase_products")
            report_path = self.storage.write_csv(master, "product_master", compress=False)
            reason_counts = master["exclusion_reason"].fillna("included").value_counts().to_dict()
            summary = {
                "rows": int(len(master)),
                "eligible_rows": int(master["eligibility_flag"].sum()),
                "source_rows": source_rows,
                "deduplicated_rows_removed": max(source_rows - deduped_rows, 0),
                "artifact": str(report_path.name),
                "mock_mode": self.coinbase.mock_mode,
                "exclusion_reason_counts": reason_counts,
                "strict_coinbase_tradability_filters": self.settings.strict_coinbase_tradability_filters,
            }
            self.storage.update_status(step, "completed", **summary)
            return summary
        except Exception as exc:  # pragma: no cover - exercised in failure only
            self.storage.update_status(step, "failed", error=str(exc), traceback=traceback.format_exc())
            raise

    def refresh_mappings(self) -> dict[str, Any]:
        step = "mapping_refresh"
        self.storage.update_status(step, "running", message="Refreshing CoinAPI symbol mappings")
        try:
            products = self.storage.read_frame("coinbase_products")
            if products.empty:
                raise RuntimeError("coinbase_products dataset is missing. Refresh universe first.")
            eligible = products.loc[products["eligibility_flag"]].copy()
            if "product_id" in eligible.columns:
                eligible = eligible.drop_duplicates(subset=["product_id"], keep="first").copy()
            eligible = eligible.reset_index(drop=True)
            eligible_rows = int(len(eligible))
            self.storage.update_status(
                step,
                "running",
                message="Loading CoinAPI symbol catalog",
                eligible_rows=eligible_rows,
            )
            symbols = pd.DataFrame(self.coinapi.list_symbols())
            if symbols.empty:
                raise RuntimeError("CoinAPI symbol catalog returned no rows.")

            # Reduce symbol catalog aggressively before matching so mapping does not repeatedly scan
            # the full CoinAPI catalog for every Coinbase product.
            spot_mask = symbols["symbol_type"].eq("SPOT") if "symbol_type" in symbols.columns else True
            filtered = symbols.loc[spot_mask].copy()
            base_assets = set(eligible["base_asset"].dropna().astype(str).tolist())
            quote_assets = set(eligible["quote_asset"].dropna().astype(str).tolist())
            if "asset_id_base" in filtered.columns:
                filtered = filtered.loc[filtered["asset_id_base"].isin(base_assets)].copy()
            if "asset_id_quote" in filtered.columns:
                filtered = filtered.loc[filtered["asset_id_quote"].isin(quote_assets)].copy()

            preferred = {ex: i for i, ex in enumerate(self.settings.preferred_coinapi_exchanges)}
            filtered["exchange_rank"] = filtered.get("exchange_id").apply(lambda x: preferred.get(x, 99))
            filtered["pair_key"] = (
                filtered.get("asset_id_base", pd.Series(index=filtered.index, dtype=str)).astype(str)
                + "|"
                + filtered.get("asset_id_quote", pd.Series(index=filtered.index, dtype=str)).astype(str)
            )
            filtered = filtered.sort_values(["pair_key", "exchange_rank", "exchange_id", "symbol_id"])
            best_by_pair = filtered.groupby("pair_key", as_index=False).first()
            best_lookup = best_by_pair.set_index("pair_key").to_dict("index") if not best_by_pair.empty else {}

            self.storage.update_status(
                step,
                "running",
                message="Matching Coinbase products to CoinAPI symbols",
                eligible_rows=eligible_rows,
                coinapi_symbol_rows=int(len(symbols)),
                filtered_symbol_rows=int(len(filtered)),
                matched_so_far=0,
            )

            rows: list[dict[str, Any]] = []
            for idx, product in eligible.iterrows():
                pair_key = f"{product['base_asset']}|{product['quote_asset']}"
                best = best_lookup.get(pair_key)
                if not best:
                    rows.append(
                        {
                            "coinbase_product_id": product["product_id"],
                            "coinapi_symbol_id": None,
                            "exchange_id": None,
                            "mapping_status": "unmapped",
                            "mapping_confidence": 0.0,
                            "notes": "No CoinAPI symbol matched base/quote pair",
                        }
                    )
                else:
                    exact_exchange = best.get("exchange_id") in self.settings.preferred_coinapi_exchanges
                    rows.append(
                        {
                            "coinbase_product_id": product["product_id"],
                            "coinapi_symbol_id": best.get("symbol_id"),
                            "exchange_id": best.get("exchange_id"),
                            "mapping_status": "mapped" if exact_exchange else "mapped_cross_exchange",
                            "mapping_confidence": 1.0 if exact_exchange else 0.6,
                            "notes": "Preferred exchange match" if exact_exchange else "Base/quote matched on non-preferred exchange",
                        }
                    )
                if (idx + 1) % 50 == 0 or (idx + 1) == eligible_rows:
                    self.storage.update_status(
                        step,
                        "running",
                        message="Matching Coinbase products to CoinAPI symbols",
                        eligible_rows=eligible_rows,
                        coinapi_symbol_rows=int(len(symbols)),
                        filtered_symbol_rows=int(len(filtered)),
                        matched_so_far=int(idx + 1),
                    )
            mapping = pd.DataFrame(rows)
            if not mapping.empty:
                mapping = mapping.sort_values(
                    ["coinbase_product_id", "mapping_confidence", "mapping_status", "coinapi_symbol_id"],
                    ascending=[True, False, True, True],
                    na_position="last",
                )
                mapping = mapping.drop_duplicates(subset=["coinbase_product_id"], keep="first").reset_index(drop=True)
            self.storage.write_frame(mapping, "coinapi_symbol_mapping")
            csv_path = self.storage.write_csv(mapping, "coinapi_mapping_report", compress=False)
            summary = {
                "rows": int(len(mapping)),
                "mapped_rows": int(mapping["mapping_status"].str.startswith("mapped").sum()),
                "unmapped_rows": int((mapping["mapping_status"] == "unmapped").sum()),
                "artifact": csv_path.name,
                "mock_mode": self.coinapi.mock_mode,
                "coinapi_symbol_rows": int(len(symbols)),
                "filtered_symbol_rows": int(len(filtered)),
            }
            self.storage.update_status(step, "completed", **summary)
            return summary
        except Exception as exc:  # pragma: no cover
            self.storage.update_status(step, "failed", error=str(exc), traceback=traceback.format_exc())
            raise

    def pull_data(self, lookback_hours: int | None = None, max_products: int | None = None) -> dict[str, Any]:
        step = "data_pull"
        self.storage.update_status(step, "running", message="Pulling historical Coinbase and CoinAPI data")
        try:
            products = self.storage.read_frame("coinbase_products")
            mapping = self.storage.read_frame("coinapi_symbol_mapping")
            if products.empty or mapping.empty:
                raise RuntimeError("Universe and mappings are required before pulling data.")
            eligible = products.loc[products["eligibility_flag"]].copy()
            if max_products:
                eligible = eligible.head(max_products)
            start = datetime.now(timezone.utc) - timedelta(hours=lookback_hours or self.settings.lookback_hours)
            end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

            cb_frames: list[pd.DataFrame] = []
            ca_frames: list[pd.DataFrame] = []
            quote_frames: list[pd.DataFrame] = []

            if not mapping.empty:
                mapping = mapping.sort_values(
                    ["coinbase_product_id", "mapping_confidence", "mapping_status", "coinapi_symbol_id"],
                    ascending=[True, False, True, True],
                    na_position="last",
                )
                mapping = mapping.drop_duplicates(subset=["coinbase_product_id"], keep="first").reset_index(drop=True)
            mapping_lookup = mapping.set_index("coinbase_product_id").to_dict("index")
            total = len(eligible)
            for idx, (_, product) in enumerate(eligible.iterrows(), start=1):
                product_id = product["product_id"]
                self.storage.update_status(step, "running", progress=f"{idx}/{total}", current_product=product_id)
                cb = self.coinbase.get_candles(product_id, start=start, end=end, granularity=self.settings.preferred_bar_granularity)
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
            coinbase_bars = pd.concat(cb_frames, ignore_index=True) if cb_frames else pd.DataFrame()
            coinapi_bars = pd.concat(ca_frames, ignore_index=True) if ca_frames else pd.DataFrame()
            coinapi_quotes = pd.concat(quote_frames, ignore_index=True) if quote_frames else pd.DataFrame()
            self.storage.write_frame(coinbase_bars, "coinbase_bars", processed=False)
            self.storage.write_frame(coinapi_bars, "coinapi_bars", processed=False)
            if not coinapi_quotes.empty:
                self.storage.write_frame(coinapi_quotes, "coinapi_quotes", processed=False)
            dq = self.build_data_quality_report(write_artifacts=True)
            effective_lookback_hours = int(lookback_hours or self.settings.lookback_hours)
            effective_max_products = int(max_products or len(eligible))
            summary = {
                "coinbase_rows": int(len(coinbase_bars)),
                "coinapi_rows": int(len(coinapi_bars)),
                "quote_rows": int(len(coinapi_quotes)),
                "data_quality_rows": int(len(dq)),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "effective_lookback_hours": effective_lookback_hours,
                "effective_max_products": effective_max_products,
            }
            self.storage.update_status(step, "completed", **summary)
            return summary
        except Exception as exc:  # pragma: no cover
            self.storage.update_status(step, "failed", error=str(exc), traceback=traceback.format_exc())
            raise

    def compute_features(self) -> dict[str, Any]:
        step = "feature_compute"
        self.storage.update_status(step, "running", message="Computing compact feature families")
        try:
            products = self.storage.read_frame("coinbase_products")
            cb = self.storage.read_frame("coinbase_bars", processed=False)
            ca = self.storage.read_frame("coinapi_bars", processed=False)
            quotes = self.storage.read_frame("coinapi_quotes", processed=False)
            if cb.empty:
                raise RuntimeError("coinbase_bars dataset is missing. Pull data first.")

            cb = cb.rename(columns={"product_id": "product_id"}).copy()
            cb["ts"] = pd.to_datetime(cb["ts"], utc=True)
            cb = cb.sort_values(["product_id", "ts"])
            cb_features = add_family_features(cb[["product_id", "ts", "open", "high", "low", "close", "volume"]], "cb", include_ema=True)
            cb_features = cb_features.rename(columns={c: c for c in cb_features.columns})

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
                # Preserve only the raw close as ca_close so cs_coinbase_vs_coinapi_close_diff
                # has a real second operand. The other raw OHLCV (open/high/low/volume) are not
                # used by any cross-source feature, so we let the ca_-prefix filter below drop
                # them — keeping them would just inflate the comparative report's
                # feature_increment section with trivially-redundant entries (~1.0 corr with cb_close).
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
            feature_df["cs_cross_source_divergence_flag"] = (
                (feature_df["cs_coinbase_vs_coinapi_close_diff"].abs() >= self.settings.divergence_threshold)
                | (feature_df["cs_coinbase_vs_coinapi_return_diff"].abs() >= self.settings.divergence_threshold)
            ).astype(int)

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
            feature_df = add_future_outcomes(
                feature_df.rename(columns={"cb_close": "close", "open": "open", "high": "high", "low": "low"}),
                horizons=[1, 4, 24],
            )
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
                *[c for c in feature_df.columns if c.startswith(("cb_", "ca_", "cx_", "cs_", "ex_", "future_", "touched_"))],
            ]
            keep_columns = list(dict.fromkeys([c for c in keep_columns if c in feature_df.columns]))
            feature_df = feature_df[keep_columns]
            self.storage.write_frame(feature_df, "feature_table")
            provenance = build_provenance_dictionary(feature_df.columns.tolist())
            self.storage.write_csv(provenance, "feature_provenance_dictionary", compress=False)
            dq = self.build_data_quality_report(write_artifacts=True, feature_df=feature_df)
            summary = {
                "rows": int(len(feature_df)),
                "columns": int(len(feature_df.columns)),
                "data_quality_rows": int(len(dq)),
            }
            self.storage.update_status(step, "completed", **summary)
            return summary
        except Exception as exc:  # pragma: no cover
            self.storage.update_status(step, "failed", error=str(exc), traceback=traceback.format_exc())
            raise

    def build_data_quality_report(self, write_artifacts: bool = False, feature_df: pd.DataFrame | None = None) -> pd.DataFrame:
        products = self.storage.read_frame("coinbase_products")
        mapping = self.storage.read_frame("coinapi_symbol_mapping")
        cb = self.storage.read_frame("coinbase_bars", processed=False)
        ca = self.storage.read_frame("coinapi_bars", processed=False)
        rows: list[dict[str, Any]] = []

        if not mapping.empty:
            unmapped = mapping.loc[mapping["mapping_status"] == "unmapped"]
            rows.append({"check_name": "missing_mappings", "severity": "warn", "object_id": "universe", "metric_value": float(len(unmapped)), "status": "warn" if len(unmapped) else "pass", "notes": ",".join(unmapped["coinbase_product_id"].astype(str).tolist())[:300]})
        if not cb.empty:
            dupes = cb.duplicated(["product_id", "ts"]).sum()
            rows.append({"check_name": "coinbase_duplicate_timestamps", "severity": "error", "object_id": "coinbase_bars", "metric_value": float(dupes), "status": "fail" if dupes else "pass", "notes": ""})
            non_positive = int(((cb[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum())
            rows.append({"check_name": "coinbase_non_positive_prices", "severity": "error", "object_id": "coinbase_bars", "metric_value": float(non_positive), "status": "fail" if non_positive else "pass", "notes": ""})
            tz_ok = int(pd.to_datetime(cb["ts"], utc=True).dt.tz is not None)
            rows.append({"check_name": "coinbase_timezone_consistency", "severity": "error", "object_id": "coinbase_bars", "metric_value": float(tz_ok), "status": "pass" if tz_ok else "fail", "notes": "UTC expected"})
        if not ca.empty:
            dupes = ca.duplicated(["coinbase_product_id", "ts"]).sum()
            rows.append({"check_name": "coinapi_duplicate_timestamps", "severity": "error", "object_id": "coinapi_bars", "metric_value": float(dupes), "status": "fail" if dupes else "pass", "notes": ""})
        if not cb.empty and not ca.empty:
            cb_counts = cb.groupby("product_id").size().rename("cb_rows")
            ca_counts = ca.groupby("coinbase_product_id").size().rename("ca_rows")
            aligned = cb_counts.to_frame().join(ca_counts, how="left").fillna(0)
            aligned["gap"] = aligned["cb_rows"] - aligned["ca_rows"]
            rows.append({"check_name": "bar_alignment_gap_total", "severity": "warn", "object_id": "alignment", "metric_value": float(aligned["gap"].abs().sum()), "status": "warn" if aligned["gap"].abs().sum() else "pass", "notes": ""})
        if not products.empty:
            if "approximate_quote_24h_volume" in products.columns:
                stale_series = pd.to_numeric(products["approximate_quote_24h_volume"], errors="coerce").fillna(0)
            else:
                stale_series = pd.Series(0, index=products.index, dtype="float64")
            stale = int((stale_series <= 0).sum())
            rows.append({"check_name": "stale_products_zero_volume", "severity": "warn", "object_id": "products", "metric_value": float(stale), "status": "warn" if stale else "pass", "notes": ""})
        if feature_df is not None and not feature_df.empty:
            null_frac = float(feature_df.isna().mean().mean())
            rows.append({"check_name": "feature_null_fraction_mean", "severity": "warn", "object_id": "feature_table", "metric_value": null_frac, "status": "warn" if null_frac > 0.15 else "pass", "notes": ""})
        dq = pd.DataFrame(rows)
        if write_artifacts:
            self.storage.write_csv(dq, "data_quality_report", compress=False)
        return dq

    def build_exports(self, compress_chatgpt_csv: bool = True) -> dict[str, Any]:
        step = "export_build"
        self.storage.update_status(step, "running", message="Building comparative exports")
        try:
            features = self.storage.read_frame("feature_table")
            if features.empty:
                raise RuntimeError("feature_table dataset is missing. Compute features first.")
            base_cols = ["product_id", "base_asset", "quote_asset", "ts", "feature_version", "coinapi_symbol_id", "trades_count"]
            cb_cols = [c for c in features.columns if c.startswith(("cb_", "ex_", "future_", "touched_"))]
            ca_cols = [c for c in features.columns if c.startswith(("ca_", "cs_"))]
            cx_cols = [c for c in features.columns if c.startswith("cx_")]

            coinbase_only = features[base_cols + cb_cols].copy()
            plus_coinapi = features[base_cols + cb_cols + ca_cols].copy()
            full_scope = features[base_cols + cb_cols + ca_cols + cx_cols].copy()
            self.storage.write_frame(coinbase_only, "coinbase_only_features")
            self.storage.write_frame(plus_coinapi, "coinbase_plus_coinapi_features")
            self.storage.write_frame(full_scope, "feature_table")
            chatgpt_path = self.storage.write_csv(full_scope, "chatgpt_ready_features", compress=compress_chatgpt_csv)
            products = self.storage.read_frame("coinbase_products")
            mapping = self.storage.read_frame("coinapi_symbol_mapping")
            if not products.empty:
                self.storage.write_frame(products, "coinbase_products")
            if not mapping.empty:
                self.storage.write_csv(mapping, "coinapi_mapping_report", compress=False)
            report = self.build_comparative_insight_report(write_artifacts=True)
            run_id = self.storage.make_run_id()
            versioned_artifacts = []
            for name, suffix in [
                ("chatgpt_ready_features", ".csv.gz" if compress_chatgpt_csv else ".csv"),
                ("comparative_insight_report", ".csv"),
                ("data_quality_report", ".csv"),
                ("feature_provenance_dictionary", ".csv"),
                ("coinapi_mapping_report", ".csv"),
                ("product_master", ".csv"),
            ]:
                snap = self.storage.snapshot_export(name, suffix, run_id)
                if snap is not None:
                    versioned_artifacts.append(self.storage.file_info(snap))

            status_payload = self.storage.read_json(self.storage.status_path)
            data_pull_step = status_payload.get("steps", {}).get("data_pull", {}) if isinstance(status_payload, dict) else {}
            effective_run_settings = {
                "lookback_hours": data_pull_step.get("effective_lookback_hours", self.settings.lookback_hours),
                "max_products": data_pull_step.get("effective_max_products", self.settings.top_n_by_volume),
                "quote_currencies": self.settings.quote_currencies,
                "preferred_bar_granularity": self.settings.preferred_bar_granularity,
                "coinapi_period_id": self.settings.coinapi_period_id,
                "compress_chatgpt_csv": compress_chatgpt_csv,
            }
            summary = {
                "run_id": run_id,
                "coinbase_only_rows": int(len(coinbase_only)),
                "coinbase_plus_coinapi_rows": int(len(plus_coinapi)),
                "full_scope_rows": int(len(full_scope)),
                "comparative_report_rows": int(len(report)),
                "chatgpt_ready_artifact": chatgpt_path.name,
                "effective_run_settings": effective_run_settings,
            }
            run_summary_payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "app": self.settings.app_name,
                "version": self.settings.app_version,
                "effective_run_settings": effective_run_settings,
                "summary": summary,
                "artifacts": versioned_artifacts,
            }
            self.storage.write_json(run_summary_payload)
            summary_snapshot = self.storage.snapshot_export("run_summary", ".json", run_id)
            if summary_snapshot is not None:
                versioned_artifacts.append(self.storage.file_info(summary_snapshot))
                run_summary_payload["artifacts"] = versioned_artifacts
                self.storage.write_json(run_summary_payload)
            self.storage.write_latest_manifest(run_summary_payload)
            self.storage.update_status(step, "completed", **summary)
            return summary
        except Exception as exc:  # pragma: no cover
            self.storage.update_status(step, "failed", error=str(exc), traceback=traceback.format_exc())
            raise

    def build_comparative_insight_report(self, write_artifacts: bool = False) -> pd.DataFrame:
        features = self.storage.read_frame("feature_table")
        if features.empty:
            return pd.DataFrame()
        scope_map = {
            "coinbase_only": [c for c in features.columns if c.startswith(("cb_", "ex_", "future_", "touched_"))],
            "coinbase_plus_coinapi": [c for c in features.columns if c.startswith(("cb_", "ca_", "cs_", "ex_", "future_", "touched_"))],
            "full_scope": [c for c in features.columns if c.startswith(("cb_", "ca_", "cs_", "cx_", "ex_", "future_", "touched_"))],
        }
        rows: list[dict[str, Any]] = []
        common_rule = (
            (features["cb_ret_3"] > 0)
            & (features["cb_rel_volume_short"] > 1)
            & (features["cb_sma_5_dist"] > -0.01)
        )
        for scope, cols in scope_map.items():
            selected_cols = list(dict.fromkeys(["product_id", "ts", self.settings.rule_target_column, "touched_up_1pct_h4", *cols]))
            scope_df = features[selected_cols].copy()
            completeness = 1.0 - float(scope_df[cols].isna().mean().mean()) if cols else 0.0
            rows.append(
                {
                    "section": "scope_summary",
                    "scope": scope,
                    "metric": "row_count",
                    "value": float(len(scope_df)),
                    "notes": "",
                }
            )
            rows.append(
                {
                    "section": "scope_summary",
                    "scope": scope,
                    "metric": "feature_count",
                    "value": float(len(cols)),
                    "notes": "",
                }
            )
            rows.append(
                {
                    "section": "scope_summary",
                    "scope": scope,
                    "metric": "feature_completeness_pct",
                    "value": completeness,
                    "notes": "",
                }
            )
            matched = scope_df.loc[common_rule]
            rows.append(
                {
                    "section": "rule_template_comparison",
                    "scope": scope,
                    "metric": "common_cb_rule_match_count",
                    "value": float(len(matched)),
                    "notes": "cb_ret_3>0 & cb_rel_volume_short>1 & cb_sma_5_dist>-0.01",
                }
            )
            rows.append(
                {
                    "section": "rule_template_comparison",
                    "scope": scope,
                    "metric": f"common_cb_rule_mean_{self.settings.rule_target_column}",
                    "value": float(matched[self.settings.rule_target_column].mean()) if len(matched) else np.nan,
                    "notes": "",
                }
            )
            rows.append(
                {
                    "section": "scope_summary",
                    "scope": scope,
                    "metric": "overall_touch_rate_h4_1pct",
                    "value": float(scope_df["touched_up_1pct_h4"].mean()),
                    "notes": "",
                }
            )
        target = self.settings.rule_target_column
        numeric = features.select_dtypes(include=[np.number]).copy()
        for feature in [c for c in numeric.columns if c.startswith(("ca_", "cs_", "cx_"))]:
            if numeric[feature].notna().sum() < 20 or numeric[target].notna().sum() < 20:
                continue
            target_corr = numeric[[feature, target]].corr().iloc[0, 1]
            cb_candidates = [c for c in numeric.columns if c.startswith("cb_")]
            best_cb = None
            best_corr = -1.0
            for cb_feature in cb_candidates:
                valid = numeric[[feature, cb_feature]].dropna()
                if len(valid) < 20:
                    continue
                corr = abs(valid.corr().iloc[0, 1])
                if corr > best_corr:
                    best_corr = corr
                    best_cb = cb_feature
            novelty = abs(target_corr) * (1 - max(best_corr, 0)) if pd.notna(target_corr) else np.nan
            rows.append(
                {
                    "section": "feature_increment",
                    "scope": "coinapi_enrichment",
                    "metric": feature,
                    "value": float(target_corr) if pd.notna(target_corr) else np.nan,
                    "notes": f"best_cb_feature={best_cb}; redundancy_abs_corr={best_corr:.4f}; novelty_score={novelty:.6f}",
                }
            )
        report = pd.DataFrame(rows)
        if write_artifacts:
            self.storage.write_csv(report, "comparative_insight_report", compress=False)
        return report
