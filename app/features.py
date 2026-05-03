from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

FEATURE_VERSION = "v1.3.0"


def safe_div(a, b):
    if isinstance(b, pd.Series):
        b = b.replace(0, np.nan)
    elif b == 0:
        b = np.nan
    out = a / b
    if isinstance(out, pd.Series):
        return out.replace([np.inf, -np.inf], np.nan)
    if np.isinf(out):
        return np.nan
    return out


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    def pct(values: np.ndarray) -> float:
        arr = pd.Series(values)
        if arr.isna().all():
            return np.nan
        return float(arr.rank(pct=True).iloc[-1])

    return series.rolling(window, min_periods=max(3, window // 3)).apply(pct, raw=True)


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        x = np.arange(len(values), dtype=float)
        x_centered = x - x.mean()
        y = values.astype(float)
        y_centered = y - y.mean()
        denom = float((x_centered**2).sum())
        if denom == 0 or float(y.mean()) == 0:
            return np.nan
        return float((x_centered * y_centered).sum() / denom) / float(y.mean())

    return series.rolling(window, min_periods=max(3, window // 2)).apply(slope, raw=True)


def _add_group_features(group: pd.DataFrame, prefix: str, include_ema: bool) -> pd.DataFrame:
    g = group.sort_values("ts").copy()
    close = g["close"]
    high = g["high"]
    low = g["low"]
    open_ = g["open"]
    volume = g["volume"]
    prev_close = close.shift(1)

    g[f"{prefix}_ret_1"] = close.pct_change(1)
    for horizon in [3, 6, 12, 24]:
        g[f"{prefix}_ret_{horizon}"] = close.pct_change(horizon)

    g[f"{prefix}_intrabar_range_pct"] = safe_div(high - low, open_)
    g[f"{prefix}_close_location_in_bar"] = safe_div(close - low, high - low)
    g[f"{prefix}_breakout_distance_short"] = safe_div(close, high.shift(1).rolling(6, min_periods=3).max()) - 1
    g[f"{prefix}_breakout_distance_medium"] = safe_div(close, high.shift(1).rolling(24, min_periods=6).max()) - 1

    for window in [5, 10, 20]:
        sma = close.rolling(window, min_periods=max(2, window // 2)).mean()
        g[f"{prefix}_sma_{window}_dist"] = safe_div(close, sma) - 1

    if include_ema:
        for span in [12, 26]:
            ema = close.ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()
            g[f"{prefix}_ema_{span}_dist"] = safe_div(close, ema) - 1

    g[f"{prefix}_rolling_slope_short"] = rolling_slope(close, 6)
    g[f"{prefix}_rolling_slope_medium"] = rolling_slope(close, 24)

    true_range = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    g[f"{prefix}_atr_like_short"] = safe_div(true_range.rolling(6, min_periods=3).mean(), close)
    g[f"{prefix}_realized_vol_short"] = g[f"{prefix}_ret_1"].rolling(6, min_periods=3).std()
    g[f"{prefix}_realized_vol_medium"] = g[f"{prefix}_ret_1"].rolling(24, min_periods=6).std()
    g[f"{prefix}_volatility_percentile"] = rolling_percentile(g[f"{prefix}_realized_vol_short"], 100)

    vol_mean_6 = volume.rolling(6, min_periods=3).mean()
    vol_std_6 = volume.rolling(6, min_periods=3).std()
    g[f"{prefix}_volume_zscore_short"] = safe_div(volume - vol_mean_6, vol_std_6)
    g[f"{prefix}_rel_volume_short"] = safe_div(volume, vol_mean_6)
    g[f"{prefix}_rel_volume_medium"] = safe_div(volume, volume.rolling(24, min_periods=6).mean())
    g[f"{prefix}_dollar_volume_proxy"] = close * volume
    return g


def add_family_features(df: pd.DataFrame, prefix: str, include_ema: bool = False) -> pd.DataFrame:
    return (
        df.groupby("product_id", group_keys=False)
        .apply(
            lambda g: _add_group_features(g.assign(product_id=g.name), prefix=prefix, include_ema=include_ema),
            include_groups=False,
        )
        .reset_index(drop=True)
    )


def add_relative_features(feature_df: pd.DataFrame, prefix: str, benchmarks: Iterable[str]) -> pd.DataFrame:
    out = feature_df.copy()
    for benchmark in benchmarks:
        bench = out.loc[out["base_asset"] == benchmark, ["ts", f"{prefix}_ret_1", f"{prefix}_ret_6"]].rename(
            columns={
                f"{prefix}_ret_1": f"_bench_{benchmark}_ret_1",
                f"{prefix}_ret_6": f"_bench_{benchmark}_ret_6",
            }
        )
        # Deduplicate bench by ts to prevent a Cartesian blow-up when the universe
        # contains multiple products sharing a base asset (e.g. BTC-USD + BTC-USDC
        # under a multi-quote-currency config). Without this guard, the merge
        # multiplies row count by the number of benchmark-quote variants per ts,
        # and the effect compounds across benchmarks (8x with 2 benchmarks × 2 quotes).
        bench = bench.drop_duplicates(subset=["ts"], keep="first")
        out = out.merge(bench, on="ts", how="left")
        out[f"{prefix}_rel_to_{benchmark.lower()}_ret_1"] = out[f"{prefix}_ret_1"] - out[f"_bench_{benchmark}_ret_1"]
        out[f"{prefix}_rel_to_{benchmark.lower()}_ret_6"] = out[f"{prefix}_ret_6"] - out[f"_bench_{benchmark}_ret_6"]
        out = out.drop(columns=[f"_bench_{benchmark}_ret_1", f"_bench_{benchmark}_ret_6"])
    return out


def add_context_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    out = feature_df.copy()
    for asset in ["BTC", "ETH"]:
        asset_rows = out.loc[out["base_asset"] == asset, ["ts", "ca_ret_24", "ca_sma_20_dist"]].rename(
            columns={"ca_ret_24": f"_{asset.lower()}_ret_24", "ca_sma_20_dist": f"_{asset.lower()}_sma_20_dist"}
        )
        # See comment in add_relative_features: dedup by ts to avoid Cartesian
        # blow-up under a multi-quote-currency universe.
        asset_rows = asset_rows.drop_duplicates(subset=["ts"], keep="first")
        out = out.merge(asset_rows, on="ts", how="left")
        out[f"cx_{asset.lower()}_regime_flag"] = np.select(
            [
                (out[f"_{asset.lower()}_ret_24"] > 0) & (out[f"_{asset.lower()}_sma_20_dist"] > 0),
                (out[f"_{asset.lower()}_ret_24"] < 0) & (out[f"_{asset.lower()}_sma_20_dist"] < 0),
            ],
            [1, -1],
            default=0,
        )
        out = out.drop(columns=[f"_{asset.lower()}_ret_24", f"_{asset.lower()}_sma_20_dist"])
    return out


def add_future_outcomes(df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    out = df.copy()
    frames = []
    for _, group in out.groupby("product_id", group_keys=False):
        g = group.sort_values("ts").copy()
        closes = g["close"].to_numpy(dtype=float)
        highs = g["high"].to_numpy(dtype=float)
        group_len = len(g)
        for horizon in horizons:
            max_future_highs = np.full(group_len, np.nan, dtype=float)
            future_close_returns = np.full(group_len, np.nan, dtype=float)
            for i in range(group_len):
                end_idx = i + horizon
                if end_idx >= group_len:
                    continue
                future_window = slice(i + 1, end_idx + 1)
                max_future_highs[i] = np.nanmax(highs[future_window]) / closes[i] - 1
                future_close_returns[i] = closes[end_idx] / closes[i] - 1
            g[f"future_max_up_pct_h{horizon}"] = max_future_highs
            g[f"future_close_return_h{horizon}"] = future_close_returns
        frames.append(g)
    out = pd.concat(frames, ignore_index=True) if frames else out
    future_h4 = out.get("future_max_up_pct_h4", pd.Series(np.nan, index=out.index, dtype=float))
    future_h24 = out.get("future_max_up_pct_h24", pd.Series(np.nan, index=out.index, dtype=float))
    out["touched_up_1pct_h4"] = np.where(future_h4.notna(), (future_h4 >= 0.01).astype(float), np.nan)
    out["touched_up_2pct_h24"] = np.where(future_h24.notna(), (future_h24 >= 0.02).astype(float), np.nan)
    return out


def build_provenance_dictionary(columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column.startswith("cb_"):
            family = "coinbase_native"
            scope = "A,B,C"
        elif column.startswith(("ca_", "cs_")):
            family = "coinapi_coinbase_mapped"
            scope = "B,C"
        elif column.startswith("cx_"):
            family = "coinapi_cross_venue_context"
            scope = "C"
        elif column.startswith("ex_"):
            family = "execution_relevance"
            scope = "A,B,C"
        elif column.startswith(("future_", "touched_")):
            family = "offline_outcomes"
            scope = "A,B,C"
        else:
            family = "metadata"
            scope = "A,B,C"
        rows.append(
            {
                "feature_name": column,
                "feature_family": family,
                "available_scopes": scope,
                "feature_version": FEATURE_VERSION,
            }
        )
    return pd.DataFrame(rows)
