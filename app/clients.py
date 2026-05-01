from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
import numpy as np
import pandas as pd
import requests
from cryptography.hazmat.primitives import serialization

from .settings import Settings


class APIClientError(RuntimeError):
    pass


@dataclass
class CoinbaseProduct:
    product_id: str
    base_asset: str
    quote_asset: str
    product_type: str
    volume_24h: float
    approximate_quote_24h_volume: float
    price: float
    base_increment: float
    quote_increment: float
    base_min_size: float
    quote_min_size: float
    base_max_size: float
    quote_max_size: float
    display_name: str
    status: str
    is_disabled: bool
    trading_disabled: bool
    cancel_only: bool
    limit_only: bool
    post_only: bool
    auction_mode: bool
    view_only: bool
    product_venue: str
    new_at: str | None


class CoinbaseAdvancedClient:
    base_url = "https://api.coinbase.com"
    request_host = "api.coinbase.com"

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def mock_mode(self) -> bool:
        return self.settings.use_mock_data or not (
            self.settings.coinbase_api_key_name and self.settings.coinbase_api_private_key
        )

    def _build_headers(self, method: str, path: str) -> dict[str, str]:
        if self.settings.coinbase_bearer_token:
            return {"Authorization": f"Bearer {self.settings.coinbase_bearer_token}"}

        key_name = self.settings.coinbase_api_key_name
        key_secret = self.settings.coinbase_api_private_key
        if not key_name or not key_secret:
            raise APIClientError("Coinbase credentials not configured.")

        private_key = serialization.load_pem_private_key(key_secret.encode("utf-8"), password=None)
        payload = {
            "sub": key_name,
            "iss": "cdp",
            "nbf": int(time.time()),
            "exp": int(time.time()) + 120,
            "uri": f"{method.upper()} {self.request_host}{path}",
        }
        token = jwt.encode(
            payload,
            private_key,
            algorithm="ES256",
            headers={"kid": key_name, "nonce": secrets.token_hex()},
        )
        return {"Authorization": f"Bearer {token}"}

    def _request(self, method: str, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = requests.request(method, url, headers=self._build_headers(method, path), params=params, timeout=60)
        if resp.status_code >= 400:
            raise APIClientError(f"Coinbase API error {resp.status_code}: {resp.text[:400]}")
        return resp.json()

    def list_products(self) -> list[dict[str, Any]]:
        if self.mock_mode:
            return self._mock_products()
        path = "/api/v3/brokerage/products"
        cursor = None
        out: list[dict[str, Any]] = []
        while True:
            params = {
                "product_type": "SPOT",
                "get_tradability_status": "true",
                "products_sort_order": "PRODUCTS_SORT_ORDER_VOLUME_24H_DESCENDING",
                "limit": 250,
            }
            if cursor:
                params["cursor"] = cursor
            payload = self._request("GET", path, params=params)
            out.extend(payload.get("products", []))
            cursor = payload.get("pagination", {}).get("next_cursor")
            if not payload.get("pagination", {}).get("has_next"):
                break
        return out

    def get_candles(
        self,
        product_id: str,
        start: datetime,
        end: datetime,
        granularity: str,
    ) -> pd.DataFrame:
        if self.mock_mode:
            return self._mock_bars(provider="coinbase", symbol=product_id, start=start, end=end)

        path = f"/api/v3/brokerage/products/{product_id}/candles"
        max_bars = 350
        step_seconds = {
            "ONE_MINUTE": 60,
            "FIVE_MINUTE": 300,
            "FIFTEEN_MINUTE": 900,
            "THIRTY_MINUTE": 1800,
            "ONE_HOUR": 3600,
            "TWO_HOUR": 7200,
            "FOUR_HOUR": 14400,
            "SIX_HOUR": 21600,
            "ONE_DAY": 86400,
        }[granularity]
        chunk_seconds = max_bars * step_seconds
        rows: list[dict[str, Any]] = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(seconds=chunk_seconds), end)
            params = {
                "start": int(cursor.timestamp()),
                "end": int(chunk_end.timestamp()),
                "granularity": granularity,
                "limit": max_bars,
            }
            payload = self._request("GET", path, params=params)
            rows.extend(payload.get("candles", []))
            cursor = chunk_end
        if not rows:
            return pd.DataFrame(columns=["product_id", "ts", "open", "high", "low", "close", "volume", "bar_granularity"])
        df = pd.DataFrame(rows)
        df = df.rename(columns={"start": "ts"})
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="s", utc=True)
        df["product_id"] = product_id
        df["bar_granularity"] = granularity
        return df[["product_id", "ts", "open", "high", "low", "close", "volume", "bar_granularity"]].sort_values("ts")

    def _mock_products(self) -> list[dict[str, Any]]:
        rows = []
        product_specs = [
            ("BTC-USD", "BTC", 600000000, 95000.0),
            ("ETH-USD", "ETH", 300000000, 4500.0),
            ("SOL-USD", "SOL", 120000000, 190.0),
            ("ADA-USD", "ADA", 80000000, 1.05),
            ("DOGE-USD", "DOGE", 70000000, 0.22),
            ("XRP-USD", "XRP", 65000000, 0.85),
        ]
        for idx, (product_id, base, vol, price) in enumerate(product_specs):
            rows.append(
                {
                    "product_id": product_id,
                    "price": f"{price}",
                    "volume_24h": f"{vol / max(price, 0.1):.4f}",
                    "approximate_quote_24h_volume": f"{vol}",
                    "base_increment": "0.00000001",
                    "quote_increment": "0.0001" if price < 10 else "0.01",
                    "quote_min_size": "1",
                    "quote_max_size": "1000000",
                    "base_min_size": "0.0001" if price > 100 else "1",
                    "base_max_size": "1000000",
                    "base_name": base,
                    "quote_name": "US Dollar",
                    "is_disabled": False,
                    "status": "online",
                    "cancel_only": False,
                    "limit_only": False,
                    "post_only": False,
                    "trading_disabled": False,
                    "auction_mode": False,
                    "base_display_symbol": base,
                    "quote_display_symbol": "USD",
                    "product_type": "SPOT",
                    "quote_currency_id": "USD",
                    "base_currency_id": base,
                    "view_only": False,
                    "price_increment": "0.0001" if price < 10 else "0.01",
                    "display_name": product_id,
                    "product_venue": "CBE",
                    "new_at": (datetime.now(timezone.utc) - timedelta(days=idx + 30)).isoformat(),
                }
            )
        return rows

    def _mock_bars(self, provider: str, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        rng = pd.date_range(start=start, end=end, freq="1h", inclusive="left", tz="UTC")
        seed = sum(ord(x) for x in f"{provider}:{symbol}") % (2**32 - 1)
        rs = np.random.default_rng(seed)
        base_price = {
            "BTC": 95000.0,
            "ETH": 4500.0,
            "SOL": 190.0,
            "ADA": 1.05,
            "DOGE": 0.22,
            "XRP": 0.85,
        }[symbol.split("-")[0] if "-" in symbol else symbol.split("_")[-2]]
        drift = 0.0008 if provider == "coinapi" else 0.0006
        noise = rs.normal(drift, 0.02, size=len(rng))
        price = base_price * np.exp(np.cumsum(noise / 6.0))
        close = pd.Series(price, index=rng)
        open_ = close.shift(1).fillna(close.iloc[0] * (1 - 0.002))
        high = pd.concat([open_, close], axis=1).max(axis=1) * (1 + rs.uniform(0.0005, 0.01, size=len(rng)))
        low = pd.concat([open_, close], axis=1).min(axis=1) * (1 - rs.uniform(0.0005, 0.01, size=len(rng)))
        volume = np.abs(rs.normal(1.0, 0.25, size=len(rng))) * (1500 if base_price < 10 else 150)
        if provider == "coinapi":
            close = close * (1 + rs.normal(0.0, 0.0015, size=len(rng)))
        df = pd.DataFrame(
            {
                "ts": rng,
                "open": open_.values,
                "high": high.values,
                "low": low.values,
                "close": close.values,
                "volume": volume,
            }
        )
        if provider == "coinbase":
            df["product_id"] = symbol
            df["bar_granularity"] = "ONE_HOUR"
            return df[["product_id", "ts", "open", "high", "low", "close", "volume", "bar_granularity"]]
        df["coinbase_product_id"] = symbol.replace("_", "-") if "_" not in symbol else symbol
        return df


class CoinAPIClient:
    base_url = "https://rest.coinapi.io"

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def mock_mode(self) -> bool:
        return self.settings.use_mock_data or not self.settings.coinapi_api_key

    @staticmethod
    def _format_time(dt: datetime) -> str:
        """CoinAPI accepts ISO 8601 query timestamps best as whole-second UTC without a trailing offset."""
        return dt.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%S")

    def _request(self, path: str, params: dict[str, Any] | None = None) -> Any:
        api_key = self.settings.coinapi_api_key or ""
        headers = {"X-CoinAPI-Key": api_key, "Authorization": api_key}
        url = f"{self.base_url}{path}"
        resp = requests.get(url, headers=headers, params=params, timeout=60)
        if resp.status_code >= 400:
            raise APIClientError(f"CoinAPI error {resp.status_code}: {resp.text[:400]}")
        return resp.json()

    def list_symbols(self) -> list[dict[str, Any]]:
        if self.mock_mode:
            return self._mock_symbols()
        return self._request("/v1/symbols")

    def get_ohlcv(self, symbol_id: str, start: datetime, end: datetime, period_id: str) -> pd.DataFrame:
        if self.mock_mode:
            product_id = symbol_id.split("_SPOT_")[-1].replace("_", "-")
            df = CoinbaseAdvancedClient(self.settings)._mock_bars("coinapi", product_id, start, end)
            df["coinapi_symbol_id"] = symbol_id
            df["coinbase_product_id"] = product_id
            df["bar_granularity"] = period_id
            df["trades_count"] = np.linspace(100, 1000, len(df)).astype(int)
            return df[[
                "coinbase_product_id",
                "coinapi_symbol_id",
                "ts",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "trades_count",
                "bar_granularity",
            ]]
        payload = self._request(
            f"/v1/ohlcv/{symbol_id}/history",
            params={
                "period_id": period_id,
                "time_start": self._format_time(start),
                "time_end": self._format_time(end),
                "limit": 100000,
            },
        )
        df = pd.DataFrame(payload)
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "coinbase_product_id",
                    "coinapi_symbol_id",
                    "ts",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "trades_count",
                    "bar_granularity",
                ]
            )
        df = df.rename(
            columns={
                "time_period_start": "ts",
                "price_open": "open",
                "price_high": "high",
                "price_low": "low",
                "price_close": "close",
                "volume_traded": "volume",
            }
        )
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["coinapi_symbol_id"] = symbol_id
        df["bar_granularity"] = period_id
        return df[["coinapi_symbol_id", "ts", "open", "high", "low", "close", "volume", "trades_count", "bar_granularity"]]

    def get_quote_history(self, symbol_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        if self.mock_mode:
            rng = pd.date_range(start=start, end=end, freq="1h", inclusive="left", tz="UTC")
            rs = np.random.default_rng(sum(ord(x) for x in symbol_id) % (2**32 - 1))
            mid = np.exp(np.cumsum(rs.normal(0, 0.005, size=len(rng))))
            spread = rs.uniform(0.0002, 0.003, size=len(rng))
            bid = 100 * mid * (1 - spread / 2)
            ask = 100 * mid * (1 + spread / 2)
            return pd.DataFrame(
                {
                    "coinapi_symbol_id": symbol_id,
                    "ts": rng,
                    "bid_price": bid,
                    "ask_price": ask,
                }
            )
        payload = self._request(
            f"/v1/quotes/{symbol_id}/history",
            params={
                "time_start": self._format_time(start),
                "time_end": self._format_time(end),
                "limit": 100000,
            },
        )
        df = pd.DataFrame(payload)
        if df.empty:
            return pd.DataFrame(columns=["coinapi_symbol_id", "ts", "bid_price", "ask_price"])
        df["ts"] = pd.to_datetime(df["time_exchange"], utc=True).dt.floor("1h")
        df["coinapi_symbol_id"] = symbol_id
        return (
            df.groupby(["coinapi_symbol_id", "ts"], as_index=False)[["bid_price", "ask_price"]].mean()
        )

    def _mock_symbols(self) -> list[dict[str, Any]]:
        rows = []
        for exchange in ["COINBASE", "KRAKEN"]:
            for base in ["BTC", "ETH", "SOL", "ADA", "DOGE"]:
                rows.append(
                    {
                        "symbol_id": f"{exchange}_SPOT_{base}_USD",
                        "exchange_id": exchange,
                        "symbol_type": "SPOT",
                        "asset_id_base": base,
                        "asset_id_quote": "USD",
                        "data_start": "2020-01-01",
                        "data_end": None,
                    }
                )
        rows.append(
            {
                "symbol_id": "KRAKEN_SPOT_XRP_USD",
                "exchange_id": "KRAKEN",
                "symbol_type": "SPOT",
                "asset_id_base": "XRP",
                "asset_id_quote": "USD",
                "data_start": "2020-01-01",
                "data_end": None,
            }
        )
        return rows
