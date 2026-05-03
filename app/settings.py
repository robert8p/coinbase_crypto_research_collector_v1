from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "coinbase_crypto_research_collector"
    app_version: str = "1.4.1"
    live_shadow_lookback_hours: int = 72
    live_shadow_max_products: int = 50
    live_shadow_auto_refresh_references: bool = True
    data_dir: Path = Path("./runtime/data")
    output_dir_name: str = "exports"
    raw_dir_name: str = "raw"
    processed_dir_name: str = "processed"
    state_dir_name: str = "state"

    use_mock_data: bool = True
    coinbase_api_key_name: str | None = None
    coinbase_api_private_key: str | None = None
    coinbase_bearer_token: str | None = None
    coinapi_api_key: str | None = None

    quote_currencies: List[str] = Field(default_factory=lambda: ["USD"])
    preferred_bar_granularity: str = "ONE_HOUR"
    coinapi_period_id: str = "1HRS"
    lookback_hours: int = 24 * 14
    export_format: str = "parquet"
    benchmark_assets: List[str] = Field(default_factory=lambda: ["BTC", "ETH"])
    max_universe_size: int = 500
    top_n_by_volume: int = 500
    preferred_coinapi_exchanges: List[str] = Field(
        default_factory=lambda: ["COINBASE", "GDAX", "COINBASEEXCHANGE"]
    )
    enable_coinapi_quotes: bool = False
    min_notional_reference_usd: float = 25.0
    divergence_threshold: float = 0.005
    rule_target_column: str = "future_close_return_h4"
    strict_coinbase_tradability_filters: bool = False

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / self.raw_dir_name

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / self.processed_dir_name

    @property
    def state_dir(self) -> Path:
        return self.data_dir / self.state_dir_name

    @property
    def export_dir(self) -> Path:
        return self.data_dir / self.output_dir_name


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
