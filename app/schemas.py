from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DataPullRequest(BaseModel):
    lookback_hours: int | None = Field(default=None, ge=24, le=24 * 365)
    max_products: int | None = Field(default=None, ge=1, le=1000)


class ExportBuildRequest(BaseModel):
    compress_chatgpt_csv: bool = True


class PipelineRunRequest(DataPullRequest):
    compress_chatgpt_csv: bool = True


class RuleCondition(BaseModel):
    feature: str
    operator: Literal[">", ">=", "<", "<=", "==", "!="]
    value: float | int | bool


class RuleEvalRequest(BaseModel):
    rule_name: str = Field(default="rule_eval", pattern=r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")
    conditions: list[RuleCondition] = Field(default_factory=list)
    expression: str | None = None
    scopes: list[Literal["coinbase_only", "coinbase_plus_coinapi", "full_scope"]] = Field(
        default_factory=lambda: ["coinbase_only", "coinbase_plus_coinapi", "full_scope"]
    )
    target_column: str = "future_close_return_h4"


class RuleBacktestRequest(BaseModel):
    rule_ids: list[str] = Field(default_factory=list)
    selection_mode: Literal["selected", "all"] = "selected"
    run_mode: Literal["individual", "collective", "both"] = "individual"
    horizon: Literal["h1", "h4", "h24", "auto"] = "h4"


class LiveShadowRequest(DataPullRequest):
    rule_ids: list[str] = Field(default_factory=list)
    selection_mode: Literal["selected", "all"] = "selected"
    refresh_references: bool = True
    as_of_time_iso: str | None = None
