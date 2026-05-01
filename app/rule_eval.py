from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import RuleEvalRequest
from .storage import StorageManager


SCOPE_PREFIXES = {
    "coinbase_only": ("cb_", "ex_", "future_", "touched_"),
    "coinbase_plus_coinapi": ("cb_", "ca_", "cs_", "ex_", "future_", "touched_"),
    "full_scope": ("cb_", "ca_", "cs_", "cx_", "ex_", "future_", "touched_"),
}


@dataclass
class RuleEvaluationService:
    storage: StorageManager

    def _scope_frame(self, full_df: pd.DataFrame, scope: str) -> pd.DataFrame:
        prefixes = SCOPE_PREFIXES[scope]
        keep = [c for c in full_df.columns if c in {"product_id", "base_asset", "quote_asset", "ts", "feature_version", "coinapi_symbol_id", "trades_count"} or c.startswith(prefixes)]
        return full_df[keep].copy()

    def _build_mask(self, df: pd.DataFrame, request: RuleEvalRequest) -> pd.Series:
        if request.expression:
            expr = request.expression.replace(" and ", " & ").replace(" or ", " | ")
            return df.eval(expr, engine="python")
        if not request.conditions:
            raise ValueError("Provide either expression or conditions.")
        mask = pd.Series(True, index=df.index)
        for condition in request.conditions:
            if condition.feature not in df.columns:
                raise ValueError(f"Feature '{condition.feature}' not available in requested scope.")
            series = df[condition.feature]
            value = condition.value
            if condition.operator == ">":
                mask &= series > value
            elif condition.operator == ">=":
                mask &= series >= value
            elif condition.operator == "<":
                mask &= series < value
            elif condition.operator == "<=":
                mask &= series <= value
            elif condition.operator == "==":
                mask &= series == value
            elif condition.operator == "!=":
                mask &= series != value
        return mask

    def run(self, request: RuleEvalRequest) -> dict[str, Any]:
        full_df = self.storage.read_frame("feature_table")
        if full_df.empty:
            raise RuntimeError("feature_table dataset is missing. Compute features first.")
        summaries: list[dict[str, Any]] = []
        artifact_names: list[str] = []
        for scope in request.scopes:
            scope_df = self._scope_frame(full_df, scope)
            mask = self._build_mask(scope_df, request)
            matched = scope_df.loc[mask].copy()
            summary = {
                "scope": scope,
                "matched_rows": int(len(matched)),
                "mean_target": float(matched[request.target_column].mean()) if len(matched) and request.target_column in matched.columns else None,
                "median_target": float(matched[request.target_column].median()) if len(matched) and request.target_column in matched.columns else None,
                "touch_rate_h4_1pct": float(matched["touched_up_1pct_h4"].mean()) if len(matched) and "touched_up_1pct_h4" in matched.columns else None,
            }
            summaries.append(summary)
            match_path = self.storage.write_csv(matched, f"{request.rule_name}_{scope}_matched_rows", compress=True)
            artifact_names.append(match_path.name)
        summary_path = self.storage.export_path(f"{request.rule_name}_summary", ".json")
        self.storage.write_json({"rule_name": request.rule_name, "target_column": request.target_column, "summaries": summaries, "artifacts": artifact_names}, summary_path)
        artifact_names.append(summary_path.name)
        return {"rule_name": request.rule_name, "summaries": summaries, "artifacts": artifact_names}
