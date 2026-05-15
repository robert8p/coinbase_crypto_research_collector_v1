from __future__ import annotations

import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schemas import RuleBacktestRequest
from .storage import StorageManager


MAX_CUSTOM_RULES = 500

RULE_RESOURCE_PATHS = [
    Path(__file__).parent / "resources" / "merged_coinbase_coinapi_rule_validation_prompt.json",
    Path(__file__).parent / "resources" / "updated_deep_crypto_unknown_pattern_test_pack.json",
]
FIELD_ALIASES = {
    "cs_close_diff": "cs_coinbase_vs_coinapi_close_diff",
    "cb_breakout_short": "cb_breakout_distance_short",
    "cx_btc": "cx_btc_regime_flag",
    "cx_eth": "cx_eth_regime_flag",
}
ROUTING_PREDICTOR_ALIASES = {
    "trigger-bar volume intensity": ["cb_rel_volume_short", "ca_rel_volume_short", "cb_volume_zscore_short", "ca_volume_zscore_short"],
    "relative volume features": ["cb_rel_volume_short", "ca_rel_volume_short", "cb_rel_volume_medium", "ca_rel_volume_medium"],
    "CoinAPI liquidity bucket": ["ca_liquidity_bucket"],
    "liquidity tier": ["ca_liquidity_bucket"],
    "extreme same-bar divergence magnitude": ["cs_coinbase_vs_coinapi_return_diff", "cs_coinbase_vs_coinapi_close_diff"],
    "divergence magnitude and sign context": ["cs_coinbase_vs_coinapi_return_diff", "cs_coinbase_vs_coinapi_close_diff", "cs_cross_source_divergence_flag"],
    "local bar shape / close location": ["cb_close_location_in_bar", "ca_close_location_in_bar"],
    "local trend distance features": ["cb_sma_5_dist", "cb_sma_10_dist", "cb_sma_20_dist", "ca_sma_5_dist", "ca_sma_10_dist", "ca_sma_20_dist"],
    "symbol identity or symbol cohort": ["base_asset", "product_id"],
    "symbol cohort": ["base_asset", "product_id"],
    "BTC/ETH regime state": ["cx_btc_regime_flag", "cx_eth_regime_flag"],
}

CURATED_LIVE_RECOMMENDED_RULE_IDS = {
    "UPDATED_RULE_001",
    "UPDATED_RULE_002",
    "UPDATED_RULE_005",
    "UPDATED_RULE_006",
    "MERGED_RULE_001",
    "MERGED_RULE_002",
    "MERGED_RULE_005",
    "MERGED_RULE_006",
    "MERGED_RULE_007",
    "MERGED_RULE_009",
    "MERGED_RULE_010",
}
CURATED_LIVE_EXCLUDED_RULE_IDS = {
    "MERGED_RULE_003",
    "MERGED_RULE_004",
    "MERGED_RULE_008",
    "UPDATED_RULE_003",
    "UPDATED_RULE_004",
}


@dataclass
class RuleBacktestService:
    storage: StorageManager
    rule_resource_paths: list[Path] | None = None

    @property
    def custom_rule_library_path(self) -> Path:
        return self.storage.settings.state_dir / "custom_rule_library.json"

    @property
    def rule_overrides_path(self) -> Path:
        return self.storage.settings.state_dir / "rule_library_overrides.json"

    def _resource_paths(self) -> list[Path]:
        return list(self.rule_resource_paths or RULE_RESOURCE_PATHS)

    def _read_builtin_library(self) -> dict[str, Any]:
        merged_rules: list[dict[str, Any]] = []
        source_summaries: list[dict[str, Any]] = []
        for path in self._resource_paths():
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            extracted = self._extract_rules_from_payload(payload)
            normalized = [self._normalize_rule(rule, source_library="builtin") for rule in extracted]
            merged_rules.extend(normalized)
            source_summaries.append({"source": path.name, "rules_found": len(normalized)})
        return {
            "artifact_type": "builtin_merged_rule_library",
            "schema_version": "1.2",
            "candidate_rules": merged_rules,
            "sources": source_summaries,
        }

    def _read_custom_library(self) -> dict[str, Any]:
        if not self.custom_rule_library_path.exists():
            return {
                "artifact_type": "custom_rule_library",
                "schema_version": "1.0",
                "candidate_rules": [],
            }
        try:
            with self.custom_rule_library_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError:
            return {
                "artifact_type": "custom_rule_library",
                "schema_version": "1.0",
                "candidate_rules": [],
                "library_error": "custom rule library was unreadable and has been ignored",
            }
        payload.setdefault("candidate_rules", [])
        return payload

    def _write_custom_library(self, payload: dict[str, Any]) -> None:
        self.custom_rule_library_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage.write_json(payload, self.custom_rule_library_path)

    def _read_rule_overrides(self) -> dict[str, Any]:
        if not self.rule_overrides_path.exists():
            return {"rule_overrides": {}}
        try:
            with self.rule_overrides_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError:
            return {"rule_overrides": {}}
        payload.setdefault("rule_overrides", {})
        return payload

    def _write_rule_overrides(self, payload: dict[str, Any]) -> None:
        self.rule_overrides_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage.write_json(payload, self.rule_overrides_path)

    def _combined_library(self) -> dict[str, Any]:
        builtin = self._read_builtin_library()
        custom = self._read_custom_library()
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for origin, library in (("builtin", builtin), ("custom", custom)):
            for rule in library.get("candidate_rules", []):
                normalized = self._normalize_rule(rule, source_library=origin)
                rule_id = normalized["merged_rule_id"]
                if rule_id not in merged:
                    order.append(rule_id)
                merged[rule_id] = normalized
        overrides = self._read_rule_overrides().get("rule_overrides", {})
        for rule_id, fields in overrides.items():
            if rule_id in merged and isinstance(fields, dict):
                merged[rule_id].update(fields)
        rules = [merged[rule_id] for rule_id in order]
        return {
            "artifact_type": "merged_rule_library",
            "schema_version": "1.3",
            "candidate_rules": rules,
            "counts": {
                "builtin": len(builtin.get("candidate_rules", [])),
                "custom": len(custom.get("candidate_rules", [])),
                "combined": len(order),
                "live_eligible": sum(1 for rule in rules if bool(rule.get("live_eligible", False))),
                "live_candidate_recommended": sum(1 for rule in rules if bool(rule.get("live_candidate_recommended", False))),
            },
        }

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").upper()
        return slug or "UPLOADED_RULE"

    def _normalize_horizon_token(self, token: Any) -> str | None:
        if token is None:
            return None
        cleaned = str(token).strip().lower().replace("hours", "h").replace("hour", "h")
        cleaned = cleaned.replace(" ", "")
        direct = {"1h": "h1", "h1": "h1", "4h": "h4", "h4": "h4", "24h": "h24", "h24": "h24"}
        if cleaned in direct:
            return direct[cleaned]
        match = re.search(r"(1|4|24)h", cleaned)
        if match:
            return {"1": "h1", "4": "h4", "24": "h24"}[match.group(1)]
        return None

    def _parse_horizon_tokens(self, values: list[Any] | tuple[Any, ...] | Any) -> list[str]:
        tokens = values if isinstance(values, (list, tuple)) else [values]
        out: list[str] = []
        for token in tokens:
            normalized = self._normalize_horizon_token(token)
            if normalized and normalized not in out:
                out.append(normalized)
        return out

    def _parse_target_horizons_from_text(self, value: str | None) -> list[str]:
        if not value:
            return []
        lowered = str(value).lower()
        out: list[str] = []
        for candidate in ("1h", "h1", "4h", "h4", "24h", "h24"):
            if candidate in lowered:
                normalized = self._normalize_horizon_token(candidate)
                if normalized and normalized not in out:
                    out.append(normalized)
        return out

    def _infer_rule_kind(self, out: dict[str, Any]) -> str:
        if out.get("rule_kind"):
            return str(out["rule_kind"])
        if out.get("candidate_routing_predictors_to_test") and out.get("parent_rule"):
            return "routing_hypothesis"
        if out.get("applies_to") and not (out.get("exact_definition") or out.get("exact_definition_variants") or out.get("all_conditions")):
            return "execution_test"
        return "direct_rule"

    def _infer_live_candidate_recommendation(self, out: dict[str, Any]) -> tuple[bool, str]:
        explicit = out.get("live_candidate_recommended")
        if explicit is not None:
            return bool(explicit), str(out.get("live_candidate_reason") or "explicit_rule_setting")
        rule_id = str(out.get("merged_rule_id") or out.get("rule_id") or out.get("test_id") or out.get("id") or "")
        if rule_id in CURATED_LIVE_RECOMMENDED_RULE_IDS:
            return True, "curated_true_live_candidate"
        if rule_id in CURATED_LIVE_EXCLUDED_RULE_IDS:
            return False, "curated_experimental_or_secondary"
        if out.get("rule_kind") != "direct_rule":
            return False, "non_direct_rule"
        if out.get("test_id") or rule_id.startswith(("TEST_", "UPDATED_TEST_", "EXEC_TEST_")):
            return False, "test_or_execution_item"
        if out.get("parent_rule") or out.get("candidate_routing_predictors_to_test"):
            return False, "routing_or_subclass_hypothesis"
        if out.get("applies_to"):
            return False, "execution_bundle"
        return True, "direct_rule_default"

    def _normalize_rule(self, rule: dict[str, Any], source_library: str = "custom") -> dict[str, Any]:
        out = dict(rule)
        if "exact_definition" not in out and "exact_definition_variants" not in out and "all_conditions" in out:
            out["exact_definition"] = {"all_conditions": out.pop("all_conditions")}
        if "source_attribution" not in out or not isinstance(out.get("source_attribution"), list):
            out["source_attribution"] = []
        rule_id = out.get("merged_rule_id") or out.get("rule_id") or out.get("test_id") or out.get("id")
        if not rule_id:
            seed = out.get("name") or out.get("title") or out.get("routing_objective") or "uploaded_rule"
            rule_id = self._slugify(str(seed))
        out["merged_rule_id"] = str(rule_id)
        out["name"] = out.get("name") or out.get("merged_rule_id")
        out["priority"] = int(out.get("priority", 999))
        out["family_id"] = out.get("family_id") or "UPLOADED"
        out["target_horizon"] = out.get("target_horizon") or "user_selected"
        target_horizons = self._parse_horizon_tokens(out.get("target_horizons", []))
        if not target_horizons:
            target_horizons = self._parse_target_horizons_from_text(out.get("target_horizon"))
        recommended_primary = self._normalize_horizon_token(out.get("recommended_primary_horizon"))
        secondary_monitor = self._normalize_horizon_token(out.get("secondary_monitor_horizon"))
        if recommended_primary and recommended_primary not in target_horizons:
            target_horizons.insert(0, recommended_primary)
        if secondary_monitor and secondary_monitor not in target_horizons:
            target_horizons.append(secondary_monitor)
        out["target_horizons"] = target_horizons
        out["recommended_primary_horizon"] = recommended_primary
        out["secondary_monitor_horizon"] = secondary_monitor
        out["why_interesting"] = out.get("why_interesting") or out.get("hypothesis") or out.get("routing_objective") or "Uploaded custom rule"
        out["source_library"] = source_library
        out["rule_kind"] = self._infer_rule_kind(out)
        recommended, reason = self._infer_live_candidate_recommendation(out)
        out["live_candidate_recommended"] = bool(recommended)
        out["live_candidate_reason"] = str(reason)
        out["live_eligible"] = bool(out.get("live_eligible", recommended))
        return out

    def _extract_rules_from_payload(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            extracted: list[dict[str, Any]] = []
            for key in ("candidate_rules", "rules"):
                if key in payload and isinstance(payload[key], list):
                    extracted.extend([item for item in payload[key] if isinstance(item, dict)])
            if "orthogonality_and_follow_up_tests" in payload and isinstance(payload["orthogonality_and_follow_up_tests"], list):
                for item in payload["orthogonality_and_follow_up_tests"]:
                    if not isinstance(item, dict):
                        continue
                    synthetic = dict(item)
                    synthetic.setdefault("name", item.get("test_id") or item.get("name") or "FOLLOW_UP_TEST")
                    if item.get("definition"):
                        synthetic["exact_definition"] = item["definition"]
                    synthetic["rule_kind"] = "analysis_test"
                    synthetic["live_candidate_recommended"] = False
                    synthetic["live_candidate_reason"] = "follow_up_or_orthogonality_test"
                    extracted.append(synthetic)
            if "execution_tests" in payload and isinstance(payload["execution_tests"], list):
                for item in payload["execution_tests"]:
                    if not isinstance(item, dict):
                        continue
                    synthetic = dict(item)
                    synthetic.setdefault("name", item.get("test_id") or item.get("name") or "EXECUTION_TEST")
                    synthetic["rule_kind"] = "execution_test"
                    extracted.append(synthetic)
            if extracted:
                return extracted
            if any(key in payload for key in ("merged_rule_id", "rule_id", "test_id", "id", "name", "exact_definition", "exact_definition_variants", "all_conditions")):
                return [payload]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        raise ValueError("Uploaded JSON must be a rule object, a list of rules, or an object containing candidate_rules/rules.")

    def upload_rules(self, file_payloads: list[tuple[str, str]] | None = None, pasted_json: str | None = None) -> dict[str, Any]:
        sources: list[tuple[str, str]] = list(file_payloads or [])
        if pasted_json and pasted_json.strip():
            sources.append(("pasted_json", pasted_json))
        if not sources:
            raise ValueError("No JSON rules supplied.")

        custom = self._read_custom_library()
        existing_by_id = {
            self._normalize_rule(rule, source_library="custom")["merged_rule_id"]: self._normalize_rule(rule, source_library="custom")
            for rule in custom.get("candidate_rules", [])
            if isinstance(rule, dict)
        }

        uploaded_rule_ids: list[str] = []
        source_summaries: list[dict[str, Any]] = []
        for source_name, raw_text in sources:
            payload = json.loads(raw_text)
            extracted = self._extract_rules_from_payload(payload)
            count_before = len(existing_by_id)
            for rule in extracted:
                normalized = self._normalize_rule(rule, source_library="custom")
                normalized["uploaded_from"] = source_name
                existing_by_id[normalized["merged_rule_id"]] = normalized
                uploaded_rule_ids.append(normalized["merged_rule_id"])
            source_summaries.append(
                {
                    "source": source_name,
                    "rules_found": len(extracted),
                    "rules_added_or_updated": len(existing_by_id) - count_before,
                }
            )

        if len(existing_by_id) > MAX_CUSTOM_RULES:
            raise ValueError(f"Too many custom rules. Limit is {MAX_CUSTOM_RULES}.")

        new_payload = {
            "artifact_type": "custom_rule_library",
            "schema_version": "1.2",
            "candidate_rules": sorted(existing_by_id.values(), key=lambda row: (row.get("priority", 999), row["merged_rule_id"])),
        }
        self._write_custom_library(new_payload)
        combined = self._combined_library()
        self.storage.update_status(
            "rule_library_upload",
            "completed",
            message="Rule JSON uploaded",
            uploaded_rule_ids=sorted(set(uploaded_rule_ids)),
            custom_rule_count=combined.get("counts", {}).get("custom", 0),
            combined_rule_count=combined.get("counts", {}).get("combined", 0),
        )
        return {
            "status": "completed",
            "uploaded_rule_ids": sorted(set(uploaded_rule_ids)),
            "sources": source_summaries,
            "counts": combined.get("counts", {}),
            "custom_rule_library_path": str(self.custom_rule_library_path),
        }

    def list_rules(self) -> list[dict[str, Any]]:
        library = self._combined_library()
        out: list[dict[str, Any]] = []
        for rule in library.get("candidate_rules", []):
            out.append(
                {
                    "merged_rule_id": rule.get("merged_rule_id"),
                    "name": rule.get("name"),
                    "priority": rule.get("priority"),
                    "family_id": rule.get("family_id"),
                    "target_horizon": rule.get("target_horizon"),
                    "target_horizons": rule.get("target_horizons", []),
                    "recommended_primary_horizon": rule.get("recommended_primary_horizon"),
                    "secondary_monitor_horizon": rule.get("secondary_monitor_horizon"),
                    "source_attribution": rule.get("source_attribution", []),
                    "source_library": rule.get("source_library", "builtin"),
                    "rule_kind": rule.get("rule_kind", "direct_rule"),
                    "live_eligible": bool(rule.get("live_eligible", False)),
                    "live_candidate_recommended": bool(rule.get("live_candidate_recommended", False)),
                    "live_candidate_reason": rule.get("live_candidate_reason"),
                    "has_variants": bool(rule.get("exact_definition_variants")) or bool(rule.get("exact_definition", {}).get("quantiles_to_test")),
                    "why_interesting": rule.get("why_interesting"),
                }
            )
        return out

    def update_live_eligibility(self, rule_ids: list[str], live_eligible: bool) -> dict[str, Any]:
        if not rule_ids:
            raise ValueError("Provide at least one rule ID.")
        library_ids = {rule["merged_rule_id"] for rule in self.list_rules()}
        missing = sorted(set(rule_ids) - library_ids)
        if missing:
            raise ValueError(f"Unknown rule IDs: {', '.join(missing[:10])}")
        payload = self._read_rule_overrides()
        overrides = payload.setdefault("rule_overrides", {})
        for rule_id in rule_ids:
            current = overrides.get(rule_id, {}) if isinstance(overrides.get(rule_id), dict) else {}
            current["live_eligible"] = bool(live_eligible)
            overrides[rule_id] = current
        self._write_rule_overrides(payload)
        combined = self._combined_library()
        self.storage.update_status(
            "rule_library_live_eligibility",
            "completed",
            message="Updated live-eligible rules",
            updated_rule_ids=sorted(set(rule_ids)),
            live_eligible=bool(live_eligible),
        )
        return {
            "status": "completed",
            "updated_rule_ids": sorted(set(rule_ids)),
            "live_eligible": bool(live_eligible),
            "counts": combined.get("counts", {}),
        }

    def apply_live_candidate_policy(self, rule_ids: list[str] | None = None) -> dict[str, Any]:
        library = self.list_rules()
        target_rules = [rule for rule in library if not rule_ids or rule.get("merged_rule_id") in set(rule_ids)]
        if not target_rules:
            raise ValueError("No rules available for automatic live-candidate selection.")
        payload = self._read_rule_overrides()
        overrides = payload.setdefault("rule_overrides", {})
        updated_ids: list[str] = []
        recommended_ids: list[str] = []
        excluded_ids: list[str] = []
        for rule in target_rules:
            rule_id = str(rule.get("merged_rule_id"))
            recommended = bool(rule.get("live_candidate_recommended", False))
            current = overrides.get(rule_id, {}) if isinstance(overrides.get(rule_id), dict) else {}
            current["live_eligible"] = recommended
            overrides[rule_id] = current
            updated_ids.append(rule_id)
            (recommended_ids if recommended else excluded_ids).append(rule_id)
        self._write_rule_overrides(payload)
        combined = self._combined_library()
        self.storage.update_status(
            "rule_library_live_eligibility",
            "completed",
            message="Applied recommended live-candidate policy",
            updated_rule_ids=sorted(set(updated_ids)),
            recommended_live_rule_ids=sorted(set(recommended_ids)),
            excluded_live_rule_ids=sorted(set(excluded_ids)),
        )
        return {
            "status": "completed",
            "updated_rule_ids": sorted(set(updated_ids)),
            "recommended_live_rule_ids": sorted(set(recommended_ids)),
            "excluded_live_rule_ids": sorted(set(excluded_ids)),
            "counts": combined.get("counts", {}),
        }

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "cs_sma_5_dist_delta" not in out.columns and {"ca_sma_5_dist", "cb_sma_5_dist"}.issubset(out.columns):
            out["cs_sma_5_dist_delta"] = out["ca_sma_5_dist"] - out["cb_sma_5_dist"]
        for alias, actual in FIELD_ALIASES.items():
            if alias not in out.columns and actual in out.columns:
                out[alias] = out[actual]
        return out

    def _target_columns(self, horizon: str) -> tuple[str, str, str | None]:
        if horizon not in {"h1", "h4", "h24"}:
            raise ValueError("horizon must be one of h1, h4, h24")
        return_col = f"future_close_return_{horizon}"
        max_up_col = f"future_max_up_pct_{horizon}"
        touch_col = {"h1": None, "h4": "touched_up_1pct_h4", "h24": "touched_up_2pct_h24"}[horizon]
        return return_col, max_up_col, touch_col

    def _field_name(self, field: str) -> str:
        return FIELD_ALIASES.get(field, field)

    def _parse_inline_logic(self, logic: str, condition: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        cleaned = str(logic or "").strip()
        match = re.match(r"^(>=|<=|>|<|==|!=)\s+(.+)$", cleaned)
        updated = dict(condition)
        if not match:
            return cleaned, updated
        operator, rhs = match.groups()
        rhs = rhs.strip()
        if "threshold_type" not in updated and "value" not in updated:
            if rhs.startswith("empirical_"):
                updated["threshold_type"] = rhs
            else:
                try:
                    updated["value"] = float(rhs)
                except ValueError:
                    updated["value"] = rhs
        return operator, updated

    def _condition_mask(self, df: pd.DataFrame, condition: dict[str, Any]) -> tuple[pd.Series | None, dict[str, Any]]:
        original_field = condition.get("field")
        field = self._field_name(original_field)
        if field not in df.columns:
            return None, {"field": original_field, "resolved_field": field, "missing": True}
        series = df[field]
        raw_logic = condition.get("logic")
        logic, normalized_condition = self._parse_inline_logic(str(raw_logic), condition)
        metadata: dict[str, Any] = {"field": original_field, "resolved_field": field, "logic": logic, "raw_logic": raw_logic}

        def _numeric_threshold_from_type(threshold_type: str) -> float:
            if threshold_type == "empirical_top_decile_threshold":
                return float(series.quantile(0.9))
            if threshold_type.startswith("empirical_q"):
                pct = float(threshold_type.replace("empirical_q", "")) / 100.0
                return float(series.quantile(pct))
            raise ValueError(f"Unsupported threshold_type '{threshold_type}'")

        if logic == "between":
            lower = normalized_condition.get("lower_bound")
            upper = normalized_condition.get("upper_bound")
            metadata.update({"lower_bound": lower, "upper_bound": upper})
            return series.between(lower, upper, inclusive="both"), metadata

        if logic == "in_top_quantile":
            quantile = float(normalized_condition.get("quantile", 0.2))
            threshold = float(series.quantile(1 - quantile))
            metadata.update({"quantile": quantile, "threshold": threshold})
            return series >= threshold, metadata

        if logic == "in_bottom_quantile":
            quantile = float(normalized_condition.get("quantile", 0.2))
            threshold = float(series.quantile(quantile))
            metadata.update({"quantile": quantile, "threshold": threshold})
            return series <= threshold, metadata

        value = normalized_condition.get("value")
        if "threshold_type" in normalized_condition:
            value = _numeric_threshold_from_type(str(normalized_condition["threshold_type"]))
            metadata.update({"threshold_type": normalized_condition["threshold_type"], "threshold": value})
        else:
            if isinstance(value, str) and not isinstance(value, bool):
                try:
                    value = float(value)
                except ValueError:
                    pass
            metadata["value"] = value

        if logic == ">":
            return series > value, metadata
        if logic == ">=":
            return series >= value, metadata
        if logic == "<":
            return series < value, metadata
        if logic == "<=":
            return series <= value, metadata

        if logic == "==":
            return series == value, metadata
        if logic == "!=":
            return series != value, metadata
        raise ValueError(f"Unsupported condition logic '{raw_logic}'")

    def _resolve_rule_variants(self, rule: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if rule.get("exact_definition_variants"):
            for variant in rule["exact_definition_variants"]:
                out.append(
                    {
                        "instance_id": f"{rule['merged_rule_id']}::{variant.get('variant_id', 'variant')}",
                        "merged_rule_id": rule["merged_rule_id"],
                        "name": rule.get("name"),
                        "variant_id": variant.get("variant_id"),
                        "conditions": variant.get("all_conditions", []),
                        "source_attribution": rule.get("source_attribution", []),
                        "family_id": rule.get("family_id"),
                        "hypothesis": rule.get("hypothesis"),
                        "source_library": rule.get("source_library", "builtin"),
                        "rule_kind": rule.get("rule_kind", "direct_rule"),
                        "priority": rule.get("priority"),
                        "recommended_primary_horizon": rule.get("recommended_primary_horizon"),
                        "secondary_monitor_horizon": rule.get("secondary_monitor_horizon"),
                        "target_horizons": rule.get("target_horizons", []),
                    }
                )
            return out

        exact = rule.get("exact_definition", {})
        if exact.get("quantiles_to_test"):
            field = exact.get("field")
            for q in exact.get("quantiles_to_test", []):
                out.append(
                    {
                        "instance_id": f"{rule['merged_rule_id']}::q{int(round(float(q) * 100)):02d}",
                        "merged_rule_id": rule["merged_rule_id"],
                        "name": rule.get("name"),
                        "variant_id": f"q{int(round(float(q) * 100)):02d}",
                        "conditions": [{"field": field, "logic": exact.get("logic"), "quantile": q}],
                        "source_attribution": rule.get("source_attribution", []),
                        "family_id": rule.get("family_id"),
                        "hypothesis": rule.get("hypothesis"),
                        "source_library": rule.get("source_library", "builtin"),
                        "rule_kind": rule.get("rule_kind", "direct_rule"),
                        "priority": rule.get("priority"),
                        "recommended_primary_horizon": rule.get("recommended_primary_horizon"),
                        "secondary_monitor_horizon": rule.get("secondary_monitor_horizon"),
                        "target_horizons": rule.get("target_horizons", []),
                    }
                )
            return out

        out.append(
            {
                "instance_id": rule["merged_rule_id"],
                "merged_rule_id": rule["merged_rule_id"],
                "name": rule.get("name"),
                "variant_id": None,
                "conditions": exact.get("all_conditions", []),
                "source_attribution": rule.get("source_attribution", []),
                "family_id": rule.get("family_id"),
                "hypothesis": rule.get("hypothesis"),
                "source_library": rule.get("source_library", "builtin"),
                "rule_kind": rule.get("rule_kind", "direct_rule"),
                "priority": rule.get("priority"),
                "recommended_primary_horizon": rule.get("recommended_primary_horizon"),
                "secondary_monitor_horizon": rule.get("secondary_monitor_horizon"),
                "target_horizons": rule.get("target_horizons", []),
            }
        )
        return out

    def _build_rule_mask(self, df: pd.DataFrame, conditions: list[dict[str, Any]]) -> tuple[pd.Series | None, list[dict[str, Any]], list[str]]:
        if not conditions:
            return pd.Series(True, index=df.index), [], []
        mask = pd.Series(True, index=df.index)
        resolved_conditions: list[dict[str, Any]] = []
        missing_features: list[str] = []
        for condition in conditions:
            cond_mask, metadata = self._condition_mask(df, condition)
            resolved_conditions.append(metadata)
            if cond_mask is None:
                missing_features.append(str(condition.get("field")))
                continue
            mask &= cond_mask.fillna(False)
        if missing_features:
            return None, resolved_conditions, missing_features
        return mask, resolved_conditions, []

    def _baseline_mask(self, df: pd.DataFrame, conditions: list[dict[str, Any]]) -> tuple[pd.Series, str]:
        if len(conditions) <= 1:
            return pd.Series(True, index=df.index), "global_dataset"
        base_mask, _, missing = self._build_rule_mask(df, conditions[:-1])
        if base_mask is None or missing:
            return pd.Series(True, index=df.index), "global_dataset_missing_baseline_fields"
        return base_mask, "all_conditions_except_last"

    def _product_concentration(self, matched: pd.DataFrame) -> tuple[int, float | None]:
        if matched.empty:
            return 0, None
        counts = matched["product_id"].value_counts(dropna=False)
        largest_share = float(counts.iloc[0] / len(matched)) if len(counts) else None
        return int(counts.size), largest_share

    def _median_per_product_uplift(self, candidate: pd.DataFrame, baseline: pd.DataFrame, return_col: str) -> float | None:
        if candidate.empty or baseline.empty or return_col not in candidate.columns or return_col not in baseline.columns:
            return None
        cand = candidate.groupby("product_id")[return_col].mean()
        base = baseline.groupby("product_id")[return_col].mean()
        joined = cand.to_frame("candidate").join(base.to_frame("baseline"), how="inner")
        if joined.empty:
            return None
        uplift = joined["candidate"] - joined["baseline"]
        return float(uplift.median()) if uplift.notna().any() else None

    def _verdict(self, support_rows: int, distinct_products: int, largest_share: float | None) -> tuple[str, str]:
        if support_rows < 20 or distinct_products < 3:
            return "fragile", "high"
        if largest_share is not None and largest_share > 0.4:
            return "concentrated", "medium"
        if support_rows >= 100 and distinct_products >= 15:
            return "broad", "medium"
        return "usable", "medium"

    def _all_metric_columns(self) -> list[str]:
        cols: list[str] = []
        for horizon in ("h1", "h4", "h24"):
            return_col, max_up_col, touch_col = self._target_columns(horizon)
            cols.extend([return_col, max_up_col])
            if touch_col:
                cols.append(touch_col)
        return cols

    def _matched_export_columns(self, df: pd.DataFrame, resolved_conditions: list[dict[str, Any]], horizon: str | None = None) -> list[str]:
        base_cols = ["product_id", "base_asset", "quote_asset", "ts", "feature_version", "coinapi_symbol_id"]
        cond_cols = [c["resolved_field"] for c in resolved_conditions if c.get("resolved_field") in df.columns]
        metrics = self._all_metric_columns()
        if horizon in {"h1", "h4", "h24"}:
            return_col, max_up_col, touch_col = self._target_columns(horizon)
            metrics = [return_col, max_up_col] + ([touch_col] if touch_col else []) + [col for col in metrics if col not in {return_col, max_up_col, touch_col}]
        keep = []
        for col in base_cols + cond_cols + metrics:
            if col in df.columns and col not in keep:
                keep.append(col)
        return keep

    def _collective_summary(self, name: str, masks: list[pd.Series], df: pd.DataFrame, horizon: str) -> tuple[dict[str, Any], pd.DataFrame]:
        return_col, max_up_col, touch_col = self._target_columns(horizon)
        if not masks:
            collective = df.iloc[0:0].copy()
        else:
            union_mask = masks[0].copy()
            for mask in masks[1:]:
                union_mask |= mask
            collective = df.loc[union_mask].copy()
        distinct_products, largest_share = self._product_concentration(collective)
        touch_rate = float(collective[touch_col].mean()) if touch_col and touch_col in collective.columns and len(collective) else None
        summary = {
            "result_type": "collective",
            "rule_instance_id": name,
            "rule_name": name,
            "variant_id": None,
            "status": "ok",
            "support_rows": int(len(collective)),
            "distinct_products": distinct_products,
            "largest_product_share": largest_share,
            "target_horizon": horizon,
            "candidate_touch_rate": touch_rate,
            "candidate_mean_forward_return": float(collective[return_col].mean()) if len(collective) else None,
            "candidate_mean_max_up_pct": float(collective[max_up_col].mean()) if len(collective) else None,
            "baseline_type": "global_dataset",
            "source_attribution": "combined",
        }
        summary.update(self._multi_horizon_metrics(collective, prefix="candidate_"))
        return summary, collective

    def _resolve_requested_horizons(self, rule: dict[str, Any], requested_horizon: str) -> list[str]:
        if requested_horizon != "auto":
            return [requested_horizon]
        horizons: list[str] = []
        for candidate in [
            rule.get("recommended_primary_horizon"),
            rule.get("secondary_monitor_horizon"),
            *(rule.get("target_horizons", []) or []),
        ]:
            normalized = self._normalize_horizon_token(candidate)
            if normalized and normalized not in horizons:
                horizons.append(normalized)
        if not horizons:
            if rule.get("rule_kind") in {"routing_hypothesis", "execution_test", "analysis_test"}:
                horizons = ["h1", "h4", "h24"]
            else:
                horizons = ["h4"]
        return horizons

    def _multi_horizon_metrics(self, subset: pd.DataFrame, prefix: str = "") -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for horizon in ("h1", "h4", "h24"):
            return_col, max_up_col, touch_col = self._target_columns(horizon)
            if return_col in subset.columns:
                metrics[f"{prefix}{horizon}_mean_forward_return"] = float(subset[return_col].mean()) if len(subset) else None
            if max_up_col in subset.columns:
                metrics[f"{prefix}{horizon}_mean_max_up_pct"] = float(subset[max_up_col].mean()) if len(subset) else None
            mean_return = metrics.get(f"{prefix}{horizon}_mean_forward_return")
            mean_max = metrics.get(f"{prefix}{horizon}_mean_max_up_pct")
            if mean_return is not None and mean_max is not None and mean_max not in (0, None) and not pd.isna(mean_max):
                metrics[f"{prefix}{horizon}_close_to_max_ratio"] = float(mean_return / mean_max) if mean_max else None
            else:
                metrics[f"{prefix}{horizon}_close_to_max_ratio"] = None
            if touch_col and touch_col in subset.columns:
                metrics[f"{prefix}{horizon}_touch_rate"] = float(subset[touch_col].mean()) if len(subset) else None
        h1 = metrics.get(f"{prefix}h1_mean_forward_return")
        h4 = metrics.get(f"{prefix}h4_mean_forward_return")
        h24 = metrics.get(f"{prefix}h24_mean_forward_return")
        metrics[f"{prefix}h1_share_of_h4_close_excess"] = float(h1 / h4) if h1 is not None and h4 not in (None, 0) and not pd.isna(h4) else None
        metrics[f"{prefix}h1_share_of_h24_close_excess"] = float(h1 / h24) if h1 is not None and h24 not in (None, 0) and not pd.isna(h24) else None
        metrics[f"{prefix}h24_minus_h4_close_return"] = float(h24 - h4) if h24 is not None and h4 is not None else None
        return metrics

    def _resolve_parent_mask(self, rule_id: str, available_rules: dict[str, dict[str, Any]], df: pd.DataFrame) -> tuple[pd.Series | None, list[dict[str, Any]], list[str], dict[str, Any] | None]:
        parent = available_rules.get(rule_id)
        if not parent:
            return None, [], [rule_id], None
        if parent.get("rule_kind") != "direct_rule":
            return None, [], [rule_id], parent
        instances = self._resolve_rule_variants(parent)
        if not instances:
            return None, [], [rule_id], parent
        return self._build_rule_mask(df, instances[0].get("conditions", [])) + (parent,)

    def _candidate_predictor_fields(self, descriptors: list[str], df: pd.DataFrame) -> list[tuple[str, str]]:
        seen: set[str] = set()
        out: list[tuple[str, str]] = []
        for descriptor in descriptors:
            actuals = ROUTING_PREDICTOR_ALIASES.get(descriptor, [descriptor])
            for actual in actuals:
                field = self._field_name(actual)
                if field in df.columns and field not in seen:
                    seen.add(field)
                    out.append((descriptor, field))
        return out

    def _subset_rows_for_field(self, parent_df: pd.DataFrame, field: str) -> list[tuple[str, pd.Series]]:
        series = parent_df[field]
        out: list[tuple[str, pd.Series]] = []
        if series.dropna().empty:
            return out
        if pd.api.types.is_numeric_dtype(series):
            q20 = float(series.quantile(0.2))
            q80 = float(series.quantile(0.8))
            out.append((f"{field}_bottom20", (series <= q20).fillna(False)))
            out.append((f"{field}_top20", (series >= q80).fillna(False)))
            if field in {"cs_coinbase_vs_coinapi_return_diff", "cs_coinbase_vs_coinapi_close_diff"}:
                abs_series = series.abs()
                q90_abs = float(abs_series.quantile(0.9))
                out.append((f"{field}_abs_top10", (abs_series >= q90_abs).fillna(False)))
        else:
            counts = series.astype("string").value_counts(dropna=True)
            for value, count in counts.head(6).items():
                out.append((f"{field}={value}", (series.astype("string") == value).fillna(False)))
        return out

    def _routing_score(self, rule: dict[str, Any], metrics: dict[str, Any]) -> float:
        h1 = metrics.get("h1_mean_forward_return") or 0.0
        h4 = metrics.get("h4_mean_forward_return") or 0.0
        h24 = metrics.get("h24_mean_forward_return") or 0.0
        h24_minus_h4 = metrics.get("h24_minus_h4_close_return") or 0.0
        h1_share_h24 = metrics.get("h1_share_of_h24_close_excess") or 0.0
        objective = " ".join(
            [
                str(rule.get("name", "")),
                str(rule.get("routing_objective", "")),
                str(rule.get("success_definition", "")),
            ]
        ).lower()
        if "fast" in objective or "early" in objective:
            return float((4 * h1) + (2 * h4) + (1.5 * h1_share_h24) - max(h24_minus_h4, 0))
        return float((5 * h24) + (3 * h24_minus_h4) + (1 * h4) - max(h1_share_h24, 0))

    def _evaluate_routing_hypothesis(
        self,
        rule: dict[str, Any],
        df: pd.DataFrame,
        available_rules: dict[str, dict[str, Any]],
        run_id: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame | None]:
        parent_id = rule.get("parent_rule")
        if not parent_id:
            row = {
                "result_type": "routing",
                "rule_instance_id": rule["merged_rule_id"],
                "merged_rule_id": rule["merged_rule_id"],
                "rule_name": rule.get("name"),
                "status": "unavailable",
                "missing_features": "parent_rule",
                "target_horizon": "multi",
            }
            return [row], [], None
        parent_mask, resolved_conditions, missing, parent_rule = self._resolve_parent_mask(parent_id, available_rules, df)
        if parent_mask is None or missing:
            row = {
                "result_type": "routing",
                "rule_instance_id": rule["merged_rule_id"],
                "merged_rule_id": rule["merged_rule_id"],
                "rule_name": rule.get("name"),
                "status": "unavailable",
                "missing_features": ", ".join(sorted(set(missing or [parent_id]))),
                "target_horizon": "multi",
            }
            return [row], [], None
        parent_df = df.loc[parent_mask].copy()
        baseline_metrics = self._multi_horizon_metrics(parent_df, prefix="baseline_")
        distinct_products_parent, largest_share_parent = self._product_concentration(parent_df)
        routing_rows: list[dict[str, Any]] = []
        predictor_fields = self._candidate_predictor_fields(rule.get("candidate_routing_predictors_to_test", []), parent_df)
        for descriptor, field in predictor_fields:
            for bucket_name, bucket_mask in self._subset_rows_for_field(parent_df, field):
                subset = parent_df.loc[bucket_mask].copy()
                if len(subset) < max(3, int(max(len(parent_df) * 0.05, 1))):
                    continue
                metrics = self._multi_horizon_metrics(subset)
                distinct_products, largest_share = self._product_concentration(subset)
                robustness_verdict, novelty_verdict = self._verdict(len(subset), distinct_products, largest_share)
                routing_score = self._routing_score(rule, metrics)
                row = {
                    "result_type": "routing",
                    "rule_instance_id": f"{rule['merged_rule_id']}::{bucket_name}",
                    "merged_rule_id": rule["merged_rule_id"],
                    "rule_name": rule.get("name"),
                    "variant_id": bucket_name,
                    "status": "ok",
                    "support_rows": int(len(subset)),
                    "distinct_products": distinct_products,
                    "largest_product_share": largest_share,
                    "target_horizon": "multi",
                    "candidate_mean_forward_return": metrics.get("h4_mean_forward_return"),
                    "candidate_mean_max_up_pct": metrics.get("h4_mean_max_up_pct"),
                    "baseline_type": f"parent_rule:{parent_id}",
                    "concentration_verdict": "concentrated" if largest_share is not None and largest_share > 0.35 else "acceptable",
                    "robustness_verdict": robustness_verdict,
                    "novelty_verdict": novelty_verdict,
                    "family_id": rule.get("family_id"),
                    "source_library": rule.get("source_library", "builtin"),
                    "source_attribution": "|".join(rule.get("source_attribution", [])),
                    "hypothesis": rule.get("hypothesis") or rule.get("routing_objective"),
                    "resolved_conditions": json.dumps(resolved_conditions),
                    "missing_features": None,
                    "routing_parent_rule": parent_id,
                    "routing_predictor_group": descriptor,
                    "routing_field": field,
                    "routing_bucket": bucket_name,
                    "routing_score": routing_score,
                    "parent_support_rows": int(len(parent_df)),
                    "parent_distinct_products": distinct_products_parent,
                    "parent_largest_product_share": largest_share_parent,
                }
                row.update(metrics)
                row.update(baseline_metrics)
                routing_rows.append(row)
        if not routing_rows:
            row = {
                "result_type": "routing",
                "rule_instance_id": rule["merged_rule_id"],
                "merged_rule_id": rule["merged_rule_id"],
                "rule_name": rule.get("name"),
                "status": "unavailable",
                "missing_features": "No available routing predictors found in feature table.",
                "target_horizon": "multi",
            }
            return [row], [], None

        routing_df = pd.DataFrame(routing_rows).sort_values(["routing_score", "support_rows"], ascending=[False, False])
        detail_path = self.storage.write_csv(routing_df, f"rule_backtest_{rule['merged_rule_id']}_routing_audit__{run_id}", compress=False)
        artifacts = [self.storage.file_info(detail_path)]
        preview = routing_df.head(10).to_dict(orient="records")
        return preview, artifacts, routing_df

    def _evaluate_execution_test(
        self,
        rule: dict[str, Any],
        df: pd.DataFrame,
        available_rules: dict[str, dict[str, Any]],
        run_id: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame | None]:
        rows: list[dict[str, Any]] = []
        applies_to = [rule_id for rule_id in rule.get("applies_to", []) if isinstance(rule_id, str)]
        if not applies_to:
            row = {
                "result_type": "execution",
                "rule_instance_id": rule["merged_rule_id"],
                "merged_rule_id": rule["merged_rule_id"],
                "rule_name": rule.get("name"),
                "status": "unavailable",
                "missing_features": "applies_to",
                "target_horizon": "multi",
            }
            return [row], [], None
        for parent_id in applies_to:
            parent_mask, resolved_conditions, missing, parent_rule = self._resolve_parent_mask(parent_id, available_rules, df)
            if parent_mask is None or missing:
                rows.append(
                    {
                        "result_type": "execution",
                        "rule_instance_id": f"{rule['merged_rule_id']}::{parent_id}",
                        "merged_rule_id": rule["merged_rule_id"],
                        "rule_name": rule.get("name"),
                        "variant_id": parent_id,
                        "status": "unavailable",
                        "missing_features": ", ".join(sorted(set(missing or [parent_id]))),
                        "target_horizon": "multi",
                    }
                )
                continue
            subset = df.loc[parent_mask].copy()
            metrics = self._multi_horizon_metrics(subset)
            distinct_products, largest_share = self._product_concentration(subset)
            robustness_verdict, novelty_verdict = self._verdict(len(subset), distinct_products, largest_share)
            row = {
                "result_type": "execution",
                "rule_instance_id": f"{rule['merged_rule_id']}::{parent_id}",
                "merged_rule_id": rule["merged_rule_id"],
                "rule_name": rule.get("name"),
                "variant_id": parent_id,
                "status": "ok",
                "support_rows": int(len(subset)),
                "distinct_products": distinct_products,
                "largest_product_share": largest_share,
                "target_horizon": "multi",
                "candidate_mean_forward_return": metrics.get("h4_mean_forward_return"),
                "candidate_mean_max_up_pct": metrics.get("h4_mean_max_up_pct"),
                "baseline_type": f"applies_to:{parent_id}",
                "concentration_verdict": "concentrated" if largest_share is not None and largest_share > 0.35 else "acceptable",
                "robustness_verdict": robustness_verdict,
                "novelty_verdict": novelty_verdict,
                "family_id": rule.get("family_id"),
                "source_library": rule.get("source_library", "builtin"),
                "source_attribution": "|".join(rule.get("source_attribution", [])),
                "hypothesis": rule.get("name"),
                "resolved_conditions": json.dumps(resolved_conditions),
                "missing_features": None,
                "applies_to_rule": parent_id,
            }
            row.update(metrics)
            rows.append(row)
        detail_df = pd.DataFrame(rows)
        detail_path = self.storage.write_csv(detail_df, f"rule_backtest_{rule['merged_rule_id']}_execution_audit__{run_id}", compress=False)
        artifacts = [self.storage.file_info(detail_path)]
        return rows, artifacts, detail_df

    def _evaluate_direct_instance(
        self,
        df: pd.DataFrame,
        instance: dict[str, Any],
        horizon: str,
        run_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, pd.Series | None]:
        mask, resolved_conditions, missing = self._build_rule_mask(df, instance["conditions"])
        baseline_mask, baseline_type = self._baseline_mask(df, instance["conditions"])
        if mask is None:
            summary_row = {
                "result_type": "individual",
                "rule_instance_id": instance["instance_id"],
                "merged_rule_id": instance["merged_rule_id"],
                "rule_name": instance["name"],
                "variant_id": instance.get("variant_id"),
                "status": "unavailable",
                "missing_features": ", ".join(sorted(set(missing))),
                "support_rows": 0,
                "distinct_products": 0,
                "largest_product_share": None,
                "target_horizon": horizon,
                "source_attribution": "|".join(instance.get("source_attribution", [])),
                "source_library": instance.get("source_library", "builtin"),
                "family_id": instance.get("family_id"),
                "baseline_type": baseline_type,
                "rule_kind": instance.get("rule_kind", "direct_rule"),
            }
            return summary_row, None, None
        candidate = df.loc[mask].copy()
        baseline = df.loc[baseline_mask].copy()
        return_col, max_up_col, touch_col = self._target_columns(horizon)
        distinct_products, largest_share = self._product_concentration(candidate)
        candidate_touch_rate = float(candidate[touch_col].mean()) if touch_col and touch_col in candidate.columns and len(candidate) else None
        base_touch_rate = float(baseline[touch_col].mean()) if touch_col and touch_col in baseline.columns and len(baseline) else None
        absolute_uplift = (
            float(candidate_touch_rate - base_touch_rate)
            if candidate_touch_rate is not None and base_touch_rate is not None
            else None
        )
        candidate_mean_return = float(candidate[return_col].mean()) if len(candidate) else None
        baseline_mean_return = float(baseline[return_col].mean()) if len(baseline) else None
        mean_forward_return_change = (
            float(candidate_mean_return - baseline_mean_return)
            if candidate_mean_return is not None and baseline_mean_return is not None
            else None
        )
        candidate_mean_max_up = float(candidate[max_up_col].mean()) if len(candidate) else None
        median_per_product_uplift = self._median_per_product_uplift(candidate, baseline, return_col)
        robustness_verdict, novelty_verdict = self._verdict(len(candidate), distinct_products, largest_share)
        summary_row = {
            "result_type": "individual",
            "rule_instance_id": instance["instance_id"],
            "merged_rule_id": instance["merged_rule_id"],
            "rule_name": instance["name"],
            "variant_id": instance.get("variant_id"),
            "status": "ok",
            "support_rows": int(len(candidate)),
            "distinct_products": distinct_products,
            "largest_product_share": largest_share,
            "target_horizon": horizon,
            "base_touch_rate": base_touch_rate,
            "candidate_touch_rate": candidate_touch_rate,
            "absolute_uplift_in_touch_rate": absolute_uplift,
            "baseline_mean_forward_return": baseline_mean_return,
            "candidate_mean_forward_return": candidate_mean_return,
            "mean_forward_return_change": mean_forward_return_change,
            "candidate_mean_max_up_pct": candidate_mean_max_up,
            "median_per_product_uplift": median_per_product_uplift,
            "baseline_type": baseline_type,
            "concentration_verdict": "concentrated" if largest_share is not None and largest_share > 0.35 else "acceptable",
            "robustness_verdict": robustness_verdict,
            "novelty_verdict": novelty_verdict,
            "family_id": instance.get("family_id"),
            "source_library": instance.get("source_library", "builtin"),
            "source_attribution": "|".join(instance.get("source_attribution", [])),
            "hypothesis": instance.get("hypothesis"),
            "resolved_conditions": json.dumps(resolved_conditions),
            "missing_features": None,
            "rule_kind": instance.get("rule_kind", "direct_rule"),
        }
        summary_row.update(self._multi_horizon_metrics(candidate, prefix="candidate_"))
        summary_row.update(self._multi_horizon_metrics(baseline, prefix="baseline_"))

        matched_export = candidate[self._matched_export_columns(candidate, resolved_conditions, horizon)].copy()
        matched_export.insert(0, "rule_instance_id", instance["instance_id"])
        matched_export.insert(1, "merged_rule_id", instance["merged_rule_id"])
        matched_export.insert(2, "rule_name", instance["name"])
        matched_export.insert(3, "source_library", instance.get("source_library", "builtin"))
        suffix = f"{instance['instance_id'].replace(':', '_')}_{horizon}"
        matched_path = self.storage.write_csv(matched_export, f"rule_backtest_{suffix}_matched_rows__{run_id}", compress=True)
        artifact = self.storage.file_info(matched_path)
        return summary_row, artifact, mask

    def run(self, request: RuleBacktestRequest) -> dict[str, Any]:
        feature_df = self.storage.read_frame("feature_table")
        if feature_df.empty:
            raise RuntimeError("feature_table dataset is missing. Compute features first.")
        library = self._combined_library()
        available_rules = {rule["merged_rule_id"]: rule for rule in library.get("candidate_rules", [])}
        selected_rule_ids = request.rule_ids or sorted(available_rules)
        if request.selection_mode == "all":
            selected_rule_ids = sorted(available_rules)
        selected_rules = [available_rules[rule_id] for rule_id in selected_rule_ids if rule_id in available_rules]
        if not selected_rules:
            raise ValueError("No valid rules selected.")

        df = self._prepare_frame(feature_df)
        run_id = self.storage.make_run_id()
        summary_rows: list[dict[str, Any]] = []
        detailed_artifacts: list[dict[str, Any]] = []
        valid_masks: list[pd.Series] = []

        def _work_units() -> int:
            total = 0
            for rule in selected_rules:
                kind = rule.get("rule_kind", "direct_rule")
                if kind == "direct_rule":
                    for instance in self._resolve_rule_variants(rule):
                        total += max(1, len(self._resolve_requested_horizons(instance, request.horizon)))
                else:
                    total += 1
            return total

        total_instances = _work_units()
        processed = 0

        self.storage.update_status(
            "rule_backtest",
            "running",
            message="Backtesting persistent rule library",
            progress=f"0/{max(total_instances, 1)}",
            current_rule=None,
            selection_mode=request.selection_mode,
            run_mode=request.run_mode,
            horizon=request.horizon,
            selected_rule_count=len(selected_rules),
        )

        for rule in selected_rules:
            kind = rule.get("rule_kind", "direct_rule")
            if kind == "direct_rule":
                for instance in self._resolve_rule_variants(rule):
                    horizons = self._resolve_requested_horizons(instance, request.horizon)
                    for horizon in horizons:
                        processed += 1
                        self.storage.update_status(
                            "rule_backtest",
                            "running",
                            message="Backtesting persistent rule library",
                            progress=f"{processed}/{total_instances}",
                            current_rule=f"{instance['instance_id']}@{horizon}",
                            selection_mode=request.selection_mode,
                            run_mode=request.run_mode,
                            horizon=request.horizon,
                        )
                        summary_row, artifact, mask = self._evaluate_direct_instance(df, instance, horizon, run_id)
                        summary_rows.append(summary_row)
                        if artifact:
                            detailed_artifacts.append(artifact)
                        if mask is not None:
                            valid_masks.append(mask)
                continue

            processed += 1
            self.storage.update_status(
                "rule_backtest",
                "running",
                message="Backtesting persistent rule library",
                progress=f"{processed}/{total_instances}",
                current_rule=rule["merged_rule_id"],
                selection_mode=request.selection_mode,
                run_mode=request.run_mode,
                horizon=request.horizon,
            )
            if kind == "routing_hypothesis":
                rows, artifacts, _ = self._evaluate_routing_hypothesis(rule, df, available_rules, run_id)
            elif kind == "execution_test":
                rows, artifacts, _ = self._evaluate_execution_test(rule, df, available_rules, run_id)
            else:
                rows = [{
                    "result_type": "analysis_test",
                    "rule_instance_id": rule["merged_rule_id"],
                    "merged_rule_id": rule["merged_rule_id"],
                    "rule_name": rule.get("name"),
                    "status": "unavailable",
                    "missing_features": "This item is a qualitative audit template rather than a direct feature-table rule.",
                    "target_horizon": "multi",
                    "rule_kind": kind,
                }]
                artifacts = []
            summary_rows.extend(rows)
            detailed_artifacts.extend(artifacts)

        if request.run_mode in {"collective", "both"}:
            collective_horizon = request.horizon if request.horizon != "auto" else "h4"
            collective_name = "COLLECTIVE_ALL_RULES" if request.selection_mode == "all" else "COLLECTIVE_SELECTED_RULES"
            collective_summary, collective_df = self._collective_summary(collective_name, valid_masks, df, collective_horizon)
            summary_rows.append(collective_summary)
            collective_path = self.storage.write_csv(
                collective_df[self._matched_export_columns(collective_df, [], collective_horizon)],
                f"rule_backtest_{collective_name.lower()}_{collective_horizon}_matched_rows__{run_id}",
                compress=True,
            )
            detailed_artifacts.append(self.storage.file_info(collective_path))

        summary_df = pd.DataFrame(summary_rows)
        summary_name = f"rule_backtest_summary__{run_id}"
        summary_path = self.storage.write_csv(summary_df, summary_name, compress=False)
        selected_payload = {
            "run_id": run_id,
            "selection_mode": request.selection_mode,
            "run_mode": request.run_mode,
            "horizon": request.horizon,
            "selected_rule_ids": selected_rule_ids,
            "selected_rules": selected_rules,
            "library_counts": library.get("counts", {}),
        }
        selected_path = self.storage.export_path(f"rule_backtest_selection__{run_id}", ".json")
        self.storage.write_json(selected_payload, selected_path)
        results_payload = {
            "run_id": run_id,
            "status": "completed",
            "horizon": request.horizon,
            "selection_mode": request.selection_mode,
            "run_mode": request.run_mode,
            "selected_rule_ids": selected_rule_ids,
            "summary_rows": int(len(summary_df)),
            "successful_rows": int((summary_df.get("status") == "ok").sum()) if not summary_df.empty and "status" in summary_df.columns else 0,
            "artifacts": [summary_path.name, selected_path.name] + [item["name"] for item in detailed_artifacts],
        }
        results_json_path = self.storage.export_path(f"rule_backtest_results__{run_id}", ".json")
        self.storage.write_json(results_payload, results_json_path)

        pack_path = self.storage.export_path(f"rule_backtest_pack__{run_id}", ".zip")
        with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in [summary_path, selected_path, results_json_path]:
                zf.write(path, arcname=path.name)
            for item in detailed_artifacts:
                source = self.storage.settings.export_dir / item["name"]
                if source.exists():
                    zf.write(source, arcname=source.name)
        manifest = {
            "generated_at": pd.Timestamp.now("UTC").isoformat(),
            "run_id": run_id,
            "app": self.storage.settings.app_name,
            "version": self.storage.settings.app_version,
            "request": selected_payload,
            "artifacts": [self.storage.file_info(summary_path), self.storage.file_info(selected_path), self.storage.file_info(results_json_path), self.storage.file_info(pack_path)] + detailed_artifacts,
            "preview": summary_df.head(50).to_dict(orient="records"),
        }
        self.storage.write_latest_rule_backtest_manifest(manifest)
        self.storage.update_status(
            "rule_backtest",
            "completed",
            message="Backtesting persistent rule library",
            progress=f"{processed}/{max(total_instances, 1)}",
            current_rule=None,
            horizon=request.horizon,
            run_mode=request.run_mode,
            selection_mode=request.selection_mode,
            result_rows=int(len(summary_df)),
            summary_artifact=summary_path.name,
            pack_artifact=pack_path.name,
        )
        return {
            "run_id": run_id,
            "status": "completed",
            "summary_rows": summary_df.to_dict(orient="records"),
            "summary_artifact": self.storage.file_info(summary_path),
            "pack_artifact": self.storage.file_info(pack_path),
            "detail_artifacts": detailed_artifacts,
        }
