from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .settings import Settings


class StorageManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        for path in [
            self.settings.data_dir,
            self.settings.raw_dir,
            self.settings.processed_dir,
            self.settings.export_dir,
            self.settings.state_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        self.status_path = self.settings.state_dir / "run_status.json"
        self.summary_path = self.settings.export_dir / "run_summary.json"
        self.latest_manifest_path = self.settings.state_dir / "latest_run_manifest.json"
        self.latest_rule_backtest_manifest_path = self.settings.state_dir / "latest_rule_backtest_manifest.json"
        self.latest_live_shadow_manifest_path = self.settings.state_dir / "latest_live_shadow_manifest.json"
        self._status_lock = threading.Lock()

    def dataset_path(self, name: str, processed: bool = True) -> Path:
        base = self.settings.processed_dir if processed else self.settings.raw_dir
        return base / f"{name}.parquet"

    def pickle_dataset_path(self, name: str, processed: bool = True) -> Path:
        base = self.settings.processed_dir if processed else self.settings.raw_dir
        return base / f"{name}.pkl"

    def export_path(self, name: str, suffix: str) -> Path:
        return self.settings.export_dir / f"{name}{suffix}"

    def _atomic_write(self, target: Path, mode: str, writer: Callable[[Any], None], *, encoding: str | None = None) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = ''.join(target.suffixes) if target.suffixes else target.suffix
        tmp = target.with_name(f"{target.name}.tmp") if suffix else target.with_suffix('.tmp')
        try:
            with tmp.open(mode, encoding=encoding) as handle:
                writer(handle)
            os.replace(tmp, target)
            return target
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def write_frame(self, df: pd.DataFrame, name: str, processed: bool = True) -> Path:
        path = self.dataset_path(name, processed=processed)
        parquet_tmp = path.with_name(f"{path.name}.tmp")
        try:
            df.to_parquet(parquet_tmp, index=False)
            os.replace(parquet_tmp, path)
            return path
        except (ImportError, ModuleNotFoundError, ValueError):
            pickle_path = self.pickle_dataset_path(name, processed=processed)
            pickle_tmp = pickle_path.with_name(f"{pickle_path.name}.tmp")
            df.to_pickle(pickle_tmp)
            os.replace(pickle_tmp, pickle_path)
            return pickle_path
        finally:
            for tmp in (parquet_tmp, self.pickle_dataset_path(name, processed=processed).with_name(f"{self.pickle_dataset_path(name, processed=processed).name}.tmp")):
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass

    def read_frame(self, name: str, processed: bool = True) -> pd.DataFrame:
        path = self.dataset_path(name, processed=processed)
        pickle_path = self.pickle_dataset_path(name, processed=processed)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except (ImportError, ModuleNotFoundError, ValueError):
                if pickle_path.exists():
                    return pd.read_pickle(pickle_path)
                raise
        if pickle_path.exists():
            return pd.read_pickle(pickle_path)
        return pd.DataFrame()

    def write_csv(self, df: pd.DataFrame, name: str, compress: bool = False) -> Path:
        suffix = ".csv.gz" if compress else ".csv"
        path = self.export_path(name, suffix)
        kwargs = {"index": False}
        if compress:
            kwargs["compression"] = "gzip"
        self._atomic_write(path, "wb" if compress else "w", lambda handle: df.to_csv(handle, **kwargs), encoding=None if compress else "utf-8")
        return path

    def write_json(self, payload: Any, path: Path | None = None) -> Path:
        target = path or self.summary_path
        return self._atomic_write(target, "w", lambda f: json.dump(payload, f, indent=2, default=str), encoding="utf-8")

    def read_json(self, path: Path | None = None) -> dict[str, Any]:
        target = path or self.status_path
        if not target.exists():
            return {}
        try:
            with target.open("r", encoding="utf-8") as f:
                return json.load(f)
        except JSONDecodeError:
            return {}

    def make_run_id(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def versioned_export_path(self, name: str, suffix: str, run_id: str) -> Path:
        return self.settings.export_dir / f"{name}__{run_id}{suffix}"

    def snapshot_export(self, name: str, suffix: str, run_id: str) -> Path | None:
        source = self.export_path(name, suffix)
        if not source.exists():
            return None
        target = self.versioned_export_path(name, suffix, run_id)
        shutil.copy2(source, target)
        return target

    def file_info(self, path: Path) -> dict[str, Any]:
        return {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            "download_url": f"/download/{path.name}?v={int(path.stat().st_mtime)}",
        }

    def write_latest_manifest(self, payload: dict[str, Any]) -> Path:
        return self.write_json(payload, self.latest_manifest_path)

    def read_latest_manifest(self) -> dict[str, Any]:
        return self.read_json(self.latest_manifest_path)

    def write_latest_rule_backtest_manifest(self, payload: dict[str, Any]) -> Path:
        return self.write_json(payload, self.latest_rule_backtest_manifest_path)

    def read_latest_rule_backtest_manifest(self) -> dict[str, Any]:
        return self.read_json(self.latest_rule_backtest_manifest_path)

    def write_latest_live_shadow_manifest(self, payload: dict[str, Any]) -> Path:
        return self.write_json(payload, self.latest_live_shadow_manifest_path)

    def read_latest_live_shadow_manifest(self) -> dict[str, Any]:
        return self.read_json(self.latest_live_shadow_manifest_path)

    def list_latest_run_artifacts(self) -> list[dict[str, Any]]:
        manifest = self.read_latest_manifest()
        artifacts = manifest.get("artifacts", []) if isinstance(manifest, dict) else []
        if artifacts:
            return artifacts
        return self.list_exports()

    def update_status(self, step: str, status: str, **extra: Any) -> dict[str, Any]:
        with self._status_lock:
            payload = self.read_json(self.status_path)
            payload.setdefault("steps", {})
            payload["app"] = self.settings.app_name
            payload["version"] = self.settings.app_version
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            step_state = payload["steps"].get(step, {}).copy()
            if status in {"running", "completed", "queued"}:
                step_state.pop("error", None)
                step_state.pop("traceback", None)
            step_state.update({"status": status, **extra, "updated_at": payload["updated_at"]})
            payload["steps"][step] = step_state
            self.write_json(payload, self.status_path)
            return payload

    def list_exports(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in sorted(self.settings.export_dir.glob("*")):
            if path.is_file():
                out.append(self.file_info(path))
        return out
