from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi import Request

from .pipeline import ResearchPipeline
from .live_shadow import LiveShadowService
from .live_scanner import LiveScannerService
from .rule_backtests import RuleBacktestService
from .rule_eval import RuleEvaluationService
from .schemas import DataPullRequest, ExportBuildRequest, LiveScanRequest, LiveShadowRequest, PipelineRunRequest, RuleBacktestRequest, RuleEvalRequest
from .settings import get_settings
from .storage import StorageManager

settings = get_settings()
storage = StorageManager(settings)
pipeline = ResearchPipeline(settings)
rule_service = RuleEvaluationService(storage)
rule_backtest_service = RuleBacktestService(storage)
live_shadow_service = LiveShadowService(settings, storage, pipeline, rule_backtest_service)
live_scan_service = LiveScannerService(settings, storage, live_shadow_service, rule_backtest_service)

app = FastAPI(title=settings.app_name, version=settings.app_version)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
MAX_UPLOAD_BYTES = 2 * 1024 * 1024


def _cache_busting_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }


def _snapshot_suffix() -> str:
    latest_ids = [
        storage.read_latest_live_scan_manifest().get("run_id"),
        storage.read_latest_live_shadow_manifest().get("run_id"),
        storage.read_latest_rule_backtest_manifest().get("run_id"),
        storage.read_latest_manifest().get("run_id"),
    ]
    for run_id in latest_ids:
        if run_id:
            return str(run_id)
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_download_response(filename: str, payload: dict) -> Response:
    headers = _cache_busting_headers()
    headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return Response(content=json.dumps(payload, indent=2, default=str), media_type="application/json", headers=headers)


def _status_payload() -> dict:
    latest_manifest = storage.read_latest_manifest()
    latest_rule_backtest = storage.read_latest_rule_backtest_manifest()
    latest_live_shadow = storage.read_latest_live_shadow_manifest()
    latest_live_scan = storage.read_latest_live_scan_manifest()
    return {
        "status": storage.read_json(storage.status_path),
        "exports": storage.list_latest_run_artifacts(),
        "settings": {
            "quote_currencies": settings.quote_currencies,
            "lookback_hours": settings.lookback_hours,
            "preferred_bar_granularity": settings.preferred_bar_granularity,
            "coinapi_period_id": settings.coinapi_period_id,
            "top_n_by_volume": settings.top_n_by_volume,
            "max_universe_size": settings.max_universe_size,
            "live_shadow_lookback_hours": settings.live_shadow_lookback_hours,
            "live_shadow_max_products": settings.live_shadow_max_products,
            "live_scan_lookback_hours": settings.live_scan_lookback_hours,
            "live_scan_max_products": settings.live_scan_max_products,
            "mock_mode": settings.use_mock_data,
        },
        "effective_run_settings": latest_manifest.get("effective_run_settings", {}),
        "latest_run": {
            "run_id": latest_manifest.get("run_id"),
            "generated_at": latest_manifest.get("generated_at"),
            "app_version": latest_manifest.get("version"),
        },
        "latest_rule_backtest": {
            "run_id": latest_rule_backtest.get("run_id"),
            "generated_at": latest_rule_backtest.get("generated_at"),
            "app_version": latest_rule_backtest.get("version"),
            "horizon": latest_rule_backtest.get("request", {}).get("horizon"),
            "artifacts": latest_rule_backtest.get("artifacts", []),
        },
        "latest_live_shadow": {
            "run_id": latest_live_shadow.get("run_id"),
            "generated_at": latest_live_shadow.get("generated_at"),
            "app_version": latest_live_shadow.get("version"),
            "request": latest_live_shadow.get("request", {}),
            "summary": latest_live_shadow.get("summary", {}),
            "artifacts": latest_live_shadow.get("artifacts", []),
            "summary_rows": latest_live_shadow.get("summary_rows", []),
        },
        "latest_live_scan": {
            "run_id": latest_live_scan.get("run_id"),
            "generated_at": latest_live_scan.get("generated_at"),
            "app_version": latest_live_scan.get("version"),
            "request": latest_live_scan.get("request", {}),
            "summary": latest_live_scan.get("summary", {}),
            "artifacts": latest_live_scan.get("artifacts", []),
            "preview": latest_live_scan.get("preview", []),
        },
    }


def _health_payload() -> dict:
    cb_mock = pipeline.coinbase.mock_mode
    ca_mock = pipeline.coinapi.mock_mode
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "use_mock_data_flag": settings.use_mock_data,
        "effective_mock_mode_coinbase": cb_mock,
        "effective_mock_mode_coinapi": ca_mock,
        "credentials_configured": {
            "coinbase": bool(settings.coinbase_api_key_name and settings.coinbase_api_private_key),
            "coinapi": bool(settings.coinapi_api_key),
        },
    }


def _operator_snapshot_payload() -> dict:
    generated_at = datetime.now(timezone.utc).isoformat()
    return {
        "generated_at": generated_at,
        "app": settings.app_name,
        "version": settings.app_version,
        "health": _health_payload(),
        "status": _status_payload(),
        "latest_manifests": {
            "latest_run_manifest": storage.read_latest_manifest(),
            "latest_rule_backtest_manifest": storage.read_latest_rule_backtest_manifest(),
            "latest_live_shadow_manifest": storage.read_latest_live_shadow_manifest(),
            "latest_live_scan_manifest": storage.read_latest_live_scan_manifest(),
        },
    }


def _operator_snapshot_zip_response() -> Response:
    snapshot = _operator_snapshot_payload()
    suffix = _snapshot_suffix()
    readme = (
        "Operator snapshot bundle for share-back support.\n\n"
        "Contents:\n"
        "- health.json: current /health response\n"
        "- status.json: current /api/status response\n"
        "- latest_*_manifest.json: latest run manifests for pipeline, backtest, live shadow, and live scan\n"
        "- operator_snapshot_meta.json: bundle metadata\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("health.json", json.dumps(snapshot["health"], indent=2, default=str))
        zf.writestr("status.json", json.dumps(snapshot["status"], indent=2, default=str))
        for name, payload in snapshot["latest_manifests"].items():
            zf.writestr(f"{name}.json", json.dumps(payload, indent=2, default=str))
        zf.writestr(
            "operator_snapshot_meta.json",
            json.dumps({
                "generated_at": snapshot["generated_at"],
                "app": snapshot["app"],
                "version": snapshot["version"],
                "snapshot_suffix": suffix,
            }, indent=2, default=str),
        )
        zf.writestr("README.txt", readme)
    headers = _cache_busting_headers()
    headers["Content-Disposition"] = f'attachment; filename="operator_snapshot__{suffix}.zip"'
    return Response(content=buffer.getvalue(), media_type="application/zip", headers=headers)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "status": storage.read_json(storage.status_path),
            "exports": storage.list_latest_run_artifacts(),
            "latest_rule_backtest": storage.read_latest_rule_backtest_manifest(),
            "latest_live_shadow": storage.read_latest_live_shadow_manifest(),
            "latest_live_scan": storage.read_latest_live_scan_manifest(),
            "rule_library": {"rules": rule_backtest_service.list_rules()},
        },
    )


@app.get("/health")
def health() -> dict:
    return _health_payload()


@app.get("/api/status")
def api_status() -> dict:
    return _status_payload()




@app.get("/api/health/download")
def api_health_download() -> Response:
    suffix = _snapshot_suffix()
    return _json_download_response(f"health__{suffix}.json", _health_payload())


@app.get("/api/status/download")
def api_status_download() -> Response:
    suffix = _snapshot_suffix()
    return _json_download_response(f"status__{suffix}.json", _status_payload())


@app.get("/api/operator/snapshot/download")
def api_operator_snapshot_download() -> Response:
    return _operator_snapshot_zip_response()


@app.post("/api/universe/refresh")
def api_refresh_universe() -> dict:
    try:
        return pipeline.refresh_universe()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/mappings/refresh")
def api_refresh_mappings() -> dict:
    try:
        return pipeline.refresh_mappings()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/data/pull")
def api_pull_data(payload: DataPullRequest) -> dict:
    try:
        return pipeline.pull_data(lookback_hours=payload.lookback_hours, max_products=payload.max_products)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/features/compute")
def api_compute_features() -> dict:
    try:
        return pipeline.compute_features()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/export/build")
def api_build_exports(payload: ExportBuildRequest) -> dict:
    try:
        return pipeline.build_exports(compress_chatgpt_csv=payload.compress_chatgpt_csv)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/pipeline/run")
def api_run_pipeline(payload: PipelineRunRequest) -> dict:
    try:
        universe = pipeline.refresh_universe()
        mappings = pipeline.refresh_mappings()
        data_pull = pipeline.pull_data(lookback_hours=payload.lookback_hours, max_products=payload.max_products)
        features = pipeline.compute_features()
        exports = pipeline.build_exports(compress_chatgpt_csv=payload.compress_chatgpt_csv)
        return {
            "status": "completed",
            "requested": {
                "lookback_hours": payload.lookback_hours,
                "max_products": payload.max_products,
                "compress_chatgpt_csv": payload.compress_chatgpt_csv,
            },
            "steps": {
                "universe_refresh": universe,
                "mapping_refresh": mappings,
                "data_pull": data_pull,
                "feature_compute": features,
                "export_build": exports,
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/export/latest")
def api_export_latest() -> dict:
    manifest = storage.read_latest_manifest()
    return {
        "run_id": manifest.get("run_id"),
        "generated_at": manifest.get("generated_at"),
        "effective_run_settings": manifest.get("effective_run_settings", {}),
        "artifacts": storage.list_latest_run_artifacts(),
    }


@app.get("/api/reports/data-quality")
def api_data_quality() -> JSONResponse:
    report_path = storage.export_path("data_quality_report", ".csv")
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Data-quality report not found.")
    return JSONResponse(content={"artifact": report_path.name, "preview": report_path.read_text(encoding="utf-8").splitlines()[:20]})


@app.get("/api/reports/comparative-insight")
def api_comparative_insight() -> JSONResponse:
    report_path = storage.export_path("comparative_insight_report", ".csv")
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Comparative insight report not found.")
    return JSONResponse(content={"artifact": report_path.name, "preview": report_path.read_text(encoding="utf-8").splitlines()[:20]})


@app.post("/api/rule-eval/run")
def api_rule_eval(payload: RuleEvalRequest) -> dict:
    try:
        return rule_service.run(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/rule-backtests/library")
def api_rule_backtest_library() -> dict:
    try:
        return {"rules": rule_backtest_service.list_rules()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/rule-backtests/library/upload")
async def api_rule_backtest_library_upload(files: list[UploadFile] = File(default=[]), pasted_json: str | None = Form(default=None)) -> dict:
    try:
        file_payloads: list[tuple[str, str]] = []
        for upload in files:
            if upload.content_type not in {None, "", "application/json", "text/plain"}:
                raise HTTPException(status_code=415, detail=f"Unsupported content type for {upload.filename or 'upload'}.")
            data = await upload.read()
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="File too large (> 2 MB).")
            file_payloads.append((upload.filename or "uploaded_rules.json", data.decode("utf-8")))
        if pasted_json and len(pasted_json.encode("utf-8")) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Pasted JSON too large (> 2 MB).")
        return rule_backtest_service.upload_rules(file_payloads=file_payloads, pasted_json=pasted_json)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/rule-backtests/library/live-eligibility")
def api_rule_backtest_library_live_eligibility(payload: dict) -> dict:
    try:
        rule_ids = payload.get("rule_ids") or []
        live_eligible = bool(payload.get("live_eligible", True))
        return rule_backtest_service.update_live_eligibility(rule_ids=rule_ids, live_eligible=live_eligible)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/rule-backtests/library/live-eligibility/auto")
def api_rule_backtest_library_live_eligibility_auto(payload: dict | None = None) -> dict:
    try:
        payload = payload or {}
        rule_ids = payload.get("rule_ids") or None
        return rule_backtest_service.apply_live_candidate_policy(rule_ids=rule_ids)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/rule-backtests/run")
def api_rule_backtests_run(payload: RuleBacktestRequest) -> dict:
    try:
        return rule_backtest_service.run(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/rule-backtests/latest")
def api_rule_backtests_latest() -> dict:
    return storage.read_latest_rule_backtest_manifest()


@app.post("/api/live/scan/run")
def api_live_scan_run(payload: LiveScanRequest, background_tasks: BackgroundTasks) -> dict:
    try:
        run_id = storage.make_run_id()
        storage.update_status(
            "live_scan_cycle",
            "queued",
            message="Live scanner queued",
            run_id=run_id,
            phase="queued",
        )
        background_tasks.add_task(live_scan_service.run_cycle, payload, run_id)
        return {"run_id": run_id, "status": "queued", "version": settings.app_version}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/live/scan/latest")
def api_live_scan_latest() -> dict:
    return live_scan_service.latest_manifest()


@app.post("/api/live/shadow/run")
def api_live_shadow_run(payload: LiveShadowRequest, background_tasks: BackgroundTasks) -> dict:
    try:
        run_id = storage.make_run_id()
        storage.update_status(
            "live_shadow_cycle",
            "queued",
            message="Live shadow validation cycle queued",
            run_id=run_id,
            phase="queued",
        )
        background_tasks.add_task(live_shadow_service.run_cycle, payload, run_id)
        return {"run_id": run_id, "status": "queued", "version": settings.app_version}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/live/shadow/latest")
def api_live_shadow_latest() -> dict:
    return live_shadow_service.latest_manifest()


@app.get("/download/{filename}")
def download_file(filename: str):
    path = settings.export_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(
        path,
        filename=path.name,
        headers=_cache_busting_headers(),
    )
