from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient


def build_client(tmp_path: Path) -> TestClient:
    os.environ["USE_MOCK_DATA"] = "true"
    os.environ["DATA_DIR"] = str(tmp_path / "runtime_data")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import app.settings as app_settings
    importlib.reload(app_settings)
    import app.main as app_main
    importlib.reload(app_main)
    return TestClient(app_main.app)


def test_end_to_end_pipeline(tmp_path: Path):
    client = build_client(tmp_path)

    assert client.get("/health").status_code == 200
    assert client.post("/api/universe/refresh").status_code == 200
    assert client.post("/api/mappings/refresh").status_code == 200
    assert client.post("/api/data/pull", json={"lookback_hours": 72, "max_products": 5}).status_code == 200
    assert client.post("/api/features/compute").status_code == 200
    export_resp = client.post("/api/export/build", json={"compress_chatgpt_csv": True})
    assert export_resp.status_code == 200

    rule_resp = client.post(
        "/api/rule-eval/run",
        json={
            "rule_name": "sample_rule",
            "conditions": [
                {"feature": "cb_ret_3", "operator": ">", "value": 0},
                {"feature": "cb_rel_volume_short", "operator": ">", "value": 1},
            ],
            "scopes": ["coinbase_only", "coinbase_plus_coinapi", "full_scope"],
            "target_column": "future_close_return_h4",
        },
    )
    assert rule_resp.status_code == 200
    payload = rule_resp.json()
    assert len(payload["summaries"]) == 3

    latest = client.get("/api/export/latest").json()["artifacts"]
    names = {item["name"] for item in latest}
    assert any(name.startswith("comparative_insight_report__") for name in names)
    assert any(name.startswith("chatgpt_ready_features__") for name in names)
    assert client.get("/api/reports/data-quality").status_code == 200
    assert client.get("/api/reports/comparative-insight").status_code == 200




def test_health_reports_effective_mock_modes_and_credentials(tmp_path: Path):
    client = build_client(tmp_path)
    payload = client.get('/health').json()
    assert payload['status'] == 'ok'
    assert payload['use_mock_data_flag'] is True
    assert payload['effective_mock_mode_coinbase'] is True
    assert payload['effective_mock_mode_coinapi'] is True
    assert payload['credentials_configured']['coinbase'] is False


def test_rule_eval_rejects_path_traversal_rule_name(tmp_path: Path):
    client = build_client(tmp_path)
    client.post('/api/pipeline/run', json={'lookback_hours': 72, 'max_products': 5, 'compress_chatgpt_csv': True})
    resp = client.post('/api/rule-eval/run', json={
        'rule_name': '../../../../tmp/PWNED',
        'conditions': [{'feature': 'cb_ret_1', 'operator': '>', 'value': -10}],
        'scopes': ['coinbase_only'],
        'target_column': 'future_close_return_h4',
    })
    assert resp.status_code == 422


def test_mock_coinapi_bars_preserve_ohlcv_ordering(tmp_path: Path):
    client = build_client(tmp_path)
    client.post('/api/pipeline/run', json={'lookback_hours': 72, 'max_products': 5, 'compress_chatgpt_csv': True})
    import app.main as app_main
    ca = app_main.storage.read_frame('coinapi_bars', processed=False)
    assert not ca.empty
    violations = ((ca['close'] > ca['high']) | (ca['close'] < ca['low']) | (ca['open'] > ca['high']) | (ca['open'] < ca['low'])).sum()
    assert int(violations) == 0

def test_universe_refresh_keeps_view_only_rows_by_default(tmp_path, monkeypatch):
    from app.pipeline import ResearchPipeline
    from app.settings import Settings

    settings = Settings(
        use_mock_data=False,
        coinbase_api_key_name='organizations/test/apiKeys/test',
        coinbase_api_private_key='dummy',
        coinapi_api_key='dummy',
        data_dir=tmp_path,
    )
    pipeline = ResearchPipeline(settings)

    monkeypatch.setattr(
        pipeline.coinbase,
        'list_products',
        lambda: [
            {
                'product_id': 'BTC-USD',
                'quote_currency_id': 'USD',
                'base_currency_id': 'BTC',
                'quote_display_symbol': 'USD',
                'base_display_symbol': 'BTC',
                'product_type': 'SPOT',
                'is_disabled': False,
                'trading_disabled': True,
                'view_only': True,
                'price': '100000',
                'volume_24h': '10',
                'approximate_quote_24h_volume': '1000000',
                'display_name': 'BTC-USD',
            }
        ],
    )

    summary = pipeline.refresh_universe()
    assert summary['rows'] == 1
    assert summary['eligible_rows'] == 1
    master = pipeline.storage.read_frame('coinbase_products')
    assert bool(master.loc[0, 'eligibility_flag']) is True


def test_pull_data_deduplicates_mapping_rows(tmp_path, monkeypatch):
    import pandas as pd
    from app.pipeline import ResearchPipeline
    from app.settings import Settings

    settings = Settings(
        use_mock_data=False,
        coinbase_api_key_name='organizations/test/apiKeys/test',
        coinbase_api_private_key='dummy',
        coinapi_api_key='dummy',
        data_dir=tmp_path,
    )
    pipeline = ResearchPipeline(settings)

    products = pd.DataFrame([
        {
            'product_id': 'BTC-USD',
            'base_asset': 'BTC',
            'quote_asset': 'USD',
            'eligibility_flag': True,
        }
    ])
    mapping = pd.DataFrame([
        {
            'coinbase_product_id': 'BTC-USD',
            'coinapi_symbol_id': 'COINBASE_SPOT_BTC_USD',
            'exchange_id': 'COINBASE',
            'mapping_status': 'mapped',
            'mapping_confidence': 1.0,
            'notes': 'Preferred exchange match',
        },
        {
            'coinbase_product_id': 'BTC-USD',
            'coinapi_symbol_id': 'GDAX_SPOT_BTC_USD',
            'exchange_id': 'GDAX',
            'mapping_status': 'mapped_cross_exchange',
            'mapping_confidence': 0.6,
            'notes': 'Fallback exchange match',
        },
    ])

    pipeline.storage.write_frame(products, 'coinbase_products')
    pipeline.storage.write_frame(mapping, 'coinapi_symbol_mapping')

    monkeypatch.setattr(
        pipeline.coinbase,
        'get_candles',
        lambda product_id, start, end, granularity: pd.DataFrame([
            {'product_id': product_id, 'ts': start, 'open': 1.0, 'high': 1.1, 'low': 0.9, 'close': 1.05, 'volume': 10.0}
        ]),
    )
    monkeypatch.setattr(
        pipeline.coinapi,
        'get_ohlcv',
        lambda symbol_id, start, end, period_id: pd.DataFrame([
            {'coinbase_product_id': 'BTC-USD', 'coinapi_symbol_id': symbol_id, 'ts': start, 'open': 1.0, 'high': 1.1, 'low': 0.9, 'close': 1.04, 'volume': 9.0, 'trades_count': 5}
        ]),
    )

    summary = pipeline.pull_data(lookback_hours=1, max_products=1)
    assert summary['coinbase_rows'] == 1
    assert summary['coinapi_rows'] == 1


def test_coinapi_time_format_is_whole_second_utc_without_offset(tmp_path):
    from datetime import datetime, timezone
    from app.clients import CoinAPIClient
    from app.settings import Settings

    client = CoinAPIClient(Settings(use_mock_data=False, coinapi_api_key='dummy', data_dir=tmp_path))
    formatted = client._format_time(datetime(2026, 4, 29, 22, 19, 26, 240278, tzinfo=timezone.utc))
    assert formatted == '2026-04-29T22:19:26'


def test_cross_source_close_diff_is_populated(tmp_path: Path):
    client = build_client(tmp_path)

    assert client.post("/api/universe/refresh").status_code == 200
    assert client.post("/api/mappings/refresh").status_code == 200
    assert client.post("/api/data/pull", json={"lookback_hours": 72, "max_products": 5}).status_code == 200
    assert client.post("/api/features/compute").status_code == 200

    import app.main as app_main
    feature_df = app_main.storage.read_frame('feature_table')

    assert "ca_close" in feature_df.columns
    assert feature_df["ca_close"].notna().sum() > 0
    assert "cs_coinbase_vs_coinapi_close_diff" in feature_df.columns
    assert feature_df["cs_coinbase_vs_coinapi_close_diff"].notna().sum() > 0


def test_relative_features_no_blowup_under_multi_quote():
    """Regression: add_relative_features must not Cartesian-blow-up when multiple
    products share a base asset (e.g. BTC-USD + BTC-USDC under multi-quote-currency
    universe). Pre-fix, the merge `on='ts'` matched each row in `out` to N benchmark
    rows per ts (where N = #quote variants), and the effect compounded across
    benchmarks (8x duplicate factor with 2 benchmarks × 2 quotes). In a real
    pipeline run this OOMs the worker."""
    import pandas as pd
    from app.features import add_relative_features, add_context_features

    rows = []
    for base in ["BTC", "ETH", "SOL"]:
        for quote in ["USD", "USDC"]:
            for ts_h in range(4):
                rows.append({
                    "product_id": f"{base}-{quote}",
                    "base_asset": base,
                    "quote_asset": quote,
                    "ts": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=ts_h),
                    "cb_ret_1": 0.001 * ts_h,
                    "cb_ret_6": 0.002 * ts_h,
                    "ca_ret_24": 0.003 * ts_h,
                    "ca_sma_20_dist": 0.0001 * ts_h,
                })
    df = pd.DataFrame(rows)
    expected_rows = len(df)
    expected_keys = df[["product_id", "ts"]].drop_duplicates().shape[0]

    out = add_relative_features(df, "cb", ["BTC", "ETH"])
    assert len(out) == expected_rows, (
        f"add_relative_features blew rows up from {expected_rows} to {len(out)}"
    )
    assert out[["product_id", "ts"]].drop_duplicates().shape[0] == expected_keys

    out2 = add_context_features(df)
    assert len(out2) == expected_rows, (
        f"add_context_features blew rows up from {expected_rows} to {len(out2)}"
    )


def test_feature_table_has_only_ca_close_not_other_raw_ca_ohlcv(tmp_path: Path):
    """Regression: after the v1.0.3 fix that renamed close→ca_close to feed
    cs_close_diff, only ca_close should be persisted. ca_open/ca_high/ca_low/
    ca_volume aren't used by any cross-source feature, so persisting them just
    inflates chatgpt_ready_features.csv and the comparative report's
    feature_increment section with trivially-redundant entries (~1.0 corr with
    cb_close). This test guards against that bloat reappearing."""
    client = build_client(tmp_path)
    assert client.post("/api/universe/refresh").status_code == 200
    assert client.post("/api/mappings/refresh").status_code == 200
    assert client.post("/api/data/pull", json={"lookback_hours": 72, "max_products": 5}).status_code == 200
    assert client.post("/api/features/compute").status_code == 200

    import app.main as app_main
    ft = app_main.storage.read_frame('feature_table')
    assert "ca_close" in ft.columns, "ca_close is required by cs_coinbase_vs_coinapi_close_diff"
    for unwanted in ("ca_open", "ca_high", "ca_low", "ca_volume"):
        assert unwanted not in ft.columns, (
            f"{unwanted} is raw input data not used by any cross-source feature; "
            "it inflates the comparative report's feature_increment section."
        )


def test_future_outcomes_require_full_horizon_and_preserve_tail_nans():
    import pandas as pd
    from app.features import add_future_outcomes

    df = pd.DataFrame(
        {
            "product_id": ["BTC-USD"] * 6,
            "ts": pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC"),
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100, 101, 102, 103, 104, 105],
        }
    )

    out = add_future_outcomes(df, horizons=[1, 4])

    assert out["future_close_return_h1"].isna().sum() == 1
    assert out["future_close_return_h4"].isna().sum() == 4
    assert out["future_max_up_pct_h4"].isna().sum() == 4
    assert out["touched_up_1pct_h4"].isna().sum() == 4
    assert out.loc[0, "future_close_return_h4"] == (104 / 100) - 1


def test_status_update_clears_stale_error_and_traceback(tmp_path):
    from app.settings import Settings
    from app.storage import StorageManager

    storage = StorageManager(Settings(data_dir=tmp_path))
    storage.update_status("data_pull", "failed", error="old error", traceback="old traceback")
    payload = storage.update_status("data_pull", "running", message="retrying")
    step = payload["steps"]["data_pull"]
    assert "error" not in step
    assert "traceback" not in step

    payload = storage.update_status("data_pull", "completed", rows=10)
    step = payload["steps"]["data_pull"]
    assert "error" not in step
    assert "traceback" not in step
    assert step["status"] == "completed"


def test_persisted_feature_artifact_has_unique_keys_and_headline_columns(tmp_path: Path):
    client = build_client(tmp_path)
    assert client.post("/api/universe/refresh").status_code == 200
    assert client.post("/api/mappings/refresh").status_code == 200
    assert client.post("/api/data/pull", json={"lookback_hours": 72, "max_products": 5}).status_code == 200
    assert client.post("/api/features/compute").status_code == 200

    import app.main as app_main
    ft = app_main.storage.read_frame('feature_table')
    assert ft.duplicated(["product_id", "ts"]).sum() == 0
    for col in [
        "cb_ret_1",
        "cs_coinbase_vs_coinapi_close_diff",
        "cs_coinbase_vs_coinapi_return_diff",
        "cs_cross_source_divergence_flag",
        "future_close_return_h4",
    ]:
        assert col in ft.columns
        assert ft[col].notna().sum() > 0


def test_pipeline_run_endpoint_uses_requested_pull_settings(tmp_path: Path):
    client = build_client(tmp_path)
    resp = client.post('/api/pipeline/run', json={'lookback_hours': 72, 'max_products': 5, 'compress_chatgpt_csv': True})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload['status'] == 'completed'
    assert payload['requested']['lookback_hours'] == 72
    assert payload['requested']['max_products'] == 5

    import app.main as app_main
    ft = app_main.storage.read_frame('feature_table')
    assert ft['product_id'].nunique() == 5


def test_index_merges_pipeline_controls_and_removes_custom_data_pull_card(tmp_path: Path):
    client = build_client(tmp_path)
    response = client.get('/')
    assert response.status_code == 200
    html = response.text
    assert 'Run full pipeline' in html
    assert 'Custom data pull' not in html
    assert '/api/pipeline/run' in html


def test_status_and_latest_exports_reflect_effective_run_settings_and_versioned_artifacts(tmp_path: Path):
    client = build_client(tmp_path)
    resp = client.post('/api/pipeline/run', json={'lookback_hours': 72, 'max_products': 5, 'compress_chatgpt_csv': True})
    assert resp.status_code == 200

    status_payload = client.get('/api/status').json()
    assert status_payload['effective_run_settings']['lookback_hours'] == 72
    assert status_payload['effective_run_settings']['max_products'] == 5
    assert status_payload['latest_run']['app_version'] == '1.8.1'

    latest = client.get('/api/export/latest').json()
    names = {item['name'] for item in latest['artifacts']}
    assert any(name.startswith('chatgpt_ready_features__') for name in names)
    assert any(name.startswith('comparative_insight_report__') for name in names)


def test_download_headers_disable_caching_for_latest_artifacts(tmp_path: Path):
    client = build_client(tmp_path)
    resp = client.post('/api/pipeline/run', json={'lookback_hours': 72, 'max_products': 5, 'compress_chatgpt_csv': True})
    assert resp.status_code == 200
    latest = client.get('/api/export/latest').json()
    artifact = latest['artifacts'][0]
    path = artifact.get('download_url') or f"/download/{artifact['name']}"
    download = client.get(path)
    assert download.status_code == 200
    assert 'no-store' in download.headers.get('cache-control', '').lower()


def test_rule_backtest_library_and_run(tmp_path: Path):
    client = build_client(tmp_path)
    assert client.post("/api/pipeline/run", json={"lookback_hours": 72, "max_products": 5, "compress_chatgpt_csv": True}).status_code == 200

    library_resp = client.get('/api/rule-backtests/library')
    assert library_resp.status_code == 200
    rules = library_resp.json()['rules']
    assert any(rule['merged_rule_id'] == 'MERGED_RULE_001' for rule in rules)

    run_resp = client.post('/api/rule-backtests/run', json={
        'rule_ids': ['MERGED_RULE_001', 'MERGED_RULE_002'],
        'selection_mode': 'selected',
        'run_mode': 'individual',
        'horizon': 'h4',
    })
    assert run_resp.status_code == 200
    payload = run_resp.json()
    assert payload['status'] == 'completed'
    assert any(row['rule_instance_id'].startswith('MERGED_RULE_001') for row in payload['summary_rows'])
    assert payload['pack_artifact']['name'].startswith('rule_backtest_pack__')

    latest = client.get('/api/rule-backtests/latest')
    assert latest.status_code == 200
    latest_payload = latest.json()
    assert latest_payload['version'] == '1.8.1'
    assert latest_payload['request']['horizon'] == 'h4'


def test_rule_backtest_collective_mode_returns_collective_row(tmp_path: Path):
    client = build_client(tmp_path)
    assert client.post("/api/pipeline/run", json={"lookback_hours": 72, "max_products": 5, "compress_chatgpt_csv": True}).status_code == 200

    run_resp = client.post('/api/rule-backtests/run', json={
        'rule_ids': ['MERGED_RULE_001', 'MERGED_RULE_005'],
        'selection_mode': 'selected',
        'run_mode': 'collective',
        'horizon': 'h4',
    })
    assert run_resp.status_code == 200
    payload = run_resp.json()
    assert any(row['result_type'] == 'collective' for row in payload['summary_rows'])



def test_rule_backtest_supports_inline_empirical_logic(tmp_path: Path):
    client = build_client(tmp_path)
    assert client.post("/api/pipeline/run", json={"lookback_hours": 72, "max_products": 5, "compress_chatgpt_csv": True}).status_code == 200

    run_resp = client.post('/api/rule-backtests/run', json={
        'rule_ids': ['MERGED_RULE_003'],
        'selection_mode': 'selected',
        'run_mode': 'individual',
        'horizon': 'h4',
    })
    assert run_resp.status_code == 200
    payload = run_resp.json()
    assert payload['status'] == 'completed'
    assert any(row['rule_instance_id'] == 'MERGED_RULE_003' for row in payload['summary_rows'])


def test_rule_library_batch_upload_persists_and_is_listed(tmp_path: Path):
    client = build_client(tmp_path)
    files = [
        ('files', ('one.json', '{"merged_rule_id":"CUSTOM_RULE_001","name":"Custom one","exact_definition":{"all_conditions":[{"field":"cb_ret_3","logic":">","value":0}]}}', 'application/json')),
        ('files', ('two.json', '{"candidate_rules":[{"merged_rule_id":"CUSTOM_RULE_002","name":"Custom two","exact_definition":{"all_conditions":[{"field":"cb_rel_volume_short","logic":">= empirical_top_decile_threshold"}]}}]}', 'application/json')),
    ]
    upload = client.post('/api/rule-backtests/library/upload', files=files)
    assert upload.status_code == 200
    payload = upload.json()
    assert payload['counts']['custom'] >= 2

    library = client.get('/api/rule-backtests/library')
    assert library.status_code == 200
    ids = {row['merged_rule_id'] for row in library.json()['rules']}
    assert 'CUSTOM_RULE_001' in ids
    assert 'CUSTOM_RULE_002' in ids

    # Rebuild client on same persisted data dir and confirm custom rules remain.
    client2 = build_client(tmp_path)
    ids2 = {row['merged_rule_id'] for row in client2.get('/api/rule-backtests/library').json()['rules']}
    assert 'CUSTOM_RULE_001' in ids2
    assert 'CUSTOM_RULE_002' in ids2


def test_index_has_batch_rule_upload_and_mobile_friendly_markup(tmp_path: Path):
    client = build_client(tmp_path)
    response = client.get('/')
    assert response.status_code == 200
    html = response.text
    assert 'Batch JSON upload' in html
    assert 'multiple' in html
    assert '@media (max-width: 640px)' in html



def test_rule_library_includes_updated_pack_and_auto_horizons(tmp_path: Path):
    client = build_client(tmp_path)
    library_resp = client.get('/api/rule-backtests/library')
    assert library_resp.status_code == 200
    rules = library_resp.json()['rules']
    updated = {row['merged_rule_id']: row for row in rules}
    assert 'UPDATED_RULE_001' in updated
    assert updated['UPDATED_RULE_001']['recommended_primary_horizon'] == 'h4'
    assert 'UPDATED_RULE_005' in updated
    assert set(updated['UPDATED_RULE_005']['target_horizons']) == {'h1', 'h4', 'h24'}


def test_updated_pack_direct_rule_and_routing_and_execution_backtests(tmp_path: Path):
    client = build_client(tmp_path)
    assert client.post('/api/pipeline/run', json={'lookback_hours': 72, 'max_products': 5, 'compress_chatgpt_csv': True}).status_code == 200

    direct = client.post('/api/rule-backtests/run', json={
        'rule_ids': ['UPDATED_RULE_001'],
        'selection_mode': 'selected',
        'run_mode': 'individual',
        'horizon': 'h4',
    })
    assert direct.status_code == 200
    direct_rows = direct.json()['summary_rows']
    assert any(row['rule_instance_id'] == 'UPDATED_RULE_001' for row in direct_rows)

    routing = client.post('/api/rule-backtests/run', json={
        'rule_ids': ['UPDATED_RULE_003'],
        'selection_mode': 'selected',
        'run_mode': 'individual',
        'horizon': 'auto',
    })
    assert routing.status_code == 200
    routing_rows = routing.json()['summary_rows']
    assert any(row.get('result_type') == 'routing' and row.get('status') == 'ok' for row in routing_rows)

    execution = client.post('/api/rule-backtests/run', json={
        'rule_ids': ['EXEC_TEST_001'],
        'selection_mode': 'selected',
        'run_mode': 'individual',
        'horizon': 'auto',
    })
    assert execution.status_code == 200
    execution_rows = execution.json()['summary_rows']
    assert any(row.get('result_type') == 'execution' and row.get('status') == 'ok' for row in execution_rows)


def test_updated_pack_upload_supports_candidate_rules_and_execution_tests(tmp_path: Path):
    client = build_client(tmp_path)
    payload = ((Path(__file__).resolve().parents[1] / "app" / "resources" / "updated_deep_crypto_unknown_pattern_test_pack.json")).read_text()
    upload = client.post('/api/rule-backtests/library/upload', files=[('files', ('updated_pack.json', payload, 'application/json'))])
    assert upload.status_code == 200
    ids = set(upload.json()['uploaded_rule_ids'])
    assert 'UPDATED_RULE_001' in ids
    assert 'EXEC_TEST_001' in ids



def test_live_shadow_cycle_and_latest_manifest(tmp_path: Path):
    client = build_client(tmp_path)
    assert client.post('/api/universe/refresh').status_code == 200
    assert client.post('/api/mappings/refresh').status_code == 200
    run_resp = client.post('/api/live/shadow/run', json={
        'selection_mode': 'selected',
        'rule_ids': ['MERGED_RULE_001'],
        'lookback_hours': 72,
        'max_products': 5,
        'refresh_references': False,
        'as_of_time_iso': '2026-05-01T12:00:00Z',
    })
    assert run_resp.status_code == 200
    payload = run_resp.json()
    assert payload['version'] == '1.8.1'
    assert payload['status'] == 'queued'

    latest = client.get('/api/live/shadow/latest')
    assert latest.status_code == 200
    latest_payload = latest.json()
    assert latest_payload['version'] == '1.8.1'
    assert latest_payload['request']['lookback_hours'] == 72
    assert any(item['name'].startswith('live_validation_pack__') for item in latest_payload['artifacts'])


def test_live_shadow_isolates_summary_from_legacy_and_other_scope_rows(tmp_path: Path):
    import pandas as pd
    import app.main as app_main

    client = build_client(tmp_path)
    assert client.post('/api/universe/refresh').status_code == 200
    assert client.post('/api/mappings/refresh').status_code == 200

    legacy_signal_log = pd.DataFrame([
        {
            'signal_id': 'legacy-sig-1',
            'signal_ts': pd.Timestamp('2026-04-30T12:00:00Z'),
            'product_id': 'BTC-USD',
            'rule_instance_id': 'LEGACY_RULE__v1',
            'merged_rule_id': 'LEGACY_RULE',
            'rule_name': 'Legacy Rule',
            'signal_price': 100.0,
        },
        {
            'signal_id': 'other-scope-sig-1',
            'signal_ts': pd.Timestamp('2026-04-30T13:00:00Z'),
            'product_id': 'ETH-USD',
            'rule_instance_id': 'OTHER_SCOPE_RULE__v1',
            'merged_rule_id': 'OTHER_SCOPE_RULE',
            'rule_name': 'Other Scope Rule',
            'signal_price': 200.0,
            'validation_scope_key': 'selected_rules__1__otherscope',
            'validation_scope_rule_ids': 'OTHER_SCOPE_RULE',
        },
    ])
    legacy_outcomes = pd.DataFrame([
        {
            'signal_id': 'legacy-sig-1',
            'signal_ts': pd.Timestamp('2026-04-30T12:00:00Z'),
            'product_id': 'BTC-USD',
            'rule_instance_id': 'LEGACY_RULE__v1',
            'merged_rule_id': 'LEGACY_RULE',
            'rule_name': 'Legacy Rule',
            'signal_price': 100.0,
            'future_close_return_h1': 0.01,
        },
        {
            'signal_id': 'other-scope-sig-1',
            'signal_ts': pd.Timestamp('2026-04-30T13:00:00Z'),
            'product_id': 'ETH-USD',
            'rule_instance_id': 'OTHER_SCOPE_RULE__v1',
            'merged_rule_id': 'OTHER_SCOPE_RULE',
            'rule_name': 'Other Scope Rule',
            'signal_price': 200.0,
            'future_close_return_h1': 0.02,
            'validation_scope_key': 'selected_rules__1__otherscope',
            'validation_scope_rule_ids': 'OTHER_SCOPE_RULE',
        },
    ])
    app_main.storage.write_frame(legacy_signal_log, app_main.live_shadow_service.signal_log_name)
    app_main.storage.write_frame(legacy_outcomes, app_main.live_shadow_service.outcomes_name)

    run_resp = client.post('/api/live/shadow/run', json={
        'selection_mode': 'selected',
        'rule_ids': ['MERGED_RULE_001'],
        'lookback_hours': 72,
        'max_products': 5,
        'refresh_references': False,
        'as_of_time_iso': '2026-05-01T12:00:00Z',
    })
    assert run_resp.status_code == 200

    latest_payload = client.get('/api/live/shadow/latest').json()
    assert latest_payload['version'] == '1.8.1'
    assert latest_payload['request']['validation_scope_key'].startswith('selected_rules__1__')
    assert latest_payload['state_scope']['scope_isolated_by_rule_set'] is True
    assert latest_payload['state_scope']['legacy_signal_rows_ignored'] == 1
    assert latest_payload['state_scope']['other_scope_signal_rows_ignored'] == 1
    assert latest_payload['state_scope']['legacy_outcome_rows_ignored'] == 1
    assert latest_payload['state_scope']['other_scope_outcome_rows_ignored'] == 1
    assert all(row['merged_rule_id'] != 'LEGACY_RULE' for row in latest_payload['summary_rows'])
    assert all(row['merged_rule_id'] != 'OTHER_SCOPE_RULE' for row in latest_payload['summary_rows'])

    status_payload = client.get('/api/status').json()
    assert status_payload['latest_live_shadow']['state_scope']['scope_isolated_by_rule_set'] is True


def test_index_contains_live_shadow_tab_and_controls(tmp_path: Path):
    client = build_client(tmp_path)
    response = client.get('/')
    assert response.status_code == 200
    html = response.text
    assert 'data-tab-target="live"' in html
    assert 'Run live cycle for selected rules' in html
    assert '/api/live/shadow/run' in html


def test_live_shadow_resolve_outcomes_accepts_tz_aware_signal_ts(tmp_path: Path):
    import os
    import sys
    from datetime import datetime, timezone
    import pandas as pd
    import pytest

    os.environ["USE_MOCK_DATA"] = "true"
    os.environ["DATA_DIR"] = str(tmp_path / "runtime_data")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from app.settings import Settings
    from app.storage import StorageManager
    from app.rule_backtests import RuleBacktestService
    from app.live_shadow import LiveShadowService
    from app.pipeline import ResearchPipeline

    settings = Settings(data_dir=tmp_path / "runtime_data", use_mock_data=True)
    settings.app_version = "1.6.3"
    storage = StorageManager(settings)
    rule_service = RuleBacktestService(storage=storage)
    pipeline = ResearchPipeline(settings)
    service = LiveShadowService(settings=settings, storage=storage, pipeline=pipeline, rule_service=rule_service)

    signal_log = pd.DataFrame([{
        "signal_id": "sig-1",
        "signal_ts": pd.Timestamp("2026-05-03T20:00:00+00:00"),
        "product_id": "BTC-USD",
        "rule_instance_id": "rule-1",
        "merged_rule_id": "rule-1",
        "rule_name": "Rule 1",
        "signal_price": 100.0,
    }])
    existing_outcomes = pd.DataFrame()
    cb = pd.DataFrame([
        {"product_id": "BTC-USD", "ts": pd.Timestamp("2026-05-03T20:00:00+00:00"), "close": 100.0, "high": 101.0},
        {"product_id": "BTC-USD", "ts": pd.Timestamp("2026-05-03T21:00:00+00:00"), "close": 102.0, "high": 103.0},
        {"product_id": "BTC-USD", "ts": pd.Timestamp("2026-05-04T00:00:00+00:00"), "close": 105.0, "high": 106.0},
        {"product_id": "BTC-USD", "ts": pd.Timestamp("2026-05-04T20:00:00+00:00"), "close": 110.0, "high": 112.0},
    ])

    outcomes, resolved = service._resolve_outcomes(
        signal_log=signal_log,
        existing_outcomes=existing_outcomes,
        cb=cb,
        latest_available_ts=pd.Timestamp("2026-05-04T20:00:00+00:00"),
        resolved_at=datetime(2026, 5, 4, 20, 5, tzinfo=timezone.utc),
    )

    assert resolved == 1
    assert len(outcomes) == 1
    row = outcomes.iloc[0]
    assert row["future_close_return_h1"] == pytest.approx(0.02)
    assert row["future_close_return_h4"] == pytest.approx(0.05)
    assert row["future_close_return_h24"] == pytest.approx(0.10)


def test_live_rule_eligibility_update_and_scan_manifest(tmp_path: Path):
    client = build_client(tmp_path)
    library = client.get('/api/rule-backtests/library').json()['rules']
    direct_rule_ids = [rule['merged_rule_id'] for rule in library if rule.get('rule_kind', 'direct_rule') == 'direct_rule']
    assert direct_rule_ids

    eligibility = client.post('/api/rule-backtests/library/live-eligibility', json={
        'rule_ids': direct_rule_ids,
        'live_eligible': True,
    })
    assert eligibility.status_code == 200

    run_resp = client.post('/api/live/scan/run', json={
        'selection_mode': 'all',
        'lookback_hours': 72,
        'max_products': 6,
        'refresh_references': False,
        'as_of_time_iso': '2026-05-01T12:00:00Z',
    })
    assert run_resp.status_code == 200
    payload = run_resp.json()
    assert payload['status'] == 'queued'
    assert payload['version'] == '1.8.1'
    queued_run_id = payload['run_id']

    latest = client.get('/api/live/scan/latest')
    assert latest.status_code == 200
    latest_payload = latest.json()
    assert latest_payload['version'] == '1.8.1'
    assert latest_payload['run_id'] == queued_run_id
    assert latest_payload['request']['lookback_hours'] == 72
    assert latest_payload['summary']['rule_hits'] > 0
    assert latest_payload['summary']['shortlist_rows'] > 0
    assert 'near_match_rows' in latest_payload['summary']
    artifact_names = {item['name'] for item in latest_payload['artifacts']}
    assert any(name.startswith('live_scan_near_matches__') for name in artifact_names)
    assert any(name.startswith('rule_coverage_summary__') for name in artifact_names)
    assert any(name.startswith('near_match_replay_summary__') for name in artifact_names)
    assert any(name.startswith('rule_relaxation_candidates__') for name in artifact_names)
    assert any(name.startswith('coverage_quality_frontier__') for name in artifact_names)
    assert any(name.startswith('live_scan_pack__') for name in artifact_names)
    assert 'coverage_quality_frontier_preview' in latest_payload
    assert 'relaxation_candidate_preview' in latest_payload


def test_index_contains_live_scan_tab_and_controls(tmp_path: Path):
    client = build_client(tmp_path)
    response = client.get('/')
    assert response.status_code == 200
    html = response.text
    assert 'data-tab-target="scan"' in html
    assert 'Scan selected live-eligible rules' in html
    assert '/api/live/scan/run' in html
    assert 'Closest current candidates' in html
    assert 'Coverage health' in html
    assert 'Adaptive near-match replay' in html
    assert 'Relaxation candidates' in html


def test_live_scan_uses_live_scan_defaults_and_matching_path(tmp_path: Path):
    client = build_client(tmp_path)
    import app.main as app_main
    app_main.settings.live_scan_lookback_hours = 24
    app_main.settings.live_scan_max_products = 6

    library = client.get('/api/rule-backtests/library').json()['rules']
    direct_rule_ids = [rule['merged_rule_id'] for rule in library if rule.get('rule_kind', 'direct_rule') == 'direct_rule']
    client.post('/api/rule-backtests/library/live-eligibility', json={'rule_ids': direct_rule_ids, 'live_eligible': True})

    run = client.post('/api/live/scan/run', json={
        'selection_mode': 'all',
        'refresh_references': False,
        'as_of_time_iso': '2026-05-01T12:00:00Z',
    })
    assert run.status_code == 200
    queued_run_id = run.json()['run_id']

    latest = client.get('/api/live/scan/latest').json()
    assert latest.get('run_id') == queued_run_id
    assert latest['request']['lookback_hours'] == 24
    assert latest['summary']['rule_hits'] > 0
    assert latest['summary']['shortlist_rows'] > 0
    assert 'near_match_preview' in latest
    assert 'coverage_preview' in latest
    assert 'near_match_replay_preview' in latest
    assert 'relaxation_candidate_preview' in latest
    assert 'coverage_quality_frontier_preview' in latest


def test_live_scan_select_per_product_latest_includes_staggered_products():
    import pandas as pd
    from app.live_scanner import _select_per_product_latest

    df = pd.DataFrame({
        'product_id': ['A', 'A', 'B'],
        'ts': pd.to_datetime(['2024-01-01 11:00', '2024-01-01 12:00', '2024-01-01 11:00'], utc=True),
        'cb_close': [100, 101, 50],
    })
    snapshot, stale_products, latest_ts = _select_per_product_latest(df, freshness_hours=2)
    assert str(latest_ts) == '2024-01-01 12:00:00+00:00'
    assert set(snapshot['product_id']) == {'A', 'B'}
    assert stale_products == []


def test_adaptive_replay_generates_relaxation_outputs_from_near_misses(tmp_path: Path):
    client = build_client(tmp_path)
    import pandas as pd
    import app.main as app_main

    # Historical feature table with future outcomes: original rule is sparse; relaxed threshold adds coverage.
    hist = pd.DataFrame([
        {'product_id': 'A-USD', 'base_asset': 'A', 'quote_asset': 'USD', 'ts': pd.Timestamp('2026-01-01T00:00:00Z'), 'cb_ret_1': -0.030, 'future_close_return_h4': 0.030, 'future_max_up_pct_h4': 0.050, 'touched_up_1pct_h4': 1.0},
        {'product_id': 'B-USD', 'base_asset': 'B', 'quote_asset': 'USD', 'ts': pd.Timestamp('2026-01-01T01:00:00Z'), 'cb_ret_1': -0.018, 'future_close_return_h4': 0.020, 'future_max_up_pct_h4': 0.030, 'touched_up_1pct_h4': 1.0},
        {'product_id': 'C-USD', 'base_asset': 'C', 'quote_asset': 'USD', 'ts': pd.Timestamp('2026-01-01T02:00:00Z'), 'cb_ret_1': -0.012, 'future_close_return_h4': 0.010, 'future_max_up_pct_h4': 0.020, 'touched_up_1pct_h4': 1.0},
        {'product_id': 'D-USD', 'base_asset': 'D', 'quote_asset': 'USD', 'ts': pd.Timestamp('2026-01-01T03:00:00Z'), 'cb_ret_1': 0.010, 'future_close_return_h4': -0.010, 'future_max_up_pct_h4': 0.005, 'touched_up_1pct_h4': 0.0},
    ])
    app_main.storage.write_frame(hist, 'feature_table')
    snapshot = pd.DataFrame([
        {'product_id': 'A-USD', 'base_asset': 'A', 'quote_asset': 'USD', 'ts': pd.Timestamp('2026-02-01T00:00:00Z'), 'cb_ret_1': -0.016},
        {'product_id': 'B-USD', 'base_asset': 'B', 'quote_asset': 'USD', 'ts': pd.Timestamp('2026-02-01T00:00:00Z'), 'cb_ret_1': -0.013},
    ])
    rules = [{
        'merged_rule_id': 'RELAX_RULE',
        'name': 'Relax test rule',
        'priority': 1,
        'rule_kind': 'direct_rule',
        'live_eligible': True,
        'recommended_primary_horizon': 'h4',
        'target_horizons': ['h4'],
        'exact_definition': {'all_conditions': [{'field': 'cb_ret_1', 'logic': '<=', 'value': -0.02}]},
    }]

    replay, relaxation, frontier = app_main.live_scan_service._build_adaptive_replay_tables(
        snapshot=snapshot,
        rules=rules,
        latest_ts=pd.Timestamp('2026-02-01T00:00:00Z'),
    )

    assert not replay.empty
    assert not relaxation.empty
    assert not frontier.empty
    assert {'near_match_replay_summary__', 'rule_relaxation_candidates__', 'coverage_quality_frontier__'}
    assert relaxation['live_current_promoted_count'].max() >= 1
    assert 'recommendation' in frontier.columns


def test_rule_backtest_condition_mask_coerces_string_value_for_comparison_operators(tmp_path: Path):
    client = build_client(tmp_path)
    import app.main as app_main
    client.post('/api/pipeline/run', json={'lookback_hours': 72, 'max_products': 5, 'compress_chatgpt_csv': True})
    feature_df = app_main.storage.read_frame('feature_table')
    prepared = app_main.rule_backtest_service._prepare_frame(feature_df)
    mask, metadata = app_main.rule_backtest_service._condition_mask(prepared, {
        'field': 'cb_ret_1',
        'logic': '>',
        'value': '0.0',
    })
    assert mask is not None
    assert metadata['value'] == 0.0
    assert int(mask.sum()) > 0

def test_rule_library_exposes_live_candidate_recommendations(tmp_path: Path):
    client = build_client(tmp_path)
    rules = client.get('/api/rule-backtests/library').json()['rules']
    by_id = {rule['merged_rule_id']: rule for rule in rules}
    assert by_id['UPDATED_RULE_001']['live_candidate_recommended'] is True
    assert by_id['UPDATED_TEST_001']['rule_kind'] == 'analysis_test'
    assert by_id['UPDATED_TEST_001']['live_candidate_recommended'] is False
    assert by_id['MERGED_RULE_003']['live_candidate_recommended'] is False


def test_auto_apply_recommended_live_set_updates_live_eligibility(tmp_path: Path):
    client = build_client(tmp_path)
    resp = client.post('/api/rule-backtests/library/live-eligibility/auto', json={})
    assert resp.status_code == 200
    payload = resp.json()
    assert 'recommended_live_rule_ids' in payload
    rules = client.get('/api/rule-backtests/library').json()['rules']
    by_id = {rule['merged_rule_id']: rule for rule in rules}
    assert by_id['UPDATED_RULE_001']['live_eligible'] is True
    assert by_id['UPDATED_TEST_001']['live_eligible'] is False
    assert by_id['MERGED_RULE_003']['live_eligible'] is False


def test_downloadable_health_and_status_snapshots(tmp_path: Path):
    client = build_client(tmp_path)
    client.post('/api/pipeline/run', json={'lookback_hours': 72, 'max_products': 5, 'compress_chatgpt_csv': True})

    health = client.get('/api/health/download')
    assert health.status_code == 200
    assert 'attachment; filename="health__' in health.headers.get('content-disposition', '')
    assert health.json()['status'] == 'ok'

    status = client.get('/api/status/download')
    assert status.status_code == 200
    assert 'attachment; filename="status__' in status.headers.get('content-disposition', '')
    payload = status.json()
    assert payload['latest_run']['app_version'] == '1.8.1'
    assert payload['effective_run_settings']['lookback_hours'] == 72


def test_operator_snapshot_zip_contains_health_status_and_latest_manifests(tmp_path: Path):
    import io
    import json
    import zipfile

    client = build_client(tmp_path)
    client.post('/api/pipeline/run', json={'lookback_hours': 72, 'max_products': 5, 'compress_chatgpt_csv': True})

    response = client.get('/api/operator/snapshot/download')
    assert response.status_code == 200
    assert response.headers.get('content-type') == 'application/zip'
    assert 'attachment; filename="operator_snapshot__' in response.headers.get('content-disposition', '')

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        names = set(zf.namelist())
        assert 'health.json' in names
        assert 'status.json' in names
        assert 'latest_run_manifest.json' in names
        assert 'latest_rule_backtest_manifest.json' in names
        assert 'latest_live_shadow_manifest.json' in names
        assert 'latest_live_scan_manifest.json' in names
        status_payload = json.loads(zf.read('status.json').decode('utf-8'))
        assert status_payload['latest_run']['app_version'] == '1.8.1'


def test_status_normalizes_current_app_version_and_marks_stale_failures(tmp_path: Path):
    import app.main as app_main

    client = build_client(tmp_path)
    app_main.storage.write_json({
        'app': app_main.settings.app_name,
        'version': '1.6.3',
        'updated_at': '2026-05-11T20:00:00+00:00',
        'steps': {
            'live_shadow_cycle': {
                'status': 'failed',
                'error': "DataFrame index must be unique for orient='index'.",
                'updated_at': '2026-05-05T22:05:32.647564+00:00',
            }
        },
    }, app_main.storage.status_path)
    app_main.storage.write_latest_live_shadow_manifest({
        'run_id': '20260505T080327Z',
        'generated_at': '2026-05-05T08:05:33.736141+00:00',
        'version': '1.6.1',
        'summary': {},
        'artifacts': [],
        'summary_rows': [],
    })

    payload = client.get('/api/status').json()
    assert payload['status']['version'] == '1.8.1'
    step = payload['status']['steps']['live_shadow_cycle']
    assert step['stale_failure'] is True
    assert step['latest_manifest_version'] == '1.6.1'
    assert step['latest_manifest_current_version'] is False
    assert payload['latest_live_shadow']['is_current_version'] is False


def test_index_contains_operator_snapshot_download_controls(tmp_path: Path):
    client = build_client(tmp_path)
    response = client.get('/')
    assert response.status_code == 200
    html = response.text
    assert "Download status" in html
    assert "Download health" in html
    assert "/api/operator/snapshot/download" in html


def test_live_shadow_outcomes_deduplicate_duplicate_signal_ids_before_index_lookup(tmp_path: Path):
    import pandas as pd
    import app.main as app_main

    client = build_client(tmp_path)
    client.post('/api/universe/refresh')
    client.post('/api/mappings/refresh')

    signal_log = pd.DataFrame([
        {
            'signal_id': 'sig-1',
            'signal_ts': pd.Timestamp('2026-05-01T00:00:00Z'),
            'product_id': 'BTC-USD',
            'rule_instance_id': 'rule-1',
            'merged_rule_id': 'RULE-1',
            'rule_name': 'Rule 1',
            'signal_price': 100.0,
        }
    ])
    existing_outcomes = pd.DataFrame([
        {
            'signal_id': 'sig-1',
            'signal_ts': pd.Timestamp('2026-05-01T00:00:00Z'),
            'product_id': 'BTC-USD',
            'rule_instance_id': 'rule-1',
            'merged_rule_id': 'RULE-1',
            'rule_name': 'Rule 1',
            'future_close_return_h1': None,
            'resolved_at': '2026-05-01T01:00:00+00:00',
        },
        {
            'signal_id': 'sig-1',
            'signal_ts': pd.Timestamp('2026-05-01T00:00:00Z'),
            'product_id': 'BTC-USD',
            'rule_instance_id': 'rule-1',
            'merged_rule_id': 'RULE-1',
            'rule_name': 'Rule 1',
            'future_close_return_h1': 0.02,
            'resolved_at': '2026-05-01T02:00:00+00:00',
        },
    ])
    cb = pd.DataFrame([
        {'product_id': 'BTC-USD', 'ts': pd.Timestamp('2026-05-01T00:00:00Z'), 'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1.0},
        {'product_id': 'BTC-USD', 'ts': pd.Timestamp('2026-05-01T01:00:00Z'), 'open': 100.0, 'high': 103.0, 'low': 99.0, 'close': 102.0, 'volume': 1.0},
        {'product_id': 'BTC-USD', 'ts': pd.Timestamp('2026-05-01T04:00:00Z'), 'open': 102.0, 'high': 104.0, 'low': 101.0, 'close': 103.0, 'volume': 1.0},
        {'product_id': 'BTC-USD', 'ts': pd.Timestamp('2026-05-02T00:00:00Z'), 'open': 103.0, 'high': 105.0, 'low': 102.0, 'close': 104.0, 'volume': 1.0},
    ])

    outcomes, resolved = app_main.live_shadow_service._resolve_outcomes(
        signal_log=signal_log,
        existing_outcomes=existing_outcomes,
        cb=cb,
        latest_available_ts=pd.Timestamp('2026-05-02T00:00:00Z'),
        resolved_at=pd.Timestamp('2026-05-02T00:00:00Z').to_pydatetime(),
    )

    assert len(outcomes) == 1
    assert outcomes.iloc[0]['signal_id'] == 'sig-1'
    assert abs(float(outcomes.iloc[0]['future_close_return_h1']) - 0.02) < 1e-9
    assert resolved >= 0


def test_index_defines_download_snapshot_helper(tmp_path: Path):
    client = build_client(tmp_path)
    response = client.get('/')
    assert response.status_code == 200
    assert 'async function downloadSnapshot(url)' in response.text


def test_live_scan_adaptive_replay_failure_is_fail_soft(tmp_path: Path, monkeypatch):
    client = build_client(tmp_path)
    import app.main as app_main
    import pandas as pd

    library = client.get('/api/rule-backtests/library').json()['rules']
    direct_rule_ids = [rule['merged_rule_id'] for rule in library if rule.get('rule_kind', 'direct_rule') == 'direct_rule']
    client.post('/api/rule-backtests/library/live-eligibility', json={'rule_ids': direct_rule_ids, 'live_eligible': True})

    def boom(*args, **kwargs):
        raise RuntimeError('adaptive replay boom')

    monkeypatch.setattr(app_main.live_scan_service, '_build_adaptive_replay_tables', boom)
    run = client.post('/api/live/scan/run', json={
        'selection_mode': 'all',
        'lookback_hours': 72,
        'max_products': 6,
        'refresh_references': False,
        'as_of_time_iso': '2026-05-01T12:00:00Z',
    })
    assert run.status_code == 200
    latest = client.get('/api/live/scan/latest').json()
    assert latest['version'] == '1.8.1'
    assert latest['summary']['adaptive_replay_warning'] == 'adaptive replay boom'
    artifact_names = {item['name'] for item in latest['artifacts']}
    assert any(name.startswith('live_scan_pack__') for name in artifact_names)
    assert any(name.startswith('near_match_replay_summary__') for name in artifact_names)
    assert latest['near_match_replay_preview'][0]['status'] == 'adaptive_replay_failed'


def test_status_marks_long_running_steps_as_stale(tmp_path: Path):
    client = build_client(tmp_path)
    import app.main as app_main

    app_main.storage.write_json({
        'app': app_main.settings.app_name,
        'version': '1.8.0',
        'updated_at': '2026-05-15T22:00:00+00:00',
        'steps': {
            'live_scan_cycle': {
                'status': 'running',
                'run_id': 'old-run',
                'phase': 'adaptive_replay',
                'updated_at': '2020-01-01T00:00:00+00:00',
            }
        },
    }, app_main.storage.status_path)

    payload = client.get('/api/status').json()
    step = payload['status']['steps']['live_scan_cycle']
    assert step['status'] == 'running'
    assert step['stale_running'] is True
    assert 'more than 30 minutes' in step['stale_running_reason']
