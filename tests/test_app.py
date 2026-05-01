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
            "ts": pd.date_range("2024-01-01", periods=6, freq="H", tz="UTC"),
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
    assert status_payload['latest_run']['app_version'] == '1.2.0'

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
    assert latest_payload['version'] == '1.2.0'
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
    payload = Path('/mnt/data/updated_deep_crypto_unknown_pattern_test_pack.json').read_text()
    upload = client.post('/api/rule-backtests/library/upload', files=[('files', ('updated_pack.json', payload, 'application/json'))])
    assert upload.status_code == 200
    ids = set(upload.json()['uploaded_rule_ids'])
    assert 'UPDATED_RULE_001' in ids
    assert 'EXEC_TEST_001' in ids
