# Coinbase Crypto Research Collector v1.6.4

A FastAPI research app for collecting a Coinbase-defined spot crypto universe, mapping it to CoinAPI symbols, computing compact research features, and exporting comparison-ready datasets that preserve feature provenance.

## What this app is for

This app is built to answer one operational question cleanly:

**Do CoinAPI-derived compact features add incremental research value beyond Coinbase Advanced Trade data alone?**

It does that by keeping feature families separated and exportable across three scopes:

1. **Coinbase-only**
2. **Coinbase + CoinAPI mapped enrichment**
3. **Coinbase + CoinAPI mapped enrichment + context**

## Core capabilities

- Pulls Coinbase Advanced Trade spot products and filters to configurable quote currencies.
- Preserves product eligibility reasoning.
- Maps Coinbase products to CoinAPI symbols with explicit confidence and status.
- Pulls Coinbase candles and CoinAPI OHLCV.
- Optionally pulls CoinAPI quotes for spread proxy enrichment.
- Computes compact technical/context features with source-explicit names:
  - `cb_*` = Coinbase native
  - `ca_*` = CoinAPI mapped enrichment
  - `cs_*` = cross-source comparison
  - `cx_*` = context features
  - `ex_*` = execution relevance
- Builds flat exports optimized for downstream ChatGPT analysis.
- Produces data-quality and comparative-insight reports.
- Runs the same simple rule across multiple feature scopes.
- Backtests the attached merged rule library on historic feature data with selectable horizons, individual/collective execution modes, progress reporting, result tables, and downloadable result packs.
- Runs live shadow validation on the latest completed hourly bar, freezes signal snapshots, and resolves H1/H4/H24 outcomes over later cycles with downloadable live validation packs.
- Provides one-click downloadable health/status snapshots and an operator snapshot ZIP for easy share-back from active Scan/Live workflows.

## Main endpoints

- `/`
- `/health`
- `/api/status`
- `/api/universe/refresh`
- `/api/mappings/refresh`
- `/api/data/pull`
- `/api/features/compute`
- `/api/export/build`
- `/api/export/latest`
- `/api/reports/data-quality`
- `/api/reports/comparative-insight`
- `/api/rule-eval/run`
- `/api/rule-backtests/library`
- `/api/rule-backtests/run`
- `/api/rule-backtests/latest`
- `/api/live/shadow/run`
- `/api/live/shadow/latest`
- `/download/{filename}`
- `/api/health/download`
- `/api/status/download`
- `/api/operator/snapshot/download`

## Important implementation choices

- Future outcome labels now use **strict full-horizon semantics**. For example, `future_close_return_h24` is only populated when a full 24 forward bars exist; tail rows remain null instead of using a shorter available window.
- Step status updates clear stale `error` and `traceback` fields on successful reruns so `/api/status` reflects the current run cleanly.



- **Read-only only.** No trading, alerting, or execution logic.
- **Mock mode supported.** If external API credentials are unavailable, the app still runs end to end using deterministic fixture generation.
- **Parquet first.** Internal persisted tables use Parquet. Flat exports use CSV or CSV.GZ where useful.
- **Simple UI.** The UI is intentionally lightweight and operator-focused.

## Artifacts produced

Typical exports:

- `coinbase_products.parquet`
- `coinapi_mapping_report.csv`
- `coinbase_bars.parquet`
- `coinapi_bars.parquet`
- `feature_table.parquet`
- `coinbase_only_features.parquet`
- `coinbase_plus_coinapi_features.parquet`
- `chatgpt_ready_features.csv.gz`
- `feature_provenance_dictionary.csv`
- `comparative_insight_report.csv`
- `data_quality_report.csv`
- `run_summary.json`

## Local quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000`.

## Recommended first run

In local mock mode:

1. Refresh Coinbase universe
2. Refresh symbol mappings
3. Pull historical data
4. Compute features
5. Build research export
6. Run the sample rule across scopes

## Key limitations in v1

- CoinAPI quote enrichment is optional and off by default to avoid unnecessary API cost and latency.
- Comparative insight is deliberately transparent and heuristic-based, not model-based.
- Cross-source value is measured via compact coverage, rule comparisons, and feature-vs-target / redundancy diagnostics rather than black-box training.
- Coinbase authentication uses per-request JWT generation and assumes an ECDSA-compatible Coinbase App/CDP key.

See `DEPLOYMENT.md`, `POST_DEPLOY_STEPS.md`, and `TEST_RESULTS.md`.


## Eligibility note

By default, v1.0.1 keeps quote-matching SPOT products even if Coinbase marks them `view_only` or `trading_disabled` for the API key context. Those flags are preserved as metadata for later filtering, but they no longer zero out the research universe by default. Set `STRICT_COINBASE_TRADABILITY_FILTERS=true` if you want to exclude those rows at universe-build time.


## v1.6.4
- Fixed the Diagnostics/Scan/Live snapshot download buttons by wiring the missing browser-side download helper.
- Hardened live-shadow outcome lookup against duplicate historical signal IDs so stale duplicate rows no longer fail the cycle with `orient='index'` errors.
