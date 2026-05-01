# Deployment

## Render blueprint included

This ZIP includes `render.yaml` for a Render web service deployment.

## Render settings used

- Runtime: Python
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`
- Persistent disk mount: `/var/data`
- App runtime data dir: `/var/data/runtime`

## Required environment variables for real API mode

These are the key secrets/settings for production-style use:

- `USE_MOCK_DATA=false`
- `COINBASE_API_KEY_NAME`
- `COINBASE_API_PRIVATE_KEY`
- `COINAPI_API_KEY`

Optional:

- `COINBASE_BEARER_TOKEN` (override; usually leave blank if using key-based JWT generation)
- `ENABLE_COINAPI_QUOTES` (`true` if you want quote-spread proxy enrichment)
- `QUOTE_CURRENCIES` (default `[
"USD"]`)
- `TOP_N_BY_VOLUME`
- `MAX_UNIVERSE_SIZE`
- `LOOKBACK_HOURS`

## Local run in mock mode

Use mock mode first if you want to verify the pipeline before adding secrets.

```bash
cp .env.example .env
uvicorn app.main:app --reload
```

## Real API mode notes

### Coinbase

This app expects a Coinbase API key that works with per-request JWT generation. The private key must preserve PEM newlines.

### CoinAPI

The app expects a CoinAPI key for symbol metadata and OHLCV. Quotes are optional.

## Suggested first production configuration

- `USE_MOCK_DATA=false`
- `QUOTE_CURRENCIES=["USD"]`
- `TOP_N_BY_VOLUME=500`
- `MAX_UNIVERSE_SIZE=500`
- `LOOKBACK_HOURS=336`
- `PREFERRED_BAR_GRANULARITY=ONE_HOUR`
- `COINAPI_PERIOD_ID=1HRS`
- `ENABLE_COINAPI_QUOTES=false`

## Where data lands

With the included Render config, runtime data and exports are written under:

- `/var/data/runtime/raw`
- `/var/data/runtime/processed`
- `/var/data/runtime/exports`
- `/var/data/runtime/state`

## After deploy

Follow `POST_DEPLOY_STEPS.md` exactly.


STRICT_COINBASE_TRADABILITY_FILTERS=false keeps the research universe from collapsing to zero when Coinbase marks products `view_only` or `trading_disabled` for the key context. Turn it on only if you explicitly want those flags to exclude rows during universe refresh.


## New in v1.1.1

- Added historic rule-library backtests driven by the attached merged validation prompt rules
- Added selectable horizon testing (`h1`, `h4`, `h24`)
- Added individual, collective, and all-rules execution modes
- Added downloadable rule-backtest result packs
