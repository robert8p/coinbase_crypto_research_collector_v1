# Post-deploy steps

## 1) Deploy the ZIP / repo to Render

Use the included `render.yaml`.

## 2) Set secrets in Render

Set these before first real-data run:

- `COINBASE_API_KEY_NAME`
- `COINBASE_API_PRIVATE_KEY`
- `COINAPI_API_KEY`

Set:

- `USE_MOCK_DATA=false`

Leave `COINBASE_BEARER_TOKEN` blank unless you intentionally want to override JWT generation.

## 3) Confirm the service is healthy

Open:

- `/health`

Expected result: status `ok`.

## 4) Open the UI

Open `/` and run the pipeline in this order:

1. **Refresh Coinbase universe**
2. **Refresh symbol mappings**
3. **Pull historical data**
4. **Compute features**
5. **Build research export**
6. **Run sample rule across scopes**
7. **Load the historic rule backtest library and run selected or all attached rules**

## 5) Download the first review pack

From the artifacts table, download at minimum:

- `product_master.csv`
- `coinapi_mapping_report.csv`
- `data_quality_report.csv`
- `feature_provenance_dictionary.csv`
- `comparative_insight_report.csv`
- `chatgpt_ready_features.csv.gz`
- `run_summary.json`

## 6) What to check immediately

### Universe

- Are the pairs the expected Coinbase spot universe?
- Are quote currencies filtered correctly?
- Are disabled / unsuitable pairs excluded with reasons?

### Mapping

- Are mappings mostly on preferred Coinbase-related CoinAPI exchanges?
- Are there any important unmapped products?

### Data quality

- Duplicate timestamps should be zero.
- Non-positive prices should be zero.
- Missing mappings should be understandable.
- Bar alignment gaps should be small enough to trust the comparison.

### Comparative report

- Review scope summary rows first.
- Then review `feature_increment` rows to see which CoinAPI / cross-source features appear both relevant and non-redundant.

## 7) What to upload back for analysis

For the next analysis cycle, the best files to upload back are:

- `chatgpt_ready_features.csv.gz`
- `comparative_insight_report.csv`
- `data_quality_report.csv`
- `coinapi_mapping_report.csv`
- `feature_provenance_dictionary.csv`
- `run_summary.json`

## 8) First real adjustment if API usage is heavy

If the initial run is too slow or too expensive:

- reduce `TOP_N_BY_VOLUME`
- reduce `LOOKBACK_HOURS`
- keep `ENABLE_COINAPI_QUOTES=false`

## 9) First research move after clean deployment

Use the comparative report plus the ChatGPT-ready feature export to decide whether CoinAPI-derived compact features appear:

- additive
- redundant
- noisy
- useful only under certain market regimes


STRICT_COINBASE_TRADABILITY_FILTERS=false keeps the research universe from collapsing to zero when Coinbase marks products `view_only` or `trading_disabled` for the key context. Turn it on only if you explicitly want those flags to exclude rows during universe refresh.


UI update in v1.0.9: the old separate "Custom data pull" section has been merged into the main pipeline card. Set Lookback hours and Max products there, then use Run full pipeline or Pull historical data so the same configured values are applied consistently.


## 10) Historic rule backtests

The UI now includes a dedicated historic rule-backtest section.

Use it like this:

1. Compute features first.
2. Pick a testing horizon (`h1`, `h4`, or `h24`).
3. Select individual rules, or run all rules.
4. Choose whether to test them individually or collectively.
5. Download the generated `rule_backtest_pack__<run_id>.zip` and upload it back for review.

The pack contains summary tables, matched-row extracts, and the exact selection payload used for the run.
