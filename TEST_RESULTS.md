# Test Results

Version: 1.4.0

Command run:

```bash
pytest -q
```

Result:

- 27 passed
- 0 failed
- 0 skipped

Notes:
- Includes hardening coverage for `/health` effective mock-mode reporting, rule-name traversal rejection, and CoinAPI mock OHLCV ordering.
- Live shadow execution is now queued via background task and validated through the latest manifest endpoint.
