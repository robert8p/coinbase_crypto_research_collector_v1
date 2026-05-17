# Test Results

Version: 1.8.1

Validation performed in this build:

- Live-scan adaptive replay is now bounded and fail-soft so the core scan pack can complete reliably.
- Adaptive replay exceptions now produce downloadable diagnostic rows instead of failing the whole live scan.
- Long-running / interrupted background tasks are now classified as stale-running warnings after 30 minutes.
- Existing v1.8.0 adaptive replay outputs remain present in scan packs.
- Regression coverage added for adaptive replay fail-soft behavior and stale-running status classification.

Results:

- 15 passed
- 15 passed
- 15 passed

Combined: all 45 collected tests passed across split runs.
