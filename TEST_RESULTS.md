Version: 1.6.4
Validation performed:
- Python compile check passed
- Manual live-scan matching-path validation passed
- Manual staggered-timestamp snapshot validation passed
- Manual comparison-operator string-to-float coercion validation passed
- Targeted regression checks for live_scan defaults and run manifest behavior passed


## v1.6.2 targeted validation
- Patched live-shadow rule selection to use live-eligible direct rules by default.
- Patched live outcome rows to preserve canonical signal_id for pending joins.
- Added robust pending-signal filtering for old malformed outcome rows.

## v1.6.4 targeted validation
- Added downloadable current health/status snapshots for operator share-back.
- Added downloadable operator snapshot ZIP bundling health, status, and latest manifests.
- Added UI download buttons for status and health in Scan, Live, and Diagnostics.
