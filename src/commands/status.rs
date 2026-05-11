//! Handler for `trusty-search status` (and the `health` alias).

use crate::run_status;
use anyhow::Result;

/// Why: thin wrapper so the dispatch table in `main()` is uniform — the actual
/// rendering lives in `run_status` and is shared with the `health` alias.
/// What: forwards `json` to `run_status` which queries `/health`, `/indexes`,
/// and per-index `/status` then renders or emits JSON.
/// Test: `cargo run -- status` against a running daemon prints the table.
pub async fn handle_status(json: bool) -> Result<()> {
    run_status(json).await
}
