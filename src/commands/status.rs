//! Handler for `trusty-search status` (and the `health` alias).

use crate::{daemon_base_url, run_status};
use anyhow::Result;

/// Why: thin wrapper so the dispatch table in `main()` is uniform — the actual
/// rendering lives in `run_status` and is shared with the `health` alias.
/// What: ensures the daemon is up (auto-starts if not), then forwards `json`
/// to `run_status` which queries `/health`, `/indexes`, and per-index
/// `/status` then renders or emits JSON.
/// Test: `cargo run -- status` against a running daemon prints the table; with
/// no daemon, it auto-starts and then prints the table.
pub async fn handle_status(json: bool) -> Result<()> {
    crate::commands::daemon_guard::ensure_daemon_running_or_exit(&daemon_base_url()).await?;
    run_status(json).await
}
