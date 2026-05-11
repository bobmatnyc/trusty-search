//! Handler for `trusty-search dashboard`.

use crate::{daemon_base_url, run_dashboard};
use anyhow::Result;

/// Why: thin wrapper. Auto-starts the daemon (if needed) so the browser
/// doesn't land on an empty page.
/// What: ensures the daemon is up, then forwards to `run_dashboard` which
/// opens `~/.trusty-search/http_addr` and launches the admin panel in the
/// default browser.
/// Test: `cargo run -- dashboard` with no daemon auto-starts it then opens
/// the browser.
pub async fn handle_dashboard() -> Result<()> {
    crate::commands::daemon_guard::ensure_daemon_running_or_exit(&daemon_base_url()).await;
    run_dashboard()
}
