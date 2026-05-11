//! Handler for `trusty-search dashboard`.

use crate::run_dashboard;
use anyhow::Result;

/// Why: thin wrapper. Behaviour unchanged.
/// What: forwards to `run_dashboard` which opens `~/.trusty-search/http_addr`
/// and launches the admin panel in the default browser.
/// Test: `cargo run -- dashboard` with a running daemon opens a browser.
pub fn handle_dashboard() -> Result<()> {
    run_dashboard()
}
