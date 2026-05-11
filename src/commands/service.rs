//! Handler for `trusty-search service` (macOS launchd integration).

use crate::{run_service_action, ServiceAction};
use anyhow::Result;

/// Why: thin wrapper for uniform dispatch in `main()`. The real platform
/// gating and per-action work lives in `run_service_action`.
/// What: forwards the action to `run_service_action`, which is a no-op-with-error
/// on non-macOS targets.
/// Test: `cargo run -- service status` on macOS runs `launchctl print`.
pub fn handle_service(action: &ServiceAction) -> Result<()> {
    run_service_action(action)
}
