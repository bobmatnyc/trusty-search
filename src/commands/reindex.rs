//! Handler for `trusty-search reindex` (bare reindex, no force/verify).

use super::daemon_utils::daemon_base_url;
use super::index_resolve::{print_index_header, resolve_index};
use super::reindex_engine::run_reindex;
use crate::detect::detect_project;
use anyhow::Result;

/// Why: extracted from `main()`; behaviour unchanged.
/// What: resolves the active index, picks the path (CLI arg > detected
/// project root), then drives `run_reindex` which renders the SSE progress
/// bar.
/// Test: `cargo run -- reindex` from inside a registered project rebuilds it.
///
/// `timeout_secs` is forwarded to the SSE stream reader; 0 = no limit.
pub async fn handle_reindex(
    explicit_index: &Option<String>,
    path: Option<std::path::PathBuf>,
    timeout_secs: u64,
) -> Result<()> {
    let (index_id, warned) = resolve_index(explicit_index);
    print_index_header(&index_id, warned);
    crate::commands::daemon_guard::ensure_daemon_running_or_exit(&daemon_base_url()).await?;
    let reindex_path = path.unwrap_or_else(|| {
        let cwd = std::env::current_dir().unwrap_or_default();
        detect_project(&cwd).root_path
    });
    run_reindex(&index_id, &reindex_path, timeout_secs).await
}
