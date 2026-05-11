//! Handler for `trusty-search reindex` (bare reindex, no force/verify).

use crate::detect::detect_project;
use crate::{print_index_header, resolve_index, run_reindex};
use anyhow::Result;

/// Why: extracted from `main()`; behaviour unchanged.
/// What: resolves the active index, picks the path (CLI arg > detected
/// project root), then drives `run_reindex` which renders the SSE progress
/// bar.
/// Test: `cargo run -- reindex` from inside a registered project rebuilds it.
pub async fn handle_reindex(
    explicit_index: &Option<String>,
    path: Option<std::path::PathBuf>,
) -> Result<()> {
    let (index_id, warned) = resolve_index(explicit_index);
    print_index_header(&index_id, warned);
    let reindex_path = path.unwrap_or_else(|| {
        let cwd = std::env::current_dir().unwrap_or_default();
        detect_project(&cwd).root_path
    });
    run_reindex(&index_id, &reindex_path).await
}
