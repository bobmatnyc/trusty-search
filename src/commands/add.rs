//! Handler for `trusty-search add <file>`.

use crate::{add_path, print_index_header, resolve_index};
use anyhow::Result;

/// Why: thin wrapper so `main()` doesn't need to know about `add_path` (which
/// handles both single-file and directory walks).
/// What: resolves the active index, prints header, delegates to `add_path`.
/// Test: `cargo run -- add src/main.rs` POSTs to `/indexes/<id>/index-file`.
pub async fn handle_add(
    explicit_index: &Option<String>,
    file: std::path::PathBuf,
) -> Result<()> {
    let (index_id, warned) = resolve_index(explicit_index);
    print_index_header(&index_id, warned);
    add_path(&index_id, &file).await
}
