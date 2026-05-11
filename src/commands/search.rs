//! Handler for `trusty-search search`.

use crate::{daemon_base_url, print_index_header, resolve_index};
use anyhow::Result;
use colored::Colorize;

/// Why: placeholder for the project-scoped `search` command — the daemon
/// connection is tracked in issue #3. Keeping a stub here preserves the CLI
/// surface so docs and shell completions remain accurate.
/// What: prints the resolved index id, query, and a "not implemented" note.
/// Test: `cargo run -- search foo` prints the header and the yellow notice.
#[allow(clippy::too_many_arguments)]
pub async fn handle_search(
    explicit_index: &Option<String>,
    query: String,
    top_k: usize,
) -> Result<()> {
    let (index_id, warned) = resolve_index(explicit_index);
    print_index_header(&index_id, warned);
    crate::commands::daemon_guard::ensure_daemon_running_or_exit(&daemon_base_url()).await;
    println!(
        "{} {} {} {}",
        "→".cyan(),
        format!("[{}]", index_id).dimmed(),
        query.bold(),
        format!("(top-{})", top_k).dimmed()
    );
    println!(
        "{}",
        "  Daemon connection not yet implemented — see issue #3".yellow()
    );
    Ok(())
}
