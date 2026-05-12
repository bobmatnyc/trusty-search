//! Handler for `trusty-search watch`.

use crate::detect::detect_project;
use crate::{daemon_base_url, print_index_header, resolve_index};
use anyhow::Result;
use colored::Colorize;

/// Why: placeholder for the FileWatcher integration tracked in issue #6.
/// What: resolves the index id, prints "watching ...", emits a not-implemented
/// notice. Kept as a separate handler so the CLI structure is consistent.
/// Test: `cargo run -- watch` prints the message without panicking.
pub async fn handle_watch(
    explicit_index: &Option<String>,
    path: Option<std::path::PathBuf>,
) -> Result<()> {
    let (index_id, warned) = resolve_index(explicit_index);
    print_index_header(&index_id, warned);
    crate::commands::daemon_guard::ensure_daemon_running_or_exit(&daemon_base_url()).await?;
    let watch_path = path.unwrap_or_else(|| {
        let cwd = std::env::current_dir().unwrap_or_default();
        detect_project(&cwd).root_path
    });
    println!(
        "{} Watching {} as index {}",
        "◉".green(),
        watch_path.display().to_string().cyan(),
        format!("'{}'", index_id).bold()
    );
    println!(
        "{}",
        "  FileWatcher not yet implemented — see issue #6".yellow()
    );
    Ok(())
}
