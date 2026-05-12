//! Handler for `trusty-search dashboard` — open the admin panel in a browser.

use super::daemon_utils::{daemon_base_url, http_addr_path};
use anyhow::Result;
use colored::Colorize;

/// Open the admin panel of the running daemon in the default browser.
///
/// Why: provides a one-command path from "is the daemon up?" to "show me the
/// UI" without the user having to memorize ports or paths.
/// What: ensures the daemon is up (auto-starts if needed), reads
/// `~/.trusty-search/http_addr`, then opens `http://<addr>/ui` (falling back
/// to printing the URL when `open` fails — e.g. headless environments).
/// Test: `cargo run -- dashboard` with no daemon auto-starts it then opens
/// the browser.
pub async fn handle_dashboard() -> Result<()> {
    crate::commands::daemon_guard::ensure_daemon_running_or_exit(&daemon_base_url()).await?;
    open_dashboard_in_browser()
}

/// Read the http_addr file, then launch the browser at /ui.
fn open_dashboard_in_browser() -> Result<()> {
    let Some(path) = http_addr_path() else {
        anyhow::bail!("could not resolve $HOME — set HOME and try again");
    };
    let addr = match std::fs::read_to_string(&path) {
        Ok(s) => s.trim().to_string(),
        Err(_) => {
            anyhow::bail!(
                "No daemon running ({} not found). Start one with `trusty-search start`.",
                path.display(),
            );
        }
    };
    if addr.is_empty() {
        anyhow::bail!("{} is empty — daemon may be shutting down", path.display());
    }
    let url = format!("http://{addr}/ui");
    println!("{} Opening {} …", "◉".green(), url.cyan());
    if let Err(e) = open::that(&url) {
        eprintln!(
            "{} could not launch browser ({e}). Open this URL manually: {}",
            "⚠".yellow(),
            url
        );
    }
    Ok(())
}
