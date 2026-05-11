//! Handler for `trusty-search ui` — open the web admin panel in a browser.

use crate::daemon_port_path;
use anyhow::Result;
use colored::Colorize;

/// Why: extracted from `main()`. The port resolution (explicit > port-file >
/// default 7878) plus a `/health` probe make this just slightly too noisy to
/// keep inline.
/// What: resolves the daemon port, probes `/health`, and shells out to the
/// system `open` to launch a browser pointed at `/ui`. Exits 1 if the daemon
/// is unreachable so the user gets a friendly hint instead of a confusing
/// browser error page.
/// Test: `cargo run -- ui` with a running daemon opens the browser; with no
/// daemon it prints the "Daemon not reachable" message and exits 1.
pub async fn handle_ui(port: Option<u16>) -> Result<()> {
    // Resolve port: explicit > port file > 7878.
    let port = port
        .or_else(|| {
            daemon_port_path()
                .and_then(|p| std::fs::read_to_string(p).ok())
                .and_then(|s| s.trim().parse::<u16>().ok())
        })
        .unwrap_or(7878);
    let url = format!("http://127.0.0.1:{port}/ui");

    // Probe the daemon — if it's not running, surface a friendly hint
    // instead of a confusing browser error page.
    let probe_url = format!("http://127.0.0.1:{port}/health");
    let ui_probe_client = trusty_common::server::daemon_http_client()?;
    let healthy = ui_probe_client
        .get(&probe_url)
        .send()
        .await
        .ok()
        .map(|r| r.status().is_success())
        .unwrap_or(false);
    if !healthy {
        eprintln!(
            "{} Daemon not reachable at {}. Run {} first.",
            "✗".red(),
            format!("http://127.0.0.1:{port}").cyan(),
            "trusty-search start".cyan(),
        );
        std::process::exit(1);
    }

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
