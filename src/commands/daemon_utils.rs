//! Daemon discovery + reachability helpers shared across CLI subcommands.
//!
//! Why: every subcommand that talks to the running daemon needs the same
//! "where is it listening?" logic — preferring the canonical
//! `~/.trusty-search/http_addr` file, falling back to the legacy port lockfile,
//! and finally to the compiled-in default port. Centralising it removes
//! duplication and gives `main.rs` a thinner footprint.
//! What: pure path resolvers and one async TCP probe.
//! Test: covered indirectly by every CLI subcommand that calls into the
//! daemon — `status`, `index`, `query`, `doctor`, etc.

use std::time::Duration;

/// Resolve the daemon's base URL.
///
/// Why: stdio MCP servers and CLI subcommands need to find the running daemon
/// without configuration. We check the canonical `~/.trusty-search/http_addr`
/// first (the new address-discovery contract, aligned with trusty-memory),
/// then fall back to the legacy port file
/// (`~/.local/share/trusty-search/daemon.port`) for backward compatibility,
/// and finally to `127.0.0.1:7878` if neither exists.
/// What: returns `http://{host}:{port}` (no trailing slash).
pub fn daemon_base_url() -> String {
    if let Some(addr) = read_http_addr_file() {
        return format!("http://{addr}");
    }
    let port = daemon_port_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| s.trim().parse::<u16>().ok())
        .unwrap_or(trusty_search::service::DEFAULT_PORT);
    format!("http://127.0.0.1:{port}")
}

/// Read the canonical address-discovery file. Returns `Some("host:port")`
/// when the daemon has written it; `None` otherwise.
pub fn read_http_addr_file() -> Option<String> {
    let path = http_addr_path()?;
    let raw = std::fs::read_to_string(&path).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Path to `~/.trusty-search/http_addr` — the canonical address-discovery
/// file. Mirrors `crate::service::daemon::http_addr_path` so the CLI doesn't
/// need to depend on the service crate for path resolution.
pub fn http_addr_path() -> Option<std::path::PathBuf> {
    dirs::home_dir().map(|h| h.join(".trusty-search").join("http_addr"))
}

/// Path to `~/.local/share/trusty-search/daemon.port` (or platform equivalent).
pub fn daemon_port_path() -> Option<std::path::PathBuf> {
    dirs::data_local_dir().map(|d| d.join("trusty-search").join("daemon.port"))
}

/// Check whether a TCP port is open (non-blocking connect with 500 ms timeout).
pub async fn port_reachable(host: &str, port: u16) -> bool {
    let addr = format!("{}:{}", host, port);
    tokio::time::timeout(
        Duration::from_millis(500),
        tokio::net::TcpStream::connect(&addr),
    )
    .await
    .ok()
    .and_then(|r| r.ok())
    .is_some()
}
