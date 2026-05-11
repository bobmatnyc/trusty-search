//! Auto-start the daemon when a CLI command needs it.
//!
//! Why: most CLI subcommands (query, index, status, etc.) silently fail or
//! emit a confusing connection error when the daemon isn't running. This
//! guard probes `/health`; if the daemon is down, it spawns
//! `trusty-search start` in the background and polls `/health` until the
//! daemon is ready (or a 10s budget is exhausted). Users get a single
//! informational line ("Starting trusty-search daemon…") and the command
//! they typed Just Works.
//!
//! What: `ensure_daemon_running(base)` returns `Ok(())` once the daemon is
//! responding to `/health`. Returns `Err(...)` when the spawn fails or the
//! daemon doesn't become ready within 10s.
//!
//! Test: with no daemon running, `cargo run -- list` prints the "Starting…"
//! line, the daemon boots, and the registered indexes are listed. With the
//! daemon already running, no informational line is printed and behaviour is
//! unchanged.
//!
//! Note: only call this from commands that *require* the daemon. Commands
//! like `start`, `stop`, `serve`, `service`, `init`, and `completions`
//! deliberately do not call this guard.

use anyhow::{anyhow, Result};
use colored::Colorize;
use std::time::{Duration, Instant};

/// Total wall-clock budget for the daemon to become ready after we spawn it.
const READY_TIMEOUT: Duration = Duration::from_secs(10);

/// Polling interval between `/health` probes while we wait.
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Per-probe HTTP timeout. Short so a hung daemon doesn't blow our budget.
const PROBE_TIMEOUT: Duration = Duration::from_millis(750);

/// Probe `GET {base}/health`. Returns `true` on any 2xx response.
async fn probe_health(base: &str) -> bool {
    // Build a lightweight client per probe so we don't share connection pool
    // state between an unhealthy run and the next probe. Probes are infrequent
    // (every 500ms) so the cost is negligible.
    let client = match reqwest::Client::builder()
        .timeout(PROBE_TIMEOUT)
        .connect_timeout(PROBE_TIMEOUT)
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    match client.get(format!("{}/health", base)).send().await {
        Ok(r) => r.status().is_success(),
        Err(_) => false,
    }
}

/// Spawn `trusty-search start` as a detached background process.
///
/// Why: we want the daemon to outlive this CLI invocation. We use the
/// currently-running executable so a `cargo run` debugging session boots its
/// own debug daemon and a production install boots the production binary.
fn spawn_daemon() -> Result<()> {
    let exe = std::env::current_exe().map_err(|e| anyhow!("could not resolve current_exe: {e}"))?;
    // Detach stdio — we don't want the daemon's logs streaming into the
    // user's terminal session while they're waiting on a `query` result.
    std::process::Command::new(&exe)
        .arg("start")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| anyhow!("could not spawn `{} start`: {e}", exe.display()))?;
    Ok(())
}

/// Ensure the daemon at `base` is running and ready. Spawns `trusty-search
/// start` and polls `/health` for up to 10 seconds if not.
///
/// On success (already-running case), prints nothing and returns `Ok(())`
/// quickly. On the auto-start path, prints a single line to stderr so
/// stdout stays clean for tools that pipe JSON.
pub async fn ensure_daemon_running(base: &str) -> Result<()> {
    // Fast path: already up.
    if probe_health(base).await {
        return Ok(());
    }

    eprintln!("{} Starting trusty-search daemon…", "◉".cyan());
    spawn_daemon()?;

    let deadline = Instant::now() + READY_TIMEOUT;
    loop {
        tokio::time::sleep(POLL_INTERVAL).await;
        if probe_health(base).await {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(anyhow!(
                "daemon did not become ready within {}s at {} — \
                 try `trusty-search start` manually to see the error",
                READY_TIMEOUT.as_secs(),
                base
            ));
        }
    }
}

/// Convenience wrapper used by command handlers: prints a friendly error and
/// exits 1 on failure. Returns on success.
///
/// Why: every caller of `ensure_daemon_running` would otherwise duplicate the
/// "print red error / exit 1" boilerplate.
pub async fn ensure_daemon_running_or_exit(base: &str) {
    if let Err(e) = ensure_daemon_running(base).await {
        eprintln!("{} {}", "✗".red(), e);
        std::process::exit(1);
    }
}
