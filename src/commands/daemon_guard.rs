//! Auto-start the daemon when a CLI command needs it.
//!
//! Why: most CLI subcommands (query, index, status, etc.) silently fail or
//! emit a confusing connection error when the daemon isn't running. This
//! guard probes `/health`; if the daemon is down, it spawns
//! `trusty-search start` in the background and polls `/health` until the
//! daemon is ready (or a 60s budget is exhausted). Users get a single
//! informational line ("Starting trusty-search daemon…") and the command
//! they typed Just Works.
//!
//! What: `ensure_daemon_running(base)` returns `Ok(())` once the daemon is
//! responding to `/health`. Returns `Err(...)` when the spawn fails or the
//! daemon doesn't become ready within the budget.
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
use std::io::Write;
use std::time::{Duration, Instant};

/// Total wall-clock budget for the daemon to become ready after we spawn it.
///
/// Why 60s: with the v0.3.12 deferred-embedder-init fix the HTTP port binds
/// in ~1s, so the readiness probe normally returns near-instantly. However if
/// an older daemon binary is running (or any other slow boot path is hit) we
/// want headroom so the user sees a friendlier outcome than a hard timeout.
/// ONNX/CoreML model loading on first run can take 15–30s, and we'd rather
/// wait than fail spuriously.
const READY_TIMEOUT: Duration = Duration::from_secs(60);

/// Polling interval between `/health` probes while we wait. 500ms keeps the
/// spinner feeling responsive without hammering the daemon.
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Per-probe HTTP timeout. Short so a hung daemon doesn't blow our budget.
const PROBE_TIMEOUT: Duration = Duration::from_millis(750);

/// Spinner frames cycled while waiting for `/health` to return 2xx.
const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

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
/// start` (only when no daemon process is already running) and polls
/// `/health` for up to `READY_TIMEOUT` if not.
///
/// On success (already-running case), prints nothing and returns `Ok(())`
/// quickly. On the auto-start path, prints a single line + spinner to stderr
/// so stdout stays clean for tools that pipe JSON.
///
/// Why (v0.3.12): previously this unconditionally spawned a daemon when the
/// initial `/health` probe returned false. If a daemon process existed but
/// hadn't bound its HTTP port yet (e.g. mid-boot while loading the embedder),
/// the spawn would print "already running (pid …)" to stderr and exit, then
/// the poll would still wait up to the full timeout. Now we check the PID
/// lockfile first: if a daemon is already running, skip the spawn entirely
/// and go straight to polling — we just need to wait for `/health` to flip.
pub async fn ensure_daemon_running(base: &str) -> Result<()> {
    // Fast path: already up.
    if probe_health(base).await {
        return Ok(());
    }

    // Detect existing daemon process before spawning a duplicate. If a
    // daemon is already running but `/health` hasn't responded yet (e.g.
    // because the embedder is still loading and the HTTP listener hasn't
    // bound — only relevant for pre-v0.3.12 binaries, but the check is
    // cheap regardless), skip the spawn and just wait for it to come up.
    let already_running = crate::service::running_daemon_pid().is_some();

    if already_running {
        eprintln!(
            "{} trusty-search daemon already running, waiting for it to become ready…",
            "◉".cyan()
        );
    } else {
        eprintln!("{} Starting trusty-search daemon…", "◉".cyan());
        spawn_daemon()?;
    }

    let deadline = Instant::now() + READY_TIMEOUT;
    let start = Instant::now();
    let mut frame = 0usize;
    loop {
        // Render a spinner so the user knows we're still waiting (ONNX/CoreML
        // model load can take 15-30s on first run; without feedback users
        // assume the daemon hung).
        let elapsed = start.elapsed().as_secs();
        let glyph = SPINNER_FRAMES[frame % SPINNER_FRAMES.len()];
        eprint!(
            "\r{} Waiting for embedder to initialize… ({}s) ",
            glyph.cyan(),
            elapsed
        );
        let _ = std::io::stderr().flush();
        frame = frame.wrapping_add(1);

        tokio::time::sleep(POLL_INTERVAL).await;
        if probe_health(base).await {
            // Erase the spinner line so subsequent output starts fresh.
            eprint!("\r\x1b[2K");
            let _ = std::io::stderr().flush();
            eprintln!(
                "{} Daemon ready ({}s)",
                "✓".green(),
                start.elapsed().as_secs()
            );
            return Ok(());
        }
        if Instant::now() >= deadline {
            // Erase the spinner line before the error message.
            eprint!("\r\x1b[2K");
            let _ = std::io::stderr().flush();
            return Err(anyhow!(
                "daemon did not become ready within {}s at {} — \
                 try `trusty-search start` manually to see the error",
                READY_TIMEOUT.as_secs(),
                base
            ));
        }
    }
}

/// Convenience wrapper used by command handlers: returns a contextualized
/// error on failure. Returns `Ok(())` on success.
///
/// Why: every caller of `ensure_daemon_running` would otherwise duplicate the
/// error-handling boilerplate. Returns `Result` (not `process::exit`) so command
/// handlers stay testable — the central `main()` dispatcher prints the friendly
/// error and chooses the exit code.
pub async fn ensure_daemon_running_or_exit(base: &str) -> Result<()> {
    ensure_daemon_running(base).await
}
