//! Handler for `trusty-search stop`.

use crate::daemon_port_path;
use anyhow::Result;
use colored::Colorize;

/// Why: extracted from `main()`. Stopping involves PID-file lookup, SIGTERM,
/// and a poll loop — clearer in its own function.
/// What: reads `~/.local/share/trusty-search/daemon.lock` for the PID, sends
/// SIGTERM, then waits up to 5 s for the daemon's port file to disappear.
/// Exits 1 if there's no PID file or if SIGTERM fails to deliver.
/// Test: with a running daemon → "Daemon stopped" within 5 s.
pub async fn handle_stop() -> Result<()> {
    // The daemon writes its PID into the fs4 lockfile at startup
    // (see trusty-search-service/src/daemon.rs). Read the PID, send
    // SIGTERM, then poll for the port file to disappear as a signal
    // that shutdown completed cleanly.
    let lock_path = dirs::data_local_dir().map(|d| d.join("trusty-search").join("daemon.lock"));
    let port_path = daemon_port_path();

    let pid = lock_path
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| s.trim().parse::<u32>().ok());

    match pid {
        None => {
            eprintln!("{} No daemon running (no PID file)", "✗".red());
            std::process::exit(1);
        }
        Some(pid) => {
            println!("{} Stopping daemon (PID {})…", "⟳".cyan(), pid);
            let status = std::process::Command::new("kill")
                .arg("-TERM")
                .arg(pid.to_string())
                .status();
            match status {
                Ok(s) if s.success() => {
                    // Poll up to 5s for the port file to disappear.
                    for _ in 0..50 {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        if port_path.as_ref().map(|p| !p.exists()).unwrap_or(true) {
                            println!("{} Daemon stopped", "✓".green());
                            return Ok(());
                        }
                    }
                    println!("{} Daemon may still be shutting down", "⚠".yellow());
                }
                _ => {
                    eprintln!(
                        "{} Failed to send SIGTERM (process may already be gone)",
                        "✗".red()
                    );
                    std::process::exit(1);
                }
            }
        }
    }
    Ok(())
}
