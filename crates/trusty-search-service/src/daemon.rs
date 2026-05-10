//! Background HTTP daemon: PID lockfile + auto-port + graceful shutdown.
//!
//! Why: `trusty-search daemon` is the long-lived process that owns every
//! index for a machine. Two invariants matter:
//!
//! 1. **Singleton.** Only one daemon may run per machine. We enforce this
//!    via an OS-level advisory exclusive lock on a lockfile in the user's
//!    data-local dir. If the lock is held, `run_daemon` returns
//!    [`DaemonError::AlreadyRunning`] and `main` exits 1.
//!
//! 2. **Discoverable port.** The MCP server (and `trusty-search status`)
//!    needs to know what port the daemon picked. We bind a `TcpListener`
//!    starting at the requested port and walking forward until something
//!    is free, then write the chosen port to a file siblings to the lock.
//!
//! Graceful shutdown: axum's `with_graceful_shutdown` is wired to a tokio
//! signal future that resolves on SIGTERM or SIGINT. On exit we delete the
//! port file (the lockfile is unlinked by drop semantics on Unix; on
//! Windows the `Drop` of `File` releases the lock).
//!
//! What:
//! - [`daemon_lock_path`] / [`daemon_port_path`] resolve XDG-style paths.
//! - [`run_daemon`] is the one-shot entry point used by `main`.
//! - [`DaemonHandle`] returned for tests/embedding.
//!
//! Test: `cargo test -p trusty-search-service` covers (a) port-file
//! round-trip, (b) lockfile contention (second `try_lock_exclusive` on the
//! same path errors), (c) auto-port selection when the requested port is
//! taken.

use crate::server::{build_router, SearchAppState};
use fs4::FileExt;
use std::{
    fs::{File, OpenOptions},
    io::Write,
    net::{SocketAddr, TcpListener as StdTcpListener},
    path::PathBuf,
};
use thiserror::Error;
use tokio::net::TcpListener;

/// Errors raised by [`run_daemon`].
#[derive(Debug, Error)]
pub enum DaemonError {
    #[error("another trusty-search daemon is already running (lock held at {0})")]
    AlreadyRunning(PathBuf),
    #[error("could not determine data-local directory")]
    NoDataDir,
    #[error("could not find a free port starting at {0}")]
    NoFreePort(u16),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("server error: {0}")]
    Server(String),
}

/// Path to the advisory PID lockfile (`~/.local/share/trusty-search/daemon.lock`
/// on Linux, the platform equivalent elsewhere).
pub fn daemon_lock_path() -> Result<PathBuf, DaemonError> {
    Ok(daemon_dir()?.join("daemon.lock"))
}

/// Path to the file that records the listening port.
pub fn daemon_port_path() -> Result<PathBuf, DaemonError> {
    Ok(daemon_dir()?.join("daemon.port"))
}

fn daemon_dir() -> Result<PathBuf, DaemonError> {
    let dir = dirs::data_local_dir()
        .ok_or(DaemonError::NoDataDir)?
        .join("trusty-search");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Handle returned by [`run_daemon`] (mostly for tests).
pub struct DaemonHandle {
    pub port: u16,
    pub addr: SocketAddr,
}

/// Try to bind a `TcpListener` starting at `start_port`, walking forward up
/// to `max_attempts` ports. `0` means "let the OS pick" — handled directly.
fn bind_with_auto_port(start_port: u16, max_attempts: u16) -> Result<StdTcpListener, DaemonError> {
    if start_port == 0 {
        return StdTcpListener::bind("127.0.0.1:0").map_err(DaemonError::Io);
    }
    let mut last: Option<std::io::Error> = None;
    for offset in 0..max_attempts {
        let port = start_port.saturating_add(offset);
        match StdTcpListener::bind(("127.0.0.1", port)) {
            Ok(l) => return Ok(l),
            Err(e) => last = Some(e),
        }
    }
    if let Some(e) = last {
        tracing::warn!("auto-port exhausted from {start_port}: {e}");
    }
    Err(DaemonError::NoFreePort(start_port))
}

/// Acquire an exclusive advisory lock on the daemon lockfile. The returned
/// `File` must outlive the daemon — drop releases the lock.
fn acquire_lock(lock_path: &PathBuf) -> Result<File, DaemonError> {
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(lock_path)?;
    if file.try_lock_exclusive().is_err() {
        return Err(DaemonError::AlreadyRunning(lock_path.clone()));
    }
    Ok(file)
}

/// Future that resolves on SIGTERM or SIGINT.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("install SIGTERM handler failed: {e}");
                let _ = tokio::signal::ctrl_c().await;
                return;
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

/// Start the daemon: acquire the lock, bind a port, write the port file,
/// serve the axum router until SIGTERM/SIGINT, then clean up the port file.
pub async fn run_daemon(state: SearchAppState, requested_port: u16) -> Result<(), DaemonError> {
    let lock_path = daemon_lock_path()?;
    let port_path = daemon_port_path()?;

    // Lock first — second daemon must error before binding a port.
    let mut lock_file = acquire_lock(&lock_path)?;
    let pid_string = std::process::id().to_string();
    // Best-effort: write PID into the lockfile so `ps`/`lsof` can confirm.
    let _ = lock_file.set_len(0);
    let _ = lock_file.write_all(pid_string.as_bytes());

    let std_listener = bind_with_auto_port(requested_port, 64)?;
    std_listener.set_nonblocking(true)?;
    let addr = std_listener.local_addr()?;
    let port = addr.port();

    // Atomically write the port file (write + rename).
    write_port_file(&port_path, port)?;

    // Why: The embedded UI needs to know the actual port at runtime so it
    // can call back to the daemon (window.__DAEMON_PORT__). Stamp it onto
    // the state right before building the router.
    let state = state.with_daemon_port(port);
    let router = build_router(state);
    let listener = TcpListener::from_std(std_listener)?;

    tracing::info!("daemon listening on {addr} (lock {})", lock_path.display());

    let serve_result = axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await;

    // Best-effort cleanup; ignore errors so the lockfile drop is what frees
    // the next daemon, not our cleanup.
    let _ = std::fs::remove_file(&port_path);

    serve_result.map_err(|e| DaemonError::Server(e.to_string()))?;
    drop(lock_file);
    Ok(())
}

fn write_port_file(path: &PathBuf, port: u16) -> Result<(), DaemonError> {
    let tmp = path.with_extension("port.tmp");
    {
        let mut f = File::create(&tmp)?;
        writeln!(f, "{port}")?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener as StdTcpListener;

    #[test]
    fn port_file_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("daemon.port");
        write_port_file(&path, 12345).unwrap();
        let read = std::fs::read_to_string(&path).unwrap();
        assert_eq!(read.trim(), "12345");
    }

    #[test]
    fn lockfile_contention_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("daemon.lock");
        let _first = acquire_lock(&path).unwrap();
        let err = acquire_lock(&path).unwrap_err();
        assert!(matches!(err, DaemonError::AlreadyRunning(_)));
    }

    #[test]
    fn auto_port_walks_forward() {
        // Bind a port, then ask the auto-port allocator to start there.
        let occupied = StdTcpListener::bind("127.0.0.1:0").unwrap();
        let occupied_port = occupied.local_addr().unwrap().port();
        let next = bind_with_auto_port(occupied_port, 64).unwrap();
        assert_ne!(next.local_addr().unwrap().port(), occupied_port);
    }

    #[test]
    fn auto_port_zero_uses_os() {
        let l = bind_with_auto_port(0, 1).unwrap();
        assert!(l.local_addr().unwrap().port() > 0);
    }
}
