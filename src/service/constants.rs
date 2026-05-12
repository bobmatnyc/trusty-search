//! Service-wide constants.
//!
//! Why: avoid magic numbers scattered across CLI args, port-discovery
//! fallbacks, and UI runtime config injection. A single home for these
//! values makes future port changes a one-line edit.
//! What: currently exports [`DEFAULT_PORT`], the loopback port the daemon
//! binds when no explicit `--port` / `port.lock` override is in play.
//! Test: integration tests start the daemon on auto-selected ports; this
//! constant is exercised indirectly via `cli::Start::port` defaults and
//! `read_daemon_port` fallback.

/// Default loopback port for the trusty-search daemon.
///
/// Used as the CLI `--port` default, the fallback when
/// `~/Library/Application Support/trusty-search/port.lock` is missing or
/// unreadable, and the value injected into the embedded UI when
/// `SearchAppState::daemon_port` is `None`.
pub const DEFAULT_PORT: u16 = 7878;
