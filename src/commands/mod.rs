//! CLI subcommand handlers.
//!
//! Why: `main()` was a 945-line function with cyclomatic complexity ~113 because
//! every subcommand was inlined into one giant `match`. This module splits each
//! `Commands::*` variant into its own handler function so `main()` becomes a
//! thin dispatcher and each handler stays under ~150 lines with cyclo < 20.
//!
//! What: one module per subcommand. Handlers take the parsed argument fields
//! plus any global flags they need (`index`, `json`). They return `Result<()>`
//! and bubble user-facing errors via `anyhow::bail!` / `Err(...)` — the
//! central `main()` dispatcher prints the friendly red-✗ line and chooses the
//! exit code (issue #104, so handlers are testable without forking a process).
//!
//! Test: `cargo build && cargo test --workspace` — no behaviour change; the
//! refactor is purely structural.

pub mod add;
pub mod convert;
pub mod daemon_guard;
pub mod dashboard;
pub mod doctor;
pub mod index;
pub mod init;
pub mod list;
pub mod query;
pub mod reindex;
pub mod remove;
pub mod search;
pub mod serve;
pub mod service;
pub mod start;
pub mod status;
pub mod stop;
pub mod watch;
