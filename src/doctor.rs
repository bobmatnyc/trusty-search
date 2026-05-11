//! `trusty-search doctor` check pipeline, decomposed into a `DoctorCheck`
//! trait so each diagnostic is an independently-testable unit.
//!
//! Why: the original `run_doctor_checks` was an inline orchestrator with
//! cyclomatic complexity 48 — hard to test, hard to extend, and hard to read.
//! Decomposing into a `Vec<Box<dyn DoctorCheck>>` driven by a single loop
//! drops the orchestrator's CC to ~3, makes each check unit-testable, and
//! lets new checks be added by implementing one trait method.
//! What: defines the [`DoctorCheck`] trait, a shared [`DoctorState`] passed
//! to every check, and the concrete check structs that wrap the existing
//! pure helper functions in `main.rs`. The orchestrator [`run_doctor_checks`]
//! walks the trait-object list and aggregates results.
//! Test: `cargo test --workspace` exercises the existing doctor integration
//! tests; `cargo run -- doctor` produces byte-identical output to the
//! pre-refactor implementation.

use crate::{
    check_daemon_running, check_data_dir, check_lock_file, check_model_cache,
    check_port_reachable, daemon_base_url, doctor_data_dir, fetch_index_names,
    fetch_index_statuses, print_index_breakdown, probe_daemon_health, read_daemon_port,
    summarize_indexes, CheckResult, EmptyIndex,
};
use async_trait::async_trait;
use std::sync::Mutex;

// ── Trait + shared state ──────────────────────────────────────────────────

/// One diagnostic in the doctor pipeline.
///
/// Why: gives each check a stable, narrow interface (`name` for diagnostics,
/// `run` for execution) so the orchestrator can iterate uniformly without
/// caring about each check's internal dependencies.
/// What: async trait returning zero or more [`CheckResult`]s. Most checks
/// return exactly one result; the indexes check returns one summary plus
/// any "no indexes registered" warning and performs its per-index printing
/// as a side effect via [`DoctorState`].
/// Test: see the concrete impls below; each is independently exercised by
/// `cargo run -- doctor` against a known daemon state.
#[async_trait]
pub(crate) trait DoctorCheck: Send + Sync {
    /// Human-readable name of the check (used in debug logs / future
    /// `--verbose` output). Not currently rendered to the user because the
    /// existing output format embeds the name in the result message.
    #[allow(dead_code)]
    fn name(&self) -> &str;

    /// Run the check, returning every `CheckResult` it produced (zero or
    /// more). Side effects (e.g. per-index breakdown prints, recording
    /// empty indexes) flow through [`DoctorState`].
    async fn run(&self, state: &DoctorState) -> Vec<CheckResult>;
}

/// Shared state threaded through every [`DoctorCheck`].
///
/// Why: checks have heterogeneous dependencies (HTTP client, daemon base URL,
/// data directory, daemon liveness). Bundling them into one struct keeps the
/// trait signature stable as new checks are added.
/// What: holds immutable runtime configuration plus interior-mutable slots
/// for state that later checks depend on (`daemon_running`, `daemon_version`)
/// and side-channel outputs (`empty_indexes` consumed by `--fix`).
/// Test: constructed once in [`run_doctor_checks`]; verified by the doctor
/// integration tests via end-to-end output.
pub(crate) struct DoctorState {
    pub client: reqwest::Client,
    pub base: String,
    pub port: u16,
    pub data_dir: std::path::PathBuf,
    /// Set by [`DaemonHealthCheck`]; read by [`LockFileCheck`] and
    /// [`IndexesCheck`]. Mutex (not RwLock) because the critical section is
    /// a single bool/string assignment.
    daemon_running: Mutex<bool>,
    daemon_version: Mutex<String>,
    /// Populated by [`IndexesCheck`]; consumed by `--fix` after the doctor
    /// run completes.
    empty_indexes: Mutex<Vec<EmptyIndex>>,
}

impl DoctorState {
    fn new(client: reqwest::Client) -> Self {
        Self {
            client,
            base: daemon_base_url(),
            port: read_daemon_port(),
            data_dir: doctor_data_dir(),
            daemon_running: Mutex::new(false),
            daemon_version: Mutex::new(String::new()),
            empty_indexes: Mutex::new(Vec::new()),
        }
    }

    fn set_daemon_health(&self, running: bool, version: String) {
        *self.daemon_running.lock().expect("doctor state poisoned") = running;
        *self.daemon_version.lock().expect("doctor state poisoned") = version;
    }

    fn daemon_running(&self) -> bool {
        *self.daemon_running.lock().expect("doctor state poisoned")
    }

    #[allow(dead_code)]
    fn daemon_version(&self) -> String {
        self.daemon_version
            .lock()
            .expect("doctor state poisoned")
            .clone()
    }

    fn push_empty_indexes(&self, mut items: Vec<EmptyIndex>) {
        self.empty_indexes
            .lock()
            .expect("doctor state poisoned")
            .append(&mut items);
    }

    fn take_empty_indexes(&self) -> Vec<EmptyIndex> {
        std::mem::take(
            &mut *self
                .empty_indexes
                .lock()
                .expect("doctor state poisoned"),
        )
    }
}

// ── Concrete checks ───────────────────────────────────────────────────────

/// Probe `/health` and record the daemon's liveness + version for later
/// checks to read.
pub(crate) struct DaemonHealthCheck;

#[async_trait]
impl DoctorCheck for DaemonHealthCheck {
    fn name(&self) -> &str {
        "daemon_health"
    }

    async fn run(&self, state: &DoctorState) -> Vec<CheckResult> {
        let (running, version) = probe_daemon_health(&state.client, &state.base).await;
        state.set_daemon_health(running, version.clone());
        vec![check_daemon_running(running, &state.base, &version)]
    }
}

/// Verify the fastembed ONNX model cache is present (or warn that it will
/// be downloaded on first start).
pub(crate) struct ModelCacheCheck;

#[async_trait]
impl DoctorCheck for ModelCacheCheck {
    fn name(&self) -> &str {
        "model_cache"
    }

    async fn run(&self, _state: &DoctorState) -> Vec<CheckResult> {
        vec![check_model_cache()]
    }
}

/// Check that the per-user data directory exists and is writable.
pub(crate) struct DataDirCheck;

#[async_trait]
impl DoctorCheck for DataDirCheck {
    fn name(&self) -> &str {
        "data_dir"
    }

    async fn run(&self, state: &DoctorState) -> Vec<CheckResult> {
        vec![check_data_dir(&state.data_dir)]
    }
}

/// Look for a stale `daemon.lock` (PID present but process dead).
pub(crate) struct LockFileCheck;

#[async_trait]
impl DoctorCheck for LockFileCheck {
    fn name(&self) -> &str {
        "lock_file"
    }

    async fn run(&self, state: &DoctorState) -> Vec<CheckResult> {
        vec![check_lock_file(&state.data_dir, state.daemon_running())]
    }
}

/// List registered indexes, fan-out status calls, print the breakdown, and
/// record any zero-chunk indexes for `--fix`.
pub(crate) struct IndexesCheck;

#[async_trait]
impl DoctorCheck for IndexesCheck {
    fn name(&self) -> &str {
        "indexes"
    }

    async fn run(&self, state: &DoctorState) -> Vec<CheckResult> {
        if !state.daemon_running() {
            return vec![CheckResult::Warn(
                "Indexes: skipped (daemon not running)".into(),
            )];
        }

        let names = fetch_index_names(&state.client, &state.base).await;
        if names.is_empty() {
            return vec![CheckResult::Warn(
                "No indexes registered — run `trusty-search index` to add a project".into(),
            )];
        }

        let per_index = fetch_index_statuses(&state.client, &state.base, &names).await;
        let zero_count = per_index
            .iter()
            .filter(|(_, b)| b.get("chunk_count").and_then(|v| v.as_u64()).unwrap_or(0) == 0)
            .count();
        let summary = summarize_indexes(per_index.len(), zero_count);

        let mut empty_buf: Vec<EmptyIndex> = Vec::new();
        print_index_breakdown(&per_index, &mut empty_buf);
        state.push_empty_indexes(empty_buf);

        vec![summary]
    }
}

/// TCP-reachability check for the daemon's listen port.
pub(crate) struct PortReachableCheck;

#[async_trait]
impl DoctorCheck for PortReachableCheck {
    fn name(&self) -> &str {
        "port_reachable"
    }

    async fn run(&self, state: &DoctorState) -> Vec<CheckResult> {
        vec![check_port_reachable(state.port).await]
    }
}

// ── Orchestrator ──────────────────────────────────────────────────────────

/// Construct the default pipeline. Order matters: the health check must run
/// first so [`LockFileCheck`] and [`IndexesCheck`] can read its result.
fn default_checks() -> Vec<Box<dyn DoctorCheck>> {
    vec![
        Box::new(DaemonHealthCheck),
        Box::new(ModelCacheCheck),
        Box::new(DataDirCheck),
        Box::new(LockFileCheck),
        Box::new(IndexesCheck),
        Box::new(PortReachableCheck),
    ]
}

/// Drive the doctor pipeline and return `(checks, empty_indexes)` for the
/// caller (and `--fix`) to consume.
///
/// Why: this is the cyclomatic-complexity-48 function that issue #60 asked
/// us to decompose. After the trait refactor the orchestrator is a 3-line
/// loop over `default_checks()`; new checks plug in by adding one Box.
/// What: builds the HTTP client (returning an early error if that fails),
/// constructs shared [`DoctorState`], iterates the pipeline in order, and
/// flattens each check's result vector into a single output vector.
/// Test: `cargo run -- doctor` produces the same six headings (daemon,
/// model, data dir, lock, indexes, port) in the same order as before; the
/// existing CLI integration tests cover the exit-code branch.
pub(crate) async fn run_doctor_checks() -> (Vec<CheckResult>, Vec<EmptyIndex>) {
    let client = match trusty_common::server::daemon_http_client() {
        Ok(c) => c,
        Err(e) => {
            return (
                vec![CheckResult::Error(format!(
                    "failed to build HTTP client: {e}"
                ))],
                Vec::new(),
            );
        }
    };

    let state = DoctorState::new(client);
    let mut checks: Vec<CheckResult> = Vec::new();

    for check in default_checks() {
        checks.extend(check.run(&state).await);
    }

    (checks, state.take_empty_indexes())
}
