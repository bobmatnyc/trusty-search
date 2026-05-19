//! Handler for `trusty-search service` (macOS launchd integration).
//!
//! Why: launchd is the canonical way to keep a long-lived foreground service
//! alive on macOS — it survives logout, restarts on crash, and integrates with
//! `launchctl` for diagnostics. Wrapping the plist mechanics in `service`
//! subcommands keeps users from having to hand-edit XML.
//! What: macOS routes to `service_install` / `service_uninstall` /
//! `service_status` / `service_logs`. Non-macOS prints "not supported" and
//! exits 1.
//! Test: on Linux, every action returns Err with the platform message;
//! on macOS, `service status` runs `launchctl list` without crashing.

use anyhow::Result;
use clap::Subcommand;
#[cfg(target_os = "macos")]
use colored::Colorize;

/// Subcommands for `trusty-search service` (macOS launchd integration).
#[derive(Debug, Clone, Subcommand)]
pub enum ServiceAction {
    /// Install the LaunchAgent plist and load it
    Install,
    /// Unload the LaunchAgent and remove the plist
    Uninstall,
    /// Show launchd status for the agent
    Status,
    /// Tail the launchd stdout / stderr logs
    Logs,
}

/// Reverse-DNS label for the LaunchAgent. Used as the plist filename and the
/// `Label` key — both must match for `launchctl` lookups to work.
#[cfg(target_os = "macos")]
const LAUNCHD_LABEL: &str = "com.trusty.trusty-search";

/// Dispatch a `trusty-search service <action>` invocation.
pub fn handle_service(action: &ServiceAction) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        match action {
            ServiceAction::Install => service_install(),
            ServiceAction::Uninstall => service_uninstall(),
            ServiceAction::Status => service_status(),
            ServiceAction::Logs => service_logs(),
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = action;
        anyhow::bail!(
            "`trusty-search service` is not supported on this platform — \
             use your distro's service manager (systemd, OpenRC, etc.) directly."
        );
    }
}

#[cfg(target_os = "macos")]
fn launchd_log_dir() -> Result<std::path::PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("could not resolve $HOME"))?;
    let dir = home.join("Library").join("Logs").join("trusty-search");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Collect the environment variables embedded in the LaunchAgent plist.
///
/// Why: launchd re-spawns the daemon without the user's shell environment.
/// Embedding env vars directly in the plist provides a belt-and-suspenders
/// guarantee for operator tunables, and pins `HF_HOME` to the user's standard
/// Hugging Face cache directory so fastembed-rs never inherits a non-standard
/// or read-only `HF_HOME` that was set in an earlier shell session (fixes #86).
/// What: always emits an `HF_HOME` entry resolved at install time, plus any
/// `PERSISTED_ENV_VARS` that are currently set. XML escaping is handled by
/// `LaunchdConfig::render_plist`.
/// Test: covered indirectly by `service_install`; the canonical plist renderer
/// is unit-tested in `trusty_common::launchd`.
#[cfg(target_os = "macos")]
fn launchd_env_vars() -> Vec<(String, String)> {
    use crate::service::PERSISTED_ENV_VARS;

    let mut pairs: Vec<(String, String)> = Vec::new();

    // Always pin HF_HOME to $HOME/.cache/huggingface resolved at install time.
    // fastembed-rs follows HF_HOME when present; if it points at a read-only
    // location (e.g. a previous admin install) the embedder silently falls
    // back to BM25-only mode. Setting it here guarantees the correct writable
    // path regardless of what the operator's shell had in HF_HOME.
    if let Some(home) = dirs::home_dir() {
        let hf_home = home.join(".cache").join("huggingface");
        pairs.push(("HF_HOME".to_string(), hf_home.display().to_string()));
    }

    // Append operator tunables (TRUSTY_* vars) that are currently set.
    for key in PERSISTED_ENV_VARS {
        if let Ok(val) = std::env::var(key) {
            pairs.push((key.to_string(), val));
        }
    }

    pairs
}

/// Build the [`LaunchdConfig`] describing the trusty-search LaunchAgent.
///
/// Why: centralises the daemon's launchd description in one place so the
/// canonical `trusty_common::launchd` renderer produces a correct plist.
/// What: foreground `start --foreground` invocation, `KeepAlive::OnFailure`
/// (a clean exit — e.g. the idempotent fast-path when a daemon is already
/// running — is honoured, but a crash triggers a restart), a 30s throttle to
/// avoid crash-loop hammering, and the embedded env vars from
/// [`launchd_env_vars`].
/// Test: covered indirectly by `service_install`.
#[cfg(target_os = "macos")]
fn launchd_config(
    exe: &std::path::Path,
    log_dir: &std::path::Path,
) -> trusty_common::launchd::LaunchdConfig {
    use trusty_common::launchd::{KeepAlive, LaunchdConfig};

    LaunchdConfig {
        label: LAUNCHD_LABEL.to_string(),
        program: exe.to_path_buf(),
        program_args: vec!["start".to_string(), "--foreground".to_string()],
        stdout_path: log_dir.join("stdout.log"),
        stderr_path: log_dir.join("stderr.log"),
        env_vars: launchd_env_vars(),
        keep_alive: KeepAlive::OnFailure,
        throttle_interval: Some(30),
        working_directory: None,
    }
}

#[cfg(target_os = "macos")]
fn service_install() -> Result<()> {
    let exe = std::env::current_exe()
        .map_err(|e| anyhow::anyhow!("could not resolve current exe: {e}"))?;
    let log_dir = launchd_log_dir()?;
    let config = launchd_config(&exe, &log_dir);

    // `install()` creates `~/Library/LaunchAgents` if missing and writes the
    // rendered plist via the canonical `trusty_common::launchd` renderer.
    config.install()?;
    let plist_path = config.plist_path()?;
    println!(
        "{} Wrote LaunchAgent plist: {}",
        "✓".green(),
        plist_path.display()
    );

    // Bootstrap into the GUI domain of the current user. `bootout` first
    // (ignoring errors) so a re-install replaces a previously-loaded agent
    // cleanly.
    let _ = config.bootout();
    config.bootstrap()?;
    println!(
        "{} Loaded {} into gui/{} — daemon will start automatically.",
        "✓".green(),
        LAUNCHD_LABEL,
        trusty_common::launchd::current_uid(),
    );

    // Issue #127: install log rotation for the launchd-managed stderr.log so
    // it never grows unbounded. Non-fatal — a failure here still leaves a
    // working service; `trusty-search doctor --fix` can install it later.
    match crate::commands::log_rotation::install_rotation() {
        Ok(()) => println!(
            "{} Installed stderr.log rotation (1 MB × 7 archives, daily check)",
            "✓".green()
        ),
        Err(e) => eprintln!(
            "{} Could not install log rotation ({e}) — run `trusty-search doctor --fix` later",
            "⚠".yellow()
        ),
    }

    println!(
        "  Logs:    {}\n  Status:  {}",
        log_dir.display().to_string().dimmed(),
        "trusty-search service status".cyan(),
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn service_uninstall() -> Result<()> {
    // `bootout` / `plist_path` only need the label; the program / log paths
    // are irrelevant when tearing the service down, so a minimal config
    // suffices here.
    let config = launchd_config(std::path::Path::new(""), std::path::Path::new(""));
    let plist_path = config.plist_path()?;
    let domain = format!("gui/{}", trusty_common::launchd::current_uid());
    if plist_path.exists() {
        let _ = config.bootout();
        std::fs::remove_file(&plist_path)
            .map_err(|e| anyhow::anyhow!("remove {}: {e}", plist_path.display()))?;
        println!(
            "{} Unloaded and removed {}",
            "✓".green(),
            plist_path.display()
        );

        // Issue #127: also tear down the log-rotation LaunchAgent + config so
        // an uninstall leaves no orphaned launchd job behind.
        if let Ok(rot_plist) = crate::commands::log_rotation::rotation_plist_path() {
            if rot_plist.exists() {
                let _ = std::process::Command::new("launchctl")
                    .args(["bootout", &domain])
                    .arg(&rot_plist)
                    .status();
                let _ = std::fs::remove_file(&rot_plist);
            }
        }
        if let Ok(conf) = crate::commands::log_rotation::newsyslog_conf_path() {
            let _ = std::fs::remove_file(&conf);
        }
    } else {
        println!(
            "{} {} not installed — nothing to do",
            "·".dimmed(),
            plist_path.display()
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn service_status() -> Result<()> {
    let uid = trusty_common::launchd::current_uid();
    let target = format!("gui/{uid}/{LAUNCHD_LABEL}");
    let output = std::process::Command::new("launchctl")
        .args(["print", &target])
        .output()
        .map_err(|e| anyhow::anyhow!("launchctl print failed: {e}"))?;
    if output.status.success() {
        println!("{}", String::from_utf8_lossy(&output.stdout));
    } else {
        // `launchctl print` exits non-zero when the service isn't loaded.
        // Print the install hint before bailing so the user sees both lines.
        eprintln!("  Install with: trusty-search service install");
        anyhow::bail!(
            "{} is not loaded ({})",
            target,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn service_logs() -> Result<()> {
    let log_dir = launchd_log_dir()?;
    let stdout = log_dir.join("stdout.log");
    let stderr = log_dir.join("stderr.log");
    if !stdout.exists() && !stderr.exists() {
        eprintln!(
            "{} No logs at {} yet — start the service first.",
            "·".dimmed(),
            log_dir.display()
        );
        return Ok(());
    }
    // Defer to `tail -F` so the user gets a familiar follow-mode experience
    // and we don't have to re-implement log rotation handling.
    let status = std::process::Command::new("tail")
        .arg("-F")
        .arg(&stdout)
        .arg(&stderr)
        .status()
        .map_err(|e| anyhow::anyhow!("tail failed: {e}"))?;
    if !status.success() {
        anyhow::bail!("tail exited with {status}");
    }
    Ok(())
}
