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
fn launchd_plist_path() -> Result<std::path::PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("could not resolve $HOME"))?;
    Ok(home
        .join("Library")
        .join("LaunchAgents")
        .join(format!("{LAUNCHD_LABEL}.plist")))
}

#[cfg(target_os = "macos")]
fn launchd_log_dir() -> Result<std::path::PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("could not resolve $HOME"))?;
    let dir = home.join("Library").join("Logs").join("trusty-search");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Render the `<key>EnvironmentVariables</key>` plist fragment.
///
/// Why: launchd re-spawns the daemon without the user's shell environment.
/// Embedding env vars directly in the plist provides a belt-and-suspenders
/// guarantee for operator tunables, and pins `HF_HOME` to the user's standard
/// Hugging Face cache directory so fastembed-rs never inherits a non-standard
/// or read-only `HF_HOME` that was set in an earlier shell session (fixes #86).
/// What: always emits an `HF_HOME` entry resolved at install time, plus any
/// `PERSISTED_ENV_VARS` that are currently set.
/// Test: call `launchd_env_vars_plist()` with HOME set; assert output contains
/// `<key>HF_HOME</key>` and the resolved path ends in `.cache/huggingface`.
#[cfg(target_os = "macos")]
fn launchd_env_vars_plist() -> String {
    use crate::service::PERSISTED_ENV_VARS;

    let xml_escape = |s: &str| -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    };

    let mut pairs: Vec<String> = Vec::new();

    // Always pin HF_HOME to $HOME/.cache/huggingface resolved at install time.
    // fastembed-rs follows HF_HOME when present; if it points at a read-only
    // location (e.g. a previous admin install) the embedder silently falls
    // back to BM25-only mode. Setting it here guarantees the correct writable
    // path regardless of what the operator's shell had in HF_HOME.
    if let Some(home) = dirs::home_dir() {
        let hf_home = home.join(".cache").join("huggingface");
        let escaped = xml_escape(&hf_home.display().to_string());
        pairs.push(format!(
            "        <key>HF_HOME</key>\n        <string>{escaped}</string>"
        ));
    }

    // Append operator tunables (TRUSTY_* vars) that are currently set.
    for key in PERSISTED_ENV_VARS {
        if let Ok(val) = std::env::var(key) {
            let escaped = xml_escape(&val);
            pairs.push(format!(
                "        <key>{key}</key>\n        <string>{escaped}</string>"
            ));
        }
    }

    if pairs.is_empty() {
        String::new()
    } else {
        format!(
            "    <key>EnvironmentVariables</key>\n    <dict>\n{}\n    </dict>\n",
            pairs.join("\n")
        )
    }
}

/// Render the LaunchAgent plist body. Foreground mode (launchd owns lifecycle).
#[cfg(target_os = "macos")]
fn launchd_plist_body(exe: &std::path::Path, log_dir: &std::path::Path) -> String {
    let exe = exe.display();
    let stdout = log_dir.join("stdout.log");
    let stderr = log_dir.join("stderr.log");
    let env_vars_section = launchd_env_vars_plist();
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{exe}</string>
        <string>start</string>
        <string>--foreground</string>
    </array>
{env_vars_section}    <key>RunAtLoad</key>
    <true/>
    <!-- KeepAlive=SuccessfulExit:false means launchd only restarts the daemon
         on a non-zero exit. The `start` command exits 0 when a live daemon is
         already running (idempotent fast-path); without this, launchd would
         immediately re-spawn and crash-loop on the existing lockfile. -->
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>ThrottleInterval</key>
    <integer>30</integer>
    <key>StandardOutPath</key>
    <string>{}</string>
    <key>StandardErrorPath</key>
    <string>{}</string>
    <key>ProcessType</key>
    <string>Interactive</string>
</dict>
</plist>
"#,
        stdout.display(),
        stderr.display(),
    )
}

#[cfg(target_os = "macos")]
fn service_install() -> Result<()> {
    let exe = std::env::current_exe()
        .map_err(|e| anyhow::anyhow!("could not resolve current exe: {e}"))?;
    let plist_path = launchd_plist_path()?;
    if let Some(parent) = plist_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let log_dir = launchd_log_dir()?;
    let body = launchd_plist_body(&exe, &log_dir);
    std::fs::write(&plist_path, body)
        .map_err(|e| anyhow::anyhow!("write {}: {e}", plist_path.display()))?;
    println!(
        "{} Wrote LaunchAgent plist: {}",
        "✓".green(),
        plist_path.display()
    );

    // Bootstrap into the GUI domain of the current user. `bootout` first
    // (ignoring errors) so a re-install replaces a previously-loaded agent
    // cleanly.
    let uid = nix::unistd::getuid().as_raw();
    let domain = format!("gui/{uid}");
    let _ = std::process::Command::new("launchctl")
        .args(["bootout", &domain])
        .arg(&plist_path)
        .status();
    let status = std::process::Command::new("launchctl")
        .args(["bootstrap", &domain])
        .arg(&plist_path)
        .status()
        .map_err(|e| anyhow::anyhow!("launchctl bootstrap failed: {e}"))?;
    if !status.success() {
        anyhow::bail!("launchctl bootstrap exited with {status}");
    }
    println!(
        "{} Loaded {} into {} — daemon will start automatically.",
        "✓".green(),
        LAUNCHD_LABEL,
        domain
    );
    println!(
        "  Logs:    {}\n  Status:  {}",
        log_dir.display().to_string().dimmed(),
        "trusty-search service status".cyan(),
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn service_uninstall() -> Result<()> {
    let plist_path = launchd_plist_path()?;
    let uid = nix::unistd::getuid().as_raw();
    let domain = format!("gui/{uid}");
    if plist_path.exists() {
        let _ = std::process::Command::new("launchctl")
            .args(["bootout", &domain])
            .arg(&plist_path)
            .status();
        std::fs::remove_file(&plist_path)
            .map_err(|e| anyhow::anyhow!("remove {}: {e}", plist_path.display()))?;
        println!(
            "{} Unloaded and removed {}",
            "✓".green(),
            plist_path.display()
        );
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
    let uid = nix::unistd::getuid().as_raw();
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
