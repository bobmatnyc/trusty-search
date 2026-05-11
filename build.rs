//! build.rs — build the Svelte UI before compiling so include_dir! has assets.
//!
//! Why: The web admin panel ships embedded in the binary via `include_dir!`
//! (see `src/ui.rs`); the source of truth is `<workspace>/ui/src/`, and
//! Vite produces the static bundle in `<workspace>/ui/dist/`. The compiled
//! crate consumes `crates/trusty-search-service/ui-dist/` so the bundle
//! travels with the crate on `cargo publish`. Running the UI build here
//! means a plain `cargo build` always produces a binary with up-to-date
//! assets, with no separate UI build step.
//! What: Skips entirely if `SKIP_UI_BUILD=1` (CI / first-time bootstrap when
//! the JS toolchain is unavailable). Otherwise prefers `pnpm` (matching
//! trusty-memory conventions), falling back to `npm` for compatibility with
//! the existing `package-lock.json`. After a successful build, mirrors
//! `<workspace>/ui/dist/` into the crate-local `ui-dist/` so the
//! `include_dir!("$CARGO_MANIFEST_DIR/ui-dist")` macro in `src/ui.rs`
//! picks the fresh bundle up at compile time. Emits `cargo:rerun`
//! directives so a `cargo build` only re-runs the JS pipeline when UI
//! sources change.
//! Test: `SKIP_UI_BUILD=1 cargo check -p trusty-search-service` exits
//! without invoking the JS toolchain; a normal `cargo build` populates
//! both `<workspace>/ui/dist/index.html` and
//! `<crate>/ui-dist/index.html`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // CARGO_MANIFEST_DIR is the crate root (= workspace root after
    // consolidation), where `ui/` lives directly underneath.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let workspace_root = PathBuf::from(&manifest_dir);
    let ui_dir = workspace_root.join("ui");
    let dist_dir = ui_dir.join("dist");
    let pkg_json = ui_dir.join("package.json");

    println!("cargo:rerun-if-env-changed=SKIP_UI_BUILD");
    println!("cargo:rerun-if-changed=ui/package.json");
    println!("cargo:rerun-if-changed=ui/vite.config.js");
    println!("cargo:rerun-if-changed=ui/index.html");
    println!("cargo:rerun-if-changed=ui/src");

    if std::env::var("SKIP_UI_BUILD").as_deref() == Ok("1") {
        println!("cargo:warning=SKIP_UI_BUILD=1 — skipping Svelte UI build");
        // When skipping the build, the embedded ui-dist/ must already be
        // populated (typically by `make release-prep` before `cargo publish`).
        // Warn loudly if it is missing or empty so stale releases are caught.
        let crate_ui_dist = Path::new(&manifest_dir).join("ui-dist");
        if !crate_ui_dist.join("index.html").exists() {
            println!(
                "cargo:warning=ui-dist/ is missing or empty. Run `make release-prep` \
                 to build and sync the Svelte UI before publishing."
            );
        }
        // Do NOT call ensure_dist_placeholder(dist_dir) here:
        // include_dir! reads from ui-dist/ (not ui/dist/), and creating
        // ui/dist/ during `cargo publish --verify` would trigger cargo's
        // "source directory was modified by build.rs" check.
        return;
    }

    if !pkg_json.exists() {
        println!("cargo:warning=ui/package.json missing — skipping UI build");
        ensure_dist_placeholder(&dist_dir);
        return;
    }

    // Pick a package manager: pnpm (preferred, matches trusty-memory) then
    // npm. If neither is on PATH, emit a placeholder so the embed still
    // compiles (the daemon then serves a stub page).
    let pm = pick_pkg_manager();
    let Some((cmd, lockfile_name)) = pm else {
        println!(
            "cargo:warning=neither pnpm nor npm found on PATH — skipping UI build (set SKIP_UI_BUILD=1 to silence)"
        );
        ensure_dist_placeholder(&dist_dir);
        return;
    };

    // Install dependencies. Use the frozen-lockfile equivalent when a
    // lockfile is present to keep CI builds reproducible.
    let lockfile = ui_dir.join(lockfile_name);
    let install_args: Vec<&str> = if lockfile.exists() {
        match cmd {
            "pnpm" => vec!["install", "--frozen-lockfile"],
            "npm" => vec!["ci"],
            _ => vec!["install"],
        }
    } else {
        vec!["install"]
    };

    match Command::new(cmd)
        .args(&install_args)
        .current_dir(&ui_dir)
        .status()
    {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!("cargo:warning={cmd} install failed with status {s:?}");
            ensure_dist_placeholder(&dist_dir);
            return;
        }
        Err(e) => {
            println!("cargo:warning=failed to spawn {cmd} install: {e}");
            ensure_dist_placeholder(&dist_dir);
            return;
        }
    }

    match Command::new(cmd)
        .args(["run", "build"])
        .current_dir(&ui_dir)
        .status()
    {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!("cargo:warning={cmd} run build failed with status {s:?}");
            ensure_dist_placeholder(&dist_dir);
            return;
        }
        Err(e) => {
            println!("cargo:warning=failed to spawn {cmd} run build: {e}");
            ensure_dist_placeholder(&dist_dir);
            return;
        }
    }

    // Mirror the fresh bundle into the crate-local ui-dist/ so include_dir!
    // picks it up. The directory is committed for crates.io publishing —
    // we replace its contents on every successful build.
    let crate_ui_dist = Path::new(&manifest_dir).join("ui-dist");
    if let Err(e) = sync_dir(&dist_dir, &crate_ui_dist) {
        println!("cargo:warning=failed to sync ui/dist → crate ui-dist: {e}");
        ensure_dist_placeholder(&crate_ui_dist);
    }
}

/// Replace `dst` with a fresh copy of `src` — deletes any stale assets that
/// no longer appear in the new build, then recursively copies the tree.
///
/// Why: Vite emits hashed asset filenames (e.g. `index-AbC123.js`); stale
/// hashes would otherwise pile up in `ui-dist/` and bloat the published
/// crate. Deleting `dst` first keeps the embedded bundle minimal.
/// What: `fs::remove_dir_all(dst)` (if present) then a recursive copy.
/// Test: Build once, run a no-op edit + rebuild, observe that
/// `ui-dist/assets/` contains only the latest hashed files.
fn sync_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    if !src.exists() {
        return Ok(());
    }
    if dst.exists() {
        std::fs::remove_dir_all(dst)?;
    }
    std::fs::create_dir_all(dst)?;
    copy_tree(src, dst)
}

fn copy_tree(src: &Path, dst: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if ty.is_dir() {
            std::fs::create_dir_all(&to)?;
            copy_tree(&from, &to)?;
        } else if ty.is_file() {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Return the preferred package manager command and its lockfile filename.
///
/// Why: trusty-memory builds with `pnpm`; trusty-search currently has a
/// `package-lock.json` from `npm`. Both work — prefer pnpm when available
/// for parity with the sibling repo, fall back to npm otherwise.
/// What: Returns `Some(("pnpm", "pnpm-lock.yaml"))` if pnpm is on PATH,
/// else `Some(("npm", "package-lock.json"))` if npm is, else `None`.
/// Test: On a host with neither installed, returns `None` and the caller
/// emits a placeholder so the cargo build still succeeds.
fn pick_pkg_manager() -> Option<(&'static str, &'static str)> {
    if which("pnpm").is_some() {
        return Some(("pnpm", "pnpm-lock.yaml"));
    }
    if which("npm").is_some() {
        return Some(("npm", "package-lock.json"));
    }
    None
}

/// Ensure `ui/dist/` exists with at least an index.html so include_dir! can
/// embed *something* — a missing folder makes the include fail at compile
/// time.
fn ensure_dist_placeholder(dist_dir: &Path) {
    if dist_dir.join("index.html").exists() {
        return;
    }
    let _ = std::fs::create_dir_all(dist_dir);
    let _ = std::fs::write(
        dist_dir.join("index.html"),
        "<!doctype html><html><body><p>trusty-search: UI assets not built. \
         Run <code>pnpm --dir ui install &amp;&amp; pnpm --dir ui build</code> \
         (or the npm equivalent) and rebuild.</p></body></html>",
    );
}

/// Cheap which() — avoid a heavy dep just for build.rs.
fn which(cmd: &str) -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(cmd);
        if candidate.is_file() {
            return Some(candidate);
        }
        // On Windows, also try with .cmd / .exe.
        for ext in ["cmd", "exe"] {
            let with_ext = dir.join(format!("{cmd}.{ext}"));
            if with_ext.is_file() {
                return Some(with_ext);
            }
        }
    }
    None
}
