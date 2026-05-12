//! build.rs — thin delegation to `make release-prep` for the Svelte UI bundle.
//!
//! Why: previously this file re-implemented package-manager detection and a
//! recursive `ui/dist → ui-dist` sync (issue #109). All of that logic now
//! lives in the `Makefile`'s `release-prep` target, which is the documented
//! prerequisite for `cargo publish`. Keeping `build.rs` as a thin wrapper
//! avoids duplicate logic and shrinks the publish-time surface.
//! What: emits `cargo:rerun` directives so UI source changes still trigger a
//! rebuild, then either honours `SKIP_UI_BUILD=1` (CI / `cargo publish`
//! flow where `make release-prep` has already populated `ui-dist/`) or
//! shells out to `make release-prep`. If `make` is unavailable or the target
//! fails, falls back to emitting a placeholder `ui-dist/index.html` so
//! `include_dir!` still compiles.
//! Test: `SKIP_UI_BUILD=1 cargo check` exits without invoking any JS
//! toolchain; a plain `cargo build` on a host with `make` + pnpm/npm
//! installed populates `ui-dist/index.html` via the Makefile.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let crate_root = crate_root();
    let ui_dist = crate_root.join("ui-dist");

    emit_rerun_directives();

    // Step 1: honour explicit skip (CI / `cargo publish --verify`).
    if skip_requested() {
        handle_skip(&ui_dist);
        return;
    }

    // Step 2: no `ui/` tree at all (e.g. running build.rs inside an extracted
    // crate tarball that already shipped `ui-dist/`) → nothing to build.
    if !has_ui_sources(&crate_root) {
        ensure_placeholder(&ui_dist);
        return;
    }

    // Step 3: delegate to `make release-prep` (build-ui + sync-ui).
    run_make_release_prep(&crate_root, &ui_dist);
}

/// Resolve the crate root from Cargo's environment.
fn crate_root() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    PathBuf::from(manifest_dir)
}

/// Tell Cargo to re-run this build script when any UI source or the Makefile
/// changes (or the `SKIP_UI_BUILD` flag flips).
fn emit_rerun_directives() {
    println!("cargo:rerun-if-env-changed=SKIP_UI_BUILD");
    println!("cargo:rerun-if-changed=ui/package.json");
    println!("cargo:rerun-if-changed=ui/vite.config.js");
    println!("cargo:rerun-if-changed=ui/index.html");
    println!("cargo:rerun-if-changed=ui/src");
    println!("cargo:rerun-if-changed=Makefile");
}

/// True when the operator opted out of the JS build (`SKIP_UI_BUILD=1`).
fn skip_requested() -> bool {
    std::env::var("SKIP_UI_BUILD").as_deref() == Ok("1")
}

/// Skip-flag branch: emit a warning if `ui-dist/` is empty, then ensure the
/// stub `index.html` exists so `include_dir!` still compiles.
fn handle_skip(ui_dist: &Path) {
    if !ui_dist.join("index.html").exists() {
        println!(
            "cargo:warning=SKIP_UI_BUILD=1 but ui-dist/ is empty. \
             Run `make release-prep` before publishing."
        );
        ensure_placeholder(ui_dist);
    }
}

/// True when the crate has a `ui/package.json` (i.e. it's a working copy, not
/// an extracted publish tarball).
fn has_ui_sources(crate_root: &Path) -> bool {
    crate_root.join("ui").join("package.json").exists()
}

/// Invoke `make release-prep` and downgrade any failure to a placeholder
/// `ui-dist/index.html` plus a `cargo:warning` so the Rust build still
/// completes.
fn run_make_release_prep(crate_root: &Path, ui_dist: &Path) {
    let status = Command::new("make")
        .arg("release-prep")
        .current_dir(crate_root)
        .status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!("cargo:warning=`make release-prep` exited with {s:?}");
            ensure_placeholder(ui_dist);
        }
        Err(e) => {
            println!(
                "cargo:warning=failed to spawn `make release-prep` ({e}); \
                 set SKIP_UI_BUILD=1 or run `make release-prep` manually"
            );
            ensure_placeholder(ui_dist);
        }
    }
}

/// Emit a stub `ui-dist/index.html` so `include_dir!("$CARGO_MANIFEST_DIR/ui-dist")`
/// still compiles even when the JS build did not run.
fn ensure_placeholder(ui_dist: &Path) {
    if ui_dist.join("index.html").exists() {
        return;
    }
    let _ = std::fs::create_dir_all(ui_dist);
    let _ = std::fs::write(
        ui_dist.join("index.html"),
        "<!doctype html><html><body><p>trusty-search: UI assets not built. \
         Run <code>make release-prep</code> and rebuild.</p></body></html>",
    );
}
