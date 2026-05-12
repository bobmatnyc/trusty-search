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
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let crate_root = PathBuf::from(&manifest_dir);
    let ui_dist = crate_root.join("ui-dist");

    println!("cargo:rerun-if-env-changed=SKIP_UI_BUILD");
    println!("cargo:rerun-if-changed=ui/package.json");
    println!("cargo:rerun-if-changed=ui/vite.config.js");
    println!("cargo:rerun-if-changed=ui/index.html");
    println!("cargo:rerun-if-changed=ui/src");
    println!("cargo:rerun-if-changed=Makefile");

    // Honour the explicit skip flag (CI / `cargo publish --verify`).
    if std::env::var("SKIP_UI_BUILD").as_deref() == Ok("1") {
        if !ui_dist.join("index.html").exists() {
            println!(
                "cargo:warning=SKIP_UI_BUILD=1 but ui-dist/ is empty. \
                 Run `make release-prep` before publishing."
            );
            ensure_placeholder(&ui_dist);
        }
        return;
    }

    // No ui/ tree at all → nothing to build (e.g. cargo publish tarball
    // running build.rs against an extracted crate that already shipped
    // `ui-dist/`).
    if !crate_root.join("ui").join("package.json").exists() {
        if !ui_dist.join("index.html").exists() {
            ensure_placeholder(&ui_dist);
        }
        return;
    }

    // Delegate to `make release-prep` (build-ui + sync-ui).
    let status = Command::new("make")
        .arg("release-prep")
        .current_dir(&crate_root)
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!("cargo:warning=`make release-prep` exited with {s:?}");
            ensure_placeholder(&ui_dist);
        }
        Err(e) => {
            println!(
                "cargo:warning=failed to spawn `make release-prep` ({e}); \
                 set SKIP_UI_BUILD=1 or run `make release-prep` manually"
            );
            ensure_placeholder(&ui_dist);
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
