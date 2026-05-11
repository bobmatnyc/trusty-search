//! Handler for `trusty-search serve` — MCP server (stdio + optional HTTP/SSE).

use crate::{daemon_base_url, http_addr_path};
use anyhow::Result;
use colored::Colorize;

/// Why: extracted from `main()`. The HTTP path involves a discovery file
/// (`~/.trusty-search/http_addr`) and cleanup-on-exit logic that's easier to
/// follow in isolation.
/// What: routes between three modes: explicit `--http <addr>`, port-based
/// HTTP, or stdio-only via `--no-http`.
/// Test: `cargo run -- serve --no-http` runs MCP over stdio; with HTTP, the
/// discovery file appears at `~/.trusty-search/http_addr` then is removed on
/// shutdown.
pub async fn handle_serve(no_http: bool, port: u16, http: Option<String>) -> Result<()> {
    let daemon_url = daemon_base_url();

    // Resolve the HTTP bind address. Precedence:
    //   1. `--no-http`              → disabled
    //   2. legacy `--http <addr>`   → explicit bind
    //   3. `--port <p>`             → 127.0.0.1:p (p=0 → OS picks)
    let bind_addr: Option<String> = if no_http {
        None
    } else if let Some(addr) = http {
        Some(addr)
    } else {
        Some(format!("127.0.0.1:{port}"))
    };

    let server = trusty_search_mcp::McpServer::new(daemon_url.clone());

    match bind_addr {
        Some(addr) => serve_http(server, addr, &daemon_url).await,
        None => {
            eprintln!(
                "{} MCP stdio (no HTTP) → daemon {}",
                "◉".green(),
                daemon_url.dimmed()
            );
            trusty_search_mcp::stdio::run(server).await?;
            Ok(())
        }
    }
}

/// Run the MCP HTTP/SSE listener on `addr`. Writes the discovery file before
/// serving and removes it on exit (clean or crashed).
async fn serve_http(
    server: trusty_search_mcp::McpServer,
    addr: String,
    daemon_url: &str,
) -> Result<()> {
    // Bind first so we can report the OS-chosen port when 0.
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    let local = listener.local_addr()?;

    // Write `~/.trusty-search/http_addr` so `trusty-search dashboard` (and
    // other clients) can find this MCP server's HTTP transport. Best-effort:
    // a missing $HOME is reported but doesn't abort.
    let addr_file = http_addr_path();
    if let Some(ref path) = addr_file {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = std::fs::write(path, format!("{local}\n")) {
            eprintln!("{} could not write {}: {e}", "⚠".yellow(), path.display());
        }
    }

    eprintln!(
        "trusty-search v{} — HTTP admin panel: http://{}",
        env!("CARGO_PKG_VERSION"),
        local,
    );
    eprintln!(
        "{} MCP HTTP/SSE on {} → daemon {}",
        "◉".green(),
        local.to_string().cyan(),
        daemon_url.dimmed()
    );

    let app = trusty_search_mcp::sse::router(server);
    let serve_result = axum::serve(listener, app).await;

    // Clean up the discovery file regardless of the serve outcome so a
    // crashed `serve` doesn't leave a stale pointer.
    if let Some(path) = addr_file {
        let _ = std::fs::remove_file(&path);
    }
    serve_result?;
    Ok(())
}
