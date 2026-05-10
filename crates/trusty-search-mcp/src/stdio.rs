//! Line-delimited JSON-RPC loop on stdin/stdout.
//!
//! Why: Claude Code launches MCP servers as subprocesses and communicates
//! over stdio with one JSON-RPC message per line. This loop reads each line,
//! parses, dispatches, and writes the response — flushing immediately so the
//! parent never blocks on a buffered pipe.
//!
//! What: [`run`] takes an [`McpServer`] and runs until stdin closes (EOF).
//! Parse errors stay on stderr; protocol errors are returned as JSON-RPC
//! error responses with id=null per the spec.
//!
//! Test: covered indirectly by `tools::tests` plus a smoke test in
//! `tests/stdio.rs` that pipes a `tools/list` request through the loop.

use crate::tools::{error_codes, McpServer, Request, Response};
use anyhow::Result;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Read JSON-RPC requests line-by-line from stdin, dispatch via `server`,
/// and write responses to stdout. Returns when stdin reaches EOF.
pub async fn run(server: McpServer) -> Result<()> {
    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin).lines();

    while let Some(line) = reader.next_line().await? {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Request>(trimmed) {
            Ok(req) => server.dispatch(req).await,
            Err(e) => Response::err(
                Value::Null,
                error_codes::PARSE_ERROR,
                format!("invalid JSON-RPC: {e}"),
            ),
        };
        // Notifications (e.g. `notifications/initialized`) carry no id and
        // require no reply. Skip emission entirely so we don't desync the
        // client's request/response pairing.
        if response.suppress {
            continue;
        }
        let serialized = serde_json::to_string(&response)?;
        stdout.write_all(serialized.as_bytes()).await?;
        stdout.write_all(b"\n").await?;
        stdout.flush().await?;
    }
    Ok(())
}
