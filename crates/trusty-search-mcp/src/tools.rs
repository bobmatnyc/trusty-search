//! MCP tool dispatcher: JSON-RPC 2.0 over a daemon HTTP back-end.
//!
//! Why: Claude Code speaks MCP/JSON-RPC; the trusty-search daemon speaks
//! REST. This module is a pure translator. It owns no state beyond a
//! `reqwest::Client` and a base URL, so the same dispatcher can be driven
//! from `stdio` (one process per session) or `sse` (long-lived axum task).
//!
//! What: [`McpServer::dispatch`] takes a [`Request`] and returns a
//! [`Response`]. Tool calls map 1:1 to daemon endpoints:
//!
//! | MCP tool        | Daemon endpoint                           |
//! |-----------------|-------------------------------------------|
//! | `search_code`   | `POST /indexes/:id/search`                |
//! | `index_file`    | `POST /indexes/:id/index-file`            |
//! | `remove_file`   | `POST /indexes/:id/remove-file`           |
//! | `list_indexes`  | `GET  /indexes`                           |
//! | `create_index`  | `POST /indexes`                           |
//! | `search_health` | `GET  /health`                            |
//!
//! Test: `cargo test -p trusty-search-mcp` covers JSON-RPC parsing, error
//! shapes (-32600 invalid request, -32601 method not found, -32602 invalid
//! params), and dispatch routing without hitting a real daemon.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON-RPC 2.0 error codes used by this server.
pub mod error_codes {
    pub const PARSE_ERROR: i32 = -32700;
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL_ERROR: i32 = -32603;
}

/// JSON-RPC 2.0 request envelope.
///
/// `id` is optional only for notifications; tool calls must supply one.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Request {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

/// JSON-RPC 2.0 response envelope. Exactly one of `result` / `error` is set.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Response {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl Response {
    pub fn ok(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn err(id: Value, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

/// Tool dispatcher backed by an HTTP client targeting the daemon.
#[derive(Clone)]
pub struct McpServer {
    base_url: String,
    http: reqwest::Client,
}

impl McpServer {
    /// Construct a dispatcher pointing at the daemon's base URL
    /// (e.g. `http://127.0.0.1:7878`).
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            http: reqwest::Client::new(),
        }
    }

    /// Inject a pre-built reqwest client (useful for tests / pooling).
    pub fn with_client(base_url: impl Into<String>, http: reqwest::Client) -> Self {
        Self {
            base_url: base_url.into(),
            http,
        }
    }

    /// Daemon base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Translate a JSON-RPC request into a daemon HTTP call and wrap the
    /// response. Always returns a `Response`; transport / daemon failures are
    /// reported as `INTERNAL_ERROR` rather than panicking.
    pub async fn dispatch(&self, req: Request) -> Response {
        let id = req.id.clone().unwrap_or(Value::Null);

        if req.jsonrpc != "2.0" {
            return Response::err(
                id,
                error_codes::INVALID_REQUEST,
                "jsonrpc must be \"2.0\"",
            );
        }

        // MCP "tools/call" wraps tool name + arguments. We also accept the
        // bare method name for ergonomics (`search_code` directly).
        let (tool, arguments) = match req.method.as_str() {
            "tools/call" => {
                let name = req
                    .params
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                let args = req
                    .params
                    .get("arguments")
                    .cloned()
                    .unwrap_or(Value::Object(Default::default()));
                match name {
                    Some(n) => (n, args),
                    None => {
                        return Response::err(
                            id,
                            error_codes::INVALID_PARAMS,
                            "tools/call requires a 'name' field",
                        )
                    }
                }
            }
            "tools/list" => {
                return Response::ok(id, serde_json::json!({ "tools": tool_descriptors() }));
            }
            other => (other.to_string(), req.params.clone()),
        };

        match self.call_tool(&tool, &arguments).await {
            Ok(value) => Response::ok(id, wrap_text_content(&value)),
            Err(DispatchError::UnknownTool) => {
                Response::err(id, error_codes::METHOD_NOT_FOUND, format!("unknown tool: {tool}"))
            }
            Err(DispatchError::InvalidParams(msg)) => {
                Response::err(id, error_codes::INVALID_PARAMS, msg)
            }
            Err(DispatchError::Transport(msg)) => {
                Response::err(id, error_codes::INTERNAL_ERROR, msg)
            }
        }
    }

    async fn call_tool(&self, tool: &str, args: &Value) -> Result<Value, DispatchError> {
        match tool {
            "search_code" => {
                let index_id = require_str(args, "index_id")?;
                let body = args
                    .get("query")
                    .cloned()
                    .ok_or_else(|| DispatchError::InvalidParams("missing 'query'".into()))?;
                self.post(&format!("/indexes/{index_id}/search"), &body).await
            }
            "index_file" => {
                let index_id = require_str(args, "index_id")?;
                let path = require_str(args, "path")?;
                let content = require_str(args, "content")?;
                self.post(
                    &format!("/indexes/{index_id}/index-file"),
                    &serde_json::json!({ "path": path, "content": content }),
                )
                .await
            }
            "remove_file" => {
                let index_id = require_str(args, "index_id")?;
                let path = require_str(args, "path")?;
                self.post(
                    &format!("/indexes/{index_id}/remove-file"),
                    &serde_json::json!({ "path": path }),
                )
                .await
            }
            "list_indexes" => self.get("/indexes").await,
            "create_index" => {
                let id = require_str(args, "id")?;
                let root_path = require_str(args, "root_path")?;
                self.post(
                    "/indexes",
                    &serde_json::json!({ "id": id, "root_path": root_path }),
                )
                .await
            }
            "search_health" => self.get("/health").await,
            _ => Err(DispatchError::UnknownTool),
        }
    }

    async fn get(&self, path: &str) -> Result<Value, DispatchError> {
        let url = format!("{}{}", self.base_url, path);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| DispatchError::Transport(format!("GET {url}: {e}")))?;
        let status = resp.status();
        let body: Value = resp
            .json()
            .await
            .map_err(|e| DispatchError::Transport(format!("decode {url}: {e}")))?;
        if !status.is_success() {
            return Err(DispatchError::Transport(format!(
                "GET {url} returned {status}: {body}"
            )));
        }
        Ok(body)
    }

    async fn post(&self, path: &str, body: &Value) -> Result<Value, DispatchError> {
        let url = format!("{}{}", self.base_url, path);
        let resp = self
            .http
            .post(&url)
            .json(body)
            .send()
            .await
            .map_err(|e| DispatchError::Transport(format!("POST {url}: {e}")))?;
        let status = resp.status();
        let body: Value = resp
            .json()
            .await
            .map_err(|e| DispatchError::Transport(format!("decode {url}: {e}")))?;
        if !status.is_success() {
            return Err(DispatchError::Transport(format!(
                "POST {url} returned {status}: {body}"
            )));
        }
        Ok(body)
    }
}

#[derive(Debug)]
enum DispatchError {
    UnknownTool,
    InvalidParams(String),
    Transport(String),
}

fn require_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, DispatchError> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| DispatchError::InvalidParams(format!("missing or non-string '{key}'")))
}

/// Wrap a structured JSON result in MCP's `content[]` envelope so downstream
/// LLM clients can render it directly.
fn wrap_text_content(value: &Value) -> Value {
    serde_json::json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string()),
        }]
    })
}

/// Static metadata for `tools/list`. Keep in sync with [`McpServer::call_tool`].
pub fn tool_descriptors() -> Value {
    serde_json::json!([
        {
            "name": "search_code",
            "description": "Hybrid BM25 + vector + KG search over an index",
            "inputSchema": {
                "type": "object",
                "required": ["index_id", "query"],
                "properties": {
                    "index_id": { "type": "string" },
                    "query": { "type": "object" }
                }
            }
        },
        {
            "name": "index_file",
            "description": "Add or update one file in an index",
            "inputSchema": {
                "type": "object",
                "required": ["index_id", "path", "content"],
                "properties": {
                    "index_id": { "type": "string" },
                    "path":     { "type": "string" },
                    "content":  { "type": "string" }
                }
            }
        },
        {
            "name": "remove_file",
            "description": "Remove a file's chunks from an index",
            "inputSchema": {
                "type": "object",
                "required": ["index_id", "path"],
                "properties": {
                    "index_id": { "type": "string" },
                    "path":     { "type": "string" }
                }
            }
        },
        {
            "name": "list_indexes",
            "description": "List all registered indexes on this daemon",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "create_index",
            "description": "Register a new (empty) index",
            "inputSchema": {
                "type": "object",
                "required": ["id", "root_path"],
                "properties": {
                    "id":        { "type": "string" },
                    "root_path": { "type": "string" }
                }
            }
        },
        {
            "name": "search_health",
            "description": "Probe daemon liveness and version",
            "inputSchema": { "type": "object", "properties": {} }
        }
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(method: &str, params: Value) -> Request {
        Request {
            jsonrpc: "2.0".into(),
            id: Some(Value::from(1u64)),
            method: method.into(),
            params,
        }
    }

    #[tokio::test]
    async fn rejects_wrong_jsonrpc_version() {
        let server = McpServer::new("http://127.0.0.1:1");
        let r = Request {
            jsonrpc: "1.0".into(),
            id: Some(Value::from(7u64)),
            method: "search_health".into(),
            params: Value::Null,
        };
        let resp = server.dispatch(r).await;
        let err = resp.error.expect("expected error");
        assert_eq!(err.code, error_codes::INVALID_REQUEST);
        assert_eq!(resp.id, Value::from(7u64));
    }

    #[tokio::test]
    async fn unknown_tool_returns_method_not_found() {
        let server = McpServer::new("http://127.0.0.1:1");
        let resp = server.dispatch(req("not_a_tool", Value::Null)).await;
        let err = resp.error.expect("expected error");
        assert_eq!(err.code, error_codes::METHOD_NOT_FOUND);
    }

    #[tokio::test]
    async fn missing_params_returns_invalid_params() {
        let server = McpServer::new("http://127.0.0.1:1");
        let resp = server.dispatch(req("index_file", serde_json::json!({}))).await;
        let err = resp.error.expect("expected error");
        assert_eq!(err.code, error_codes::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn tools_list_returns_all_six() {
        let server = McpServer::new("http://127.0.0.1:1");
        let resp = server.dispatch(req("tools/list", Value::Null)).await;
        let result = resp.result.expect("expected result");
        let tools = result.get("tools").and_then(Value::as_array).expect("array");
        assert_eq!(tools.len(), 6);
    }

    #[tokio::test]
    async fn tools_call_without_name_returns_invalid_params() {
        let server = McpServer::new("http://127.0.0.1:1");
        let resp = server.dispatch(req("tools/call", serde_json::json!({}))).await;
        let err = resp.error.expect("expected error");
        assert_eq!(err.code, error_codes::INVALID_PARAMS);
    }
}
