//! Minimal MCP (Model Context Protocol) JSON-RPC 2.0 framing over stdio.
//!
//! Only what a tool server needs: initialize, tools/list, tools/call, ping.
//! Line-delimited JSON (the stdio transport used by Claude Code / Codex).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// An inbound JSON-RPC request or notification.
#[derive(Debug, Clone, Deserialize)]
pub struct RpcRequest {
    pub jsonrpc: String,
    #[serde(default)]
    pub id: Option<Value>,
    #[serde(default)]
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

impl RpcRequest {
    pub fn is_notification(&self) -> bool {
        self.id.is_none()
    }
}

/// An outbound JSON-RPC response.
#[derive(Debug, Serialize)]
pub struct RpcResponse {
    pub jsonrpc: &'static str,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

#[derive(Debug, Serialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
}

impl RpcResponse {
    pub fn ok(id: Value, result: Value) -> Self {
        Self { jsonrpc: "2.0", id, result: Some(result), error: None }
    }

    pub fn err(id: Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(RpcError { code, message: message.into() }),
        }
    }
}

/// Standard JSON-RPC error codes.
pub const PARSE_ERROR: i64 = -32700;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;
pub const INTERNAL_ERROR: i64 = -32603;

/// A tool definition advertised by tools/list (MCP 2024-11-05 shape).
#[derive(Debug, Serialize)]
pub struct ToolDef {
    pub name: &'static str,
    pub description: &'static str,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// The three tools the Python headroom MCP server exposes.
pub fn tools() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "headroom_compress",
            description: "Compress large content (tool outputs, logs, search results, JSON arrays). Returns the compressed text plus a hash; retrieve the original with headroom_retrieve.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"]
            }),
        },
        ToolDef {
            name: "headroom_retrieve",
            description: "Retrieve the original (pre-compression) content by hash from a headroom_compress marker.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"hash": {"type": "string"}},
                "required": ["hash"]
            }),
        },
        ToolDef {
            name: "headroom_stats",
            description: "Show compression savings statistics (session + lifetime).",
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        },
        ToolDef {
            name: "headroom_read",
            description: "Read a file with content-hash caching (requires HEADROOM_MCP_READ=on).",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "fresh": {"type": "boolean"}
                },
                "required": ["file_path"]
            }),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_request_line() {
        let req: RpcRequest =
            serde_json::from_str(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#)
                .unwrap();
        assert_eq!(req.method, "initialize");
        assert!(!req.is_notification());
        let note: RpcRequest =
            serde_json::from_str(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#)
                .unwrap();
        assert!(note.is_notification());
    }

    #[test]
    fn tool_defs_have_required_fields() {
        let tools = tools();
        for t in &tools {
            assert!(t.input_schema.get("type").is_some());
        }
        assert_eq!(tools.len(), 4);
    }
}
