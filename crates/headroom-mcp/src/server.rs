//! MCP stdio server loop: reads JSON-RPC lines, dispatches the three tools.

use std::io::{BufRead, Write};

use serde_json::{json, Value};

use crate::protocol::{RpcRequest, RpcResponse, METHOD_NOT_FOUND, PARSE_ERROR, tools};

/// Dependencies the tools need (store + ledger), injected by the CLI.
pub struct McpContext {
    pub store: Option<headroom_store::ccr::CompressionStore>,
    pub ledger: headroom_store::ledger::SavingsLedger,
    /// Proxy URL for proxy-store fallback (M2: local only for now).
    pub proxy_url: String,
    /// Content-hash cache for headroom_read.
    pub read_cache: ReadCache,
}

/// path -> (content hash) cache for headroom_read (last write wins).
#[derive(Default)]
pub struct ReadCache {
    inner: std::sync::Mutex<std::collections::HashMap<String, String>>,
}

impl ReadCache {
    fn read(&self, path: &str, _hash: &str) -> Option<String> {
        self.inner.lock().unwrap().get(path).cloned()
    }

    fn write(&self, path: &str, hash: &str) {
        self.inner
            .lock()
            .unwrap()
            .insert(path.to_string(), hash.to_string());
    }
}

/// Run the stdio loop until EOF. Each line is one JSON-RPC message.
pub fn serve_stdio(ctx: McpContext) -> anyhow::Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<RpcRequest>(&line) {
            Ok(req) => {
                if req.is_notification() {
                    continue; // no response for notifications
                }
                dispatch(&ctx, &req)
            }
            Err(_) => RpcResponse::err(
                Value::Null,
                PARSE_ERROR,
                "invalid JSON-RPC request",
            ),
        };
        writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
        stdout.flush()?;
    }
    Ok(())
}

fn dispatch(ctx: &McpContext, req: &RpcRequest) -> RpcResponse {
    let id = req.id.clone().unwrap_or(Value::Null);
    match req.method.as_str() {
        "initialize" => RpcResponse::ok(
            id,
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "headroom", "version": headroom_core::VERSION},
            }),
        ),
        "ping" => RpcResponse::ok(id, json!({})),
        "tools/list" => RpcResponse::ok(
            id,
            json!({"tools": tools()}),
        ),
        "tools/call" => {
            let name = req
                .params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let args = req.params.get("arguments").cloned().unwrap_or(json!({}));
            let result = call_tool(ctx, &name, &args);
            match result {
                Ok(text) => RpcResponse::ok(
                    id,
                    json!({
                        "content": [{"type": "text", "text": text}],
                        "isError": false,
                    }),
                ),
                Err(e) => RpcResponse::ok(
                    id,
                    json!({
                        "content": [{"type": "text", "text": format!("{{\"error\": {e:?}}}")}],
                        "isError": true,
                    }),
                ),
            }
        }
        other => RpcResponse::err(id, METHOD_NOT_FOUND, format!("unknown method: {other}")),
    }
}

fn call_tool(ctx: &McpContext, name: &str, args: &Value) -> Result<String, anyhow::Error> {
    match name {
        "headroom_compress" => {
            let content = args
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("missing required argument: content"))?;
            Ok(compress_tool(ctx, content))
        }
        "headroom_retrieve" => {
            let hash = args
                .get("hash")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("missing required argument: hash"))?;
            Ok(retrieve_tool(ctx, hash))
        }
        "headroom_stats" => Ok(stats_tool(ctx)),
        "headroom_read" => {
            let file_path = args
                .get("file_path")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("missing required argument: file_path"))?;
            let fresh = args.get("fresh").and_then(Value::as_bool).unwrap_or(false);
            Ok(read_tool(ctx, file_path, fresh))
        }
        other => Err(anyhow::anyhow!("Unknown tool: {other}")),
    }
}

fn compress_tool(ctx: &McpContext, content: &str) -> String {
    use headroom_core::content::{detect_content_type, ContentType};
    use headroom_core::tokens::EstimatingTokenCounter;
    use headroom_core::transforms::diff_compressor::DiffCompressor;
    use headroom_core::transforms::log_compressor::LogCompressor;
    use headroom_core::transforms::search_compressor::SearchCompressor;
    use headroom_core::transforms::smart_crusher::SmartCrusher;
    use headroom_core::transforms::text_crusher::TextCrusher;

    let counter = EstimatingTokenCounter::default();
    let tokens_before = counter.count_text(content) as i64;

    let (compressed, transforms) = match detect_content_type(content) {
        ContentType::JsonArray => {
            match serde_json::from_str::<Value>(content.trim()) {
                Ok(Value::Array(items)) => {
                    if let Some(densified) =
                        headroom_core::compaction::encode_compacted(&items)
                    {
                        ("compaction".to_string(), densified)
                    } else {
                        let r = SmartCrusher::default().crush_array(&items, "", 1.0);
                        if let (Some(key), Some(orig)) = (&r.cache_key, &r.ccr_original) {
                            stash(ctx, key, orig, "", "smart_crusher");
                        }
                        (
                            "smart_crusher".to_string(),
                            serde_json::to_string(&r.compressed)
                                .unwrap_or_else(|_| content.to_string()),
                        )
                    }
                }
                _ => ("none".to_string(), content.to_string()),
            }
        }
        ContentType::SearchResults => {
            let r = SearchCompressor::default().compress(content, "", 1.0);
            if let Some(key) = &r.cache_key {
                stash(ctx, key, content, &r.compressed, "search_compressor");
            }
            ("search_compressor".to_string(), r.compressed)
        }
        ContentType::BuildOutput => {
            let r = LogCompressor::default().compress(content, "", 1.0);
            if let Some(key) = &r.cache_key {
                stash(ctx, key, content, &r.compressed, "log_compressor");
            }
            ("log_compressor".to_string(), r.compressed)
        }
        ContentType::GitDiff => {
            let r = DiffCompressor::default().compress(content, "");
            if let Some(key) = &r.cache_key {
                stash(ctx, key, content, &r.compressed, "diff_compressor");
            }
            ("diff_compressor".to_string(), r.compressed)
        }
        ContentType::PlainText => {
            let r = TextCrusher::default().compress(content, "", None);
            ("text_crusher".to_string(), r.compressed)
        }
        // Conservative: leave structured/unknown formats untouched.
        ContentType::Html
        | ContentType::Tabular
        | ContentType::StructuredConfig
        | ContentType::SourceCode => ("none".to_string(), content.to_string()),
    };

    // Only stash + report when something was actually saved.
    let tokens_after = counter.count_text(&compressed) as i64;
    if tokens_after >= tokens_before || compressed == content {
        return json!({
            "compressed": content,
            "hash": null,
            "original_tokens": tokens_before,
            "compressed_tokens": tokens_before,
            "tokens_saved": 0,
            "savings_percent": 0.0,
            "transforms": [transforms],
            "note": "No compression applied (content not compressible or already small).",
        })
        .to_string();
    }

    let _ = transforms;
    let hash = if let Some(store) = &ctx.store {
        store
            .store_original(content, &compressed, tokens_before, tokens_after, None, Some("mcp"))
            .unwrap_or_else(|_| headroom_core::ccr::ccr_hash(content))
    } else {
        headroom_core::ccr::ccr_hash(content)
    };
    let savings_percent = if tokens_before > 0 {
        (1.0 - tokens_after as f64 / tokens_before as f64) * 100.0
    } else {
        0.0
    };
    // Record the MCP-source savings event.
    let event = headroom_store::ledger::SavingsEvent::new_proxy(
        "claude-sonnet-4-5-20250929",
        "mcp",
        tokens_before,
        tokens_after,
    );
    ctx.ledger.record(&event);

    json!({
        "compressed": compressed,
        "hash": hash,
        "original_tokens": tokens_before,
        "compressed_tokens": tokens_after,
        "tokens_saved": tokens_before - tokens_after,
        "savings_percent": (savings_percent * 10.0).round() / 10.0,
        "transforms": [transforms],
        "note": "Original stored with hash=... Use mcp__headroom__headroom_retrieve to get it back.",
    })
    .to_string()
}

fn stash(ctx: &McpContext, key: &str, original: &str, compressed: &str, strategy: &str) {
    if let Some(store) = &ctx.store {
        let counter = headroom_core::tokens::EstimatingTokenCounter::default();
        let _ = store.store_original(
            original,
            compressed,
            counter.count_text(original) as i64,
            counter.count_text(compressed) as i64,
            None,
            Some(strategy),
        );
        let _ = key;
    }
}

/// headroom_read: read a file with content-hash caching (Python parity;
/// gated behind HEADROOM_MCP_READ=on). Unchanged re-reads return a light
/// cache marker; fresh=true bypasses the cache.
fn read_tool(ctx: &McpContext, file_path: &str, fresh: bool) -> String {
    let enabled = std::env::var("HEADROOM_MCP_READ")
        .map(|v| v == "on" || v == "1")
        .unwrap_or(false);
    if !enabled {
        return json!({
            "error": "headroom_read is disabled (set HEADROOM_MCP_READ=on to enable)",
        })
        .to_string();
    }
    let content = match std::fs::read_to_string(file_path) {
        Ok(c) => c,
        Err(e) => return json!({"error": format!("read failed: {e}")}).to_string(),
    };
    let hash = headroom_core::ccr::ccr_hash(&content);
    if !fresh {
        if let Some(cached) = ctx.read_cache.read(file_path, &hash) {
            if cached == hash {
                return json!({
                    "file_path": file_path,
                    "cache": "unchanged",
                    "hash": hash,
                    "note": "File unchanged since last read; content cached. Use fresh=true to bypass.",
                })
                .to_string();
            }
        }
    }
    if let Some(store) = &ctx.store {
        let counter = headroom_core::tokens::EstimatingTokenCounter::default();
        let _ = store.store_original(
            &content,
            "",
            counter.count_text(&content) as i64,
            counter.count_text(&content) as i64,
            Some("headroom_read"),
            Some("read"),
        );
    }
    ctx.read_cache.write(file_path, &hash);
    json!({
        "file_path": file_path,
        "content": content,
        "hash": hash,
    })
    .to_string()
}

fn retrieve_tool(ctx: &McpContext, hash: &str) -> String {
    let normalized = hash
        .strip_prefix("<<ccr:")
        .and_then(|r| r.split([',', ' ']).next())
        .unwrap_or(hash)
        .to_lowercase();
    if let Some(store) = &ctx.store {
        if let Some(entry) = store.retrieve(&normalized) {
            return json!({
                "hash": entry.hash,
                "source": "local",
                "original_content": entry.original_content,
                "original_item_count": entry.original_item_count,
                "compressed_item_count": entry.compressed_item_count,
                "retrieval_count": entry.retrieval_count,
            })
            .to_string();
        }
    }
    json!({
        "error": "Content not found (expired or never stored). Re-run the source command to get fresh output.",
        "hash": normalized,
    })
    .to_string()
}

fn stats_tool(ctx: &McpContext) -> String {
    let events = ctx.ledger.read_all();
    let agg = headroom_store::ledger::aggregate(&events);
    let lifetime = &agg["lifetime"];
    let saved = lifetime["tokens_saved"].as_i64().unwrap_or(0);
    let cost = lifetime["cost_usd"].as_f64().unwrap_or(0.0);
    json!({
        "session_duration_seconds": 0,
        "compressions": lifetime["calls"].as_i64().unwrap_or(0),
        "retrievals": 0,
        "total_input_tokens": lifetime["tokens_before"].as_i64().unwrap_or(0),
        "total_output_tokens": 0,
        "total_tokens_saved": saved,
        "savings_percent": lifetime["tokens_before"].as_f64().map(|b| if b > 0.0 { saved as f64 / b * 100.0 } else { 0.0 }).unwrap_or(0.0),
        "estimated_cost_saved_usd": cost,
        "recent_events": [],
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_initialize_and_list() {
        let ctx = McpContext {
            store: None,
            ledger: headroom_store::ledger::SavingsLedger::open(std::env::temp_dir().join("mcp-test-ledger.jsonl")),
            proxy_url: "http://127.0.0.1:8787".into(),
            read_cache: Default::default(),
        };
        let req = RpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(json!(1)),
            method: "initialize".into(),
            params: json!({}),
        };
        let resp = dispatch(&ctx, &req);
        assert!(resp.result.is_some());

        let req = RpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(json!(2)),
            method: "tools/list".into(),
            params: json!({}),
        };
        let resp = dispatch(&ctx, &req);
        let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
        assert_eq!(tools.len(), 4);
    }

    #[test]
    fn unknown_method_is_error() {
        let ctx = McpContext {
            store: None,
            ledger: headroom_store::ledger::SavingsLedger::open(std::env::temp_dir().join("mcp-test-ledger2.jsonl")),
            proxy_url: String::new(),
            read_cache: Default::default(),
        };
        let req = RpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(json!(3)),
            method: "bogus".into(),
            params: json!({}),
        };
        let resp = dispatch(&ctx, &req);
        assert_eq!(resp.error.unwrap().code, METHOD_NOT_FOUND);
    }
}
