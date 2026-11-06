//! Request-body compression for the Anthropic and OpenAI wire formats.
//!
//! Walks tool-result content, applies the protection gates, dispatches to the
//! matching compressor by content type, stashes dropped originals in the CCR
//! store, and re-serializes. Everything here is fail-open: any error returns
//! the body untouched.

use headroom_core::content::{detect_content_type, ContentType};
use headroom_store::ccr::{CompressionStore, StoreError};
use headroom_core::tokens::EstimatingTokenCounter;
use headroom_core::transforms::diff_compressor::DiffCompressor;
use headroom_core::transforms::log_compressor::LogCompressor;
use headroom_core::transforms::search_compressor::SearchCompressor;
use headroom_core::transforms::smart_crusher::SmartCrusher;
use headroom_core::transforms::text_crusher::TextCrusher;
use serde_json::{json, Value};

/// Tools whose results are never compressed (mirrors the Python
/// DEFAULT_EXCLUDE_TOOLS; lossless-only folding is out of M1 scope).
pub const EXCLUDE_TOOLS: &[&str] = &[
    "read", "glob", "grep", "write", "edit", "skill", "web_search", "web_fetch",
    "headroom_retrieve", "caveman_retrieve", "view", "read_file",
];

/// Gates and knobs (defaults mirror the Python CompressConfig /
/// ContentRouterConfig).
#[derive(Debug, Clone)]
pub struct RequestCompressConfig {
    /// Last N messages are never touched.
    pub protect_recent: usize,
    /// Leading messages assumed provider-prompt-cached; never touched.
    pub frozen_message_count: usize,
    /// Min chars of a tool result before compression is considered.
    pub min_chars_for_block_compression: usize,
    /// Min estimated tokens before compression is considered.
    pub min_tokens_to_compress: usize,
    /// Error-ish outputs up to this many chars stay verbatim.
    pub error_protection_max_chars: usize,
    /// TextCrusher target keep-ratio for plain text.
    pub plain_target_ratio: f64,
}

impl Default for RequestCompressConfig {
    fn default() -> Self {
        Self {
            protect_recent: 4,
            frozen_message_count: 0,
            min_chars_for_block_compression: 500,
            min_tokens_to_compress: 250,
            error_protection_max_chars: 8000,
            plain_target_ratio: 0.5,
        }
    }
}

/// Which wire format the request body speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireFormat {
    Anthropic,
    OpenAi,
}

/// Summary of one compression pass (for stats + the savings ledger).
#[derive(Debug, Clone, Default)]
pub struct CompressionSummary {
    pub tokens_before: i64,
    pub tokens_after: i64,
    pub strategies: Vec<String>,
    pub ccr_hashes: Vec<String>,
}

impl CompressionSummary {
    pub fn tokens_saved(&self) -> i64 {
        (self.tokens_before - self.tokens_after).max(0)
    }
}

/// Outcome of compressing a request body.
pub enum CompressOutcome {
    /// Body unchanged (nothing to do, gated, or fail-open).
    Unchanged { reason: &'static str },
    /// Body rewritten: carries the full replacement body and the summary.
    Compressed { body: Value, summary: CompressionSummary },
}

/// Errors are internal-only; callers map them to fail-open passthrough.
#[derive(Debug, thiserror::Error)]
pub enum CompressError {
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("store: {0}")]
    Store(#[from] StoreError),
}

/// Compress one request body (either wire format). Fail-open: on error the
/// caller should forward the original bytes.
pub fn compress_request_body(
    body: &Value,
    format: WireFormat,
    config: &RequestCompressConfig,
    store: Option<&CompressionStore>,
) -> Result<CompressOutcome, CompressError> {
    let messages = match body.get("messages") {
        Some(Value::Array(m)) => m.clone(),
        _ => return Ok(CompressOutcome::Unchanged { reason: "no messages" }),
    };
    if messages.is_empty() {
        return Ok(CompressOutcome::Unchanged { reason: "empty messages" });
    }

    let counter = EstimatingTokenCounter::default();
    let query = extract_user_query(&messages, format);
    let tool_names = collect_tool_names(&messages, format);

    let mut messages = messages;
    let mut summary = CompressionSummary::default();
    let freeze = config.frozen_message_count.min(messages.len());
    let protect_from = messages.len().saturating_sub(config.protect_recent);

    let mut changed = false;
    for idx in freeze..protect_from {
        let (role, is_tool_output) = message_shape(&messages[idx], format);
        // Only tool outputs are compressed on the request path (M1).
        if !is_tool_output {
            let _ = role;
            continue;
        }
        let tool_name = message_tool_name(&messages[idx], &tool_names, format)
            .unwrap_or_default();
        if EXCLUDE_TOOLS.iter().any(|t| t.eq_ignore_ascii_case(&tool_name)) {
            continue;
        }
        let before = counter.count_message(&messages[idx]) as i64;
        let compressed = match format {
            WireFormat::Anthropic => compress_anthropic_message(
                &mut messages[idx], &query, config, store, &mut summary,
            )?,
            WireFormat::OpenAi => {
                compress_openai_tool_message(
                    &mut messages[idx], &query, config, store, &mut summary,
                )?
            }
        };
        if compressed {
            changed = true;
            let after = counter.count_message(&messages[idx]) as i64;
            summary.tokens_before += before;
            summary.tokens_after += after;
        }
    }

    if !changed {
        return Ok(CompressOutcome::Unchanged { reason: "no compressible content" });
    }
    // Inflation guard: never grow the request (byte-level, stricter than the
    // token estimate alone).
    if summary.tokens_after > summary.tokens_before {
        return Ok(CompressOutcome::Unchanged { reason: "inflation guard" });
    }
    let mut rewritten = body.clone();
    rewritten["messages"] = Value::Array(messages);
    Ok(CompressOutcome::Compressed { body: rewritten, summary })
}

/// Anthropic: compress the text inside tool_result content blocks.
fn compress_anthropic_message(
    message: &mut Value,
    query: &str,
    config: &RequestCompressConfig,
    store: Option<&CompressionStore>,
    summary: &mut CompressionSummary,
) -> Result<bool, CompressError> {
    let Some(Value::Array(blocks)) = message.get_mut("content") else {
        return Ok(false);
    };
    let mut changed = false;
    for block in blocks.iter_mut() {
        if block.get("type").and_then(Value::as_str) != Some("tool_result") {
            continue;
        }
        // Error results stay verbatim (protect_error_outputs).
        if block.get("is_error").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        match block.get_mut("content") {
            Some(Value::String(text)) => {
                if compress_text_in_place(text, query, config, store, summary)? {
                    changed = true;
                }
            }
            Some(Value::Array(inner)) => {
                for cell in inner.iter_mut() {
                    if cell.get("type").and_then(Value::as_str) == Some("text") {
                        if let Some(Value::String(text)) = cell.get_mut("text") {
                            if compress_text_in_place(text, query, config, store, summary)? {
                                changed = true;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(changed)
}

/// OpenAI: compress the content string of a role=tool message.
fn compress_openai_tool_message(
    message: &mut Value,
    query: &str,
    config: &RequestCompressConfig,
    store: Option<&CompressionStore>,
    summary: &mut CompressionSummary,
) -> Result<bool, CompressError> {
    match message.get_mut("content") {
        Some(Value::String(text)) => {
            compress_text_in_place(text, query, config, store, summary)
        }
        Some(Value::Array(parts)) => {
            let mut changed = false;
            for part in parts.iter_mut() {
                if part.get("type").and_then(Value::as_str) == Some("text") {
                    if let Some(Value::String(text)) = part.get_mut("text") {
                        if compress_text_in_place(text, query, config, store, summary)? {
                            changed = true;
                        }
                    }
                }
            }
            Ok(changed)
        }
        _ => Ok(false),
    }
}

/// Compress one tool-result text in place. Returns whether it changed.
pub(crate) fn compress_text_in_place(
    text: &mut String,
    query: &str,
    config: &RequestCompressConfig,
    store: Option<&CompressionStore>,
    summary: &mut CompressionSummary,
) -> Result<bool, CompressError> {
    let original = text.clone();
    if original.len() < config.min_chars_for_block_compression {
        return Ok(false);
    }
    let counter = EstimatingTokenCounter::default();
    if counter.count_text(&original) < config.min_tokens_to_compress {
        return Ok(false);
    }
    // Error-looking output stays verbatim up to the protection cap.
    if looks_like_error(&original) && original.len() <= config.error_protection_max_chars {
        return Ok(false);
    }
    // Never recompress content that already carries a CCR marker.
    if headroom_core::ccr::contains_ccr_marker(&original) {
        return Ok(false);
    }

    let compressed = match detect_content_type(&original) {
        ContentType::JsonArray => compress_json_array(&original, query, store, summary)?,
        ContentType::SearchResults => {
            let r = SearchCompressor::default().compress(&original, query, 1.0);
            if let Some(key) = &r.cache_key {
                stash(store, key, &original, &r.compressed, "search_compressor", summary)?;
            }
            summary.strategies.push("search_compressor".into());
            r.compressed
        }
        ContentType::BuildOutput => {
            let r = LogCompressor::default().compress(&original, query, 1.0);
            if let Some(key) = &r.cache_key {
                stash(store, key, &original, &r.compressed, "log_compressor", summary)?;
            }
            summary.strategies.push("log_compressor".into());
            r.compressed
        }
        ContentType::GitDiff => {
            let r = DiffCompressor::default().compress(&original, query);
            if let Some(key) = &r.cache_key {
                stash(store, key, &original, &r.compressed, "diff_compressor", summary)?;
            }
            summary.strategies.push("diff_compressor".into());
            r.compressed
        }
        // M1: conservative — unknown shapes keep their text.
        ContentType::Html
        | ContentType::Tabular
        | ContentType::StructuredConfig
        | ContentType::SourceCode => return Ok(false),
        ContentType::PlainText => {
            let r = TextCrusher::default().compress(&original, query, None);
            summary.strategies.push("text_crusher".into());
            r.compressed
        }
    };

    if compressed.len() >= original.len() {
        return Ok(false); // local inflation guard
    }
    *text = compressed;
    Ok(true)
}

/// JSON arrays: lossless compaction first, then lossy crush + CCR sentinel.
fn compress_json_array(
    original: &str,
    query: &str,
    store: Option<&CompressionStore>,
    summary: &mut CompressionSummary,
) -> Result<String, CompressError> {
    let parsed: Value = match serde_json::from_str(original.trim()) {
        Ok(v) => v,
        Err(_) => return Ok(original.to_string()),
    };
    let Value::Array(items) = &parsed else {
        return Ok(original.to_string());
    };

    // Lossless densification first (no CCR, byte-stable).
    if let Some(densified) = headroom_core::compaction::encode_compacted(items) {
        // Wrap as a JSON string like the Python ContentRouter does when the
        // tool result content was a JSON array.
        if let Ok(wrapped) = serde_json::to_string(&densified) {
            if wrapped.len() < original.len() {
                summary.strategies.push("smart_crusher:compaction".into());
                return Ok(wrapped);
            }
        }
        if densified.len() < original.len() {
            summary.strategies.push("smart_crusher:compaction".into());
            return Ok(densified);
        }
    }

    // Lossy statistical crush with CCR stash.
    let crusher = SmartCrusher::default();
    let result = crusher.crush_array(items, query, 1.0);
    if !result.was_modified {
        return Ok(original.to_string());
    }
    if let (Some(key), Some(original_stash)) = (&result.cache_key, &result.ccr_original) {
        stash(store, key, original_stash, "", "smart_crusher", summary)?;
    }
    summary.strategies.push("smart_crusher".into());
    Ok(serde_json::to_string(&result.compressed).unwrap_or_else(|_| original.to_string()))
}

/// Persist a dropped original under its CCR hash (fail-open on store errors).
pub(crate) fn stash(
    store: Option<&CompressionStore>,
    key: &str,
    original: &str,
    compressed: &str,
    strategy: &str,
    summary: &mut CompressionSummary,
) -> Result<(), CompressError> {
    summary.ccr_hashes.push(key.to_string());
    if let Some(store) = store {
        let counter = EstimatingTokenCounter::default();
        let _ = store.store_original(
            original,
            compressed,
            counter.count_text(original) as i64,
            counter.count_text(compressed) as i64,
            None,
            Some(strategy),
        );
    }
    Ok(())
}

fn looks_like_error(text: &str) -> bool {
    let head = &text[..text.len().min(2000)];
    head.to_lowercase().contains("error")
        || head.contains("Traceback")
        || head.contains("FAILED")
        || head.contains("panic:")
}

/// Extract the last user message text as the relevance query.
pub(crate) fn extract_user_query(messages: &[Value], format: WireFormat) -> String {
    for msg in messages.iter().rev() {
        let is_user = match format {
            WireFormat::Anthropic => msg.get("role") == Some(&json!("user")),
            WireFormat::OpenAi => msg.get("role") == Some(&json!("user")),
        };
        if !is_user {
            continue;
        }
        return match msg.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(blocks)) => blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
    }
    String::new()
}

/// tool_use_id / tool_call_id -> tool name, from assistant messages.
fn collect_tool_names(messages: &[Value], format: WireFormat) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    for msg in messages {
        let Some(Value::Array(blocks)) = msg.get("content") else {
            continue;
        };
        match format {
            WireFormat::Anthropic => {
                for b in blocks {
                    if b.get("type") == Some(&json!("tool_use")) {
                        if let (Some(id), Some(name)) = (
                            b.get("id").and_then(Value::as_str),
                            b.get("name").and_then(Value::as_str),
                        ) {
                            map.insert(id.to_string(), name.to_string());
                        }
                    }
                }
            }
            WireFormat::OpenAi => {
                for b in blocks {
                    if b.get("type") == Some(&json!("tool_call")) {
                        if let (Some(id), Some(func)) = (
                            b.get("id").and_then(Value::as_str),
                            b.get("function"),
                        ) {
                            if let Some(name) = func.get("name").and_then(Value::as_str) {
                                map.insert(id.to_string(), name.to_string());
                            }
                        }
                    }
                }
            }
        }
    }
    map
}

fn message_shape(msg: &Value, format: WireFormat) -> (String, bool) {
    let role = msg.get("role").and_then(Value::as_str).unwrap_or("").to_string();
    match format {
        WireFormat::Anthropic => {
            let is_tool = msg
                .get("content")
                .and_then(Value::as_array)
                .map(|blocks| {
                    blocks
                        .iter()
                        .any(|b| b.get("type") == Some(&json!("tool_result")))
                })
                .unwrap_or(false);
            (role, is_tool)
        }
        WireFormat::OpenAi => {
            let is_tool = role == "tool";
            (role, is_tool)
        }
    }
}

fn message_tool_name(
    msg: &Value,
    names: &std::collections::HashMap<String, String>,
    format: WireFormat,
) -> Option<String> {
    match format {
        WireFormat::Anthropic => msg
            .get("content")
            .and_then(Value::as_array)?
            .iter()
            .filter(|b| b.get("type") == Some(&json!("tool_result")))
            .find_map(|b| b.get("tool_use_id").and_then(Value::as_str))
            .and_then(|id| names.get(id).cloned()),
        WireFormat::OpenAi => msg
            .get("tool_call_id")
            .and_then(Value::as_str)
            .and_then(|id| names.get(id).cloned()),
    }
}

// Re-export for callers that need the raw hash helper.
pub use headroom_core::ccr::ccr_hash as hash_of;

/// The retrieve tool name (matches the Python headroom).
pub const RETRIEVE_TOOL_NAME: &str = "headroom_retrieve";

fn retrieve_tool_schema() -> Value {
    json!({
        "type": "object",
        "properties": {"hash": {"type": "string"}},
        "required": ["hash"]
    })
}

fn retrieve_tool_description() -> String {
    "Retrieve the original (pre-compression) content by hash from a compression marker. Use when you need the full text behind a 'Retrieve more: hash=...' or '<<ccr:...>>' marker.".to_string()
}

/// Inject the retrieve tool into a request body when the compression pass
/// emitted CCR markers (mirrors the Python CCRToolInjector). Idempotent.
pub fn inject_retrieve_tool(body: &mut Value, format: WireFormat) {
    let tool = match format {
        WireFormat::Anthropic => json!({
            "name": RETRIEVE_TOOL_NAME,
            "description": retrieve_tool_description(),
            "input_schema": retrieve_tool_schema(),
        }),
        WireFormat::OpenAi => json!({
            "type": "function",
            "function": {
                "name": RETRIEVE_TOOL_NAME,
                "description": retrieve_tool_description(),
                "parameters": retrieve_tool_schema(),
            }
        }),
    };
    let Some(obj) = body.as_object_mut() else { return };
    let tools = obj.entry("tools").or_insert_with(|| Value::Array(Vec::new()));
    let Value::Array(arr) = tools else { return };
    let exists = arr.iter().any(|t| {
        t.get("name") == Some(&json!(RETRIEVE_TOOL_NAME))
            || t.get("function").and_then(|f| f.get("name"))
                == Some(&json!(RETRIEVE_TOOL_NAME))
    });
    if !exists {
        arr.push(tool);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anthropic_tool_result(text: &str, idx_from_end: usize, total: usize) -> Value {
        let mut messages = Vec::new();
        messages.push(json!({"role": "user", "content": "please summarize"}));
        messages.push(json!({"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {}}
        ]}));
        let mut tool_msg = json!({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": text}
        ]});
        for _ in 0..(total - idx_from_end - 1) {
            messages.push(json!({"role": "assistant", "content": "ok"}));
        }
        messages.push(tool_msg.take());
        json!({"model": "claude-sonnet-4-5", "messages": messages})
    }

    #[test]
    fn small_tool_result_untouched() {
        let body = anthropic_tool_result("short output", 0, 1);
        let out = compress_request_body(
            &body,
            WireFormat::Anthropic,
            &RequestCompressConfig::default(),
            None,
        )
        .unwrap();
        match out {
            CompressOutcome::Unchanged { reason } => {
                assert_eq!(reason, "no compressible content");
            }
            _ => panic!("expected unchanged"),
        }
    }

    #[test]
    fn protected_recent_tail_untouched() {
        // A huge tool result sitting inside the last 4 messages is protected.
        let big = "error: something failed\n".repeat(200);
        let body = anthropic_tool_result(&big, 0, 1);
        let out = compress_request_body(
            &body,
            WireFormat::Anthropic,
            &RequestCompressConfig::default(),
            None,
        )
        .unwrap();
        assert!(matches!(out, CompressOutcome::Unchanged { .. }));
    }

    #[test]
    fn excluded_tool_never_compressed() {
        let big = "line of log output\n".repeat(200);
        let mut body = anthropic_tool_result(&big, 3, 8);
        // Rename the tool to Grep (excluded).
        body["messages"][1]["content"][0]["name"] = json!("Grep");
        let out = compress_request_body(
            &body,
            WireFormat::Anthropic,
            &RequestCompressConfig::default(),
            None,
        )
        .unwrap();
        assert!(matches!(out, CompressOutcome::Unchanged { .. }));
    }

    #[test]
    fn error_output_protected_verbatim() {
        let err = "Traceback (most recent call last):\n  boom\n".repeat(60);
        assert!(err.len() <= 8000);
        let body = anthropic_tool_result(&err, 3, 8);
        let out = compress_request_body(
            &body,
            WireFormat::Anthropic,
            &RequestCompressConfig::default(),
            None,
        )
        .unwrap();
        assert!(matches!(out, CompressOutcome::Unchanged { .. }));
    }

    #[test]
    fn openai_tool_message_shape() {
        let msg = json!({"role": "tool", "tool_call_id": "call_1", "content": "x"});
        let (_, is_tool) = message_shape(&msg, WireFormat::OpenAi);
        assert!(is_tool);
        let mut names = std::collections::HashMap::new();
        names.insert("call_1".to_string(), "Bash".to_string());
        assert_eq!(
            message_tool_name(&msg, &names, WireFormat::OpenAi).as_deref(),
            Some("Bash")
        );
    }

    #[test]
    fn query_extraction() {
        let msgs = vec![
            json!({"role": "user", "content": "first question"}),
            json!({"role": "assistant", "content": "answer"}),
            json!({"role": "user", "content": "find the failing test"}),
        ];
        assert_eq!(extract_user_query(&msgs, WireFormat::OpenAi), "find the failing test");
    }
}
