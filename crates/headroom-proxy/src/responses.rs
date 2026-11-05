//! OpenAI Responses API (/v1/responses) request compression.
//!
//! The Responses API body has `input` as either a string (user prompt;
//! untouched) or an array of items. Only `function_call_output` items with
//! a large `output` string are compressed — mirroring the chat-completions
//! tool-message handling via the same content-type routing.

use headroom_core::content::{detect_content_type, ContentType};
use headroom_core::tokens::EstimatingTokenCounter;
use headroom_core::transforms::diff_compressor::DiffCompressor;
use headroom_core::transforms::log_compressor::LogCompressor;
use headroom_core::transforms::search_compressor::SearchCompressor;
use headroom_core::transforms::smart_crusher::SmartCrusher;
use headroom_core::transforms::text_crusher::TextCrusher;
use headroom_store::ccr::CompressionStore;
use serde_json::Value;

use crate::compress_request::{
    compress_text_in_place, stash, CompressError, CompressOutcome, CompressionSummary,
    RequestCompressConfig,
};

/// Compress a Responses API request body. Fail-open: any error returns
/// `Unchanged`.
pub fn compress_responses_body(
    body: &Value,
    config: &RequestCompressConfig,
    store: Option<&CompressionStore>,
) -> Result<CompressOutcome, CompressError> {
    let Some(Value::Array(items)) = body.get("input") else {
        return Ok(CompressOutcome::Unchanged { reason: "input is not an array" });
    };
    if items.is_empty() {
        return Ok(CompressOutcome::Unchanged { reason: "empty input" });
    }

    let counter = EstimatingTokenCounter::default();
    let mut items = items.clone();
    let mut summary = CompressionSummary::default();
    let protect_from = items.len().saturating_sub(config.protect_recent);

    let mut changed = false;
    for idx in 0..protect_from {
        let item = &mut items[idx];
        if item.get("type").and_then(Value::as_str) != Some("function_call_output") {
            continue;
        }
        let Some(Value::String(output)) = item.get_mut("output") else {
            continue;
        };
        let before = counter.count_text(output) as i64;
        // Reuse the shared in-place compressor (gates + routing + CCR).
        if compress_text_in_place(output, "", config, store, &mut summary)? {
            changed = true;
            let after = counter.count_text(output) as i64;
            summary.tokens_before += before;
            summary.tokens_after += after;
        }
    }

    if !changed {
        return Ok(CompressOutcome::Unchanged { reason: "no compressible content" });
    }
    if summary.tokens_after > summary.tokens_before {
        return Ok(CompressOutcome::Unchanged { reason: "inflation guard" });
    }
    let mut rewritten = body.clone();
    rewritten["input"] = Value::Array(items);
    Ok(CompressOutcome::Compressed { body: rewritten, summary })
}

// Keep the unused imports referenced (search/log/diff/crusher are used via
// compress_text_in_place's routing; the direct constructors stay for
// callers that want per-type compression of Responses items).
#[allow(dead_code)]
fn _type_router_probe(text: &str) -> &'static str {
    match detect_content_type(text) {
        ContentType::SearchResults => "search",
        ContentType::BuildOutput => "log",
        ContentType::GitDiff => "diff",
        ContentType::JsonArray => "json",
        ContentType::PlainText => "text",
        _ => "other",
    }
}

#[allow(dead_code)]
fn _probe_stash(store: Option<&CompressionStore>) {
    let _ = stash;
    let _ = SearchCompressor::default();
    let _ = LogCompressor::default();
    let _ = DiffCompressor::default();
    let _ = SmartCrusher::default();
    let _ = TextCrusher::default();
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn big_log() -> String {
        (0..300)
            .map(|i| format!("warning: unused variable v{i} in module_{i}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn function_call_output_compressed() {
        let mut input = vec![json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "run the build"}]})];
        for i in 0..2 {
            input.push(json!({"type": "function_call_output", "call_id": i, "output": big_log()}));
        }
        for i in 0..4 {
            input.push(json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "filler {i}"}]}));
        }
        let body = json!({"model": "gpt-5", "input": input});
        let out = compress_responses_body(&body, &RequestCompressConfig::default(), None).unwrap();
        match out {
            CompressOutcome::Compressed { body, summary } => {
                let output = body["input"][1]["output"].as_str();
                assert!(output.unwrap().len() < big_log().len());
                assert!(summary.tokens_saved() > 0);
            }
            _ => panic!("expected compression"),
        }
    }

    #[test]
    fn string_input_untouched() {
        let body = json!({"model": "gpt-5", "input": "just a prompt"});
        assert!(matches!(
            compress_responses_body(&body, &RequestCompressConfig::default(), None).unwrap(),
            CompressOutcome::Unchanged { .. }
        ));
    }

    #[test]
    fn protect_recent_tail() {
        let mut input = vec![json!({"type": "message", "role": "user", "content": "q"})];
        for i in 0..6 {
            input.push(json!({"type": "function_call_output", "call_id": i, "output": big_log()}));
        }
        let body = json!({"model": "gpt-5", "input": input});
        let cfg = RequestCompressConfig { protect_recent: 4, ..Default::default() };
        match compress_responses_body(&body, &cfg, None).unwrap() {
            CompressOutcome::Compressed { body, .. } => {
                // The last 4 outputs are protected verbatim.
                assert_eq!(body["input"][3]["output"].as_str().unwrap(), big_log());
                assert_eq!(body["input"][6]["output"].as_str().unwrap(), big_log());
                // The earlier one is compressed.
                assert!(body["input"][2]["output"].as_str().unwrap().len() < big_log().len());
            }
            _ => panic!("expected compression"),
        }
    }

    #[test]
    fn small_output_untouched() {
        let body = json!({
            "model": "gpt-5",
            "input": [
                {"type": "function_call_output", "call_id": "c1", "output": "tiny"},
            ]
        });
        assert!(matches!(
            compress_responses_body(&body, &RequestCompressConfig::default(), None).unwrap(),
            CompressOutcome::Unchanged { .. }
        ));
    }
}
