//! Whole-conversation verbatim deduplication with in-context pointers.
//!
//! dedupe_messages scans a conversation (an Anthropic-style message list)
//! and replaces every tool-result text block whose text already appeared
//! verbatim — as an exact substring of at least 200 chars — in an earlier
//! message with a short pointer note. The first occurrence always stays
//! untouched, so the full text remains available in context above. Plain
//! (non-tool-result) text blocks are collected into the seen corpus but
//! never replaced.

use serde_json::Value;

/// Minimum char count for a text block to be considered a duplicate.
const MIN_DUP_CHARS: usize = 200;

/// Deduplicates tool-result text blocks across a conversation.
///
/// Deterministic: messages are processed in order and the first occurrence
/// of any text wins. The input slice is never modified; modified copies are
/// returned.
pub fn dedupe_messages(messages: &[Value]) -> Vec<Value> {
    let mut seen: Vec<String> = Vec::new();
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());
    for message in messages {
        let mut message = message.clone();
        dedupe_tool_results(&mut message, &seen);
        collect_texts(&message, &mut seen);
        out.push(message);
    }
    out
}

/// Replaces duplicate tool-result texts in one message with pointer notes.
fn dedupe_tool_results(message: &mut Value, seen: &[String]) {
    let Some(blocks) = message.get_mut("content").and_then(Value::as_array_mut) else {
        return;
    };
    for block in blocks.iter_mut() {
        let Some(object) = block.as_object_mut() else {
            continue;
        };
        if object.get("type").and_then(Value::as_str) != Some("tool_result") {
            continue;
        }
        match object.get_mut("content") {
            // string-shaped tool result content
            Some(Value::String(text)) => {
                if is_duplicate(text, seen) {
                    let marker = duplicate_marker(text);
                    *text = marker;
                }
            }
            // block-shaped tool result content
            Some(Value::Array(items)) => {
                for item in items.iter_mut() {
                    let Some(inner) = item.as_object_mut() else {
                        continue;
                    };
                    if inner.get("type").and_then(Value::as_str) != Some("text") {
                        continue;
                    }
                    if let Some(Value::String(text)) = inner.get_mut("text") {
                        if is_duplicate(text, seen) {
                            let marker = duplicate_marker(text);
                            *text = marker;
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// Adds every text of the message to the seen corpus (call after deduping
/// so only strictly-earlier messages count).
fn collect_texts(message: &Value, seen: &mut Vec<String>) {
    match message.get("content") {
        Some(Value::Array(blocks)) => {
            for block in blocks {
                let Some(object) = block.as_object() else {
                    continue;
                };
                match object.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(Value::String(text)) = object.get("text") {
                            seen.push(text.clone());
                        }
                    }
                    Some("tool_result") => match object.get("content") {
                        Some(Value::String(text)) => seen.push(text.clone()),
                        Some(Value::Array(items)) => {
                            for item in items {
                                if let Some(Value::String(text)) = item.get("text") {
                                    seen.push(text.clone());
                                }
                            }
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
        Some(Value::String(text)) => seen.push(text.clone()),
        _ => {}
    }
}

/// True when the text is long enough and appeared verbatim earlier.
fn is_duplicate(text: &str, seen: &[String]) -> bool {
    text.chars().count() >= MIN_DUP_CHARS && seen.iter().any(|earlier| earlier.contains(text))
}

/// The in-context pointer note replacing a duplicated tool output.
fn duplicate_marker(text: &str) -> String {
    format!(
        "[Duplicate of earlier tool output ({} chars). The full text is already in context above.]",
        text.chars().count()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool_result_msg(text: &str) -> Value {
        json!({
            "role": "user",
            "content": [
                {"type": "tool_result", "content": [{"type": "text", "text": text}]}
            ]
        })
    }

    #[test]
    fn duplicate_tool_output_across_turns_is_replaced() {
        let text = "alpha ".repeat(60); // 360 chars
        let messages = vec![tool_result_msg(&text), tool_result_msg(&text)];
        let out = dedupe_messages(&messages);
        assert_eq!(out[0], messages[0]); // first occurrence untouched
        let duplicate = out[1]["content"][0]["content"][0]["text"].as_str().unwrap();
        assert_eq!(
            duplicate,
            "[Duplicate of earlier tool output (360 chars). The full text is already in context above.]"
        );
    }

    #[test]
    fn unique_outputs_are_untouched() {
        let messages = vec![
            tool_result_msg(&"alpha ".repeat(60)),
            tool_result_msg(&"beta ".repeat(60)),
        ];
        assert_eq!(dedupe_messages(&messages), messages);
    }

    #[test]
    fn short_duplicates_are_kept() {
        let text = "short".repeat(20); // 100 chars, below the threshold
        let messages = vec![tool_result_msg(&text), tool_result_msg(&text)];
        assert_eq!(dedupe_messages(&messages), messages);
    }

    #[test]
    fn substring_of_earlier_output_is_replaced() {
        let full = "the quick brown fox ".repeat(30);
        let later: String = full.chars().skip(100).take(250).collect();
        let messages = vec![tool_result_msg(&full), tool_result_msg(&later)];
        let out = dedupe_messages(&messages);
        let duplicate = out[1]["content"][0]["content"][0]["text"].as_str().unwrap();
        assert!(
            duplicate.starts_with("[Duplicate of earlier tool output (250 chars)."),
            "unexpected marker: {duplicate}"
        );
    }

    #[test]
    fn every_repeat_after_the_first_is_replaced() {
        let text = "gamma ".repeat(60);
        let messages = vec![
            tool_result_msg(&text),
            tool_result_msg(&text),
            tool_result_msg(&text),
        ];
        let out = dedupe_messages(&messages);
        assert_eq!(out[0], messages[0]);
        for message in &out[1..] {
            let body = message["content"][0]["content"][0]["text"].as_str().unwrap();
            assert!(body.starts_with("[Duplicate of earlier tool output"));
        }
    }

    #[test]
    fn plain_text_blocks_are_never_replaced() {
        let text = "delta ".repeat(60);
        let messages = vec![
            json!({"role": "assistant", "content": [{"type": "text", "text": text}]}),
            json!({"role": "assistant", "content": [{"type": "text", "text": text}]}),
        ];
        assert_eq!(dedupe_messages(&messages), messages);
    }

    #[test]
    fn string_shaped_tool_result_content_is_replaced() {
        let text = "epsilon ".repeat(60);
        let messages = vec![
            json!({"role": "user", "content": [{"type": "tool_result", "content": text}]}),
            json!({"role": "user", "content": [{"type": "tool_result", "content": text}]}),
        ];
        let out = dedupe_messages(&messages);
        assert_eq!(out[0], messages[0]);
        let duplicate = out[1]["content"][0]["content"].as_str().unwrap();
        assert_eq!(
            duplicate,
            "[Duplicate of earlier tool output (480 chars). The full text is already in context above.]"
        );
    }

    #[test]
    fn non_object_messages_pass_through() {
        let messages = vec![Value::Null, json!({"role": "user", "content": "hi"})];
        assert_eq!(dedupe_messages(&messages), messages);
    }

    #[test]
    fn dedup_is_deterministic_across_runs() {
        let text = "zeta ".repeat(60);
        let messages = vec![tool_result_msg(&text), tool_result_msg(&text)];
        assert_eq!(dedupe_messages(&messages), dedupe_messages(&messages));
    }
}
