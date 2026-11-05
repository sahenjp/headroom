//! Estimating token counter: the fallback when no exact tokenizer exists.
//!
//! Port of headroom/tokenizers/estimator.py semantics (approximate, no ML).
//! Calibrated against tiktoken cl100k_base and deliberately errs on the high
//! side: a slight overestimate is safer for context-window management.
//!
//! Estimation strategy:
//! - base chars-per-token ratio auto-detected from content: 4.0 prose,
//!   3.5 code, 3.2 JSON;
//! - CJK / Kana / Hangul runs are priced separately at 1.5 chars/token
//!   (these scripts tokenize at roughly one token per character);
//! - URL and UUID overhead is added on top;
//! - messages carry a per-message overhead plus an extra charge per
//!   tool_call / tool_use block.
//!
//! The count is rounded up (ceil) by design, so the estimate never
//! under-reports.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

/// Average chars per token for English prose.
pub const CHARS_PER_TOKEN: f64 = 4.0;
/// Chars per token for source code (code is denser).
pub const CHARS_PER_TOKEN_CODE: f64 = 3.5;
/// Chars per token for JSON (JSON has more structure).
pub const CHARS_PER_TOKEN_JSON: f64 = 3.2;
/// Chars per token for dense scripts (CJK / Kana / Hangul / fullwidth).
pub const CHARS_PER_TOKEN_CJK: f64 = 1.5;
/// Token overhead per message (role, formatting, etc.).
pub const MESSAGE_OVERHEAD: usize = 4;
/// Reply-priming overhead added once by [EstimatingTokenCounter::count_messages].
pub const REPLY_OVERHEAD: usize = 3;
/// Extra tokens per tool_call / tool_use block.
pub const TOOL_CALL_OVERHEAD: usize = 10;
/// Fixed price of an image content block (base64 blobs are never priced as
/// text; mirrors the Python base counter).
pub const IMAGE_BLOCK_TOKENS: usize = 1600;
/// Fixed price of an audio content block.
pub const AUDIO_BLOCK_TOKENS: usize = 200;

static CODE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?m)(?:def |class |function |const |let |var |import |from |if \(|for \(|while \(|switch \(|try \{|catch \(|=>|->|\{\{|\}\}|;$)",
    )
    .expect("code pattern must compile")
});

static JSON_START_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*[\[\{]").expect("json start pattern must compile"));

static URL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"https?://\S+").expect("url pattern must compile"));

static UUID_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
        .expect("uuid pattern must compile")
});

/// Approximate token counter for unknown models and fallback paths.
///
/// Create with [new] for auto-detected ratios, or with
/// [with_chars_per_token] to pin the base ratio (the Python
/// 'chars_per_token=' override). The counter is stateless and synchronous.
pub struct EstimatingTokenCounter {
    fixed_ratio: Option<f64>,
}

impl EstimatingTokenCounter {
    /// Auto-detecting counter (prose 4.0 / code 3.5 / JSON 3.2 chars/token).
    pub fn new() -> Self {
        Self { fixed_ratio: None }
    }

    /// Counter with a fixed chars-per-token ratio.
    ///
    /// Dense scripts are still priced at [CHARS_PER_TOKEN_CJK]; the fixed
    /// ratio applies to the remaining characters, and (mirroring the Python
    /// fixed-ratio path) no URL/UUID overhead is added.
    pub fn with_chars_per_token(chars_per_token: f64) -> Self {
        Self {
            fixed_ratio: Some(chars_per_token),
        }
    }

    /// Estimate the token count of 'text'. Empty input counts as 0.
    ///
    /// The estimate is rounded up and never below 1 for non-empty text.
    pub fn count_text(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        let cjk_chars = text.chars().filter(|c| is_cjk_char(*c)).count();
        let other_chars = text.chars().count() - cjk_chars;
        let ratio = match self.fixed_ratio {
            Some(r) => r,
            None => detect_ratio(text),
        };
        let base =
            (other_chars as f64 / ratio + cjk_chars as f64 / CHARS_PER_TOKEN_CJK).ceil() as usize;
        match self.fixed_ratio {
            // Python's fixed-ratio path prices the text only.
            Some(_) => base.max(1),
            None => (base + special_overhead(text)).max(1),
        }
    }

    /// Estimate the tokens of a single chat message.
    ///
    /// Adds [MESSAGE_OVERHEAD] per message, counts role / content / name
    /// text (multi-part content blocks included), and charges
    /// [TOOL_CALL_OVERHEAD] plus the rendered text per tool_call / tool_use
    /// block.
    pub fn count_message(&self, msg: &Value) -> usize {
        let mut total = MESSAGE_OVERHEAD;
        match msg.as_object() {
            Some(obj) => {
                if let Some(Value::String(role)) = obj.get("role") {
                    total += self.count_text(role);
                }
                match obj.get("content") {
                    Some(Value::String(text)) => total += self.count_text(text),
                    Some(Value::Array(parts)) => total += self.count_content_parts(parts),
                    _ => {}
                }
                if let Some(Value::String(name)) = obj.get("name") {
                    total += self.count_text(name);
                }
                if let Some(Value::Array(calls)) = obj.get("tool_calls") {
                    for call in calls {
                        total += self.count_tool_call(call);
                    }
                }
                if let Some(call) = obj.get("function_call") {
                    total += self.count_tool_call(call);
                }
            }
            None => {
                // Malformed message: price its serialized form so the
                // counter stays total instead of panicking.
                total += self.count_serialized(msg);
            }
        }
        total
    }

    /// Estimate the tokens of a whole message list.
    ///
    /// Mirrors the Python BaseTokenizer: the per-message totals plus a
    /// single [REPLY_OVERHEAD] for the assistant reply priming, i.e.
    /// 'count_message(m) == count_messages([m]) - REPLY_OVERHEAD'.
    pub fn count_messages(&self, msgs: &[Value]) -> usize {
        let total: usize = msgs.iter().map(|m| self.count_message(m)).sum();
        total.saturating_add(REPLY_OVERHEAD)
    }

    /// Price one tool_call entry: overhead plus its id / name / arguments.
    fn count_tool_call(&self, call: &Value) -> usize {
        let mut total = TOOL_CALL_OVERHEAD;
        if let Some(obj) = call.as_object() {
            if let Some(func) = obj.get("function").and_then(Value::as_object) {
                total += self.count_text(&coerce_text(func.get("name")));
                total += self.count_text(&coerce_text(func.get("arguments")));
            }
            if let Some(id) = obj.get("id") {
                total += self.count_text(&coerce_text(Some(id)));
            }
        }
        total
    }

    /// Price a multi-part content list (Anthropic-style typed blocks).
    fn count_content_parts(&self, parts: &[Value]) -> usize {
        let mut total = 0;
        for part in parts {
            match part {
                Value::String(s) => total += self.count_text(s),
                Value::Object(obj) => {
                    let kind = obj.get("type").and_then(Value::as_str).unwrap_or("");
                    match kind {
                        "text" => {
                            total += self
                                .count_text(obj.get("text").and_then(Value::as_str).unwrap_or(""));
                        }
                        "tool_use" | "tool_call" => {
                            total += TOOL_CALL_OVERHEAD;
                            total += self
                                .count_text(obj.get("name").and_then(Value::as_str).unwrap_or(""));
                            total +=
                                self.count_serialized(obj.get("input").unwrap_or(&Value::Null));
                        }
                        "tool_result" => match obj.get("content") {
                            Some(Value::String(s)) => total += self.count_text(s),
                            Some(Value::Array(inner)) => total += self.count_content_parts(inner),
                            Some(other) => total += self.count_serialized(other),
                            None => {}
                        },
                        "image_url" | "image" | "input_image" => total += IMAGE_BLOCK_TOKENS,
                        "input_audio" | "audio" => total += AUDIO_BLOCK_TOKENS,
                        _ => total += self.count_serialized(part),
                    }
                }
                other => total += self.count_serialized(other),
            }
        }
        total
    }

    /// Price a non-string blob by its compact JSON serialization.
    fn count_serialized(&self, v: &Value) -> usize {
        match serde_json::to_string(v) {
            Ok(s) => self.count_text(&s),
            Err(_) => 0,
        }
    }
}

impl Default for EstimatingTokenCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// Auto-detect the chars-per-token ratio: JSON first, then code, else prose.
fn detect_ratio(text: &str) -> f64 {
    if JSON_START_RE.is_match(text) && serde_json::from_str::<Value>(text).is_ok() {
        return CHARS_PER_TOKEN_JSON;
    }
    let code_matches = CODE_PATTERN.find_iter(text).count();
    if code_matches as f64 > text.chars().count() as f64 / 500.0 {
        return CHARS_PER_TOKEN_CODE;
    }
    CHARS_PER_TOKEN
}

/// URL and UUID overhead: URLs chop into many tokens, UUIDs run ~8-10 tokens
/// despite being 36 chars.
fn special_overhead(text: &str) -> usize {
    let urls = URL_RE.find_iter(text).count();
    let uuids = UUID_RE.find_iter(text).count();
    urls.saturating_mul(4) + uuids.saturating_mul(2)
}

/// True for dense-script codepoints: CJK symbols/punctuation, Hiragana +
/// Katakana, CJK Unified (+ Ext A), Hangul syllables, CJK compatibility
/// ideographs, fullwidth forms, and CJK Extension B.
fn is_cjk_char(c: char) -> bool {
    matches!(c,
        '\u{3000}'..='\u{303f}'
        | '\u{3040}'..='\u{30ff}'
        | '\u{3400}'..='\u{4dbf}'
        | '\u{4e00}'..='\u{9fff}'
        | '\u{ac00}'..='\u{d7af}'
        | '\u{f900}'..='\u{faff}'
        | '\u{ff00}'..='\u{ffef}'
        | '\u{20000}'..='\u{2a6df}')
}

/// Coerce a tool-call field to text safe to count.
///
/// Strings pass through; null counts as nothing; objects / arrays are
/// JSON-serialized so an object 'arguments' is priced like the JSON string it
/// should have been; anything else falls back to its JSON form.
fn coerce_text(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => serde_json::to_string(other).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_and_tiny_text() {
        let counter = EstimatingTokenCounter::new();
        assert_eq!(counter.count_text(""), 0);
        assert_eq!(counter.count_text("a"), 1);
    }

    #[test]
    fn prose_code_json_ratios_differ_sensibly() {
        let counter = EstimatingTokenCounter::new();
        // 600 chars of prose, no code/JSON shapes
        let prose = "hello world ".repeat(50);
        assert_eq!(prose.chars().count(), 600);
        let prose_count = counter.count_text(&prose);
        assert_eq!(prose_count, 150); // ceil(600 / 4.0)

        // 600 chars of code: "const x = 1;\n" x46 + "ab"
        let code = format!("{}ab", "const x = 1;\n".repeat(46));
        assert_eq!(code.chars().count(), 600);
        let code_count = counter.count_text(&code);
        assert_eq!(code_count, 172); // ceil(600 / 3.5)

        // valid JSON: ratio 3.2
        let json_text = format!("[{}]", vec!["\"aaaaaaaaaa\""; 60].join(","));
        assert!(serde_json::from_str::<Value>(&json_text).is_ok());
        let json_count = counter.count_text(&json_text);
        assert_eq!(
            json_count,
            (json_text.chars().count() as f64 / 3.2).ceil() as usize
        );

        assert!(
            json_count > code_count,
            "json {json_count} > code {code_count}"
        );
        assert!(
            code_count > prose_count,
            "code {code_count} > prose {prose_count}"
        );
    }

    #[test]
    fn cjk_runs_are_priced_separately() {
        let counter = EstimatingTokenCounter::new();
        // 7 dense-script chars, no Latin chars: ceil(7 / 1.5) = 5
        let cjk = "\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}\u{4e16}\u{754c}";
        assert_eq!(cjk.chars().count(), 7);
        assert_eq!(counter.count_text(cjk), 5);
        // mixed: 5 Latin at 4.0 + 2 CJK at 1.5 -> ceil(1.25 + 1.333...) = 3
        let mixed = "hello\u{4e16}\u{754c}";
        assert_eq!(counter.count_text(mixed), 3);
        // dense scripts are more expensive than Latin of the same length
        let latin = "hello";
        assert!(
            counter.count_text("\u{4e16}\u{754c}\u{4e16}\u{754c}\u{4e16}")
                > counter.count_text(latin)
        );
    }

    #[test]
    fn url_and_uuid_overhead() {
        let counter = EstimatingTokenCounter::new();
        // same length, one URL: +4
        let url = "https://example.com/a/bc";
        let plain = "x".repeat(url.chars().count());
        assert_eq!(url.chars().count(), 24);
        assert_eq!(counter.count_text(url) - counter.count_text(&plain), 4);
        // same length, one UUID: +2
        let plain36 = "x".repeat(36);
        let uuid = "123e4567-e89b-12d3-a456-426614174000";
        assert_eq!(uuid.chars().count(), 36);
        assert_eq!(counter.count_text(uuid) - counter.count_text(&plain36), 2);
        // two URLs and one UUID in running text
        let text =
            "see https://a.com/x and https://b.org/y ref 123e4567-e89b-12d3-a456-426614174000 end";
        assert_eq!(special_overhead(text), 2 * 4 + 2);
    }

    #[test]
    fn message_overhead() {
        let counter = EstimatingTokenCounter::new();
        let msg = json!({"role": "user", "content": "hello world"});
        // 4 (message) + ceil(4/4)=1 (role "user") + ceil(11/4)=3 (content)
        assert_eq!(counter.count_message(&msg), 8);
        // count_messages adds the +3 reply-priming overhead once
        assert_eq!(counter.count_messages(&[msg.clone()]), 11);
        assert_eq!(
            counter.count_messages(&[msg.clone()]),
            counter.count_message(&msg) + REPLY_OVERHEAD
        );
        // two messages: sum of per-message counts + 3
        let other = json!({"role": "assistant", "content": "done"});
        let expected = counter.count_message(&msg) + counter.count_message(&other);
        assert_eq!(
            counter.count_messages(&[msg, other]),
            expected + REPLY_OVERHEAD
        );
    }

    #[test]
    fn tool_call_overhead() {
        let counter = EstimatingTokenCounter::new();
        let msg = json!({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"Tokyo\"}"
                    }
                }
            ]
        });
        // 4 (message) + ceil(9/4)=3 (role "assistant")
        // + 10 (tool_call) + ceil(6/4)=2 (id) + ceil(11/4)=3 (name)
        // + ceil(16/3.2)=5 (arguments: a JSON string, priced at the JSON ratio)
        assert_eq!(counter.count_message(&msg), 27);
    }

    #[test]
    fn tool_use_block_overhead() {
        let counter = EstimatingTokenCounter::new();
        let msg = json!({
            "role": "assistant",
            "content": [
                {"type": "tool_use", "name": "get_weather", "input": {"city": "Tokyo"}}
            ]
        });
        // 4 + 3 (role) + [10 + ceil(11/4)=3 (name)
        // + ceil(16/3.2)=5 (serialized input, JSON ratio)] = 4 + 3 + 18
        assert_eq!(counter.count_message(&msg), 25);
    }

    #[test]
    fn content_parts_are_counted() {
        let counter = EstimatingTokenCounter::new();
        // text parts are counted as text
        let msg = json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "hello world"},
                {"type": "text", "text": "second part here"}
            ]
        });
        let expected = MESSAGE_OVERHEAD
            + counter.count_text("user")
            + counter.count_text("hello world")
            + counter.count_text("second part here");
        assert_eq!(counter.count_message(&msg), expected);
        // image blocks get the fixed price, not the base64 length
        let img = json!({
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        });
        assert_eq!(
            counter.count_message(&img),
            MESSAGE_OVERHEAD + counter.count_text("user") + IMAGE_BLOCK_TOKENS
        );
        // tool_result content recurses
        let tr = json!({
            "role": "user",
            "content": [{"type": "tool_result", "content": [{"type": "text", "text": "tool says hi"}]}]
        });
        let expected =
            MESSAGE_OVERHEAD + counter.count_text("user") + counter.count_text("tool says hi");
        assert_eq!(counter.count_message(&tr), expected);
    }

    #[test]
    fn fixed_ratio_path() {
        let counter = EstimatingTokenCounter::with_chars_per_token(4.0);
        assert_eq!(counter.count_text("hello world"), 3); // ceil(11/4)
                                                          // dense scripts still use the CJK ratio under a fixed ratio
        let cjk = "\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}";
        assert_eq!(counter.count_text(cjk), 4); // ceil(5 / 1.5)
                                                // fixed-ratio path adds no URL/UUID overhead (mirrors Python)
        let with_url = "https://example.com/a/b";
        assert_eq!(counter.count_text(with_url), 6); // ceil(24/4), no +4
    }

    #[test]
    fn malformed_message_is_total() {
        let counter = EstimatingTokenCounter::new();
        // non-object message does not panic; priced via serialization
        let count = counter.count_message(&json!("just a string"));
        assert!(count > MESSAGE_OVERHEAD);
        assert_eq!(counter.count_messages(&[]), REPLY_OVERHEAD);
    }
}
