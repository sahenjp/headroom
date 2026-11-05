//! CCR (Compress-Cache-Retrieve) markers: hashing, grammar, and detection.
//!
//! CCR is headroom's escape hatch for lossy compression: when a transform
//! drops content it cannot reconstruct, the original is stored in a side cache
//! and a small *marker* is left in the compressed text. A marker is either a
//! bare pointer ('<<ccr:HASH>>'), a row-drop summary
//! ('<<ccr:HASH N_rows_offloaded>>'), an opaque-blob offload
//! ('<<ccr:HASH,KIND,SIZE>>'), or a human-readable bracket phrase.
//!
//! The literal phrases 'Retrieve more: hash=' and 'Retrieve original: hash='
//! are **load-bearing**: retrieval tooling matches those exact phrases by
//! regex, so they must never be reworded.
//!
//! Hash discipline mirrors the Python store (cache/compression_store.py):
//! SHA-256 hex truncated to 24 chars ([`ccr_hash`]); the Rust SmartCrusher
//! row-drop path uses the shorter 12-char form ([`ccr_hash12`]).

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

/// Key of the dict-array drop sentinel object SmartCrusher appends to a
/// compressed array of objects:
/// '{"_ccr_dropped": "<<ccr:HASH N_rows_offloaded>>"}'.
pub const CCR_SENTINEL_KEY: &str = "_ccr_dropped";

/// Prefix of the scalar/mixed-array drop sentinel string:
/// '"\u{2026} N more items <<ccr:HASH N_items_offloaded>>"' (U+2026 + space).
pub const SCALAR_SENTINEL_PREFIX: &str = "\u{2026} ";

/// Infix of the scalar/mixed-array drop sentinel string.
pub const SCALAR_SENTINEL_INFIX: &str = " more items ";

/// Line-oriented marker detector: matches every line that carries any CCR
/// marker shape (a retrieval phrase or a '<<ccr:...>>' pointer).
///
/// Mirrors transforms/compression_units._CCR_MARKER_RE.
pub static CCR_MARKER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^.*(?:Retrieve more: hash=|Retrieve original: hash=|<<ccr:[^>]+>>).*$")
        .expect("CCR_MARKER_RE must compile")
});

/// Marker alternation without line anchoring; matches anywhere in text.
///
/// Mirrors parser.CCR_RETRIEVAL_MARKER_RE.
pub static RETRIEVAL_MARKER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"Retrieve more: hash=|Retrieve original: hash=|<<ccr:[^>]+>>")
        .expect("RETRIEVAL_MARKER_RE must compile")
});

/// Hash extractor: 12-24 lowercase hex chars following '<<ccr:' or 'hash='.
static MARKER_HASH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:<<ccr:|hash=)([a-f0-9]{12,24})").expect("MARKER_HASH_RE must compile")
});

/// SHA-256 of 'original', lowercase hex, truncated to 24 chars.
///
/// The Python compression store keys entries this way: sha256(...)[:24].
pub fn ccr_hash(original: &str) -> String {
    let digest = Sha256::digest(original.as_bytes());
    let hex_str = hex::encode(digest);
    hex_str[..24].to_string()
}

/// SHA-256 of 'original', lowercase hex, truncated to 12 chars.
///
/// The Rust SmartCrusher row-drop path emits this shorter form.
pub fn ccr_hash12(original: &str) -> String {
    let digest = Sha256::digest(original.as_bytes());
    let hex_str = hex::encode(digest);
    hex_str[..12].to_string()
}

/// Bare pointer marker: '<<ccr:HASH>>'.
pub fn pointer_marker(hash: &str) -> String {
    format!("<<ccr:{}>>", hash)
}

/// Row-drop summary marker: '<<ccr:HASH N_rows_offloaded>>'.
pub fn row_drop_marker(hash: &str, rows: usize) -> String {
    format!("<<ccr:{} {}_rows_offloaded>>", hash, rows)
}

/// Opaque-blob offload marker: '<<ccr:HASH,KIND,SIZE>>'.
///
/// The blob itself is only recoverable from the CCR store, never from text.
pub fn blob_marker(hash: &str, kind: &str, size: &str) -> String {
    format!("<<ccr:{},{},{}>>", hash, kind, size)
}

/// Dict-array drop sentinel appended to a compressed array of objects:
/// '{"_ccr_dropped": "<<ccr:HASH N_rows_offloaded>>"}'.
pub fn dict_sentinel(hash: &str, rows: usize) -> Value {
    let mut obj = Map::new();
    obj.insert(
        CCR_SENTINEL_KEY.to_string(),
        Value::String(row_drop_marker(hash, rows)),
    );
    Value::Object(obj)
}

/// Scalar/mixed-array drop sentinel, a plain string so string/number arrays
/// keep a uniform element type:
/// '"\u{2026} N more items <<ccr:HASH N_items_offloaded>>"'.
pub fn scalar_sentinel(more: usize, hash: &str, items: usize) -> String {
    format!(
        "{}{}{}<<ccr:{} {}_items_offloaded>>",
        SCALAR_SENTINEL_PREFIX, more, SCALAR_SENTINEL_INFIX, hash, items
    )
}

/// Items retrieval marker appended to a compressed JSON array:
///
/// '\n[{original_count} items compressed to {compressed_count}.{summary}
/// Retrieve more: hash={hash}. Expires in {ttl_minutes}m.]'
///
/// Mirrors the marker_template default in the Python config. The phrase
/// 'Retrieve more: hash=' is load-bearing.
pub fn items_marker(
    original_count: usize,
    compressed_count: usize,
    summary: &str,
    hash: &str,
    ttl_minutes: u64,
) -> String {
    format!(
        "\n[{} items compressed to {}.{} Retrieve more: hash={}. Expires in {}m.]",
        original_count, compressed_count, summary, hash, ttl_minutes
    )
}

/// Word-level retrieval marker appended after a lossy Kompress pass.
///
/// Mirrors transforms/kompress_compressor.ccr_retrieval_marker: the counts are
/// word counts, the source line span is reported so a reader can tell content
/// was compressed away rather than absent, and 'source line' is singular for
/// a one-line source.
pub fn kompress_words_marker(
    n_words: usize,
    compressed_count: usize,
    source_lines: usize,
    hash: &str,
) -> String {
    let line_word = if source_lines == 1 { "line" } else { "lines" };
    format!(
        "\n[{} words compressed to {} (from {} source {}). Original content preserved. Retrieve more: hash={}]",
        n_words, compressed_count, source_lines, line_word, hash
    )
}

/// Generic elision marker:
/// '\n[{count} {description} elided. Retrieve original: hash={hash}]'.
///
/// 'description' is the plural noun phrase (e.g. 'dense machine-generated
/// lines', 'comment/blank lines'). The phrase 'Retrieve original: hash=' is
/// load-bearing.
pub fn elided_marker(count: usize, description: &str, hash: &str) -> String {
    format!(
        "\n[{} {} elided. Retrieve original: hash={}]",
        count, description, hash
    )
}

/// True if 'text' carries any CCR marker (retrieval phrase or '<<ccr:...>>').
pub fn contains_ccr_marker(text: &str) -> bool {
    RETRIEVAL_MARKER_RE.is_match(text)
}

/// Extract every 12-24 lowercase-hex hash that follows '<<ccr:' or 'hash='.
///
/// Occurrences are returned in order, duplicates included. The retrieval
/// phrases carry 24-hex hashes; '<<ccr:...>>' pointers carry 12-24 hex. A blob
/// hash shorter than 12 hex chars (e.g. '<<ccr:abc,base64,4.5KB>>') is not a
/// store key and is not extracted.
pub fn find_marker_hashes(text: &str) -> Vec<String> {
    MARKER_HASH_RE
        .captures_iter(text)
        .map(|caps| caps[1].to_string())
        .collect()
}

/// True if 'item' is a CCR-dropped sentinel.
///
/// Covers both sentinel shapes: the dict object emitted for dict arrays
/// (contains [CCR_SENTINEL_KEY]) and the plain-string sentinel appended to
/// string/number/mixed arrays.
pub fn is_ccr_sentinel(item: &Value) -> bool {
    match item {
        Value::Object(obj) => obj.contains_key(CCR_SENTINEL_KEY),
        Value::String(s) => {
            s.starts_with(SCALAR_SENTINEL_PREFIX)
                && s.contains(SCALAR_SENTINEL_INFIX)
                && s.contains("<<ccr:")
        }
        _ => false,
    }
}

/// Return 'items' with any CCR-dropped sentinels filtered out.
///
/// Non-array inputs pass through unchanged so callers can hand over whatever
/// serde_json produced without first checking the shape.
pub fn strip_ccr_sentinels(items: &Value) -> Value {
    match items {
        Value::Array(arr) => Value::Array(
            arr.iter()
                .filter(|item| !is_ccr_sentinel(item))
                .cloned()
                .collect(),
        ),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const H24: &str = "2cf24dba5fb0a30e26e83b2a";
    const H12: &str = "2cf24dba5fb0";

    #[test]
    fn hashes_are_truncated_sha256() {
        // Known vector: sha256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e...
        assert_eq!(ccr_hash("hello"), "2cf24dba5fb0a30e26e83b2a");
        assert_eq!(ccr_hash12("hello"), "2cf24dba5fb0");
        assert_eq!(ccr_hash12("hello"), &ccr_hash("hello")[..12]);
        assert_eq!(ccr_hash("hello").len(), 24);
        assert_eq!(ccr_hash12("hello").len(), 12);
        // lowercase hex only
        let h = ccr_hash("world");
        assert!(h
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        // deterministic and content-sensitive
        assert_eq!(ccr_hash("a"), ccr_hash("a"));
        assert_ne!(ccr_hash("a"), ccr_hash("b"));
    }

    #[test]
    fn pointer_marker_form_round_trips() {
        let m = pointer_marker(H12);
        assert_eq!(m, "<<ccr:2cf24dba5fb0>>");
        assert!(contains_ccr_marker(&m));
        assert!(RETRIEVAL_MARKER_RE.is_match(&m));
        assert_eq!(find_marker_hashes(&m), vec![H12.to_string()]);
    }

    #[test]
    fn row_drop_marker_form_round_trips() {
        let m = row_drop_marker("abc123def456", 100);
        assert_eq!(m, "<<ccr:abc123def456 100_rows_offloaded>>");
        assert!(contains_ccr_marker(&m));
        assert!(RETRIEVAL_MARKER_RE.is_match(&m));
        assert_eq!(find_marker_hashes(&m), vec!["abc123def456".to_string()]);
    }

    #[test]
    fn blob_marker_form_round_trips() {
        let m = blob_marker(H12, "base64", "4.5KB");
        assert_eq!(m, "<<ccr:2cf24dba5fb0,base64,4.5KB>>");
        assert!(contains_ccr_marker(&m));
        assert!(RETRIEVAL_MARKER_RE.is_match(&m));
        // a 12-hex blob hash is a store key and is extracted
        assert_eq!(find_marker_hashes(&m), vec![H12.to_string()]);
        // a sub-12-hex blob hash matches the marker regex but is no store key
        let short = blob_marker("abc", "base64", "4.5KB");
        assert!(contains_ccr_marker(&short));
        assert!(find_marker_hashes(&short).is_empty());
    }

    #[test]
    fn items_marker_form_round_trips() {
        let m = items_marker(250, 10, "", H24, 30);
        assert_eq!(
            m,
            "\n[250 items compressed to 10. Retrieve more: hash=2cf24dba5fb0a30e26e83b2a. Expires in 30m.]"
        );
        let with_summary = items_marker(100, 20, " Top sellers kept.", H24, 45);
        assert!(with_summary.contains("20. Top sellers kept. Retrieve more: hash="));
        for text in [&m, &with_summary] {
            assert!(contains_ccr_marker(text));
            assert!(RETRIEVAL_MARKER_RE.is_match(text));
            assert_eq!(find_marker_hashes(text), vec![H24.to_string()]);
        }
    }

    #[test]
    fn kompress_words_marker_form_round_trips() {
        let m = kompress_words_marker(120, 30, 3, H24);
        assert_eq!(
            m,
            "\n[120 words compressed to 30 (from 3 source lines). Original content preserved. Retrieve more: hash=2cf24dba5fb0a30e26e83b2a]"
        );
        let one_line = kompress_words_marker(12, 4, 1, H24);
        assert!(one_line.contains("from 1 source line)."));
        for text in [&m, &one_line] {
            assert!(contains_ccr_marker(text));
            assert_eq!(find_marker_hashes(text), vec![H24.to_string()]);
        }
    }

    #[test]
    fn elided_marker_form_round_trips() {
        let m = elided_marker(4, "dense machine-generated lines", H24);
        assert_eq!(
            m,
            "\n[4 dense machine-generated lines elided. Retrieve original: hash=2cf24dba5fb0a30e26e83b2a]"
        );
        assert!(contains_ccr_marker(&m));
        assert!(RETRIEVAL_MARKER_RE.is_match(&m));
        assert_eq!(find_marker_hashes(&m), vec![H24.to_string()]);
    }

    #[test]
    fn line_regex_matches_only_marker_lines() {
        let text = format!(
            "intro line\n{}\ntail line\n{}\nplain tail",
            items_marker(250, 10, "", H24, 30).trim(),
            row_drop_marker(H12, 7)
        );
        let matched: Vec<&str> = CCR_MARKER_RE.find_iter(&text).map(|m| m.as_str()).collect();
        assert_eq!(
            matched,
            vec![
                "[250 items compressed to 10. Retrieve more: hash=2cf24dba5fb0a30e26e83b2a. Expires in 30m.]",
                "<<ccr:2cf24dba5fb0 7_rows_offloaded>>",
            ]
        );
    }

    #[test]
    fn finds_every_hash_in_order_with_duplicates() {
        let text = format!(
            "{} and {} then {} plus again {}",
            pointer_marker(H12),
            row_drop_marker(H24, 9),
            elided_marker(2, "lines", H24),
            pointer_marker(H12),
        );
        assert_eq!(
            find_marker_hashes(&text),
            vec![
                H12.to_string(),
                H24.to_string(),
                H24.to_string(),
                H12.to_string()
            ]
        );
    }

    #[test]
    fn plain_text_has_no_markers() {
        let text = "Just prose about hashes, e.g. hash= and ccr, but no real marker.";
        assert!(!contains_ccr_marker(text));
        assert!(find_marker_hashes(text).is_empty());
        assert!(!CCR_MARKER_RE.is_match(text));
    }

    #[test]
    fn phrase_without_valid_hash_still_matches_phrase_regex() {
        // The load-bearing phrase matches even when the hash is not hex;
        // only valid 12-24 hex hashes are extracted.
        let text = "Retrieve more: hash=not-hex";
        assert!(contains_ccr_marker(text));
        assert!(find_marker_hashes(text).is_empty());
    }

    #[test]
    fn hash_length_bounds_are_12_to_24() {
        let short = "hash=abc123def45"; // 11 hex: too short
        assert!(find_marker_hashes(short).is_empty());
        let exact12 = "hash=abc123def456";
        assert_eq!(
            find_marker_hashes(exact12),
            vec!["abc123def456".to_string()]
        );
        let exact24 = format!("hash={}", "a".repeat(24));
        assert_eq!(find_marker_hashes(&exact24), vec!["a".repeat(24)]);
        // a longer hex run is captured greedily up to 24 chars
        let long = format!("hash={}", "a".repeat(30));
        assert_eq!(find_marker_hashes(&long), vec!["a".repeat(24)]);
    }

    #[test]
    fn dict_sentinel_is_detected_and_stripped() {
        let sentinel = dict_sentinel("abc123def456", 100);
        assert_eq!(
            sentinel,
            json!({"_ccr_dropped": "<<ccr:abc123def456 100_rows_offloaded>>"})
        );
        assert!(is_ccr_sentinel(&sentinel));
        // any object carrying the key is a sentinel, mirroring the Python check
        assert!(is_ccr_sentinel(&json!({"_ccr_dropped": 1})));
        assert!(!is_ccr_sentinel(&json!({"id": 1})));
        let items = json!([{"id": 1}, sentinel, {"id": 2}]);
        assert_eq!(strip_ccr_sentinels(&items), json!([{"id": 1}, {"id": 2}]));
    }

    #[test]
    fn scalar_sentinel_is_detected_and_stripped() {
        let s = scalar_sentinel(3, "abc123def456", 3);
        assert_eq!(
            s,
            "\u{2026} 3 more items <<ccr:abc123def456 3_items_offloaded>>"
        );
        assert!(is_ccr_sentinel(&Value::String(s)));
        // near-misses are not sentinels
        assert!(!is_ccr_sentinel(&Value::String(
            "\u{2026} 3 more items".to_string()
        )));
        assert!(!is_ccr_sentinel(&Value::String(
            "\u{2026} plain text with <<ccr:abc>> but no infix".to_string()
        )));
        assert!(!is_ccr_sentinel(&Value::String(
            "3 more items <<ccr:abc>>".to_string()
        )));
        let items = json!(["a", "b", Value::String(scalar_sentinel(1, H12, 1))]);
        assert_eq!(strip_ccr_sentinels(&items), json!(["a", "b"]));
    }

    #[test]
    fn strip_passes_non_arrays_through() {
        let obj = json!({"a": 1});
        assert_eq!(strip_ccr_sentinels(&obj), obj);
        assert_eq!(strip_ccr_sentinels(&json!(null)), json!(null));
        // duplicate sentinels are all removed
        let s = dict_sentinel(H12, 1);
        let items = json!([s.clone(), {"x": 1}, s]);
        assert_eq!(strip_ccr_sentinels(&items), json!([{"x": 1}]));
    }
}
