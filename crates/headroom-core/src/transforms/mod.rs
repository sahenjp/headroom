//! Specialized content transforms for the headroom compression core.
//!
//! Every transform targets one content shape — JSON arrays, grep output,
//! build/test logs, git diffs, or prose — and rewrites it into a shorter
//! form. Lossy transforms append a CCR marker so the full original can be
//! retrieved from the store by its cache key.
//!
//! The shared scoring primitives used by several transforms (an alphanumeric
//! tokenizer and a small BM25 implementation) live here.

pub mod diff_compressor;
pub mod log_compressor;
pub mod search_compressor;
pub mod smart_crusher;
pub mod text_crusher;
pub mod config_compressor;
pub mod cross_turn_dedup;
pub mod dense_line;
pub mod error_detection;
pub mod html;
pub mod lossless;
pub mod read_lifecycle;
pub mod spreadsheet;
pub mod tabular;

pub use diff_compressor::{DiffCompressionResult, DiffCompressor, DiffCompressorConfig};
pub use log_compressor::{LogCompressionResult, LogCompressor, LogCompressorConfig, LogFormat};
pub use search_compressor::{SearchCompressionResult, SearchCompressor, SearchCompressorConfig};
pub use smart_crusher::{CrushResult, SmartCrusher, SmartCrusherConfig};
pub use text_crusher::{TextCrusher, TextCrusherConfig, TextCrusherResult};
pub use config_compressor::{ConfigCompressor, ConfigResult};
pub use cross_turn_dedup::dedupe_messages;
pub use dense_line::{DenseLineConfig, DenseResult, elide_dense_lines};
pub use error_detection::{KeywordRegistry, is_error_line, score_line};
pub use html::{HtmlCompressor, HtmlResult, extract_main_content};
pub use lossless::{collapse_log_runs, fold_grep_heading, minify_json, strip_ansi};
pub use read_lifecycle::{ReadDecision, ReadLifecycleConfig, ReadState, stale_marker, superseded_marker};
pub use spreadsheet::{render_sheet, xlsx_to_csv_messages};
pub use tabular::{TabularCompressor, TabularResult, detect_delimiter, is_tabular};

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use regex::Regex;

/// BM25 term-frequency saturation parameter (standard default).
const BM25_K1: f64 = 1.2;
/// BM25 length-normalization parameter (standard default).
const BM25_B: f64 = 0.75;

/// Lowercase needles that mark text as error-ish.
const ERRORISH_NEEDLES: &[&str] = &["error", "fail", "exception", "panic", "todo", "fixme"];

/// Tokenizes `text` into lowercase alphanumeric words (`[a-zA-Z0-9_]+`).
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    static WORD: OnceLock<Regex> = OnceLock::new();
    let word = WORD.get_or_init(|| Regex::new(r"[a-zA-Z0-9_]+").expect("static regex is valid"));
    word.find_iter(text)
        .map(|m| m.as_str().to_lowercase())
        .collect()
}

/// Returns `true` when `haystack` contains any of `needles`, ignoring case.
///
/// All `needles` must already be lowercase.
pub(crate) fn contains_any_ci(haystack: &str, needles: &[&str]) -> bool {
    let lower = haystack.to_lowercase();
    needles.iter().any(|needle| lower.contains(needle))
}

/// Returns `true` when the text mentions errors, failures, panics or TODO markers.
pub(crate) fn is_errorish(text: &str) -> bool {
    contains_any_ci(text, ERRORISH_NEEDLES)
}

/// Renders a byte count as a compact human-readable size (`"4.5KB"`).
pub(crate) fn human_size(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    let size = bytes as f64;
    if size >= MB {
        format!("{:.1}MB", size / MB)
    } else if size >= KB {
        format!("{:.1}KB", size / KB)
    } else {
        format!("{bytes}B")
    }
}

/// Rough token estimate: about one token per four characters.
pub(crate) fn estimate_tokens(chars: usize) -> usize {
    chars / 4
}

/// Joins `lines` with `\n`, restoring a trailing newline when `original` had one.
pub(crate) fn join_lines(lines: &[String], original: &str) -> String {
    let mut joined = lines.join("\n");
    if !lines.is_empty() && original.ends_with('\n') {
        joined.push('\n');
    }
    joined
}

/// A minimal BM25 index over pre-tokenized documents.
///
/// `idf = ln(1 + (N - df + 0.5) / (df + 0.5))` with standard `k1`/`b`
/// saturation and length normalization. Query terms are de-duplicated.
pub(crate) struct Bm25Index {
    docs: Vec<Vec<String>>,
    df: HashMap<String, usize>,
    avgdl: f64,
}

impl Bm25Index {
    /// Builds an index over `docs`; each doc is a token vector.
    pub(crate) fn new(docs: Vec<Vec<String>>) -> Self {
        let n = docs.len();
        let mut df: HashMap<String, usize> = HashMap::new();
        for doc in &docs {
            let mut seen: HashSet<&str> = HashSet::new();
            for token in doc {
                if seen.insert(token.as_str()) {
                    *df.entry(token.clone()).or_insert(0) += 1;
                }
            }
        }
        let total_tokens: usize = docs.iter().map(|d| d.len()).sum();
        let avgdl = if n == 0 {
            0.0
        } else {
            total_tokens as f64 / n as f64
        };
        Self { docs, df, avgdl }
    }

    /// BM25 score of document `doc_index` against the tokenized `query`.
    pub(crate) fn score(&self, doc_index: usize, query: &[String]) -> f64 {
        let Some(doc) = self.docs.get(doc_index) else {
            return 0.0;
        };
        if query.is_empty() || doc.is_empty() {
            return 0.0;
        }
        let n = self.docs.len() as f64;
        let dl = doc.len() as f64;
        let mut tf: HashMap<&str, usize> = HashMap::new();
        for token in doc {
            *tf.entry(token.as_str()).or_insert(0) += 1;
        }
        let mut seen_query: HashSet<&str> = HashSet::new();
        let mut total = 0.0;
        for term in query {
            if !seen_query.insert(term.as_str()) {
                continue;
            }
            let Some(&df) = self.df.get(term.as_str()) else {
                continue;
            };
            let tf = *tf.get(term.as_str()).unwrap_or(&0) as f64;
            if tf == 0.0 {
                continue;
            }
            let idf = (1.0 + (n - df as f64 + 0.5) / (df as f64 + 0.5)).ln();
            let denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / self.avgdl);
            total += idf * (tf * (BM25_K1 + 1.0)) / denom;
        }
        total
    }

    /// Scores every document, max-normalized into `[0, 1]`.
    ///
    /// When no document matches the query, all scores are `0.0`.
    pub(crate) fn normalized_scores(&self, query: &[String]) -> Vec<f64> {
        let raw: Vec<f64> = (0..self.docs.len()).map(|i| self.score(i, query)).collect();
        let max = raw.iter().copied().fold(0.0_f64, f64::max);
        if max > 0.0 {
            raw.iter().map(|s| s / max).collect()
        } else {
            raw
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_splits_alphanumeric_words() {
        assert_eq!(
            tokenize("Hello, World_42! done"),
            vec!["hello", "world_42", "done"]
        );
        assert!(tokenize("").is_empty());
        assert_eq!(tokenize("日本語 mix 123"), vec!["mix", "123"]);
    }

    #[test]
    fn contains_any_ci_matches_case_insensitively() {
        assert!(contains_any_ci("FATAL Error seen", &["error"]));
        assert!(!contains_any_ci("all good here", &["error"]));
    }

    #[test]
    fn is_errorish_detects_common_markers() {
        assert!(is_errorish("TODO fix later"));
        assert!(is_errorish("Panic: bad state"));
        assert!(!is_errorish("nothing to see"));
    }

    #[test]
    fn human_size_formats_compactly() {
        assert_eq!(human_size(500), "500B");
        assert_eq!(human_size(4608), "4.5KB");
        assert_eq!(human_size(1_048_576), "1.0MB");
        assert_eq!(human_size(2_097_152), "2.0MB");
    }

    #[test]
    fn estimate_tokens_uses_quarter_of_chars() {
        assert_eq!(estimate_tokens(0), 0);
        assert_eq!(estimate_tokens(7), 1);
        assert_eq!(estimate_tokens(8), 2);
    }

    #[test]
    fn join_lines_restores_trailing_newline() {
        assert_eq!(join_lines(&["a".into(), "b".into()], "x\n"), "a\nb\n");
        assert_eq!(join_lines(&["a".into(), "b".into()], "x"), "a\nb");
        assert_eq!(join_lines(&[], "x\n"), "");
    }

    #[test]
    fn bm25_ranks_matching_docs_higher() {
        let docs = vec![
            tokenize("error raised in handler"),
            tokenize("plain text only here"),
            tokenize("error raised again later"),
        ];
        let index = Bm25Index::new(docs);
        let query = tokenize("error");
        let scores = index.normalized_scores(&query);
        assert!(scores[0] > 0.0);
        assert_eq!(scores[1], 0.0);
        assert!((scores[0] - 1.0).abs() < 1e-12);
        assert!((scores[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bm25_is_zero_without_matches() {
        let docs = vec![tokenize("alpha beta"), tokenize("gamma delta")];
        let index = Bm25Index::new(docs);
        assert!(index
            .normalized_scores(&tokenize("missing"))
            .iter()
            .all(|&s| s == 0.0));
    }
}
