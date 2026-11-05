//! Deterministic extractive prose compressor.
//!
//! [`TextCrusher::compress`] splits text into sentences, scores each segment
//! by context relevance (BM25), recency and salience, then keeps segments in
//! score order until the requested character budget is reached. Kept text is
//! emitted verbatim — nothing is ever paraphrased — and near-duplicate
//! segments (word 4-shingle Jaccard) are suppressed so repeated phrasing does
//! not consume the budget.

use std::cmp::Ordering;
use std::collections::HashSet;

use super::{estimate_tokens, tokenize, Bm25Index};

/// Cue words that raise a segment's salience score.
const CUE_WORDS: &[&str] = &[
    "error",
    "fail",
    "todo",
    "note",
    "important",
    "warning",
    "bug",
    "summary",
];

/// Number of characters at which the salience length factor saturates.
const SALIENCE_LENGTH_BASIS: f64 = 200.0;

/// Configuration for [`TextCrusher`].
#[derive(Debug, Clone, PartialEq)]
pub struct TextCrusherConfig {
    /// Fraction of the original characters to keep (0.0 - 1.0).
    pub target_ratio: f64,
    /// Weight of segment recency (later segments score higher).
    pub w_recency: f64,
    /// Weight of BM25 relevance against the conversation context.
    pub w_relevance: f64,
    /// Weight of cue-word salience.
    pub w_salience: f64,
    /// Segments shorter than this are merged into the previous one.
    pub min_segment_chars: usize,
    /// Word 4-shingle Jaccard similarity at which two segments are near-duplicates.
    pub near_dup_threshold: f64,
    /// Inputs with fewer segments are returned unchanged.
    pub min_segments_for_crush: usize,
}

impl Default for TextCrusherConfig {
    fn default() -> Self {
        Self {
            target_ratio: 0.5,
            w_recency: 1.0,
            w_relevance: 2.0,
            w_salience: 1.5,
            min_segment_chars: 12,
            near_dup_threshold: 0.85,
            min_segments_for_crush: 6,
        }
    }
}

/// Deterministic extractive prose compressor.
#[derive(Debug, Clone, Default)]
pub struct TextCrusher {
    /// Tuning knobs.
    pub config: TextCrusherConfig,
}

/// Result of a [`TextCrusher`] run.
#[derive(Debug, Clone, PartialEq)]
pub struct TextCrusherResult {
    /// Compressed text: kept segments trimmed and joined with `\n`.
    pub compressed: String,
    /// Token estimate (~chars/4) of the input.
    pub original_tokens: usize,
    /// Token estimate (~chars/4) of the output.
    pub compressed_tokens: usize,
    /// `compressed_tokens / original_tokens` (1.0 when nothing was crushed).
    pub compression_ratio: f64,
    /// Number of segments kept.
    pub kept_segments: usize,
    /// Number of segments the input split into.
    pub total_segments: usize,
}

impl TextCrusher {
    /// Creates a crusher with the given configuration.
    pub fn new(config: TextCrusherConfig) -> Self {
        Self { config }
    }

    /// Crushes `content`, keeping roughly `target_ratio` (or the configured
    /// default) of its characters.
    ///
    /// The input is returned unchanged when it splits into fewer than
    /// [`TextCrusherConfig::min_segments_for_crush`] segments or when every
    /// segment survives selection.
    pub fn compress(
        &self,
        content: &str,
        context: &str,
        target_ratio: Option<f64>,
    ) -> TextCrusherResult {
        let segments = merge_short(split_sentences(content), self.config.min_segment_chars);
        let total_segments = segments.len();
        let original_tokens = estimate_tokens(content.chars().count());
        if total_segments < self.config.min_segments_for_crush {
            return TextCrusherResult {
                compressed: content.to_string(),
                original_tokens,
                compressed_tokens: original_tokens,
                compression_ratio: 1.0,
                kept_segments: total_segments,
                total_segments,
            };
        }

        let trimmed: Vec<&str> = segments.iter().map(|s| s.trim()).collect();
        let lengths: Vec<usize> = trimmed.iter().map(|s| s.chars().count()).collect();
        let total_chars: usize = lengths.iter().sum();

        let query = tokenize(context);
        let docs: Vec<Vec<String>> = trimmed.iter().map(|s| tokenize(s)).collect();
        let relevance = Bm25Index::new(docs.clone()).normalized_scores(&query);

        let mut scores = vec![0.0; total_segments];
        for i in 0..total_segments {
            let recency = if total_segments > 1 {
                i as f64 / (total_segments - 1) as f64
            } else {
                1.0
            };
            let salience = salience_of(trimmed[i], lengths[i]);
            scores[i] = self.config.w_relevance * relevance[i]
                + self.config.w_recency * recency
                + self.config.w_salience * salience;
        }

        // Score order: highest score first, ties broken by original position.
        let mut order: Vec<usize> = (0..total_segments).collect();
        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(Ordering::Equal)
                .then(a.cmp(&b))
        });

        let target = target_ratio
            .unwrap_or(self.config.target_ratio)
            .clamp(0.0, 1.0);
        let target_chars = target * total_chars as f64;
        let shingles: Vec<HashSet<String>> = docs.iter().map(|d| word_shingles(d)).collect();

        let mut kept: Vec<usize> = Vec::new();
        let mut kept_chars = 0usize;
        for &index in &order {
            if kept_chars as f64 >= target_chars {
                break;
            }
            let near_dup = kept.iter().any(|&k| {
                shingle_jaccard(&shingles[index], &shingles[k], &docs[index], &docs[k])
                    >= self.config.near_dup_threshold
            });
            if near_dup {
                continue;
            }
            kept.push(index);
            kept_chars += lengths[index];
        }

        if kept.len() == total_segments {
            return TextCrusherResult {
                compressed: content.to_string(),
                original_tokens,
                compressed_tokens: original_tokens,
                compression_ratio: 1.0,
                kept_segments: total_segments,
                total_segments,
            };
        }

        kept.sort_unstable();
        let compressed = kept
            .iter()
            .map(|&i| trimmed[i])
            .collect::<Vec<_>>()
            .join("\n");
        let compressed_tokens = estimate_tokens(compressed.chars().count());
        let compression_ratio = if original_tokens > 0 {
            compressed_tokens as f64 / original_tokens as f64
        } else {
            1.0
        };
        TextCrusherResult {
            compressed,
            original_tokens,
            compressed_tokens,
            compression_ratio,
            kept_segments: kept.len(),
            total_segments,
        }
    }
}

/// Splits text into sentence segments, keeping the trailing punctuation.
fn split_sentences(content: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    for ch in content.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?' | '。' | '！' | '？') {
            segments.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        segments.push(current);
    }
    segments
}

/// Merges segments shorter than `min_chars` into their predecessor.
///
/// A short leading segment is held back and merged with the next one; a short
/// trailing segment is appended to the last merged segment.
fn merge_short(segments: Vec<String>, min_chars: usize) -> Vec<String> {
    let mut merged: Vec<String> = Vec::new();
    let mut pending = String::new();
    for segment in segments {
        if pending.is_empty() {
            if segment.trim().chars().count() < min_chars {
                pending = segment;
            } else {
                merged.push(segment);
            }
        } else {
            pending.push_str(&segment);
            if pending.trim().chars().count() >= min_chars {
                merged.push(std::mem::take(&mut pending));
            }
        }
    }
    if !pending.is_empty() {
        match merged.last_mut() {
            Some(last) => last.push_str(&pending),
            None => merged.push(pending),
        }
    }
    merged
}

/// Cue-word, digit and length salience of a segment, normalized to `[0, 1]`.
fn salience_of(text: &str, char_len: usize) -> f64 {
    let lower = text.to_lowercase();
    let cues = CUE_WORDS
        .iter()
        .filter(|cue| lower.contains(*cue))
        .count()
        .min(4) as f64;
    let digits = if text.chars().any(|c| c.is_ascii_digit()) || text.contains('%') {
        1.0
    } else {
        0.0
    };
    let length = (char_len as f64 / SALIENCE_LENGTH_BASIS).min(1.0);
    (cues / 4.0) * 0.5 + digits * 0.25 + length * 0.25
}

/// Word 4-shingles of a token list (empty when fewer than four words).
fn word_shingles(tokens: &[String]) -> HashSet<String> {
    if tokens.len() < 4 {
        return HashSet::new();
    }
    tokens.windows(4).map(|w| w.join(" ")).collect()
}

/// Jaccard similarity of two shingle sets.
///
/// Segments with fewer than four words have no shingles; two such segments
/// compare as near-duplicates only when their word lists are identical.
fn shingle_jaccard(
    a: &HashSet<String>,
    b: &HashSet<String>,
    tokens_a: &[String],
    tokens_b: &[String],
) -> f64 {
    if a.is_empty() && b.is_empty() {
        return if tokens_a == tokens_b { 1.0 } else { 0.0 };
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    intersection / union
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crusher() -> TextCrusher {
        TextCrusher::default()
    }

    #[test]
    fn crushes_to_about_the_target_ratio() {
        let content: Vec<String> = (0..12)
            .map(|i| {
                format!(
                    "Segment number {i} talks about topic {i} with plenty of unique filler words."
                )
            })
            .collect();
        let content = content.join(" ");
        let result = crusher().compress(&content, "", Some(0.5));
        assert!(result.kept_segments >= 5, "kept too few: {:?}", result);
        assert!(result.kept_segments <= 8, "kept too many: {:?}", result);
        assert!(result.kept_segments < result.total_segments);
        assert_eq!(result.total_segments, 12);
        assert!(result.compression_ratio > 0.3 && result.compression_ratio < 0.85);
        assert_ne!(result.compressed, content);
    }

    #[test]
    fn kept_text_is_verbatim() {
        let content: Vec<String> = (0..10)
            .map(|i| format!("Sentence {i} carries unique payload words number {i} here."))
            .collect();
        let content = content.join(" ");
        let result = crusher().compress(&content, "", Some(0.5));
        for line in result.compressed.lines() {
            assert!(
                content.contains(line),
                "kept line is not a substring of the input: {line:?}"
            );
        }
    }

    #[test]
    fn near_duplicate_segments_are_suppressed() {
        let a =
            "The quick brown fox jumps over the lazy dog near the river bank every single morning.";
        let a_prime =
            "The quick brown fox jumps over the lazy dog near the river bank every single evening.";
        let mut segments = vec![a.to_string(), a_prime.to_string()];
        for i in 0..8 {
            segments.push(format!(
                "Completely unrelated sentence {i} describing entirely different subject matter."
            ));
        }
        let content = segments.join(" ");
        let result = crusher().compress(&content, "", Some(0.99));
        assert_eq!(result.total_segments, 10);
        assert_eq!(result.kept_segments, 9, "one of the dup pair must drop");
        assert!(result.compressed.contains("evening"));
        assert!(!result.compressed.contains("morning"));
    }

    #[test]
    fn context_relevance_prioritizes_matching_segments() {
        let mut segments: Vec<String> = Vec::new();
        for i in 0..10 {
            let seg = if i == 3 {
                "The database connection pool exhausted all available connections again today."
            } else if i == 7 {
                "A database connection pool leak was detected during the nightly check."
            } else {
                "Generic filler sentence about unrelated everyday topics and plain weather."
            };
            segments.push(seg.to_string());
        }
        let content = segments.join(" ");
        let result = crusher().compress(
            &content,
            "database connection pool exhausted leak",
            Some(0.3),
        );
        assert!(result.compressed.contains("exhausted all available"));
        assert!(result.compressed.contains("leak was detected"));
        assert!(result.kept_segments < result.total_segments);
    }

    #[test]
    fn passthrough_for_short_inputs() {
        let content = "One sentence here. Another one follows. Third one is last.";
        let result = crusher().compress(content, "", Some(0.5));
        assert_eq!(result.compressed, content);
        assert_eq!(result.compression_ratio, 1.0);
        assert_eq!(result.kept_segments, 3);
        assert_eq!(result.total_segments, 3);
    }

    #[test]
    fn short_segments_merge_with_previous() {
        let content = "Hi there. Ok. This is a much longer sentence with details inside.";
        let result = crusher().compress(content, "", Some(0.5));
        assert_eq!(result.total_segments, 2, "Hi there. + Ok. must merge");
        assert_eq!(result.compressed, content);
    }

    #[test]
    fn deterministic_for_identical_calls() {
        let content: Vec<String> = (0..12)
            .map(|i| format!("Sentence {i} has a unique set of words about subject {i} today."))
            .collect();
        let content = content.join(" ");
        let first = crusher().compress(&content, "subject words", Some(0.5));
        let second = crusher().compress(&content, "subject words", Some(0.5));
        assert_eq!(first, second);
    }
}
