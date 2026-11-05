//! Compressor for grep/ripgrep-style search output.
//!
//! [`SearchCompressor::compress`] parses `file:line:content` match lines
//! (plus the `file-line-content` context, `--` separator and `N-M` range
//! shapes it recognizes), groups matches by file, and keeps the first/last
//! match per file plus the highest-scored middle matches. Scores combine
//! context-keyword hits, error-ish substrings and BM25 relevance against the
//! conversation context. When matches are dropped, a CCR marker lets the
//! caller retrieve the full original output.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::OnceLock;

use regex::Regex;

use crate::ccr;

use super::{is_errorish, join_lines, tokenize, Bm25Index};

/// Weight applied to the normalized BM25 relevance before the bias multiplier.
const RELEVANCE_WEIGHT: f64 = 10.0;
/// Bonus per matched context keyword.
const KEYWORD_BONUS: f64 = 10.0;
/// Bonus for error-ish match content when `boost_errors` is on.
const ERROR_BONUS: f64 = 5.0;

/// Configuration for [`SearchCompressor`].
#[derive(Debug, Clone, PartialEq)]
pub struct SearchCompressorConfig {
    /// Maximum matches kept per file.
    pub max_matches_per_file: usize,
    /// Always keep the first match of each file.
    pub always_keep_first: bool,
    /// Always keep the last match of each file.
    pub always_keep_last: bool,
    /// Maximum matches kept across all files.
    pub max_total_matches: usize,
    /// Maximum number of files kept.
    pub max_files: usize,
    /// Keywords (checked case-insensitively) that boost a match's score.
    pub context_keywords: Vec<String>,
    /// Boost matches whose content mentions errors/failures/TODOs.
    pub boost_errors: bool,
    /// Append a CCR retrieval marker when the output shrank enough.
    pub enable_ccr: bool,
    /// Minimum total matches before a CCR marker is considered.
    pub min_matches_for_ccr: usize,
    /// Maximum kept/original match ratio for a CCR marker
    /// (kept/original <= this).
    pub min_compression_ratio_for_ccr: f64,
    /// Emit the file path once, then indented `line:content` rows.
    pub group_by_file: bool,
}

impl Default for SearchCompressorConfig {
    fn default() -> Self {
        Self {
            max_matches_per_file: 5,
            always_keep_first: true,
            always_keep_last: true,
            max_total_matches: 30,
            max_files: 15,
            context_keywords: Vec::new(),
            boost_errors: true,
            enable_ccr: true,
            min_matches_for_ccr: 10,
            min_compression_ratio_for_ccr: 0.8,
            group_by_file: false,
        }
    }
}

/// Compressor for grep/ripgrep-style search output.
#[derive(Debug, Clone, Default)]
pub struct SearchCompressor {
    /// Tuning knobs.
    pub config: SearchCompressorConfig,
}

/// Result of a [`SearchCompressor::compress`] run.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchCompressionResult {
    /// Compressed output; the original text when nothing was dropped.
    pub compressed: String,
    /// The original input.
    pub original: String,
    /// Number of parsed match lines in the input.
    pub original_match_count: usize,
    /// Number of match lines kept in the output.
    pub compressed_match_count: usize,
    /// Number of files that contained matches.
    pub files_affected: usize,
    /// Kept/original match ratio (1.0 when nothing was dropped).
    pub compression_ratio: f64,
    /// CCR cache key of the original content (`None` when no marker was added).
    pub cache_key: Option<String>,
    /// Per-file truncation notes, e.g. `"(3 of 12 matches shown)"`.
    pub summaries: Vec<String>,
}

/// One parsed `file:line:content` match.
struct ParsedMatch {
    line_no: u64,
    content: String,
    raw: String,
}

fn match_line_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^([^:\s]+):(\d+):(.*)$").expect("static regex is valid"))
}

impl SearchCompressor {
    /// Creates a compressor with the given configuration.
    pub fn new(config: SearchCompressorConfig) -> Self {
        Self { config }
    }

    /// Compresses grep-style `content`.
    ///
    /// `context` supplies BM25 relevance terms and `bias` scales them.
    /// Returns the input unchanged when every match is kept.
    pub fn compress(&self, content: &str, context: &str, bias: f64) -> SearchCompressionResult {
        let lines: Vec<&str> = content.lines().collect();

        // Parse and group matches by file, preserving first-appearance order.
        let mut files: Vec<String> = Vec::new();
        let mut matches: Vec<ParsedMatch> = Vec::new();
        let mut per_file: Vec<Vec<usize>> = Vec::new();
        for line in &lines {
            if let Some(c) = match_line_re().captures(line) {
                let file = c[1].to_string();
                let index = match files.iter().position(|f| f == &file) {
                    Some(i) => i,
                    None => {
                        files.push(file.clone());
                        per_file.push(Vec::new());
                        files.len() - 1
                    }
                };
                per_file[index].push(matches.len());
                matches.push(ParsedMatch {
                    line_no: c[2].parse().unwrap_or(0),
                    content: c[3].to_string(),
                    raw: line.to_string(),
                });
            }
            // Context lines (file-line-content), "--" separators and "N-M"
            // range lines are recognized shapes but never count as matches.
        }
        let total_matches = matches.len();
        let files_affected = files.len();

        let scores = self.match_scores(&matches, context, bias);
        let kept_per_file = self.select_per_file(&per_file, &scores);
        let files_kept = self.config.max_files.min(files.len());
        let final_kept = self.apply_global_caps(&kept_per_file, files_kept);

        let kept_total: usize = final_kept.iter().map(|v| v.len()).sum();
        let ratio = if total_matches > 0 {
            kept_total as f64 / total_matches as f64
        } else {
            1.0
        };

        if kept_total == total_matches && files_kept == files.len() {
            return SearchCompressionResult {
                compressed: content.to_string(),
                original: content.to_string(),
                original_match_count: total_matches,
                compressed_match_count: kept_total,
                files_affected,
                compression_ratio: 1.0,
                cache_key: None,
                summaries: Vec::new(),
            };
        }

        let mut out_lines: Vec<String> = Vec::new();
        let mut summaries: Vec<String> = Vec::new();
        for fi in 0..files_kept {
            let kept = &final_kept[fi];
            if self.config.group_by_file {
                out_lines.push(files[fi].clone());
                for &mi in kept {
                    let m = &matches[mi];
                    out_lines.push(format!("  {}:{}", m.line_no, m.content));
                }
            } else {
                for &mi in kept {
                    out_lines.push(matches[mi].raw.clone());
                }
            }
            if kept.len() < per_file[fi].len() {
                let summary = format!("({} of {} matches shown)", kept.len(), per_file[fi].len());
                out_lines.push(summary.clone());
                summaries.push(summary);
            }
        }

        let mut compressed = join_lines(&out_lines, content);
        let mut cache_key = None;
        if self.config.enable_ccr
            && total_matches >= self.config.min_matches_for_ccr
            && ratio <= self.config.min_compression_ratio_for_ccr
        {
            let hash = ccr::ccr_hash(content);
            compressed.push_str(&format!(
                "\n[{kept_total} of {total_matches} matches shown. Retrieve original: hash={hash}]"
            ));
            cache_key = Some(hash);
        }

        SearchCompressionResult {
            compressed,
            original: content.to_string(),
            original_match_count: total_matches,
            compressed_match_count: kept_total,
            files_affected,
            compression_ratio: ratio,
            cache_key,
            summaries,
        }
    }

    /// Per-match score: keywords + error boost + bias * BM25 relevance.
    fn match_scores(&self, matches: &[ParsedMatch], context: &str, bias: f64) -> Vec<f64> {
        let query = tokenize(context);
        let docs: Vec<Vec<String>> = matches.iter().map(|m| tokenize(&m.content)).collect();
        let relevance = Bm25Index::new(docs).normalized_scores(&query);
        let keywords: Vec<String> = self
            .config
            .context_keywords
            .iter()
            .map(|k| k.to_lowercase())
            .collect();
        matches
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let mut score = 0.0;
                for keyword in &keywords {
                    if m.content.to_lowercase().contains(keyword) {
                        score += KEYWORD_BONUS;
                    }
                }
                if self.config.boost_errors && is_errorish(&m.content) {
                    score += ERROR_BONUS;
                }
                score + bias * RELEVANCE_WEIGHT * relevance[i]
            })
            .collect()
    }

    /// Per-file selection: forced first/last, then best middle matches.
    fn select_per_file(&self, per_file: &[Vec<usize>], scores: &[f64]) -> Vec<Vec<usize>> {
        let cap = self.config.max_matches_per_file;
        per_file
            .iter()
            .map(|file_matches| {
                if file_matches.len() <= cap {
                    return file_matches.clone();
                }
                let mut kept: Vec<usize> = Vec::new();
                if self.config.always_keep_first && cap > 0 {
                    kept.push(file_matches[0]);
                }
                if self.config.always_keep_last && cap > kept.len() {
                    kept.push(*file_matches.last().expect("non-empty"));
                }
                let forced: HashSet<usize> = kept.iter().copied().collect();
                let mut rest: Vec<usize> = file_matches
                    .iter()
                    .copied()
                    .filter(|i| !forced.contains(i))
                    .collect();
                rest.sort_by(|a, b| {
                    scores[*b]
                        .partial_cmp(&scores[*a])
                        .unwrap_or(Ordering::Equal)
                        .then(a.cmp(b))
                });
                kept.extend(rest.into_iter().take(cap - kept.len()));
                kept.sort_unstable();
                kept
            })
            .collect()
    }

    /// Applies the file and total-match caps, keeping the first files in order.
    fn apply_global_caps(
        &self,
        kept_per_file: &[Vec<usize>],
        files_kept: usize,
    ) -> Vec<Vec<usize>> {
        let mut final_kept: Vec<Vec<usize>> = vec![Vec::new(); kept_per_file.len()];
        let mut total = 0usize;
        for fi in 0..files_kept {
            let mut kept: Vec<usize> = Vec::new();
            for &mi in &kept_per_file[fi] {
                if total + kept.len() >= self.config.max_total_matches {
                    break;
                }
                kept.push(mi);
            }
            total += kept.len();
            final_kept[fi] = kept;
        }
        final_kept
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grep_output(files: &[(&str, usize)]) -> String {
        let mut lines = Vec::new();
        for (file, count) in files {
            for line in 1..=*count {
                lines.push(format!("{file}:{line}:match content number {line}"));
            }
        }
        lines.join("\n")
    }

    #[test]
    fn multi_file_output_passes_through_when_nothing_drops() {
        let content = grep_output(&[("src/a.rs", 3), ("src/b.rs", 3)]);
        let compressor = SearchCompressor::default();
        let result = compressor.compress(&content, "", 1.0);
        assert_eq!(result.compressed, content, "nothing dropped");
        assert_eq!(result.original_match_count, 6);
        assert_eq!(result.compressed_match_count, 6);
        assert_eq!(result.files_affected, 2);
        assert_eq!(result.compression_ratio, 1.0);
        assert_eq!(result.cache_key, None);
        assert!(result.summaries.is_empty());
    }

    #[test]
    fn per_file_caps_keep_first_last_and_errors() {
        let mut lines: Vec<String> = (1..=12)
            .map(|i| format!("src/a.rs:{i}:match content number {i}"))
            .collect();
        lines[5] = "src/a.rs:6:match content number 6 with an error here".to_string();
        let content = lines.join("\n");
        let compressor = SearchCompressor::default();
        let result = compressor.compress(&content, "", 1.0);
        assert_eq!(result.compressed_match_count, 5);
        assert!(result.compressed.contains("src/a.rs:1:"));
        assert!(result.compressed.contains("src/a.rs:12:"));
        assert!(result.compressed.contains("src/a.rs:6:"), "error line kept");
        assert!(!result.compressed.contains("src/a.rs:9:"));
        assert!(result.compressed.contains("(5 of 12 matches shown)"));
        assert_eq!(result.summaries, vec!["(5 of 12 matches shown)"]);
        // CCR: 12 >= 10 matches and 5/12 <= 0.8.
        assert!(result
            .compressed
            .contains("[5 of 12 matches shown. Retrieve original: hash="));
        assert_eq!(result.cache_key, Some(ccr::ccr_hash(&content)));
        assert!(result.compression_ratio < 0.42);
    }

    #[test]
    fn no_ccr_marker_when_ratio_above_threshold() {
        let content = grep_output(&[("src/a.rs", 10)]);
        let compressor = SearchCompressor::new(SearchCompressorConfig {
            max_matches_per_file: 9,
            ..Default::default()
        });
        let result = compressor.compress(&content, "", 1.0);
        assert_eq!(result.compressed_match_count, 9);
        assert!(!result.compressed.contains("Retrieve original"));
        assert_eq!(result.cache_key, None);
        assert!((result.compression_ratio - 0.9).abs() < 1e-9);
    }

    #[test]
    fn context_lines_and_separators_are_not_matches() {
        let content = [
            "src/main.rs:10:fn main() {",
            "src/main.rs-9-let x = 1;",
            "--",
            "src/main.rs:11:    println!(\"hi\");",
            "12-15",
            "some unparsed line",
        ]
        .join("\n");
        let compressor = SearchCompressor::default();
        let result = compressor.compress(&content, "", 1.0);
        assert_eq!(
            result.original_match_count, 2,
            "only file:line:content lines"
        );
        assert_eq!(result.files_affected, 1);
        assert_eq!(result.compressed, content, "nothing dropped -> passthrough");
    }

    #[test]
    fn file_cap_drops_later_files() {
        let content = grep_output(&[("a.rs", 4), ("b.rs", 4), ("c.rs", 4)]);
        let compressor = SearchCompressor::new(SearchCompressorConfig {
            max_files: 2,
            ..Default::default()
        });
        let result = compressor.compress(&content, "", 1.0);
        assert_eq!(result.files_affected, 3);
        assert_eq!(result.compressed_match_count, 8);
        assert!(!result.compressed.contains("c.rs:"), "third file dropped");
        assert!(result.cache_key.is_some());
    }

    #[test]
    fn total_cap_limits_matches() {
        let files: Vec<(&str, usize)> = (0..8)
            .map(|i| (Box::leak(format!("f{i}.rs").into_boxed_str()) as &str, 5))
            .collect();
        let content = grep_output(&files);
        let compressor = SearchCompressor::new(SearchCompressorConfig {
            max_matches_per_file: 10,
            max_total_matches: 30,
            ..Default::default()
        });
        let result = compressor.compress(&content, "", 1.0);
        assert_eq!(result.original_match_count, 40);
        assert_eq!(result.compressed_match_count, 30);
    }

    #[test]
    fn group_by_file_emits_path_then_indented_rows() {
        let content = grep_output(&[("src/a.rs", 8)]);
        let compressor = SearchCompressor::new(SearchCompressorConfig {
            group_by_file: true,
            ..Default::default()
        });
        let result = compressor.compress(&content, "", 1.0);
        let lines: Vec<&str> = result.compressed.lines().collect();
        assert_eq!(lines[0], "src/a.rs");
        assert!(lines[1].starts_with("  1:match content number 1"));
        assert!(lines[2].starts_with("  2:match content number 2"));
        assert!(result.compressed.contains("(5 of 8 matches shown)"));
    }

    #[test]
    fn deterministic_for_identical_calls() {
        let content = grep_output(&[("src/a.rs", 25), ("src/b.rs", 25)]);
        let compressor = SearchCompressor::default();
        let first = compressor.compress(&content, "error content", 2.0);
        let second = compressor.compress(&content, "error content", 2.0);
        assert_eq!(first, second);
    }
}
