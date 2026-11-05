//! Compressor for build, test and lint logs.
//!
//! [`LogCompressor::compress`] classifies every line as error, warning,
//! stack-trace frame, summary or plain, then keeps: error lines (first and
//! last always, each with up to [`LogCompressorConfig::error_context_lines`]
//! following context lines), warnings (deduplicated by normalized text),
//! summary lines, and up to [`LogCompressorConfig::max_stack_traces`] stack
//! traces. Long traces are collapsed to their head plus application-code
//! frames — runtime/stdlib frames (`site-packages`, `node_modules`, cargo
//! registry, `std::`, ...) are replaced with `"[... N frames collapsed]"`.

use std::collections::HashSet;
use std::sync::OnceLock;

use regex::Regex;

use crate::ccr;

use super::join_lines;

/// Substrings (any case) that identify a runtime/stdlib stack-trace frame.
///
/// Note: a bare `~/` is deliberately not treated as a runtime marker — user
/// application code commonly lives under `~`; only toolchain locations such
/// as `.rustup` and the cargo registry are matched.
const RUNTIME_MARKERS: &[&str] = &[
    "site-packages",
    "dist-packages",
    "urllib",
    "typing.py",
    "/json/",
    "json.py",
    "/python3",
    "lib/python",
    "cargo/registry",
    "/rustc/",
    ".rustup",
    "std::",
    "core::",
    "alloc::",
    "library/std",
    "library/core",
    "node_modules",
];

/// Log format detected (or requested) for a compression run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// Auto-detection requested; `compress` always returns a concrete format.
    Auto,
    /// pytest output.
    Pytest,
    /// cargo/rustc output.
    Cargo,
    /// Jest output.
    Jest,
    /// Anything else.
    Generic,
}

/// Configuration for [`LogCompressor`].
#[derive(Debug, Clone, PartialEq)]
pub struct LogCompressorConfig {
    /// Maximum error lines kept.
    pub max_errors: usize,
    /// Plain lines kept after each error line.
    pub error_context_lines: usize,
    /// Always keep the first error line.
    pub keep_first_error: bool,
    /// Always keep the last error line.
    pub keep_last_error: bool,
    /// Maximum stack-trace blocks kept.
    pub max_stack_traces: usize,
    /// Line count at which a stack-trace block is collapsed or truncated.
    pub stack_trace_max_lines: usize,
    /// Maximum warning lines kept.
    pub max_warnings: usize,
    /// Deduplicate warnings by normalized text.
    pub dedupe_warnings: bool,
    /// Always keep summary lines.
    pub keep_summary_lines: bool,
    /// Hard cap on the compressed output line count.
    pub max_total_lines: usize,
    /// Append a CCR retrieval marker when the output shrank enough.
    pub enable_ccr: bool,
    /// Minimum original lines before a CCR marker is considered.
    pub min_lines_for_ccr: usize,
    /// Maximum compressed/original line ratio for a CCR marker
    /// (compressed/original <= this).
    pub min_compression_ratio_for_ccr: f64,
    /// Collapse runtime frames in over-long stack traces.
    pub collapse_runtime_frames: bool,
    /// Head frames kept when collapsing a stack trace.
    pub trace_head_frames: usize,
    /// Application-code frames kept when collapsing a stack trace.
    pub trace_app_frames: usize,
}

impl Default for LogCompressorConfig {
    fn default() -> Self {
        Self {
            max_errors: 10,
            error_context_lines: 3,
            keep_first_error: true,
            keep_last_error: true,
            max_stack_traces: 3,
            stack_trace_max_lines: 20,
            max_warnings: 5,
            dedupe_warnings: true,
            keep_summary_lines: true,
            max_total_lines: 100,
            enable_ccr: true,
            min_lines_for_ccr: 50,
            min_compression_ratio_for_ccr: 0.5,
            collapse_runtime_frames: true,
            trace_head_frames: 3,
            trace_app_frames: 5,
        }
    }
}

/// Compressor for build, test and lint logs.
#[derive(Debug, Clone, Default)]
pub struct LogCompressor {
    /// Tuning knobs.
    pub config: LogCompressorConfig,
}

/// Result of a [`LogCompressor::compress`] run.
#[derive(Debug, Clone, PartialEq)]
pub struct LogCompressionResult {
    /// Compressed log; the original text when nothing was dropped.
    ///
    /// The appended CCR marker line is not counted in
    /// `compressed_line_count` or `compression_ratio`.
    pub compressed: String,
    /// The original input.
    pub original: String,
    /// Line count of the input.
    pub original_line_count: usize,
    /// Line count of the compressed body (excluding the CCR marker line).
    pub compressed_line_count: usize,
    /// Format detected in the input.
    pub format_detected: LogFormat,
    /// Compressed/original line ratio (1.0 when nothing was dropped).
    pub compression_ratio: f64,
    /// CCR cache key of the original content (`None` without a marker).
    pub cache_key: Option<String>,
}

/// Internal per-line classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineClass {
    Error,
    Warning,
    TraceFrame,
    TraceHeader,
    Summary,
    Plain,
}

/// One emitted output line: an original line or an inserted note.
#[derive(Debug, Clone, PartialEq)]
enum OutLine {
    Original(usize),
    Inserted(String),
}

impl LogCompressor {
    /// Creates a compressor with the given configuration.
    pub fn new(config: LogCompressorConfig) -> Self {
        Self { config }
    }

    /// Compresses a build/test/lint log.
    ///
    /// `context` and `bias` are accepted for signature parity with the
    /// other transforms and reserved for future relevance-based selection;
    /// the current rule-based selection does not use them. Returns the input
    /// unchanged when nothing is dropped.
    pub fn compress(&self, content: &str, context: &str, bias: f64) -> LogCompressionResult {
        let _ = (context, bias);
        let lines: Vec<&str> = content.lines().collect();
        let original_line_count = lines.len();
        let format_detected = detect_format(content);

        let mut classes: Vec<LineClass> = lines.iter().map(|l| classify_line(l)).collect();
        promote_numbered_frames(&lines, &mut classes);
        let blocks = find_trace_blocks(&classes);
        let mut block_of: Vec<Option<usize>> = vec![None; lines.len()];
        for (bi, &(start, end)) in blocks.iter().enumerate() {
            for slot in &mut block_of[start..end] {
                *slot = Some(bi);
            }
        }

        let marked = self.mark_kept_lines(&lines, &classes, &block_of);
        let kept_blocks = self.config.max_stack_traces.min(blocks.len());
        let out = self.assemble(&lines, &classes, &blocks, &block_of, &marked, kept_blocks);

        let modified = out.len() < lines.len()
            || out.iter().any(|o| matches!(o, OutLine::Inserted(_)))
            || kept_blocks < blocks.len();
        // Safety: when the keep-rules retain nothing, return the input
        // unchanged instead of an empty body.
        if !modified || out.is_empty() {
            return LogCompressionResult {
                compressed: content.to_string(),
                original: content.to_string(),
                original_line_count,
                compressed_line_count: original_line_count,
                format_detected,
                compression_ratio: 1.0,
                cache_key: None,
            };
        }

        let body: Vec<String> = out
            .iter()
            .map(|o| match o {
                OutLine::Original(j) => lines[*j].to_string(),
                OutLine::Inserted(s) => s.clone(),
            })
            .collect();
        let compressed_line_count = body.len();
        let compression_ratio = if original_line_count > 0 {
            compressed_line_count as f64 / original_line_count as f64
        } else {
            1.0
        };

        let mut compressed = join_lines(&body, content);
        let mut cache_key = None;
        if self.config.enable_ccr
            && original_line_count >= self.config.min_lines_for_ccr
            && compression_ratio <= self.config.min_compression_ratio_for_ccr
        {
            let hash = ccr::ccr_hash(content);
            compressed.push_str(&format!(
                "\n[{compressed_line_count} of {original_line_count} lines shown. \
                 Retrieve original: hash={hash}]"
            ));
            cache_key = Some(hash);
        }

        LogCompressionResult {
            compressed,
            original: content.to_string(),
            original_line_count,
            compressed_line_count,
            format_detected,
            compression_ratio,
            cache_key,
        }
    }

    /// Marks the original lines that survive: errors (+ context), warnings,
    /// and summaries. Lines inside trace blocks are decided by the block
    /// handling instead, so error context never crosses into a block.
    fn mark_kept_lines(
        &self,
        lines: &[&str],
        classes: &[LineClass],
        block_of: &[Option<usize>],
    ) -> Vec<bool> {
        let mut marked = vec![false; lines.len()];

        let error_indices: Vec<usize> = (0..lines.len())
            .filter(|&i| classes[i] == LineClass::Error)
            .collect();
        let kept_errors: Vec<usize> = if error_indices.len() <= self.config.max_errors {
            error_indices
        } else {
            let mut selected: Vec<usize> = error_indices[..self.config.max_errors].to_vec();
            if self.config.keep_last_error {
                let last = *error_indices.last().expect("non-empty");
                if !selected.contains(&last) {
                    selected.pop();
                    selected.push(last);
                }
            }
            selected.sort_unstable();
            selected
        };
        for &e in &kept_errors {
            marked[e] = true;
            for c in 1..=self.config.error_context_lines {
                let j = e + c;
                if j < lines.len() && block_of[j].is_none() {
                    marked[j] = true;
                }
            }
        }

        let mut seen: HashSet<String> = HashSet::new();
        let mut kept_warnings = 0usize;
        for i in 0..lines.len() {
            if classes[i] != LineClass::Warning {
                continue;
            }
            let normalized = normalize_ws(lines[i]);
            if self.config.dedupe_warnings && !seen.insert(normalized) {
                continue;
            }
            if kept_warnings >= self.config.max_warnings {
                continue;
            }
            kept_warnings += 1;
            marked[i] = true;
        }

        if self.config.keep_summary_lines {
            for i in 0..lines.len() {
                if classes[i] == LineClass::Summary {
                    marked[i] = true;
                }
            }
        }
        marked
    }

    /// Walks the input and produces the emitted lines, handling trace blocks
    /// as units (kept, collapsed or truncated; blocks beyond
    /// `max_stack_traces` are dropped entirely).
    fn assemble(
        &self,
        lines: &[&str],
        classes: &[LineClass],
        blocks: &[(usize, usize)],
        block_of: &[Option<usize>],
        marked: &[bool],
        kept_blocks: usize,
    ) -> Vec<OutLine> {
        let mut out: Vec<OutLine> = Vec::new();
        let mut i = 0usize;
        while i < lines.len() {
            if let Some(bi) = block_of[i] {
                let &(start, end) = &blocks[bi];
                if bi < kept_blocks {
                    let block_len = end - start;
                    if block_len <= self.config.stack_trace_max_lines {
                        for j in start..end {
                            out.push(OutLine::Original(j));
                        }
                    } else if self.config.collapse_runtime_frames {
                        out.extend(self.collapse_block(lines, classes, start, end));
                    } else {
                        for j in start..start + self.config.stack_trace_max_lines {
                            out.push(OutLine::Original(j));
                        }
                        out.push(OutLine::Inserted(format!(
                            "[... {} lines elided]",
                            block_len - self.config.stack_trace_max_lines
                        )));
                    }
                }
                i = end;
            } else {
                if marked[i] {
                    out.push(OutLine::Original(i));
                }
                i += 1;
            }
        }
        out.truncate(self.config.max_total_lines);
        out
    }

    /// Collapses one over-long trace block: message/chain-head lines, the
    /// first `trace_head_frames` frames, up to `trace_app_frames`
    /// application-code frames, then `"[... N frames collapsed]"`.
    fn collapse_block(
        &self,
        lines: &[&str],
        classes: &[LineClass],
        start: usize,
        end: usize,
    ) -> Vec<OutLine> {
        let frames: Vec<usize> = (start..end)
            .filter(|&j| classes[j] == LineClass::TraceFrame)
            .collect();
        let head: Vec<usize> = frames
            .iter()
            .take(self.config.trace_head_frames)
            .copied()
            .collect();
        let mut app: Vec<usize> = Vec::new();
        for &f in frames.iter().skip(self.config.trace_head_frames) {
            if app.len() >= self.config.trace_app_frames {
                break;
            }
            if !is_runtime_frame(lines[f]) {
                app.push(f);
            }
        }
        let kept: HashSet<usize> = head.iter().chain(app.iter()).copied().collect();
        let dropped_frames = frames.len() - kept.len();

        let mut out: Vec<OutLine> = Vec::new();
        for (j, class) in classes.iter().enumerate().take(end).skip(start) {
            if *class == LineClass::TraceHeader || kept.contains(&j) {
                out.push(OutLine::Original(j));
            }
        }
        if dropped_frames > 0 {
            out.push(OutLine::Inserted(format!(
                "[... {dropped_frames} frames collapsed]"
            )));
        }
        out
    }
}

/// Classifies a single log line.
fn classify_line(line: &str) -> LineClass {
    let trimmed = line.trim_start();
    let lower = trimmed.to_lowercase();
    if is_summary_line(&lower) {
        LineClass::Summary
    } else if is_error_line(trimmed, &lower) {
        LineClass::Error
    } else if lower.contains("warning") || lower.contains("deprecated") || lower.contains("warn:") {
        LineClass::Warning
    } else if is_trace_header(trimmed) {
        LineClass::TraceHeader
    } else if is_trace_frame(trimmed) {
        LineClass::TraceFrame
    } else {
        LineClass::Plain
    }
}

/// Summary lines are checked first so that totals such as
/// `"error: aborting due to ..."` classify as summaries, not errors.
fn is_summary_line(lower: &str) -> bool {
    lower.starts_with("test result:")
        || lower.starts_with("finished")
        || lower.starts_with("compiling")
        || lower.starts_with("running")
        || lower.starts_with("doc-tests")
        || lower.starts_with("tests:")
        || lower.starts_with("test suites:")
        || lower.starts_with("suites:")
        || lower.starts_with("snapshots:")
        || lower.starts_with("time:")
        || lower.contains("error: aborting")
        || lower.starts_with("warning: `")
        || (lower.contains("passed") && (lower.contains("failed") || lower.contains("skipped")))
}

fn is_error_line(trimmed: &str, lower: &str) -> bool {
    lower.contains("error")
        || lower.contains("failed")
        || lower.contains("exception")
        || lower.contains("panicked")
        || lower.starts_with("panic")
        || (trimmed.starts_with("E ") && trimmed.len() > 2)
        || trimmed.contains('\u{2715}')
        || trimmed.contains('\u{2717}')
}

fn is_trace_header(trimmed: &str) -> bool {
    trimmed.starts_with("Traceback (most recent")
        || trimmed == "stack backtrace:"
        || trimmed.starts_with("During handling of the above")
}

fn is_trace_frame(trimmed: &str) -> bool {
    trimmed.starts_with("at ") || (trimmed.starts_with("File \"") && trimmed.contains(", line "))
}

/// Promotes rustc-style numbered backtrace frames (`"   0: foo::bar"`)
/// that sit next to an existing trace line.
fn promote_numbered_frames(lines: &[&str], classes: &mut [LineClass]) {
    static NUMBERED: OnceLock<Regex> = OnceLock::new();
    let numbered =
        NUMBERED.get_or_init(|| Regex::new(r"^\d+:\s+\S").expect("static regex is valid"));
    for i in 0..lines.len() {
        if classes[i] != LineClass::Plain {
            continue;
        }
        if !numbered.is_match(lines[i].trim_start()) {
            continue;
        }
        let prev = i > 0
            && matches!(
                classes[i - 1],
                LineClass::TraceFrame | LineClass::TraceHeader
            );
        let next = i + 1 < lines.len()
            && matches!(
                classes[i + 1],
                LineClass::TraceFrame | LineClass::TraceHeader
            );
        if prev || next {
            classes[i] = LineClass::TraceFrame;
        }
    }
}

/// Finds maximal runs of trace header/frame lines as `[start, end)` blocks.
fn find_trace_blocks(classes: &[LineClass]) -> Vec<(usize, usize)> {
    let mut blocks = Vec::new();
    let mut i = 0usize;
    while i < classes.len() {
        if matches!(classes[i], LineClass::TraceFrame | LineClass::TraceHeader) {
            let start = i;
            while i < classes.len()
                && matches!(classes[i], LineClass::TraceFrame | LineClass::TraceHeader)
            {
                i += 1;
            }
            blocks.push((start, i));
        } else {
            i += 1;
        }
    }
    blocks
}

/// Normalized warning text: trimmed, whitespace-collapsed, lowercased.
fn normalize_ws(line: &str) -> String {
    line.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Detects the log format from its content.
fn detect_format(content: &str) -> LogFormat {
    let lower = content.to_lowercase();
    if lower.contains("test suites:")
        || lower.contains("tests:")
        || lower.contains("snapshots:")
        || content.contains('\u{2715}')
        || content.contains('\u{2713}')
    {
        return LogFormat::Jest;
    }
    if lower.contains("compiling")
        || lower.contains("finished `")
        || lower.contains("error[e")
        || (lower.contains("test result:") && lower.contains("running"))
    {
        return LogFormat::Cargo;
    }
    if lower.contains("test result:")
        || lower.contains("short test summary")
        || lower.contains("collected ")
        || lower.contains("=== ")
    {
        return LogFormat::Pytest;
    }
    LogFormat::Generic
}

/// `true` when the stack-trace frame comes from runtime/stdlib code.
fn is_runtime_frame(line: &str) -> bool {
    RUNTIME_MARKERS.iter().any(|m| line.contains(m))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pytest_log_keeps_errors_and_summaries() {
        let mut lines: Vec<&str> = vec![
            "============================= test session starts ==============================",
            "collected 30 items",
        ];
        for _ in 0..25 {
            lines.push("tests/test_ok.py .");
        }
        lines.extend([
            "=================================== FAILURES ===================================",
            "_________________________________ test_widget _________________________________",
            "Traceback (most recent call last):",
            r#"  File "/app/tests/test_widget.py", line 10, in test_widget"#,
            "    assert widget.count == 2",
            "E   assert 1 == 2",
            "test result: 1 failed, 29 passed in 0.05s",
            "=========================== short test summary info ============================",
            "FAILED tests/test_widget.py::test_widget - assert 1 == 2",
            "1 failed, 29 passed in 0.05s",
        ]);
        let content = lines.join("\n");
        let compressor = LogCompressor::default();
        let result = compressor.compress(&content, "", 1.0);
        assert_eq!(result.format_detected, LogFormat::Pytest);
        assert!(result
            .compressed
            .contains("FAILED tests/test_widget.py::test_widget"));
        assert!(result
            .compressed
            .contains("test result: 1 failed, 29 passed in 0.05s"));
        assert!(result.compressed.contains("E   assert 1 == 2"));
        assert!(result
            .compressed
            .contains("Traceback (most recent call last):"));
        assert!(!result.compressed.contains("tests/test_ok.py ."));
        assert_eq!(result.original_line_count, 37);
        assert_eq!(result.compressed_line_count, 7);
        assert_eq!(result.cache_key, None, "37 < 50 lines -> no CCR");
    }

    #[test]
    fn cargo_log_keeps_errors_and_dedupes_warnings() {
        let mut lines: Vec<&str> = vec!["Compiling headroom-core v0.1.0 (/home/user/headroom)"];
        for _ in 0..30 {
            lines.push("checking filler item for the build log");
        }
        lines.extend([
            "warning: unused variable: `x`",
            "warning: unused variable: `x`",
            "warning: unused variable: `x`",
            "warning: unused import: std::collections::HashMap",
            "warning: unused variable: `y`",
            "error[E0308]: mismatched types expected `i32` found `String`",
            "  --> src/main.rs:42:5",
            "error: aborting due to 1 previous error",
            "Finished `dev` profile [unoptimized + debuginfo] target(s) in 3.14s",
        ]);
        let content = lines.join("\n");
        let compressor = LogCompressor::default();
        let result = compressor.compress(&content, "", 1.0);
        assert_eq!(result.format_detected, LogFormat::Cargo);
        assert_eq!(
            result.compressed.matches("unused variable: `x`").count(),
            1,
            "identical warnings deduplicated"
        );
        assert!(result.compressed.contains("warning: unused import"));
        assert!(result.compressed.contains("error[E0308]"));
        assert!(result
            .compressed
            .contains("error: aborting due to 1 previous error"));
        assert!(result.compressed.contains("Finished `dev` profile"));
        assert!(
            result.compressed.contains("  --> src/main.rs:42:5"),
            "error context kept"
        );
        assert!(!result.compressed.contains("checking filler item"));
        assert_eq!(result.compressed_line_count, 8);
        assert_eq!(result.cache_key, None, "40 < 50 lines -> no CCR");
    }

    #[test]
    fn long_stack_traces_collapse_runtime_frames() {
        let mut lines: Vec<&str> = vec![
            "Traceback (most recent call last):",
            r#"  File "/app/src/main.py", line 5, in run"#,
            r#"  File "/app/src/worker.py", line 9, in handle"#,
            r#"  File "/app/src/task.py", line 3, in execute"#,
        ];
        for _ in 0..25 {
            lines.push(
                r#"  File "/usr/lib/python3.11/site-packages/requests/api.py", line 115, in get"#,
            );
        }
        lines.extend([
            r#"  File "/app/src/app.py", line 20, in main"#,
            r#"  File "/app/src/entry.py", line 2, in start"#,
            "ValueError: invalid input for the worker task",
            "ERROR: job failed during processing",
        ]);
        let content = lines.join("\n");
        let compressor = LogCompressor::default();
        let result = compressor.compress(&content, "", 1.0);
        assert!(result.compressed.contains("[... 25 frames collapsed]"));
        assert!(
            result.compressed.contains(r#"/app/src/app.py"#),
            "app frames kept"
        );
        assert!(
            !result.compressed.contains("site-packages"),
            "runtime frames dropped"
        );
        assert!(result.compressed.contains("ValueError: invalid input"));
        assert_eq!(result.compressed_line_count, 9); // header + 5 frames + marker + 2 errors
    }

    #[test]
    fn warnings_are_deduplicated() {
        let mut lines: Vec<&str> = Vec::new();
        for _ in 0..10 {
            lines.push("warning: unused variable: `x`");
        }
        lines.extend([
            "warning: deprecated API usage detected",
            "warning: disk space is almost full",
            "all good in the end",
        ]);
        let content = lines.join("\n");
        let compressor = LogCompressor::default();
        let result = compressor.compress(&content, "", 1.0);
        assert_eq!(result.compressed.matches("unused variable: `x`").count(), 1);
        assert!(result.compressed.contains("deprecated API usage detected"));
        assert!(result.compressed.contains("disk space is almost full"));
    }

    #[test]
    fn error_caps_keep_first_and_last() {
        let mut lines: Vec<String> = Vec::new();
        for i in 0..20 {
            if matches!(i, 2 | 6 | 10 | 14 | 18) {
                lines.push(format!("error: failure number {}", (i - 2) / 4));
            } else {
                lines.push(format!("filler line {i}"));
            }
        }
        let content = lines.join("\n");
        let compressor = LogCompressor::new(LogCompressorConfig {
            max_errors: 2,
            ..Default::default()
        });
        let result = compressor.compress(&content, "", 1.0);
        assert!(result.compressed.contains("error: failure number 0"));
        assert!(result.compressed.contains("error: failure number 4"));
        assert!(!result.compressed.contains("error: failure number 1"));
        assert!(!result.compressed.contains("error: failure number 2"));
        assert_eq!(result.compressed_line_count, 6); // 2 errors + 3 + 1 context lines
    }

    #[test]
    fn total_line_cap_truncates_output() {
        let lines: Vec<String> = (0..30)
            .map(|i| format!("Compiling crate number {i}"))
            .collect();
        let content = lines.join("\n");
        let compressor = LogCompressor::new(LogCompressorConfig {
            max_total_lines: 10,
            ..Default::default()
        });
        let result = compressor.compress(&content, "", 1.0);
        assert_eq!(result.compressed_line_count, 10);
        assert_eq!(result.compressed.lines().count(), 10);
    }

    #[test]
    fn ccr_marker_appended_for_large_shrunk_logs() {
        let mut lines: Vec<String> = (0..50)
            .map(|i| format!("filler line {i} of the log stream"))
            .collect();
        lines.push("error: something went wrong in stage two".to_string());
        for i in 0..3 {
            lines.push(format!("plain detail line {i}"));
        }
        lines.push("error: another failure happened here".to_string());
        for i in 0..4 {
            lines.push(format!("plain tail line {i}"));
        }
        lines.push("Finished build in 2.00s".to_string());
        let content = lines.join("\n");
        assert_eq!(content.lines().count(), 60);
        let compressor = LogCompressor::default();
        let result = compressor.compress(&content, "", 1.0);
        assert_eq!(result.compressed_line_count, 9);
        assert!(result
            .compressed
            .contains("[9 of 60 lines shown. Retrieve original: hash="));
        assert_eq!(result.cache_key, Some(ccr::ccr_hash(&content)));
        assert!(!result.compressed.contains("filler line 20 of"));
        assert!(result.compression_ratio < 0.16);
    }

    #[test]
    fn nothing_dropped_passes_through() {
        let content = [
            "Compiling a",
            "Compiling b",
            "error: bad thing happened",
            "warning: watch out for this",
            "Finished dev profile fast",
        ]
        .join("\n");
        let compressor = LogCompressor::default();
        let result = compressor.compress(&content, "", 1.0);
        assert_eq!(result.compressed, content);
        assert_eq!(result.compression_ratio, 1.0);
        assert_eq!(result.cache_key, None);
        assert_eq!(result.format_detected, LogFormat::Cargo);
    }

    #[test]
    fn jest_log_detected() {
        let content = [
            "PASS src/a.test.js",
            "\u{2715} renders the button",
            "Tests: 1 failed, 2 passed, 1 skipped",
            "Test Suites: 1 failed, 1 total",
        ]
        .join("\n");
        let compressor = LogCompressor::default();
        let result = compressor.compress(&content, "", 1.0);
        assert_eq!(result.format_detected, LogFormat::Jest);
        assert!(result.compressed.contains("renders the button"));
        assert!(result
            .compressed
            .contains("Tests: 1 failed, 2 passed, 1 skipped"));
    }

    #[test]
    fn generic_log_detected() {
        let content = "just some text\nwith no format markers\nending plainly here";
        let compressor = LogCompressor::default();
        let result = compressor.compress(&content, "", 1.0);
        assert_eq!(result.format_detected, LogFormat::Generic);
        assert_eq!(result.compressed, content, "nothing kept -> passthrough");
    }

    #[test]
    fn deterministic_for_identical_calls() {
        let mut lines: Vec<String> = (0..55)
            .map(|i| format!("filler line {i} of the log stream"))
            .collect();
        lines.push("error: boom".to_string());
        lines.push("warning: careful".to_string());
        lines.push("Finished".to_string());
        let content = lines.join("\n");
        let compressor = LogCompressor::default();
        let first = compressor.compress(&content, "context", 1.5);
        let second = compressor.compress(&content, "context", 1.5);
        assert_eq!(first, second);
    }
}
