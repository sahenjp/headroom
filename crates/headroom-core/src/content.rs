//! Content-type detection for routing tool output to the right compressor.
//!
//! Port of headroom/transforms/content_detector.py detect_content_type,
//! simplified to a pure [ContentType] verdict. Detectors run in this
//! priority order (per the port spec; the Python module checks diff and
//! HTML before search/log, this port follows the spec's ordering):
//!
//! 1. JSON_ARRAY - the trimmed text parses as a JSON array of >= 5 items;
//! 2. SEARCH_RESULTS - grep-style 'file:line:content' rows and grep
//!    -A/-B/-C context shapes;
//! 3. BUILD_OUTPUT - compiler / test / lint log lines;
//! 4. GIT_DIFF - 'diff --git' headers, '--- a/' lines, '@@' hunks;
//! 5. HTML - doctype / html / head / body tags and structural markup;
//! 6. TABULAR - markdown tables, consistently delimited CSV/TSV, and
//!    space-aligned fixed-width command output;
//! 7. STRUCTURED_CONFIG - YAML key/list shapes and TOML/INI sections
//!    (structural only: no TOML/INI parser confirmation);
//! 8. SOURCE_CODE - per-language code patterns;
//! 9. PLAIN_TEXT - fallback.

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

/// The kind of content a tool result holds; routes it to a compressor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    /// Structured JSON array data (SmartCrusher's native format).
    JsonArray,
    /// grep / ripgrep style file:line:content output.
    SearchResults,
    /// Compiler, test, and lint logs.
    BuildOutput,
    /// Unified / git diff format.
    GitDiff,
    /// Web pages (need content extraction, not compression).
    Html,
    /// CSV/TSV, markdown tables, fixed-width tables.
    Tabular,
    /// YAML / TOML / INI config files.
    StructuredConfig,
    /// Python, JS/TS, Go, Rust, Java, C#, PHP source.
    SourceCode,
    /// Generic text (fallback).
    PlainText,
}

impl ContentType {
    /// Wire name, matching the Python enum values.
    pub fn as_str(self) -> &'static str {
        match self {
            ContentType::JsonArray => "json_array",
            ContentType::SearchResults => "search",
            ContentType::BuildOutput => "build",
            ContentType::GitDiff => "diff",
            ContentType::Html => "html",
            ContentType::Tabular => "tabular",
            ContentType::StructuredConfig => "structured_config",
            ContentType::SourceCode => "source_code",
            ContentType::PlainText => "text",
        }
    }
}

impl std::fmt::Display for ContentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Minimum character count before a block is worth compressing at all.
pub fn min_chars_for_block_compression() -> usize {
    500
}

// --- search-result shapes -------------------------------------------------

static SEARCH_LINE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[^\s:]+:\d+:").expect("search pattern must compile"));
static GREP_CONTEXT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?P<path>[^\s:]+?)-(?P<line>\d+)-").expect("grep context pattern must compile")
});
static GREP_COLON_DASH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?P<path>[^\s:]+):(?P<line>\d+)-").expect("grep colon-dash pattern must compile")
});
/// A timestamped log row has the same colon-delimited shape as a grep row but
/// must never be treated as one (mirrors lossless_compaction).
static TIMESTAMP_ROW_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^\s*\[?(?:\d{4}-\d{2}-\d{2}[ T]\d{1,2}:\d{2}|\d{2}/\d{2}/\d{2,4}[ T]\d{1,2}:\d{2}|[A-Z][a-z]{2}\s+\d{1,2}\s+\d{1,2}:\d{2}|\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?(?:\s|\]|$))",
    )
    .expect("timestamp row pattern must compile")
});

// --- diff shapes -----------------------------------------------------------

static DIFF_HEADER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^(diff --git|diff --combined |diff --cc |--- a/|@@\s+-\d+,\d+\s+\+\d+,\d+\s+@@|@@@+\s+-\d+(?:,\d+)?\s+(?:-\d+(?:,\d+)?\s+)+\+\d+(?:,\d+)?\s+@@@+)",
    )
    .expect("diff header pattern must compile")
});
static DIFF_CHANGE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[+-][^+-]").expect("diff change pattern must compile"));

// --- html shapes ----------------------------------------------------------

static HTML_DOCTYPE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^\s*<!doctype\s+html").expect("doctype pattern must compile")
});
static HTML_TAG_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)<html[\s>]").expect("html tag pattern must compile"));
static HTML_HEAD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)<head[\s>]").expect("head tag pattern must compile"));
static HTML_BODY_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)<body[\s>]").expect("body tag pattern must compile"));
static HTML_STRUCTURAL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)<(div|span|script|style|link|meta|nav|header|footer|aside|article|section|main)[\s>]",
    )
    .expect("html structural pattern must compile")
});

// --- tabular shapes -------------------------------------------------------

static MD_SEP_CELL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^:?-{2,}:?$").expect("md separator cell must compile"));
static FIXED_WIDTH_LIST_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*(?:[-*+\u{2022}]|\d{1,3}[.)])\s")
        .expect("fixed-width list pattern must compile")
});
static FIXED_WIDTH_PROSE_END_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"[A-Za-z][.!?]["')\]]?$"#).expect("fixed-width prose pattern must compile")
});
/// Line endings that make a row look like code rather than aligned output.
const FIXED_WIDTH_CODE_END_CHARS: [char; 8] = ['{', '}', ';', '(', ')', ',', ':', '\\'];
/// Line starts that make a row look like code rather than aligned output.
const FIXED_WIDTH_CODE_START_STRINGS: [&str; 4] = ["#", "//", "/*", "--"];
/// Column positions examined for shared gutters.
const FIXED_WIDTH_MAX_COLS: usize = 400;

// --- structured-config shapes ----------------------------------------------

static CONFIG_SECTION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"^\s*\[\[?[\w.\-"' ]+\]\]?\s*$"#).expect("config section pattern must compile")
});
static TOML_ASSIGN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"^\s*(?:[\w.\-]+|"[^"]+"|'[^']+')\s*=\s*\S"#)
        .expect("toml assign pattern must compile")
});
static INI_ASSIGN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*[\w.\-@ ]+?\s*[=:]\s*").expect("ini assign pattern must compile")
});
static YAML_KEY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"^\s*(?:-\s+)?(?:[\w.\-/]+|"[^"]+"|'[^']+')\s*:(?:\s|$)"#)
        .expect("yaml key pattern must compile")
});
static YAML_LIST_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*-\s+\S").expect("yaml list pattern must compile"));
static YAML_DOC_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^---\s*$|^\.\.\.\s*$").expect("yaml doc pattern must compile"));
static CONFIG_COMMENT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*[#;]").expect("config comment pattern must compile"));

// --- log / build-output shapes --------------------------------------------

static LOG_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    const PATTERNS: &[&str] = &[
        r"(?i)\b(ERROR|FAIL|FAILED|FATAL|CRITICAL)\b",
        r"(?i)\b(WARN|WARNING)\b",
        r"(?i)\b(INFO|DEBUG|TRACE)\b",
        r"^\s*\d{4}-\d{2}-\d{2}",
        r"^\s*\[\d{2}:\d{2}:\d{2}\]",
        r"^={3,}|^-{3,}",
        r"^\s*PASSED|^\s*FAILED|^\s*SKIPPED",
        r"^npm ERR!|^yarn error|^cargo error",
        r"Traceback \(most recent call last\)",
        r"^\w*(Error|Exception):",
        r"^\s*at\s+[\w.$/]+\(",
        r"^\s*at async \S",
        r"^(panic|fatal error): ",
        r"^goroutine \d+ \[",
        r"^\t\S+\.go:\d+ \+0x",
        r"^thread '[^']*' panicked at",
        r"^stack backtrace:",
        r"^\s+\d+: \S",
        r"^\s+at \S+:\d+:\d+$",
        r"^Unhandled exception\.",
        r"^\s*at .+\) in .+:line \d+",
        r"^Caused by: ",
        r"^\s*\.\.\. \d+ more$",
        r"^<!\[LOG\[",
        // cargo / pytest / jest shapes from the port spec
        r"^\s*(Compiling|Checking)\s",
        r"^\s*Finished\s",
        r"^\s*test result:",
        r"^test \S.*\.\.\. (ok|FAILED|ignored)",
        r"^(note|help):",
        r"^\s*running \d+ tests?",
        r"^\s*\d+ passed",
    ];
    PATTERNS
        .iter()
        .map(|p| Regex::new(p).expect("log pattern must compile"))
        .collect()
});

// --- source-code shapes ---------------------------------------------------

static CODE_PATTERNS: LazyLock<Vec<(&'static str, Vec<Regex>)>> = LazyLock::new(|| {
    const GROUPS: &[(&str, &[&str])] = &[
        (
            "python",
            &[
                r"^\s*(def|class|import|from|async def)\s+\w+",
                r#"^\s*""""#,
                r"^\s*@\w+",
                r"^\s*if __name__\s*==",
            ],
        ),
        (
            "javascript",
            &[
                r"^\s*(function|const|let|var|class|import|export)\s+",
                r"^\s*(async\s+function|=>\s*\{)",
                r"^\s*module\.exports",
            ],
        ),
        (
            "typescript",
            &[
                r"^\s*(interface|type|enum|namespace)\s+\w+",
                r":\s*(string|number|boolean|any|void)\b",
            ],
        ),
        (
            "go",
            &[
                r"^\s*(func|type|package|import)\s+",
                r"^\s*func\s+\([^)]+\)\s+\w+",
            ],
        ),
        (
            "rust",
            &[r"^\s*(fn|struct|enum|impl|mod|use|pub)\s+", r"^\s*#\["],
        ),
        (
            "java",
            &[
                r"^\s*(public|private|protected)\s+(class|interface|enum)",
                r"^\s*@\w+",
                r"^\s*package\s+[\w.]+;",
            ],
        ),
        (
            "csharp",
            &[
                r"^\s*using\s+[\w.]+\s*;",
                r"^\s*namespace\s+[\w.]+",
                r"^\s*(public|private|protected|internal|sealed|static|abstract|partial)\s+(class|struct|record|interface|enum)\b",
                r"^.*\b(get|set|init);",
            ],
        ),
        (
            "php",
            &[
                r"<\?php\b",
                r"^\s*namespace\s+[\w\\]+\s*;",
                r"^\s*use\s+[\w\\]+(\s+as\s+\w+)?\s*;",
                r"^\s*(public|private|protected|static|abstract|final)?\s*function\s+\w+\s*\(",
                r"\$this->",
            ],
        ),
    ];
    GROUPS
        .iter()
        .map(|(lang, pats)| {
            (
                *lang,
                pats.iter()
                    .map(|p| Regex::new(p).expect("code pattern must compile"))
                    .collect(),
            )
        })
        .collect()
});

/// Detect the content type of a tool result, in the priority order documented
/// on the module. Empty or whitespace-only text is plain.
pub fn detect_content_type(text: &str) -> ContentType {
    if text.trim().is_empty() {
        return ContentType::PlainText;
    }
    if let Some(t) = try_detect_json(text) {
        return t;
    }
    if let Some(t) = try_detect_search(text) {
        return t;
    }
    if let Some(t) = try_detect_log(text) {
        return t;
    }
    if let Some(t) = try_detect_diff(text) {
        return t;
    }
    if let Some(t) = try_detect_html(text) {
        return t;
    }
    if let Some(t) = try_detect_tabular(text) {
        return t;
    }
    if let Some(t) = try_detect_structured_config(text) {
        return t;
    }
    if let Some(t) = try_detect_code(text) {
        return t;
    }
    ContentType::PlainText
}

/// JSON by PARSING, not by surface patterns: the trimmed text must parse as
/// a JSON array of at least 5 items (the port's stand-in for the Python bulk
/// fraction gate).
fn try_detect_json(text: &str) -> Option<ContentType> {
    let trimmed = text.trim();
    let value: Value = serde_json::from_str(trimmed).ok()?;
    let array = value.as_array()?;
    if array.len() >= 5 {
        Some(ContentType::JsonArray)
    } else {
        None
    }
}

/// True when the pre-colon segment looks like a file path: no angle brackets
/// (markup tags) and no '=' (key=value log prefixes).
fn prefix_looks_like_path(prefix: &str) -> bool {
    !prefix.contains('<') && !prefix.contains('>') && !prefix.contains('=')
}

/// 'path-NN-content' grep -A/-B/-C context rows; the path must contain '/' or
/// '.' so dates and dashed prose stay out.
fn grep_context_line(line: &str) -> bool {
    match GREP_CONTEXT_RE.captures(line) {
        Some(caps) => {
            let prefix = caps.name("path").map(|m| m.as_str()).unwrap_or("");
            prefix_looks_like_path(prefix) && (prefix.contains('/') || prefix.contains('.'))
        }
        None => false,
    }
}

/// True for 'path:line:content' rows, 'path:NN-content' and 'path-NN-content'
/// context rows; timestamped log rows are never search rows.
fn is_search_result_line(line: &str) -> bool {
    if TIMESTAMP_ROW_RE.is_match(line) {
        return false;
    }
    if SEARCH_LINE_RE.is_match(line) || GREP_COLON_DASH_RE.is_match(line) {
        let prefix = line.split(':').next().unwrap_or("");
        return prefix_looks_like_path(prefix);
    }
    grep_context_line(line)
}

/// grep/ripgrep results: at least 2 matching lines and >= 30% of the
/// non-empty lines (first 100 lines).
fn try_detect_search(text: &str) -> Option<ContentType> {
    let lines: Vec<&str> = text.split('\n').take(100).collect();
    let matching = lines
        .iter()
        .filter(|line| !line.trim().is_empty() && is_search_result_line(line))
        .count();
    if matching < 2 {
        return None;
    }
    let non_empty = lines.iter().filter(|line| !line.trim().is_empty()).count();
    if non_empty == 0 {
        return None;
    }
    if (matching as f64) / (non_empty as f64) < 0.3 {
        return None;
    }
    Some(ContentType::SearchResults)
}

/// Build/log output: pattern-matching lines with a confidence gate,
/// mirroring the Python thresholds (ratio >= 0.1, confidence >= 0.5).
fn try_detect_log(text: &str) -> Option<ContentType> {
    let lines: Vec<&str> = text.split('\n').take(200).collect();
    let mut pattern_matches = 0usize;
    let mut error_matches = 0usize;
    for line in &lines {
        for (i, pattern) in LOG_PATTERNS.iter().enumerate() {
            if pattern.is_match(line) {
                pattern_matches += 1;
                if i < 2 {
                    error_matches += 1;
                }
                break; // one pattern per line is enough
            }
        }
    }
    if pattern_matches == 0 {
        return None;
    }
    let non_empty = lines.iter().filter(|line| !line.trim().is_empty()).count();
    if non_empty == 0 {
        return None;
    }
    let ratio = pattern_matches as f64 / non_empty as f64;
    if ratio < 0.1 {
        return None;
    }
    let confidence = (0.3 + ratio * 0.5 + error_matches as f64 * 0.05).min(1.0);
    if confidence < 0.5 {
        return None;
    }
    Some(ContentType::BuildOutput)
}

/// Unified diff: headers plus +/- change lines, confidence >= 0.7.
fn try_detect_diff(text: &str) -> Option<ContentType> {
    let lines: Vec<&str> = text.split('\n').take(500).collect();
    let mut headers = 0usize;
    let mut changes = 0usize;
    for line in &lines {
        if DIFF_HEADER_RE.is_match(line) {
            headers += 1;
        }
        if DIFF_CHANGE_RE.is_match(line) {
            changes += 1;
        }
    }
    if headers == 0 {
        return None;
    }
    let confidence = (0.5 + headers as f64 * 0.2 + changes as f64 * 0.05).min(1.0);
    if confidence < 0.7 {
        return None;
    }
    Some(ContentType::GitDiff)
}

/// HTML: doctype / html / head / body tags and structural markup density,
/// confidence >= 0.7 (first 3000 chars).
fn try_detect_html(text: &str) -> Option<ContentType> {
    let sample: String = text.chars().take(3000).collect();
    let has_doctype = HTML_DOCTYPE_RE.is_match(&sample);
    let has_html_tag = HTML_TAG_RE.is_match(&sample);
    let has_head = HTML_HEAD_RE.is_match(&sample);
    let has_body = HTML_BODY_RE.is_match(&sample);
    let structural = HTML_STRUCTURAL_RE.find_iter(&sample).count();
    if !has_doctype && !has_html_tag && structural < 3 {
        return None;
    }
    let mut confidence = 0.0;
    if has_doctype {
        confidence += 0.5;
    }
    if has_html_tag {
        confidence += 0.3;
    }
    if has_head {
        confidence += 0.1;
    }
    if has_body {
        confidence += 0.1;
    }
    confidence += (structural as f64 * 0.03).min(0.3);
    if confidence.min(1.0) < 0.7 {
        return None;
    }
    Some(ContentType::Html)
}

/// Tabular text: markdown tables first, then delimited CSV/TSV, then
/// space-aligned fixed-width output.
fn try_detect_tabular(text: &str) -> Option<ContentType> {
    let lines: Vec<&str> = text
        .split('\n')
        .filter(|line| !line.trim().is_empty())
        .take(50)
        .collect();
    if lines.len() < 3 {
        return None;
    }
    if try_detect_markdown_table(&lines).is_some() {
        return Some(ContentType::Tabular);
    }
    if try_detect_delimited(&lines).is_some() {
        return Some(ContentType::Tabular);
    }
    try_detect_fixed_width(text)
}

fn md_cell_count(row: &str) -> usize {
    row.trim().trim_matches('|').split('|').count()
}

fn is_md_separator(row: &str) -> bool {
    let cells: Vec<&str> = row
        .trim()
        .trim_matches('|')
        .split('|')
        .map(str::trim)
        .filter(|cell| !cell.is_empty())
        .collect();
    cells.len() >= 2 && cells.iter().all(|cell| MD_SEP_CELL_RE.is_match(cell))
}

fn try_detect_markdown_table(lines: &[&str]) -> Option<ContentType> {
    for pair in lines.windows(2) {
        let (header, sep) = (pair[0], pair[1]);
        if header.contains('|') && is_md_separator(sep) && md_cell_count(header) >= 2 {
            return Some(ContentType::Tabular);
        }
    }
    None
}

/// CSV/TSV by a delimiter with a consistent per-line column count, guarded
/// against prose that merely contains the delimiter.
fn try_detect_delimited(lines: &[&str]) -> Option<ContentType> {
    let sample: Vec<&str> = lines.iter().take(20).copied().collect();
    if sample.len() < 3 {
        return None;
    }
    let mut best: Option<f64> = None;
    for (delim, min_consistency) in [(',', 0.85f64), ('\t', 0.7), (';', 0.85), ('|', 0.85)] {
        let counts: Vec<usize> = sample
            .iter()
            .map(|row| row.chars().filter(|c| *c == delim).count())
            .collect();
        if counts[0] == 0 {
            continue; // header row must contain the delimiter
        }
        // most common count; ties keep the first-seen value
        let mut common_count = 0usize;
        let mut freq = 0usize;
        for &count in &counts {
            let f = counts.iter().filter(|&&c| c == count).count();
            if f > freq {
                freq = f;
                common_count = count;
            }
        }
        if common_count == 0 {
            continue;
        }
        let consistency = freq as f64 / sample.len() as f64;
        let ncols = common_count + 1;
        if ncols < 2 || consistency < min_consistency {
            continue;
        }
        if looks_like_prose(&sample, delim) {
            continue;
        }
        let confidence = (0.5 + consistency * 0.3 + ncols.min(5) as f64 * 0.03).min(0.95);
        if best.map_or(true, |b| confidence > b) {
            best = Some(confidence);
        }
    }
    match best {
        Some(confidence) if confidence >= 0.6 => Some(ContentType::Tabular),
        _ => None,
    }
}

/// Prose reads like sentences and has wordy cells; either rejects the
/// delimited candidate.
fn looks_like_prose(sample: &[&str], delim: char) -> bool {
    let enders = sample
        .iter()
        .filter(|row| {
            let row = row.trim_end();
            row.ends_with('.') || row.ends_with('!') || row.ends_with('?')
        })
        .count();
    if enders as f64 / sample.len() as f64 >= 0.5 {
        return true;
    }
    let cells: Vec<&str> = sample
        .iter()
        .flat_map(|row| row.split(delim))
        .map(str::trim)
        .collect();
    let total_words: usize = cells
        .iter()
        .map(|cell| cell.split_whitespace().count())
        .sum();
    (total_words as f64 / cells.len() as f64) > 3.0
}

/// Count the column gaps shared by at least 90% of the lines. A gutter is a
/// space position strictly inside the text on nearly every line; adjacent
/// gutter positions count as one gap.
fn fixed_width_gutters(lines: &[&str]) -> usize {
    let n = lines.len();
    if n == 0 {
        return 0;
    }
    let need = (9 * n).div_ceil(10);
    let mut hits = [0usize; FIXED_WIDTH_MAX_COLS];
    for line in lines {
        let chars: Vec<char> = line.chars().collect();
        let start = chars.iter().take_while(|c| **c == ' ').count();
        for (i, ch) in chars.iter().enumerate() {
            if i >= FIXED_WIDTH_MAX_COLS {
                break;
            }
            if i >= start && *ch == ' ' {
                hits[i] += 1;
            }
        }
    }
    let mut gaps = 0;
    let mut in_gap = false;
    for &count in &hits {
        let is_gutter = count >= need;
        if is_gutter && !in_gap {
            gaps += 1;
        }
        in_gap = is_gutter;
    }
    gaps
}

fn code_like_line(line: &str) -> bool {
    line.chars()
        .last()
        .map_or(false, |c| FIXED_WIDTH_CODE_END_CHARS.contains(&c))
        || FIXED_WIDTH_CODE_START_STRINGS
            .iter()
            .any(|s| line.trim_start().starts_with(s))
}

/// Space-aligned command output (ls -l, ps aux, docker ps): >= 2 shared
/// gutters after list/prose/code guards.
fn try_detect_fixed_width(text: &str) -> Option<ContentType> {
    let lines: Vec<&str> = text
        .split('\n')
        .filter(|line| !line.trim().is_empty())
        .map(str::trim_end)
        .take(50)
        .collect();
    if lines.len() < 4 || lines.iter().any(|line| line.contains('\t')) {
        return None;
    }
    let n = lines.len();
    let list_like = lines
        .iter()
        .filter(|line| FIXED_WIDTH_LIST_RE.is_match(line))
        .count();
    if list_like as f64 / n as f64 >= 0.5 {
        return None;
    }
    let prose_ends = lines
        .iter()
        .filter(|line| FIXED_WIDTH_PROSE_END_RE.is_match(line))
        .count();
    if prose_ends as f64 / n as f64 >= 0.3 {
        return None;
    }
    let code_like = lines.iter().filter(|line| code_like_line(line)).count();
    if code_like as f64 / n as f64 >= 0.3 {
        return None;
    }
    let mut gaps = fixed_width_gutters(&lines);
    if gaps < 2 && n >= 5 {
        // a one-line preamble (e.g. 'total 480' above ls -l rows) has no
        // gaps of its own; measure the rows without it
        gaps = fixed_width_gutters(&lines[1..]);
    }
    if gaps < 2 {
        return None;
    }
    Some(ContentType::Tabular)
}

/// Structured config: TOML/INI section shapes (structural only, no parser
/// confirmation) and the YAML key/list/document-marker heuristic with prose
/// and front-matter guards.
fn try_detect_structured_config(text: &str) -> Option<ContentType> {
    let head = text.trim_start().chars().next()?;
    if head == '{' || head == '<' {
        // JSON objects and markup are never config
        return None;
    }
    let lines: Vec<&str> = text.split('\n').take(200).collect();
    let non_empty: Vec<&str> = lines
        .iter()
        .copied()
        .filter(|line| !line.trim().is_empty())
        .collect();
    if non_empty.len() < 3 {
        return None;
    }
    // comment lines are neutral for the line-share ratio
    let body: Vec<&str> = non_empty
        .iter()
        .copied()
        .filter(|line| !CONFIG_COMMENT_RE.is_match(line))
        .collect();
    if body.len() < 3 {
        return None;
    }

    // TOML / INI: a section header plus assignment-dominant body
    let sections = body
        .iter()
        .filter(|line| CONFIG_SECTION_RE.is_match(line))
        .count();
    if sections >= 1 {
        let assigns = body
            .iter()
            .filter(|line| TOML_ASSIGN_RE.is_match(line) || INI_ASSIGN_RE.is_match(line))
            .count();
        let share = (sections + assigns) as f64 / body.len() as f64;
        if assigns >= 2 && share >= 0.6 {
            let confidence = (0.7 + share * 0.25).min(0.95);
            if confidence >= 0.6 {
                return Some(ContentType::StructuredConfig);
            }
        }
    }

    // markdown front-matter guard: a --- fence closed within 60 lines and
    // followed by non-YAML content is a markdown document
    if !lines.is_empty() && lines[0].trim() == "---" {
        for idx in 1..lines.len().min(60) {
            let stripped = lines[idx].trim();
            if stripped == "---" || stripped == "..." {
                let tail: Vec<&str> = lines[idx + 1..]
                    .iter()
                    .copied()
                    .filter(|line| !line.trim().is_empty())
                    .collect();
                if !tail.is_empty() {
                    let tail_yaml = tail
                        .iter()
                        .filter(|line| YAML_KEY_RE.is_match(line) || YAML_LIST_RE.is_match(line))
                        .count();
                    if (tail_yaml as f64) / (tail.len() as f64) < 0.3 {
                        return None;
                    }
                }
                break;
            }
        }
    }

    // YAML heuristic
    let yaml_keys = body
        .iter()
        .filter(|line| YAML_KEY_RE.is_match(line))
        .count();
    let yaml_lists = body
        .iter()
        .filter(|line| YAML_LIST_RE.is_match(line) && !YAML_KEY_RE.is_match(line))
        .count();
    let doc_marks = body
        .iter()
        .filter(|line| YAML_DOC_RE.is_match(line.trim()))
        .count();
    if yaml_keys < 3 {
        return None;
    }
    let share = (yaml_keys + yaml_lists + doc_marks) as f64 / body.len() as f64;
    if share < 0.6 {
        return None;
    }
    // prose guards: config lines are short field-ish tuples
    let enders = body
        .iter()
        .filter(|line| {
            let line = line.trim_end();
            line.ends_with('.') || line.ends_with('!') || line.ends_with('?')
        })
        .count();
    if enders as f64 / body.len() as f64 >= 0.5 {
        return None;
    }
    let avg_words = body
        .iter()
        .map(|line| line.split_whitespace().count())
        .sum::<usize>() as f64
        / body.len() as f64;
    if avg_words > 8.0 {
        return None;
    }
    // structure signal: nested indentation, a document marker, or a real list
    let mut indents: HashSet<usize> = HashSet::new();
    for line in &body {
        if YAML_KEY_RE.is_match(line) || YAML_LIST_RE.is_match(line) {
            indents.insert(line.chars().take_while(|c| *c == ' ').count());
        }
    }
    if indents.len() < 2 && doc_marks == 0 && yaml_lists < 3 {
        return None;
    }
    let confidence = (0.55 + share * 0.35).min(0.9);
    if confidence < 0.6 {
        return None;
    }
    Some(ContentType::StructuredConfig)
}

/// Python's re.match equivalent: the pattern must match at position 0.
fn anchored(pattern: &Regex, line: &str) -> bool {
    pattern.find(line).map_or(false, |m| m.start() == 0)
}

/// Source code: per-language patterns, at least 3 matching lines for the
/// best language (first 100 lines).
fn try_detect_code(text: &str) -> Option<ContentType> {
    let lines: Vec<&str> = text.split('\n').take(100).collect();
    if lines.is_empty() {
        return None;
    }
    let mut scores: Vec<(&'static str, usize)> = Vec::new();
    for line in &lines {
        for (lang, patterns) in CODE_PATTERNS.iter() {
            if patterns.iter().any(|pattern| anchored(pattern, line)) {
                let mut hit = false;
                for entry in scores.iter_mut() {
                    if entry.0 == *lang {
                        entry.1 += 1;
                        hit = true;
                        break;
                    }
                }
                if !hit {
                    scores.push((lang, 1));
                }
            }
        }
    }
    let best = scores.iter().max_by_key(|(_, score)| *score)?;
    if best.1 < 3 {
        return None;
    }
    Some(ContentType::SourceCode)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_json_array() {
        let json_array = r#"[
  {"id": 1, "title": "First issue", "state": "open", "labels": ["bug"], "comments": 3},
  {"id": 2, "title": "Second issue", "state": "closed", "labels": ["bug", "ui"], "comments": 12},
  {"id": 3, "title": "Third issue", "state": "open", "labels": [], "comments": 0},
  {"id": 4, "title": "Fourth issue", "state": "open", "labels": ["feature"], "comments": 1},
  {"id": 5, "title": "Fifth issue", "state": "closed", "labels": ["docs"], "comments": 2},
  {"id": 6, "title": "Sixth issue", "state": "open", "labels": [], "comments": 5}
]"#;
        assert_eq!(detect_content_type(json_array), ContentType::JsonArray);
    }

    #[test]
    fn detects_search_results() {
        let search = "src/main.py:42:def process():\n\
src/main.py:108:    return value\n\
src/utils.py:12:import os\n\
src/utils.py:30:def helper(x):\n\
--\n\
tests/test_main.py:5:def test_process():\n\
tests/test_main.py:22:    assert result is not None\n\
docs/readme.md:1:# Project\n\
docs/readme.md:12:## Usage\n\
src/main.py:200:x = compute()";
        assert_eq!(detect_content_type(search), ContentType::SearchResults);
    }

    #[test]
    fn detects_build_output() {
        let build = "   Compiling headroom-core v0.1.0 (/home/sahen/File/Prog/headroom/crates/headroom-core)\n\
   Compiling headroom-store v0.1.0 (/home/sahen/File/Prog/headroom/crates/headroom-store)\n\
error[E0432]: unresolved import crate::missing\n\
 --> src/lib.rs:10:5\n\
  |\n\
  | use crate::missing::Thing;\n\
  |     ^ no missing in the crate root\n\
warning: unused variable: x\n\
 --> src/lib.rs:42:9\n\
  |\n\
  |     let x = 5;\n\
  |         ^ help: if this is intentional, prefix with an underscore\n\
note: this is a note\n\
help: consider adding an underscore\n\
error: could not compile headroom-core due to 2 previous errors\n\
    Finished dev profile target(s) in 12.34s\n\
    Running unittests src/lib.rs\n\
test result: ok. 42 passed; 0 failed; 1 ignored; 0 measured";
        assert_eq!(detect_content_type(build), ContentType::BuildOutput);
    }

    #[test]
    fn detects_git_diff() {
        let diff = "diff --git a/src/lib.rs b/src/lib.rs\n\
index 3a2f1b4..9c8d7e6 100644\n\
--- a/src/lib.rs\n\
+++ b/src/lib.rs\n\
@@ -10,7 +10,10 @@ pub mod ccr;\n\
 pub mod compaction;\n\
-pub mod missing;\n\
+pub mod content;\n\
+pub mod tag_protector;\n\
+pub mod tokens;\n\
 pub mod transforms;\n\
 pub mod pipeline;\n\
diff --git a/src/main.rs b/src/main.rs\n\
index 1a2b3c4..5d6e7f8 100644\n\
--- a/src/main.rs\n\
+++ b/src/main.rs\n\
@@ -1,4 +1,5 @@\n\
 fn main() {\n\
-    println!(\"hello\");\n\
+    println!(\"hello, world\");\n\
     let x = 1;\n\
 }";
        assert_eq!(detect_content_type(diff), ContentType::GitDiff);
    }

    #[test]
    fn detects_html() {
        let html = r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Report</title>
  <link rel="stylesheet" href="style.css">
  <script src="app.js"></script>
</head>
<body>
  <div class="header">
    <nav><a href="/">Home</a></nav>
  </div>
  <main>
    <section id="content">
      <p>Hello, world.</p>
      <span class="note">A note here.</span>
    </section>
  </main>
  <footer><span>2026</span></footer>
</body>
</html>"#;
        assert_eq!(detect_content_type(html), ContentType::Html);
    }

    #[test]
    fn detects_tabular_csv() {
        let csv = "id,name,category,price,stock\n\
1,Widget,tools,9.99,120\n\
2,Gadget,electronics,24.50,80\n\
3,Gizmo,tools,15.00,45\n\
4,Doohickey,home,7.25,200\n\
5,Thingamajig,garden,32.10,12";
        assert_eq!(detect_content_type(csv), ContentType::Tabular);
    }

    #[test]
    fn detects_tabular_markdown_table() {
        let markdown = "| Name | Type | Nullable |\n\
| --- | --- | --- |\n\
| id | int | no |\n\
| name | string | yes |\n\
| path | json | no |";
        assert_eq!(detect_content_type(markdown), ContentType::Tabular);
    }

    #[test]
    fn detects_tabular_fixed_width() {
        let ls = "total 48\n\
drwxr-xr-x  2 sahen sahen 4096 Nov  5 10:12 src\n\
-rw-r--r--  1 sahen sahen  220 Nov  5 10:12 Cargo.toml\n\
-rw-r--r--  1 sahen sahen  512 Nov  4 18:30 README.md\n\
drwxr-xr-x  5 sahen sahen 4096 Nov  5 09:58 target\n\
-rw-r--r--  1 sahen sahen  128 Nov  3 22:45 LICENSE";
        assert_eq!(detect_content_type(ls), ContentType::Tabular);
    }

    #[test]
    fn detects_structured_config_yaml() {
        let yaml = "name: headroom\n\
version: 1.0\n\
dependencies:\n\
  - serde\n\
  - regex\n\
  - sha2\n\
enabled: true\n\
verbose: false\n\
retries: 3";
        assert_eq!(detect_content_type(yaml), ContentType::StructuredConfig);
    }

    #[test]
    fn detects_structured_config_toml() {
        let toml = "[package]\n\
name = \"headroom-core\"\n\
version = \"0.1.0\"\n\
edition = \"2021\"\n\
\n\
[dependencies]\n\
serde = \"1\"\n\
regex = \"1\"\n\
sha2 = \"0.10\"";
        assert_eq!(detect_content_type(toml), ContentType::StructuredConfig);
    }

    #[test]
    fn detects_source_code() {
        let python = "import os\n\
import sys\n\
from dataclasses import dataclass\n\
\n\
@dataclass\n\
class Point:\n\
    x: int\n\
    y: int\n\
\n\
def distance(p1: Point, p2: Point) -> float:\n\
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5\n\
\n\
def main():\n\
    origin = Point(0, 0)\n\
    target = Point(3, 4)\n\
    print(distance(origin, target))\n\
\n\
if __name__ == \"__main__\":\n\
    main()";
        assert_eq!(detect_content_type(python), ContentType::SourceCode);
    }

    #[test]
    fn fallback_plain_text() {
        assert_eq!(detect_content_type(""), ContentType::PlainText);
        assert_eq!(detect_content_type("   \n  "), ContentType::PlainText);
        assert_eq!(detect_content_type("ok"), ContentType::PlainText);
        let prose = "The quick brown fox jumps over the lazy dog. \
Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump!";
        assert_eq!(detect_content_type(prose), ContentType::PlainText);
        // comma-bearing prose is not CSV
        assert_eq!(
            detect_content_type("Hello, friend. This is, a test, of commas, in prose."),
            ContentType::PlainText
        );
    }

    #[test]
    fn small_or_non_array_json_is_not_json_array() {
        // fewer than 5 items falls through to plain text
        assert_eq!(detect_content_type("[1, 2]"), ContentType::PlainText);
        assert_eq!(
            detect_content_type("[{\"a\": 1}, {\"b\": 2}]"),
            ContentType::PlainText
        );
        // a JSON object is not an array under this port's rule
        assert_eq!(
            detect_content_type("{\"a\": 1, \"b\": 2}"),
            ContentType::PlainText
        );
    }

    #[test]
    fn min_chars_gate_and_wire_names() {
        assert_eq!(min_chars_for_block_compression(), 500);
        assert_eq!(ContentType::JsonArray.as_str(), "json_array");
        assert_eq!(ContentType::SearchResults.as_str(), "search");
        assert_eq!(ContentType::BuildOutput.as_str(), "build");
        assert_eq!(ContentType::GitDiff.as_str(), "diff");
        assert_eq!(ContentType::Html.as_str(), "html");
        assert_eq!(ContentType::Tabular.as_str(), "tabular");
        assert_eq!(ContentType::StructuredConfig.as_str(), "structured_config");
        assert_eq!(ContentType::SourceCode.as_str(), "source_code");
        assert_eq!(ContentType::PlainText.as_str(), "text");
    }
}
