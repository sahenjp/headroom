//! Structural folding for YAML, TOML and INI configuration files.
//!
//! `ConfigCompressor::compress` first performs a *reversible* elision:
//! blank lines and full-line comments are dropped while the indentation
//! structure and every key/value survive. When the elided document is
//! still deeply repetitive — the same key sets repeated over and over, as
//! in generated Kubernetes manifests — the repeated blocks are
//! additionally *folded*: the first occurrence is kept verbatim and each
//! further repeat is replaced by a single `"[N more identical blocks]"`
//! line. Folding is lossy (the repeated values are only recoverable
//! through the CCR store), so it only triggers on runs of
//! `FOLD_MIN_BLOCKS` or more blocks.
//!
//! `ConfigCompressor::fold_to_schema` goes one step further and emits
//! only the dotted key paths (no values), one per line, for documents too
//! large to keep even in folded form. Both paths key their retrieval
//! marker on the original content hash.

use std::sync::LazyLock;

use regex::Regex;

use crate::ccr;

use super::join_lines;

/// Consecutive key-identical blocks required before a fold applies.
const FOLD_MIN_BLOCKS: usize = 3;

/// Configuration syntax detected from the content itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConfigFormat {
    /// `key: value` with indentation scoping; `#` comments.
    Yaml,
    /// `key = value` with `[section]` headers; `#` comments.
    Toml,
    /// `key=value` with `[section]` headers; `;` and `#` comments.
    Ini,
}

impl ConfigFormat {
    /// Line-start prefixes that mark a full-line comment.
    fn comment_prefixes(self) -> &'static [&'static str] {
        match self {
            ConfigFormat::Yaml | ConfigFormat::Toml => &["#"],
            ConfigFormat::Ini => &["#", ";"],
        }
    }

    /// Separator between a key and its value on one line.
    fn key_separator(self) -> char {
        match self {
            ConfigFormat::Yaml => ':',
            ConfigFormat::Toml | ConfigFormat::Ini => '=',
        }
    }
}

static YAML_KEY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"^\s*(?:-\s+)?[A-Za-z0-9_."'-]+\s*:(\s|$)"#).expect("static regex is valid")
});
static SPACED_ASSIGN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"^\s*[A-Za-z0-9_."'-]+[ \t]+="#).expect("static regex is valid")
});
static TIGHT_ASSIGN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*[A-Za-z0-9_.-]+=\S").expect("static regex is valid")
});
static SECTION_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*\[+[^\]]+\]+\s*$").expect("static regex is valid"));
static SECTION_PATH_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*\[+([^\]]+)\]+\s*$").expect("static regex is valid"));
static KV_PATH_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*([^=\[]+?)\s*=").expect("static regex is valid"));
static YAML_SCHEMA_KEY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"^(\s*)(?:-\s+)?([^:#\[\]{}]+?)\s*:(\s|$)"#).expect("static regex is valid")
});

/// Detects the config syntax by counting signature lines; the highest score
/// wins and ties resolve to YAML, the most permissive of the three.
fn detect_format(content: &str) -> ConfigFormat {
    let mut yaml = 0usize;
    let mut toml = 0usize;
    let mut ini = 0usize;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with(';') {
            continue;
        }
        if SECTION_RE.is_match(line) {
            toml += 1;
            ini += 1;
        } else {
            if YAML_KEY_RE.is_match(line) {
                yaml += 1;
            }
            if SPACED_ASSIGN_RE.is_match(line) {
                toml += 2;
            } else if TIGHT_ASSIGN_RE.is_match(line) {
                ini += 2;
            }
        }
    }
    if yaml >= toml && yaml >= ini {
        ConfigFormat::Yaml
    } else if toml >= ini {
        ConfigFormat::Toml
    } else {
        ConfigFormat::Ini
    }
}

/// Result of a `ConfigCompressor::compress` run.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigResult {
    /// Elided document, plus fold and retrieval marker when applicable.
    pub compressed: String,
    /// Number of full-line comment lines dropped.
    pub elided_comments: usize,
    /// Number of blank lines dropped.
    pub elided_blank: usize,
    /// `true` when repeated blocks were folded to their first occurrence.
    pub folded: bool,
}

/// Structural compressor for YAML / TOML / INI configuration text.
#[derive(Debug, Clone, Default)]
pub struct ConfigCompressor;

impl ConfigCompressor {
    /// Elides comments and blank lines and folds deeply repeated blocks.
    ///
    /// The returned `ConfigResult::compressed` equals `content` when
    /// nothing could be elided or folded (fail-open passthrough). A CCR
    /// marker counting the elided lines is appended when anything was
    /// dropped.
    pub fn compress(&self, content: &str) -> ConfigResult {
        let format = detect_format(content);
        let mut kept: Vec<&str> = Vec::new();
        let mut elided_comments = 0usize;
        let mut elided_blank = 0usize;
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                elided_blank += 1;
            } else if format
                .comment_prefixes()
                .iter()
                .any(|prefix| trimmed.starts_with(prefix))
            {
                elided_comments += 1;
            } else {
                kept.push(line);
            }
        }

        let elided_total = elided_comments + elided_blank;
        let (folded_lines, folded) = fold_repeated_blocks(&kept, format);
        let mut compressed = join_lines(&folded_lines, content);
        if elided_total > 0 {
            compressed.push_str(&ccr::elided_marker(
                elided_total,
                "comment/blank lines",
                &ccr::ccr_hash(content),
            ));
        }

        ConfigResult {
            compressed,
            elided_comments,
            elided_blank,
            folded,
        }
    }

    /// Emits only the dotted key paths of the document (no values), one per
    /// line, followed by a `"[config folded to schema. Retrieve original:
    /// hash=...]"` marker.
    ///
    /// Returns `None` when no key paths can be extracted (the text is not
    /// recognized as YAML/TOML/INI).
    pub fn fold_to_schema(&self, content: &str) -> Option<String> {
        let format = detect_format(content);
        let paths = match format {
            ConfigFormat::Yaml => yaml_key_paths(content),
            ConfigFormat::Toml | ConfigFormat::Ini => sectioned_key_paths(content),
        };
        if paths.is_empty() {
            return None;
        }
        let mut folded = paths.join("\n");
        folded.push_str(&format!(
            "\n[config folded to schema. Retrieve original: hash={}]",
            ccr::ccr_hash(content)
        ));
        Some(folded)
    }
}

/// Splits `lines` into blocks and folds runs of key-identical blocks.
///
/// A block boundary is a YAML line whose indentation does not exceed the
/// previous line's, or a TOML/INI `[section]` header. Two blocks are
/// key-identical when their base indentation and their extracted key
/// sequences match. Only runs of `FOLD_MIN_BLOCKS` or more are folded:
/// the first block is kept verbatim and the remaining `N - 1` are replaced
/// by a single `"[N more identical blocks]"` line at the block indentation.
fn fold_repeated_blocks(lines: &[&str], format: ConfigFormat) -> (Vec<String>, bool) {
    let mut blocks: Vec<(usize, usize)> = Vec::new(); // [start, end) line ranges
    let mut start = 0usize;
    for i in 1..lines.len() {
        let boundary = match format {
            ConfigFormat::Yaml => indent_of(lines[i]) <= indent_of(lines[i - 1]),
            _ => SECTION_RE.is_match(lines[i]),
        };
        if boundary {
            blocks.push((start, i));
            start = i;
        }
    }
    if !lines.is_empty() {
        blocks.push((start, lines.len()));
    }

    let mut out: Vec<String> = Vec::new();
    let mut folded = false;
    let mut i = 0usize;
    while i < blocks.len() {
        let run = key_identical_run(&blocks, lines, i, format);
        if run >= FOLD_MIN_BLOCKS {
            let first = blocks[i];
            let base_indent = indent_of(lines[first.0]);
            for line in &lines[first.0..first.1] {
                out.push((*line).to_string());
            }
            out.push(format!(
                "{}[{} more identical blocks]",
                " ".repeat(base_indent),
                run - 1
            ));
            folded = true;
        } else {
            for (s, e) in &blocks[i..i + run] {
                for line in &lines[*s..*e] {
                    out.push((*line).to_string());
                }
            }
        }
        i += run;
    }
    (out, folded)
}

/// Length of the run of key-identical consecutive blocks starting at `from`.
fn key_identical_run(
    blocks: &[(usize, usize)],
    lines: &[&str],
    from: usize,
    format: ConfigFormat,
) -> usize {
    let signature = block_signature(blocks[from], lines, format);
    let mut run = 1usize;
    while from + run < blocks.len()
        && block_signature(blocks[from + run], lines, format) == signature
    {
        run += 1;
    }
    run
}

/// Run-comparison signature of one block: base indentation plus the key of
/// every line ("same key sets" — values are ignored).
fn block_signature(block: (usize, usize), lines: &[&str], format: ConfigFormat) -> String {
    let (start, end) = block;
    let mut signature = format!("{}\n", indent_of(lines[start]));
    for line in &lines[start..end] {
        signature.push_str(&line_key(line, format));
        signature.push('\n');
    }
    signature
}

/// Key part of one config line: the text before the key/value separator,
/// with any YAML `- ` list marker stripped.
fn line_key(line: &str, format: ConfigFormat) -> String {
    let mut text = line.trim();
    if format == ConfigFormat::Yaml {
        if let Some(rest) = text.strip_prefix('-') {
            text = rest.trim_start();
        }
    }
    let separator = format.key_separator();
    match text.find(separator) {
        Some(idx) => text[..idx].trim().to_string(),
        None => text.to_string(),
    }
}

/// Number of leading whitespace characters of `line`.
fn indent_of(line: &str) -> usize {
    line.chars().take_while(|c| c.is_whitespace()).count()
}

/// Extracts dotted key paths from a YAML document, tracking indentation.
fn yaml_key_paths(content: &str) -> Vec<String> {
    let mut stack: Vec<(usize, String)> = Vec::new();
    let mut paths: Vec<String> = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some(caps) = YAML_SCHEMA_KEY_RE.captures(line) else {
            continue;
        };
        // The key column sits after the leading whitespace and any `- `
        // list marker, so children of list items align with their siblings.
        let indent = caps.get(2).map(|m| m.start()).unwrap_or(0);
        let key = caps[2]
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_string();
        if key.is_empty() {
            continue;
        }
        while stack.last().map(|(i, _)| *i >= indent).unwrap_or(false) {
            stack.pop();
        }
        stack.push((indent, key));
        paths.push(
            stack
                .iter()
                .map(|(_, key)| key.as_str())
                .collect::<Vec<_>>()
                .join("."),
        );
    }
    paths
}

/// Extracts dotted key paths from a TOML/INI document: `[a.b]` sections
/// prefix their keys.
fn sectioned_key_paths(content: &str) -> Vec<String> {
    let mut prefix: Vec<String> = Vec::new();
    let mut paths: Vec<String> = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with(';') {
            continue;
        }
        if let Some(caps) = SECTION_PATH_RE.captures(line) {
            prefix = caps[1]
                .split('.')
                .map(|part| unquote(part.trim()))
                .collect();
            paths.push(prefix.join("."));
            continue;
        }
        if let Some(caps) = KV_PATH_RE.captures(line) {
            let mut path = prefix.clone();
            path.push(unquote(caps[1].trim()));
            paths.push(path.join("."));
        }
    }
    paths
}

/// Strips one layer of surrounding single or double quotes.
fn unquote(text: &str) -> String {
    text.trim_matches('"').trim_matches('\'').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const YAML_DOC: &str = "# service configuration
server:
  host: localhost   # inline comment stays
  port: 8080

# tuning
retries: 3
";

    #[test]
    fn yaml_drops_comments_and_blank_lines() {
        let result = ConfigCompressor.compress(YAML_DOC);
        assert_eq!(result.elided_comments, 2);
        assert_eq!(result.elided_blank, 1);
        assert!(!result.folded);
        assert!(result.compressed.starts_with(
            "server:\n  host: localhost   # inline comment stays\n  port: 8080\nretries: 3"
        ));
        let hash = ccr::ccr_hash(YAML_DOC);
        assert!(result.compressed.contains(&format!(
            "\n[3 comment/blank lines elided. Retrieve original: hash={hash}]"
        )));
        assert!(ccr::contains_ccr_marker(&result.compressed));
    }

    #[test]
    fn toml_drops_comments_and_keeps_section_headers() {
        let doc = "# app config\ntitle = \"demo\"\n\n[server]\nhost = \"localhost\"\n\n[db]\nurl = \"postgres://x\"\n";
        let result = ConfigCompressor.compress(doc);
        assert_eq!(result.elided_comments, 1);
        assert_eq!(result.elided_blank, 2);
        assert!(result.compressed.contains("[server]"));
        assert!(result.compressed.contains("[db]"));
        assert!(result.compressed.contains("title = \"demo\""));
        assert!(!result.compressed.contains("# app config"));
    }

    #[test]
    fn ini_drops_semicolon_comments_and_keeps_sections() {
        let doc = "; globals\n[core]\ntimeout=30\n\n[log]\nlevel=info\n";
        let result = ConfigCompressor.compress(doc);
        assert_eq!(result.elided_comments, 1);
        assert_eq!(result.elided_blank, 1);
        assert!(result.compressed.contains("[core]"));
        assert!(result.compressed.contains("[log]"));
        assert!(!result.compressed.contains("; globals"));
    }

    #[test]
    fn no_op_when_nothing_to_elide() {
        let doc = "a: 1\nb: 2\n";
        let result = ConfigCompressor.compress(doc);
        assert_eq!(result.compressed, doc);
        assert_eq!(result.elided_comments, 0);
        assert_eq!(result.elided_blank, 0);
        assert!(!result.folded);
        assert!(!ccr::contains_ccr_marker(&result.compressed));
    }

    #[test]
    fn folds_repeated_yaml_blocks() {
        let mut doc = String::from("items:\n");
        for (name, value) in [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")] {
            doc.push_str(&format!("  - name: {name}\n    value: {value}\n"));
        }
        let result = ConfigCompressor.compress(&doc);
        assert!(result.folded);
        assert!(result.compressed.contains("[2 more identical blocks]"));
        // the run's first occurrence survives verbatim, repeats are elided
        assert!(result.compressed.contains("  - name: a\n    value: 1"));
        assert!(result.compressed.contains("  - name: b\n    value: 2"));
        assert!(!result.compressed.contains("- name: c"));
        assert!(!result.compressed.contains("- name: d"));
        assert_eq!(result.compressed.matches("value:").count(), 2);
    }

    #[test]
    fn does_not_fold_short_runs() {
        // three items -> only two key-identical repeats, below the threshold
        let mut doc = String::from("items:\n");
        for name in ["a", "b", "c"] {
            doc.push_str(&format!("  - name: {name}\n    value: 1\n"));
        }
        let result = ConfigCompressor.compress(&doc);
        assert!(!result.folded);
        assert!(result.compressed.contains("- name: c"));
    }

    #[test]
    fn folds_repeated_toml_sections() {
        let mut doc = String::new();
        for name in ["a", "b", "c"] {
            doc.push_str(&format!("[[packages]]\nname = \"{name}\"\nversion = \"1\"\n"));
        }
        let result = ConfigCompressor.compress(&doc);
        assert!(result.folded);
        assert!(result.compressed.contains("[2 more identical blocks]"));
        assert!(result.compressed.contains("name = \"a\""));
    }

    #[test]
    fn fold_to_schema_emits_yaml_key_paths() {
        let doc = "# c\nserver:\n  host: localhost\n  port: 8080\nretries: 3\n";
        let folded = ConfigCompressor.fold_to_schema(doc).unwrap();
        assert!(folded.starts_with("server\n"));
        assert!(folded.contains("server.host"));
        assert!(folded.contains("server.port"));
        assert!(folded.contains("\nretries"));
        assert!(folded.contains(&format!(
            "[config folded to schema. Retrieve original: hash={}]",
            ccr::ccr_hash(doc)
        )));
    }

    #[test]
    fn fold_to_schema_handles_list_item_keys() {
        let doc = "items:\n  - name: a\n    value: 1\n";
        let folded = ConfigCompressor.fold_to_schema(doc).unwrap();
        assert!(folded.contains("items.name"));
        assert!(folded.contains("items.value"));
    }

    #[test]
    fn fold_to_schema_emits_toml_section_paths() {
        let doc = "[server]\nhost = \"localhost\"\n\n[db]\nurl = \"x\"\n";
        let folded = ConfigCompressor.fold_to_schema(doc).unwrap();
        assert!(folded.contains("server.host"));
        assert!(folded.contains("db.url"));
    }

    #[test]
    fn fold_to_schema_rejects_prose() {
        assert!(ConfigCompressor
            .fold_to_schema("just some prose\nno keys here\n")
            .is_none());
    }
}
