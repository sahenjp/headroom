//! Elides the middle of long, whitespace-free (machine-generated) lines.
//!
//! Minified JavaScript, base64 blobs and similar machine output arrive as
//! very long lines with huge whitespace-free runs. Human prose never has
//! 40+-char runs of non-whitespace, so `elide_dense_lines` keeps the head
//! and tail of every long dense line and replaces the middle with a
//! `"...[N chars elided]..."` note, then appends a CCR retrieval marker
//! keyed on the original content. Returns `None` when nothing was elided.

use crate::ccr;

/// A whitespace-free run must reach this many chars for its line to count
/// as dense (prose words stay far below this bound).
const DENSE_RUN_MIN_CHARS: usize = 40;

/// Configuration for `elide_dense_lines`.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseLineConfig {
    /// Lines shorter than this many chars are never touched.
    pub min_line_chars: usize,
    /// Leading chars kept from an elided line.
    pub keep_head: usize,
    /// Trailing chars kept from an elided line.
    pub keep_tail: usize,
}

impl Default for DenseLineConfig {
    fn default() -> Self {
        Self {
            min_line_chars: 200,
            keep_head: 40,
            keep_tail: 20,
        }
    }
}

/// Result of an `elide_dense_lines` run.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseResult {
    /// Text with dense line middles replaced and the retrieval marker
    /// appended.
    pub compressed: String,
    /// Number of lines whose middle was elided.
    pub elided: usize,
}

/// Elides the middles of long whitespace-free lines.
///
/// A line qualifies when it is at least `config.min_line_chars` chars long
/// and contains a whitespace-free run of at least 40 chars (a line with no
/// whitespace at all trivially qualifies). Returns `None` when no line was
/// elided.
pub fn elide_dense_lines(text: &str, config: &DenseLineConfig) -> Option<DenseResult> {
    let mut out: Vec<String> = Vec::new();
    let mut elided = 0usize;
    for line in text.lines() {
        match elide_line(line, config) {
            Some(replacement) => {
                out.push(replacement);
                elided += 1;
            }
            None => out.push(line.to_string()),
        }
    }
    if elided == 0 {
        return None;
    }
    let mut compressed = out.join("\n");
    if text.ends_with('\n') && !out.is_empty() {
        compressed.push('\n');
    }
    compressed.push_str(&ccr::elided_marker(
        elided,
        "dense machine-generated lines",
        &ccr::ccr_hash(text),
    ));
    Some(DenseResult { compressed, elided })
}

/// The elided form of one line, or `None` when the line stays as-is.
fn elide_line(line: &str, config: &DenseLineConfig) -> Option<String> {
    let chars: Vec<char> = line.chars().collect();
    if chars.len() < config.min_line_chars || longest_whitespace_free_run(&chars) < DENSE_RUN_MIN_CHARS
    {
        return None;
    }
    let head_end = config.keep_head.min(chars.len());
    let tail_start = chars.len().saturating_sub(config.keep_tail).max(head_end);
    let mid = tail_start - head_end;
    if mid == 0 {
        return None;
    }
    let head: String = chars[..head_end].iter().collect();
    let tail: String = chars[tail_start..].iter().collect();
    Some(format!("{head}...[{mid} chars elided]...{tail}"))
}

/// Length in chars of the longest run of non-whitespace characters.
fn longest_whitespace_free_run(chars: &[char]) -> usize {
    let mut longest = 0usize;
    let mut current = 0usize;
    for c in chars {
        if c.is_whitespace() {
            current = 0;
        } else {
            current += 1;
            longest = longest.max(current);
        }
    }
    longest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elides_minified_javascript_line() {
        let js = format!("function a(){{return {};}}", "x".repeat(300));
        let text = format!("let ok = 1;\n{js}\nlet done = 2;\n");
        let result = elide_dense_lines(&text, &DenseLineConfig::default()).unwrap();
        assert_eq!(result.elided, 1);
        let mid = js.chars().count() - 60;
        assert!(result.compressed.contains(&format!("...[{mid} chars elided]...")));
        assert!(result.compressed.starts_with("let ok = 1;\n"));
        assert!(result.compressed.contains(&js[..40]));
        assert!(result.compressed.contains(&js[js.len() - 20..]));
        assert!(result.compressed.contains("let done = 2;"));
        let hash = ccr::ccr_hash(&text);
        assert!(result.compressed.contains(&format!(
            "\n[1 dense machine-generated lines elided. Retrieve original: hash={hash}]"
        )));
    }

    #[test]
    fn elides_base64_line() {
        let b64 =
            "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdHV2d3h5ejAxMjM0NTY3ODk="
                .repeat(4);
        let text = format!("{b64}\n");
        let result = elide_dense_lines(&text, &DenseLineConfig::default()).unwrap();
        assert_eq!(result.elided, 1);
        assert!(result.compressed.starts_with(&b64[..40]));
        assert!(result.compressed.contains(&b64[b64.len() - 20..]));
    }

    #[test]
    fn normal_prose_is_untouched() {
        let prose = "this is a normal sentence with plenty of spaces and short words ".repeat(4);
        let text = format!("{prose}\n");
        assert!(elide_dense_lines(&text, &DenseLineConfig::default()).is_none());
    }

    #[test]
    fn respects_min_line_chars() {
        let dense = "a".repeat(150);
        let text = format!("{dense}\n");
        assert!(elide_dense_lines(&text, &DenseLineConfig::default()).is_none());
        let config = DenseLineConfig {
            min_line_chars: 100,
            ..DenseLineConfig::default()
        };
        let result = elide_dense_lines(&text, &config).unwrap();
        assert_eq!(result.elided, 1);
    }

    #[test]
    fn keeps_head_and_tail_exactly() {
        let line = "z".repeat(201);
        let result = elide_dense_lines(&line, &DenseLineConfig::default()).unwrap();
        let expected = format!(
            "{}...[{} chars elided]...{}",
            "z".repeat(40),
            141,
            "z".repeat(20)
        );
        assert!(result.compressed.starts_with(&expected));
    }

    #[test]
    fn only_dense_lines_change() {
        let dense = "y".repeat(250);
        let text = format!("plain prose line\n{dense}\nanother plain line\n");
        let result = elide_dense_lines(&text, &DenseLineConfig::default()).unwrap();
        assert_eq!(result.elided, 1);
        assert!(result.compressed.contains("plain prose line\n"));
        assert!(result.compressed.contains("\nanother plain line\n"));
    }
}
