//! Tabular text detection and row elision for CSV, TSV and markdown tables.
//!
//! `is_tabular` recognizes a table when every non-blank line carries the
//! same delimiter count of 2 or more (on at least 3 lines) or when a
//! markdown `|---|` separator row is present.
//! `TabularCompressor::compress` keeps the header row (and the markdown
//! separator) plus the first and last data rows within a row budget and
//! elides the middle with a `"... (N rows elided)"` line plus a CCR
//! retrieval marker keyed on the original content. Small tables pass
//! through unchanged (`None`).

use crate::ccr;

use super::join_lines;

/// Row budget used when a caller has no preference.
pub const DEFAULT_MAX_ROWS: usize = 30;

/// Delimiter candidates, tried in priority order.
const DELIMITERS: [u8; 3] = [b',', b'\t', b'|'];

/// Result of a `TabularCompressor::compress` run.
#[derive(Debug, Clone, PartialEq)]
pub struct TabularResult {
    /// Compressed table: header, kept rows, elision note and CCR marker.
    pub compressed: String,
    /// Table lines in the input (header and markdown separator included).
    pub rows_total: usize,
    /// Table lines kept (header, markdown separator and kept data rows).
    pub rows_kept: usize,
}

/// Row elider for delimited and markdown tables.
#[derive(Debug, Clone, Default)]
pub struct TabularCompressor;

/// Returns `true` when `text` looks like a delimited table (every
/// non-blank line carries the same delimiter count of 2 or more, on at
/// least 3 lines) or a markdown table (a `|---|` separator row plus at
/// least two pipe rows).
pub fn is_tabular(text: &str) -> bool {
    let lines = nonempty_lines(text);
    if lines.len() < 3 {
        return false;
    }
    if DELIMITERS
        .iter()
        .any(|&delim| consistent_count(&lines, delim).is_some())
    {
        return true;
    }
    is_markdown_table(&lines)
}

/// Returns the detected field delimiter: `b','`, `b'\t'` or `b'|'`.
///
/// A markdown separator row forces `b'|'` even when the pipe counts are
/// ragged. Returns `None` when no delimiter is used consistently.
pub fn detect_delimiter(text: &str) -> Option<u8> {
    let lines = nonempty_lines(text);
    for &delim in &DELIMITERS {
        if consistent_count(&lines, delim).is_some() {
            return Some(delim);
        }
    }
    if lines.iter().any(|line| is_markdown_separator(line)) {
        return Some(b'|');
    }
    None
}

impl TabularCompressor {
    /// Keeps the header plus the first and last data rows within
    /// `max_rows` total table lines and elides the middle.
    ///
    /// The markdown `|---|` separator row (when present) is always kept.
    /// Returns `None` when the text is not tabular or when no data row
    /// would be dropped (fail-open passthrough).
    pub fn compress(&self, content: &str, max_rows: usize) -> Option<TabularResult> {
        if !is_tabular(content) {
            return None;
        }
        let lines: Vec<&str> = content.lines().collect();
        let header_len = match lines.iter().position(|line| is_markdown_separator(line)) {
            Some(sep_idx) => sep_idx + 1, // header row(s) plus the separator
            None => 1,
        };
        let data = &lines[header_len..];
        let budget = max_rows.saturating_sub(header_len);
        if data.len() <= budget {
            return None;
        }
        let keep_first = budget / 2;
        let keep_last = budget - keep_first;
        let dropped = data.len() - budget;

        let mut out: Vec<String> = lines[..header_len]
            .iter()
            .map(|line| (*line).to_string())
            .collect();
        out.extend(data[..keep_first].iter().map(|line| (*line).to_string()));
        out.push(format!("... ({dropped} rows elided)"));
        out.extend(
            data[data.len() - keep_last..]
                .iter()
                .map(|line| (*line).to_string()),
        );

        let mut compressed = join_lines(&out, content);
        compressed.push_str(&ccr::elided_marker(
            dropped,
            "rows",
            &ccr::ccr_hash(content),
        ));

        Some(TabularResult {
            rows_total: lines.len(),
            rows_kept: header_len + keep_first + keep_last,
            compressed,
        })
    }
}

/// The non-blank lines of `text`.
fn nonempty_lines(text: &str) -> Vec<&str> {
    text.lines().filter(|line| !line.trim().is_empty()).collect()
}

/// The per-line count of `delim` when it is the same (>= 2) on every line.
fn consistent_count(lines: &[&str], delim: u8) -> Option<usize> {
    let mut count: Option<usize> = None;
    for line in lines {
        let current = line
            .as_bytes()
            .iter()
            .filter(|&&byte| byte == delim)
            .count();
        match count {
            None => count = Some(current),
            Some(previous) if previous == current => {}
            _ => return None,
        }
    }
    let count = count?;
    (count >= 2).then_some(count)
}

/// True for a markdown alignment row such as `| --- | :---: |`.
fn is_markdown_separator(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with('|')
        && trimmed.ends_with('|')
        && trimmed.contains('-')
        && trimmed
            .chars()
            .all(|c| c == '|' || c == '-' || c == ':' || c.is_whitespace())
}

/// True for a markdown table: a separator row plus at least two pipe rows.
fn is_markdown_table(lines: &[&str]) -> bool {
    lines.iter().any(|line| is_markdown_separator(line))
        && lines
            .iter()
            .filter(|line| {
                let trimmed = line.trim();
                trimmed.starts_with('|') && trimmed.ends_with('|')
            })
            .count()
            >= 3
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3-column CSV with a header and `rows` data rows.
    fn csv_rows(rows: usize) -> String {
        let mut csv = String::from("id,name,value\n");
        for i in 0..rows {
            csv.push_str(&format!("{i},row{i},{i}00\n"));
        }
        csv
    }

    #[test]
    fn detects_csv_and_elides_middle_rows() {
        let content = csv_rows(40);
        assert!(is_tabular(&content));
        assert_eq!(detect_delimiter(&content), Some(b','));

        let result = TabularCompressor.compress(&content, 10).unwrap();
        assert_eq!(result.rows_total, 41);
        assert_eq!(result.rows_kept, 10);
        assert!(result.compressed.starts_with("id,name,value\n"));
        assert!(result.compressed.contains("\n0,row0,000\n"));
        assert!(result.compressed.contains("\n39,row39,3900"));
        assert!(!result.compressed.contains("\n20,row20,"));
        assert!(result.compressed.contains("... (31 rows elided)"));
        let hash = ccr::ccr_hash(&content);
        assert!(result.compressed.contains(&format!(
            "\n[31 rows elided. Retrieve original: hash={hash}]"
        )));
    }

    #[test]
    fn compresses_tsv() {
        let content = "a\tb\tc\n1\t2\t3\n4\t5\t6\n7\t8\t9\n10\t11\t12\n13\t14\t15\n16\t17\t18\n";
        assert!(is_tabular(&content));
        assert_eq!(detect_delimiter(&content), Some(b'\t'));

        let result = TabularCompressor.compress(content, 4).unwrap();
        assert_eq!(result.rows_total, 7);
        assert_eq!(result.rows_kept, 4);
        assert!(result.compressed.starts_with("a\tb\tc\n"));
        assert!(result
            .compressed
            .contains("\n1\t2\t3\n... (3 rows elided)\n13\t14\t15\n16\t17\t18"));
        assert!(result.compressed.contains("... (3 rows elided)"));
    }

    #[test]
    fn compresses_markdown_tables_keeping_separator() {
        let mut content = String::from("| id | name |\n|----|------|\n");
        for i in 0..12 {
            content.push_str(&format!("| {i} | item{i} |\n"));
        }
        assert!(is_tabular(&content));
        assert_eq!(detect_delimiter(&content), Some(b'|'));

        let result = TabularCompressor.compress(&content, 6).unwrap();
        assert_eq!(result.rows_total, 14);
        assert_eq!(result.rows_kept, 6);
        assert!(result.compressed.contains("|----|------|"));
        assert!(result.compressed.contains("... (8 rows elided)"));
        assert!(result.compressed.contains("| 11 | item11 |"));
    }

    #[test]
    fn small_table_passes_through() {
        let content = "a,b,c\n1,2,3\n4,5,6\n";
        assert!(is_tabular(&content));
        assert!(TabularCompressor.compress(content, DEFAULT_MAX_ROWS).is_none());
    }

    #[test]
    fn prose_is_not_tabular() {
        assert!(!is_tabular("hello world\nanother line\nmore text here"));
        assert!(!is_tabular("a,b,c\n1,2,3")); // too few lines
        assert!(!is_tabular("hello, world\nfoo, bar\nbaz, qux")); // single commas
        assert_eq!(detect_delimiter("hello, world\nfoo, bar\nbaz, qux"), None);
    }

    #[test]
    fn detect_delimiter_prefers_commas_then_tabs_then_pipes() {
        assert_eq!(detect_delimiter("a,b,c\n1,2,3\n4,5,6\n"), Some(b','));
        assert_eq!(detect_delimiter("a\tb\tc\n1\t2\t3\n4\t5\t6\n"), Some(b'\t'));
        assert_eq!(
            detect_delimiter("| a | b | c |\n| 1 | 2 | 3 |\n| 4 | 5 | 6 |\n"),
            Some(b'|')
        );
        assert_eq!(detect_delimiter("no delimiters\nin this text\nat all"), None);
    }
}
