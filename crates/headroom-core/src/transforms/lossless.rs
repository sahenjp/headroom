//! Lossless, format-native folds: byte-safe cleanup with no markers.
//!
//! Unlike the lossy transforms, these helpers never drop information that
//! cannot be reconstructed locally and never append CCR markers:
//!
//! * `fold_grep_heading` rewrites runs of `path:line:content` rows into
//!   one `path:` heading plus indented `"  line:content"` rows —
//!   reversible by concatenating the heading with each row.
//! * `strip_ansi` removes ANSI escape sequences.
//! * `collapse_log_runs` folds runs of 3+ repeated log lines into the
//!   first line plus `"[repeated N times]"` (trailing digits that differ
//!   between the lines are ignored when detecting the run).
//! * `minify_json` re-serializes JSON compactly when parsing succeeds and
//!   the compact form is strictly shorter; key order is preserved via the
//!   `preserve_order` serde_json feature.

use super::join_lines;

/// Folds runs of `path:line:content` rows for the same path into one
/// `"path:"` heading plus indented rows.
///
/// A row parses as grep output when it splits into a non-empty path, an
/// all-digit line number and the remaining content. Consecutive rows for
/// the same path (two or more) become a heading plus `"  line:content"`
/// rows; anything else passes through verbatim.
pub fn fold_grep_heading(text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut group: Option<(String, Vec<String>)> = None;

    for line in text.lines() {
        match parse_grep_line(line) {
            Some((path, lineno, content)) => {
                let row = format!("{lineno}:{content}");
                let continues = matches!(&group, Some((current, _)) if *current == path);
                if continues {
                    if let Some((_, rows)) = group.as_mut() {
                        rows.push(row);
                    }
                } else {
                    flush_group(&mut out, group.take());
                    group = Some((path.to_string(), vec![row]));
                }
            }
            None => {
                flush_group(&mut out, group.take());
                out.push(line.to_string());
            }
        }
    }
    flush_group(&mut out, group.take());
    join_lines(&out, text)
}

/// Emits one folded group: a `"path:"` heading plus indented rows when the
/// group has two or more rows, the original lines otherwise.
fn flush_group(out: &mut Vec<String>, group: Option<(String, Vec<String>)>) {
    let Some((path, rows)) = group else {
        return;
    };
    if rows.len() >= 2 {
        out.push(format!("{path}:"));
        for row in rows {
            out.push(format!("  {row}"));
        }
    } else {
        for row in rows {
            out.push(format!("{path}:{row}"));
        }
    }
}

/// Splits a `path:line:content` grep row; `None` for anything else.
fn parse_grep_line(line: &str) -> Option<(&str, &str, &str)> {
    let (path, rest) = line.split_once(':')?;
    if path.is_empty() {
        return None;
    }
    let (lineno, content) = rest.split_once(':')?;
    if lineno.is_empty() || !lineno.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    Some((path, lineno, content))
}

/// Removes ANSI escape sequences: CSI (`ESC [ params letter`), OSC
/// (`ESC ] ... BEL`) and charset selection (`ESC (B` and friends).
/// Malformed escapes are kept verbatim (fail-open).
pub fn strip_ansi(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == 0x1b {
            if i + 1 < bytes.len() && bytes[i + 1] == b'[' {
                // CSI: ESC [ <digits and semicolons> <final alpha byte>
                let mut j = i + 2;
                while j < bytes.len() && (bytes[j].is_ascii_digit() || bytes[j] == b';') {
                    j += 1;
                }
                if j < bytes.len() && bytes[j].is_ascii_alphabetic() {
                    i = j + 1;
                    continue;
                }
            } else if i + 1 < bytes.len() && bytes[i + 1] == b']' {
                // OSC: ESC ] ... BEL
                if let Some(bel) = bytes[i + 2..].iter().position(|&byte| byte == 0x07) {
                    i = i + 2 + bel + 1;
                    continue;
                }
            } else if i + 2 < bytes.len()
                && (bytes[i + 1] == b'(' || bytes[i + 1] == b')')
                && matches!(bytes[i + 2], b'0' | b'1' | b'2' | b'B')
            {
                // charset selection: ESC ( B, ESC ) 0, ...
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    // Only ASCII bytes were removed, so the remainder is still valid UTF-8.
    String::from_utf8(out).unwrap_or_else(|_| text.to_string())
}

/// Folds runs of 3+ consecutive repeated log lines into the first line plus
/// a `"[repeated N times]"` note (`N` is the total number of occurrences).
///
/// Lines compare equal after trimming trailing digits, so repeated lines
/// that differ only in a trailing counter collapse together. Blank lines
/// are never collapsed.
pub fn collapse_log_runs(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        let collapsible = !lines[i].trim().is_empty();
        let key = run_key(lines[i]);
        let mut run = 1usize;
        if collapsible {
            while i + run < lines.len() && run_key(lines[i + run]) == key {
                run += 1;
            }
        }
        if run >= 3 {
            out.push(lines[i].to_string());
            out.push(format!("[repeated {run} times]"));
        } else {
            for line in &lines[i..i + run] {
                out.push((*line).to_string());
            }
        }
        i += run;
    }
    join_lines(&out, text)
}

/// Run-comparison key: the line without its trailing digits.
fn run_key(line: &str) -> &str {
    line.trim_end()
        .trim_end_matches(|c: char| c.is_ascii_digit())
        .trim_end()
}

/// Compactly re-serializes JSON when it parses and the compact form is
/// strictly shorter; `None` otherwise (fail-open passthrough).
///
/// Key order is preserved: the workspace enables serde_json's
/// `preserve_order` feature.
pub fn minify_json(text: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let compact = serde_json::to_string(&value).ok()?;
    (compact.len() < text.len()).then_some(compact)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_grep_heading_groups_same_path() {
        let text = "src/main.rs:10:fn main() {}\nsrc/main.rs:12:    println!(\"hi\");\nsrc/lib.rs:1:pub mod x;\n";
        let out = fold_grep_heading(text);
        assert!(out.contains("src/main.rs:\n  10:fn main() {}\n  12:    println!(\"hi\");"));
        assert!(out.contains("src/lib.rs:1:pub mod x;"));
    }

    #[test]
    fn fold_grep_heading_is_reversible_by_concatenation() {
        let lines = [
            "src/a.rs:1:one".to_string(),
            "src/a.rs:2:two".to_string(),
            "src/b.rs:3:three".to_string(),
        ];
        let text = format!("{}\n", lines.join("\n"));
        let folded = fold_grep_heading(&text);

        let mut rebuilt: Vec<String> = Vec::new();
        let mut current_path: Option<String> = None;
        for line in folded.lines() {
            if let Some(row) = line.strip_prefix("  ") {
                let path = current_path.as_deref().expect("row follows a heading");
                rebuilt.push(format!("{path}:{row}"));
            } else if line.ends_with(':') && !line.contains(char::is_whitespace) {
                current_path = Some(line.trim_end_matches(':').to_string());
            } else {
                rebuilt.push(line.to_string());
            }
        }
        assert_eq!(rebuilt, lines);
    }

    #[test]
    fn fold_grep_heading_keeps_single_rows_and_prose() {
        let text = "just prose\nsrc/a.rs:1:only one row\nmore prose\n";
        assert_eq!(fold_grep_heading(text), text);
    }

    #[test]
    fn strip_ansi_removes_csi_osc_and_charset_sequences() {
        assert_eq!(strip_ansi("\x1b[31mred\x1b[0m"), "red");
        assert_eq!(strip_ansi("a\x1b[1;32mgreen"), "agreen");
        assert_eq!(strip_ansi("\x1b]0;title\x07after"), "after");
        assert_eq!(strip_ansi("\x1b(Bplain"), "plain");
        assert_eq!(strip_ansi("no escapes here"), "no escapes here");
    }

    #[test]
    fn strip_ansi_keeps_malformed_escapes_verbatim() {
        assert_eq!(strip_ansi("\x1b[31"), "\x1b[31");
        assert_eq!(strip_ansi("\x1b"), "\x1b");
        assert_eq!(strip_ansi("\x1b]unterminated"), "\x1b]unterminated");
    }

    #[test]
    fn collapse_log_runs_folds_repeats() {
        let text = "GET /api 200\nGET /api 200\nGET /api 200\ndone\n";
        assert_eq!(collapse_log_runs(text), "GET /api 200\n[repeated 3 times]\ndone\n");
    }

    #[test]
    fn collapse_log_runs_folds_trailing_digit_variants() {
        let text = "req 200\nreq 201\nreq 202\nok\n";
        assert_eq!(collapse_log_runs(text), "req 200\n[repeated 3 times]\nok\n");
    }

    #[test]
    fn collapse_log_runs_keeps_pairs_and_blank_lines() {
        let text = "a\na\n\nb\nb\n";
        assert_eq!(collapse_log_runs(text), text);
    }

    #[test]
    fn minify_json_compacts_and_preserves_key_order() {
        let pretty = "{\n  \"b\": 1,\n  \"a\": [1, 2, 3]\n}\n";
        let out = minify_json(pretty).unwrap();
        assert_eq!(out, "{\"b\":1,\"a\":[1,2,3]}");

        let nested = "{\n  \"z\": {\"m\": 1, \"k\": 2},\n  \"a\": 3\n}";
        assert_eq!(minify_json(nested).unwrap(), "{\"z\":{\"m\":1,\"k\":2},\"a\":3}");
    }

    #[test]
    fn minify_json_rejects_non_shortening_or_invalid_input() {
        assert!(minify_json("{\"a\":1}").is_none()); // already compact
        assert!(minify_json("not json").is_none());
        assert!(minify_json("").is_none());
    }
}
