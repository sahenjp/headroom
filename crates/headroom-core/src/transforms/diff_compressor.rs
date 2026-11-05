//! Compressor for unified git diffs.
//!
//! [`DiffCompressor::compress`] parses file sections (`diff --git` or bare
//! `--- a/` starts) and their `@@` hunks, then keeps up to
//! [`DiffCompressorConfig::max_hunks_per_file`] hunks per file — when
//! truncating, the hunks with the largest +/- churn survive (ties broken by
//! hunk index) — trims each hunk's context lines to
//! [`DiffCompressorConfig::max_context_lines`] around the +/- runs (eliding
//! middle context with `"..."`), and caps the file count. Additions and
//! deletions are always preserved.

use crate::ccr;

use super::join_lines;

/// Configuration for [`DiffCompressor`].
#[derive(Debug, Clone, PartialEq)]
pub struct DiffCompressorConfig {
    /// Context lines kept around each +/- run (leading/trailing of a run).
    pub max_context_lines: usize,
    /// Maximum hunks kept per file.
    pub max_hunks_per_file: usize,
    /// Maximum number of file sections kept.
    pub max_files: usize,
    /// Always keep `+` addition lines.
    pub always_keep_additions: bool,
    /// Always keep `-` deletion lines.
    pub always_keep_deletions: bool,
    /// Append a CCR retrieval marker when something was dropped.
    pub enable_ccr: bool,
    /// Minimum original lines before a CCR marker is considered.
    pub min_lines_for_ccr: usize,
}

impl Default for DiffCompressorConfig {
    fn default() -> Self {
        Self {
            max_context_lines: 2,
            max_hunks_per_file: 10,
            max_files: 20,
            always_keep_additions: true,
            always_keep_deletions: true,
            enable_ccr: true,
            min_lines_for_ccr: 50,
        }
    }
}

/// Compressor for unified git diffs.
#[derive(Debug, Clone, Default)]
pub struct DiffCompressor {
    /// Tuning knobs.
    pub config: DiffCompressorConfig,
}

/// Result of a [`DiffCompressor::compress`] run.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffCompressionResult {
    /// Compressed diff; the original text when nothing was dropped.
    pub compressed: String,
    /// Line count of the input.
    pub original_line_count: usize,
    /// Line count of the compressed body (excluding the CCR marker line).
    pub compressed_line_count: usize,
    /// Number of file sections in the input.
    pub files_affected: usize,
    /// `+` lines in the input (all files, before compression).
    pub additions: usize,
    /// `-` lines in the input (all files, before compression).
    pub deletions: usize,
    /// Hunks kept (in the files that survived the file cap).
    pub hunks_kept: usize,
    /// Hunks removed by the per-file caps and the file cap.
    pub hunks_removed: usize,
    /// CCR cache key of the original content (`None` without a marker).
    pub cache_key: Option<String>,
}

/// Kind of a line inside a hunk body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BodyKind {
    Context,
    Addition,
    Deletion,
    Meta,
}

/// One `@@` hunk.
#[derive(Debug, Clone)]
struct Hunk {
    header: String,
    body: Vec<(BodyKind, String)>,
}

/// One file section: header lines plus hunks.
#[derive(Debug, Clone)]
struct FileSection {
    header: Vec<String>,
    hunks: Vec<Hunk>,
}

/// One emitted hunk line, or a `"..."` context elision.
#[derive(Debug, Clone, PartialEq)]
enum HunkOut {
    Line(String),
    Elide,
}

impl DiffCompressor {
    /// Creates a compressor with the given configuration.
    pub fn new(config: DiffCompressorConfig) -> Self {
        Self { config }
    }

    /// Compresses a unified git diff.
    ///
    /// `context` is accepted for signature parity with the other transforms
    /// and reserved for future hunk-relevance scoring; hunk selection is
    /// churn-based. Returns the input unchanged when nothing is dropped.
    pub fn compress(&self, content: &str, context: &str) -> DiffCompressionResult {
        let _ = context;
        let lines: Vec<&str> = content.lines().collect();
        let original_line_count = lines.len();
        let (_preamble, sections) = parse_diff(&lines);

        let files_affected = sections.len();
        let additions = count_body_kind(&sections, BodyKind::Addition);
        let deletions = count_body_kind(&sections, BodyKind::Deletion);
        let total_hunks: usize = sections.iter().map(|s| s.hunks.len()).sum();

        if sections.is_empty() {
            // Not a recognizable diff: leave it alone.
            return DiffCompressionResult {
                compressed: content.to_string(),
                original_line_count,
                compressed_line_count: original_line_count,
                files_affected,
                additions,
                deletions,
                hunks_kept: 0,
                hunks_removed: 0,
                cache_key: None,
            };
        }

        let files_kept = self.config.max_files.min(sections.len());
        let mut out: Vec<String> = Vec::new();
        let mut hunks_kept = 0usize;
        let mut modified = false;

        for section in sections.iter().take(files_kept) {
            out.extend(section.header.iter().cloned());
            let selected: Vec<usize> = if section.hunks.len() > self.config.max_hunks_per_file {
                modified = true;
                // Largest churn first; ties broken by the lower hunk index.
                let mut indices: Vec<usize> = (0..section.hunks.len()).collect();
                indices.sort_by(|a, b| {
                    hunk_churn(&section.hunks[*b])
                        .cmp(&hunk_churn(&section.hunks[*a]))
                        .then(a.cmp(b))
                });
                indices.truncate(self.config.max_hunks_per_file);
                indices.sort_unstable();
                indices
            } else {
                (0..section.hunks.len()).collect()
            };
            let elided_hunks = section.hunks.len() - selected.len();

            for &hi in &selected {
                let hunk = &section.hunks[hi];
                out.push(hunk.header.clone());
                let (body_out, any_elided) =
                    trim_hunk_context(&hunk.body, self.config.max_context_lines);
                if any_elided {
                    modified = true;
                }
                for item in body_out {
                    out.push(match item {
                        HunkOut::Line(s) => s,
                        HunkOut::Elide => "...".to_string(),
                    });
                }
            }
            if elided_hunks > 0 {
                out.push(format!(
                    "({elided_hunks} hunk{} elided)",
                    if elided_hunks == 1 { "" } else { "s" }
                ));
            }
            hunks_kept += selected.len();
        }

        if files_kept < sections.len() {
            modified = true;
            let extra_files = sections.len() - files_kept;
            out.push(format!(
                "… {extra_files} more file{}",
                if extra_files == 1 { "" } else { "s" }
            ));
        }
        let hunks_removed = total_hunks - hunks_kept;

        if !modified {
            return DiffCompressionResult {
                compressed: content.to_string(),
                original_line_count,
                compressed_line_count: original_line_count,
                files_affected,
                additions,
                deletions,
                hunks_kept: total_hunks,
                hunks_removed: 0,
                cache_key: None,
            };
        }

        let compressed_line_count = out.len();
        let mut compressed = join_lines(&out, content);
        let mut cache_key = None;
        if self.config.enable_ccr && original_line_count >= self.config.min_lines_for_ccr {
            let hash = ccr::ccr_hash(content);
            compressed.push_str(&format!(
                "\n[{compressed_line_count} of {original_line_count} diff lines shown. \
                 Retrieve original: hash={hash}]"
            ));
            cache_key = Some(hash);
        }

        DiffCompressionResult {
            compressed,
            original_line_count,
            compressed_line_count,
            files_affected,
            additions,
            deletions,
            hunks_kept,
            hunks_removed,
            cache_key,
        }
    }
}

/// Parses unified diff lines into preamble lines and file sections.
fn parse_diff(lines: &[&str]) -> (Vec<String>, Vec<FileSection>) {
    let mut preamble: Vec<String> = Vec::new();
    let mut sections: Vec<FileSection> = Vec::new();
    let mut current: Option<FileSection> = None;
    let mut in_hunk = false;

    for (i, line) in lines.iter().enumerate() {
        let line = *line;
        if line.starts_with("diff --git ") {
            flush_section(&mut current, &mut sections);
            current = Some(FileSection {
                header: vec![line.to_string()],
                hunks: Vec::new(),
            });
            in_hunk = false;
        } else if line.starts_with("@@") {
            if current.is_none() {
                current = Some(FileSection {
                    header: Vec::new(),
                    hunks: Vec::new(),
                });
            }
            let section = current.as_mut().expect("section just created");
            section.hunks.push(Hunk {
                header: line.to_string(),
                body: Vec::new(),
            });
            in_hunk = true;
        } else if !in_hunk
            && current.is_none()
            && line.starts_with("--- ")
            && i + 1 < lines.len()
            && lines[i + 1].starts_with("+++ ")
        {
            // Headerless file start: "--- a/x" without a preceding "diff --git".
            current = Some(FileSection {
                header: vec![line.to_string()],
                hunks: Vec::new(),
            });
        } else if in_hunk {
            let section = current.as_mut().expect("section exists while in hunk");
            let hunk = section.hunks.last_mut().expect("open hunk");
            hunk.body.push((classify_body_line(line), line.to_string()));
        } else if let Some(section) = current.as_mut() {
            section.header.push(line.to_string());
        } else {
            preamble.push(line.to_string());
        }
    }
    flush_section(&mut current, &mut sections);
    (preamble, sections)
}

fn flush_section(current: &mut Option<FileSection>, sections: &mut Vec<FileSection>) {
    if let Some(section) = current.take() {
        sections.push(section);
    }
}

fn classify_body_line(line: &str) -> BodyKind {
    if line.starts_with('+') {
        BodyKind::Addition
    } else if line.starts_with('-') {
        BodyKind::Deletion
    } else if line.starts_with(' ') {
        BodyKind::Context
    } else {
        BodyKind::Meta // e.g. "\ No newline at end of file"
    }
}

/// +/- churn of a hunk.
fn hunk_churn(hunk: &Hunk) -> usize {
    hunk.body
        .iter()
        .filter(|(kind, _)| matches!(kind, BodyKind::Addition | BodyKind::Deletion))
        .count()
}

fn count_body_kind(sections: &[FileSection], kind: BodyKind) -> usize {
    sections
        .iter()
        .flat_map(|s| s.hunks.iter())
        .flat_map(|h| h.body.iter())
        .filter(|(k, _)| *k == kind)
        .count()
}

/// Trims context runs to `max_context` lines adjacent to +/- changes,
/// eliding the remainder with `"..." ` markers.
///
/// A leading run keeps its tail (the lines next to the following change), a
/// trailing run keeps its head, and a middle run keeps both sides.
fn trim_hunk_context(body: &[(BodyKind, String)], max_context: usize) -> (Vec<HunkOut>, bool) {
    let mut out: Vec<HunkOut> = Vec::new();
    let mut any_elided = false;
    let mut i = 0usize;
    while i < body.len() {
        if body[i].0 != BodyKind::Context {
            out.push(HunkOut::Line(body[i].1.clone()));
            i += 1;
            continue;
        }
        let mut j = i;
        while j < body.len() && body[j].0 == BodyKind::Context {
            j += 1;
        }
        let run_len = j - i;
        let leading = i == 0 && j < body.len();
        let trailing = j == body.len() && i > 0;
        let keep = if leading || trailing {
            run_len.min(max_context)
        } else {
            2 * max_context
        };
        if run_len <= keep {
            for item in &body[i..j] {
                out.push(HunkOut::Line(item.1.clone()));
            }
        } else {
            any_elided = true;
            if leading {
                out.push(HunkOut::Elide);
                for item in &body[j - max_context..j] {
                    out.push(HunkOut::Line(item.1.clone()));
                }
            } else if trailing {
                for item in &body[i..i + max_context] {
                    out.push(HunkOut::Line(item.1.clone()));
                }
                out.push(HunkOut::Elide);
            } else {
                for item in &body[i..i + max_context] {
                    out.push(HunkOut::Line(item.1.clone()));
                }
                out.push(HunkOut::Elide);
                for item in &body[j - max_context..j] {
                    out.push(HunkOut::Line(item.1.clone()));
                }
            }
        }
        i = j;
    }
    (out, any_elided)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_file_diff_passes_through_when_under_caps() {
        let content = [
            "diff --git a/src/one.rs b/src/one.rs",
            "index 1111111..2222222 100644",
            "--- a/src/one.rs",
            "+++ b/src/one.rs",
            "@@ -1,4 +1,4 @@",
            " line one",
            "-old line",
            "+new line",
            " line two",
            "diff --git a/src/two.rs b/src/two.rs",
            "index 3333333..4444444 100644",
            "--- a/src/two.rs",
            "+++ b/src/two.rs",
            "@@ -10,3 +10,4 @@",
            " context here",
            "-gone",
            "+here",
            "+added extra",
        ]
        .join("\n");
        let compressor = DiffCompressor::default();
        let result = compressor.compress(&content, "");
        assert_eq!(result.compressed, content, "nothing dropped -> passthrough");
        assert_eq!(result.files_affected, 2);
        assert_eq!(result.additions, 3);
        assert_eq!(result.deletions, 2);
        assert_eq!(result.hunks_kept, 2);
        assert_eq!(result.hunks_removed, 0);
        assert_eq!(result.cache_key, None);
    }

    #[test]
    fn context_runs_are_trimmed_with_elisions() {
        let mut lines: Vec<String> = vec![
            "diff --git a/big.txt b/big.txt".to_string(),
            "--- a/big.txt".to_string(),
            "+++ b/big.txt".to_string(),
            "@@ -1,33 +1,33 @@".to_string(),
        ];
        for i in 1..=10 {
            lines.push(format!(" c{i:02}"));
        }
        lines.push("+added line".to_string());
        for i in 11..=20 {
            lines.push(format!(" c{i:02}"));
        }
        lines.push("-deleted line".to_string());
        for i in 21..=30 {
            lines.push(format!(" c{i:02}"));
        }
        let content = lines.join("\n");
        assert_eq!(content.lines().count(), 36);
        let compressor = DiffCompressor::default();
        let result = compressor.compress(&content, "");
        assert_ne!(result.compressed, content);
        assert!(result.compressed.contains("+added line"));
        assert!(result.compressed.contains("-deleted line"));
        assert!(
            !result.compressed.contains("c01"),
            "far leading context elided"
        );
        assert!(!result.compressed.contains("c15"), "middle context elided");
        assert!(
            !result.compressed.contains("c25"),
            "far trailing context elided"
        );
        assert!(
            result.compressed.contains("c09"),
            "near leading context kept"
        );
        assert!(result.compressed.contains("c10"));
        assert!(
            result.compressed.contains("c11"),
            "near middle context kept"
        );
        assert!(result.compressed.contains("c20"));
        assert!(
            result.compressed.contains("c21"),
            "near trailing context kept"
        );
        assert!(result.compressed.contains("c22"));
        let elisions = result.compressed.lines().filter(|l| *l == "...").count();
        assert_eq!(elisions, 3);
        assert_eq!(result.cache_key, None, "38 < 50 lines -> no CCR");
    }

    #[test]
    fn hunk_cap_keeps_highest_churn_hunks() {
        let mut lines: Vec<String> = vec![
            "diff --git a/f.rs b/f.rs".to_string(),
            "--- a/f.rs".to_string(),
            "+++ b/f.rs".to_string(),
        ];
        for i in 0..15 {
            lines.push(format!("@@ -{i},2 +{i},2 @@"));
            lines.push(format!(" shared context {i}"));
            if i >= 10 {
                for tag in ["a", "b", "c"] {
                    lines.push(format!("+add-{i}-{tag}"));
                }
            } else {
                lines.push(format!("+add-{i}"));
            }
        }
        let content = lines.join("\n");
        assert_eq!(content.lines().count(), 58);
        let compressor = DiffCompressor::default();
        let result = compressor.compress(&content, "");
        assert!(result.compressed.contains("(5 hunks elided)"));
        assert!(
            result.compressed.contains("+add-12-b"),
            "high-churn hunk kept"
        );
        assert!(
            !result.compressed.contains("+add-7"),
            "low-churn hunk dropped"
        );
        assert_eq!(result.hunks_kept, 10);
        assert_eq!(result.hunks_removed, 5);
        assert!(result
            .compressed
            .contains("diff lines shown. Retrieve original: hash="));
        assert_eq!(result.cache_key, Some(ccr::ccr_hash(&content)));
        assert_eq!(result.additions, 25);
        assert_eq!(result.deletions, 0);
    }

    #[test]
    fn file_cap_appends_more_files_line() {
        let mut lines: Vec<String> = Vec::new();
        for i in 0..25 {
            lines.push(format!("diff --git a/f{i}.txt b/f{i}.txt"));
            lines.push(format!("--- a/f{i}.txt"));
            lines.push(format!("+++ b/f{i}.txt"));
            lines.push("@@ -1,2 +1,2 @@".to_string());
            lines.push(format!("+change {i}"));
            lines.push(format!(" tail {i}"));
        }
        let content = lines.join("\n");
        assert_eq!(content.lines().count(), 150);
        let compressor = DiffCompressor::default();
        let result = compressor.compress(&content, "");
        assert!(result.compressed.contains("… 5 more files"));
        assert!(result.compressed.contains("+change 19"));
        assert!(!result.compressed.contains("+change 20"), "file 20 dropped");
        assert_eq!(result.files_affected, 25);
        assert_eq!(result.hunks_kept, 20);
        assert_eq!(result.hunks_removed, 5);
        assert_eq!(result.cache_key, Some(ccr::ccr_hash(&content)));
    }

    #[test]
    fn headerless_diff_is_parsed() {
        let content = [
            "--- a/x.txt",
            "+++ b/x.txt",
            "@@ -1,2 +1,2 @@",
            "-old line",
            "+new line",
            " shared tail",
        ]
        .join("\n");
        let compressor = DiffCompressor::default();
        let result = compressor.compress(&content, "");
        assert_eq!(result.compressed, content, "nothing dropped -> passthrough");
        assert_eq!(result.files_affected, 1);
        assert_eq!(result.hunks_kept, 1);
        assert_eq!(result.additions, 1);
        assert_eq!(result.deletions, 1);
    }

    #[test]
    fn non_diff_content_passes_through() {
        let content = "just some text\nnot a diff at all";
        let compressor = DiffCompressor::default();
        let result = compressor.compress(&content, "");
        assert_eq!(result.compressed, content);
        assert_eq!(result.files_affected, 0);
        assert_eq!(result.additions, 0);
        assert_eq!(result.deletions, 0);
        assert_eq!(result.cache_key, None);
    }

    #[test]
    fn deterministic_for_identical_calls() {
        let mut lines: Vec<String> = vec![
            "diff --git a/f.rs b/f.rs".to_string(),
            "--- a/f.rs".to_string(),
            "+++ b/f.rs".to_string(),
        ];
        for i in 0..15 {
            lines.push(format!("@@ -{i},2 +{i},2 @@"));
            lines.push(format!(" shared context {i}"));
            lines.push(format!("+add-{i}"));
        }
        let content = lines.join("\n");
        let compressor = DiffCompressor::default();
        let first = compressor.compress(&content, "some context");
        let second = compressor.compress(&content, "some context");
        assert_eq!(first, second);
    }
}
