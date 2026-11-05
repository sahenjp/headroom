//! headroom audit-reads — estimate how much Claude Code transcript bytes
//! are tool results (compression candidates).
//!
//! Scans a directory (default ~/.claude/projects) for *.jsonl transcript
//! files and counts lines containing the tool_result marker. The savings
//! estimate is a DOCUMENTED HEURISTIC, not a measurement: files larger than
//! 2000 bytes are assumed 60% compressible, smaller files 0%.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// Substring that marks a transcript line as a tool result.
const TOOL_RESULT_MARKER: &str = "\"type\":\"tool_result\"";
/// Files larger than this are treated as compression candidates.
const LARGE_FILE_BYTES: u64 = 2_000;
/// Share of a large transcript assumed recoverable by compression.
/// Documented heuristic, not a measurement.
const ESTIMATED_SAVINGS_SHARE: f64 = 0.6;

/// Per-file audit result.
#[derive(Debug, Clone, PartialEq)]
pub struct FileAudit {
    pub file: PathBuf,
    pub tool_result_lines: u64,
    pub bytes: u64,
    pub estimated_savings_bytes: f64,
}

/// Audit one transcript file. Returns Err only on IO errors (the caller
/// skips unreadable files).
pub fn audit_file(path: &Path) -> std::io::Result<FileAudit> {
    let content = std::fs::read_to_string(path)?;
    let tool_result_lines =
        content.lines().filter(|l| l.contains(TOOL_RESULT_MARKER)).count() as u64;
    let bytes = content.len() as u64;
    let estimated_savings_bytes = if bytes > LARGE_FILE_BYTES {
        bytes as f64 * ESTIMATED_SAVINGS_SHARE
    } else {
        0.0
    };
    Ok(FileAudit {
        file: path.to_path_buf(),
        tool_result_lines,
        bytes,
        estimated_savings_bytes,
    })
}

/// Recursively collect audits for every *.jsonl file under dir.
/// Unreadable files are skipped (fail-open). Sorted by path.
pub fn scan_dir(dir: &Path) -> Vec<FileAudit> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                if let Ok(audit) = audit_file(&path) {
                    out.push(audit);
                }
            }
        }
    }
    out.sort_by(|a, b| a.file.cmp(&b.file));
    out
}

/// Default scan root: ~/.claude/projects.
pub fn projects_dir() -> PathBuf {
    std::env::var("HOME")
        .map(|home| PathBuf::from(home).join(".claude").join("projects"))
        .unwrap_or_else(|_| PathBuf::from(".claude/projects"))
}

/// Entry point for headroom audit-reads.
pub fn run(path: Option<String>, json: bool) -> anyhow::Result<()> {
    let dir = path.map(PathBuf::from).unwrap_or_else(projects_dir);
    if !dir.is_dir() {
        println!("no claude projects dir at {}", dir.display());
        return Ok(());
    }
    let audits = scan_dir(&dir);
    if audits.is_empty() {
        println!("no .jsonl transcripts under {}", dir.display());
        return Ok(());
    }
    if json {
        let files: Vec<serde_json::Value> = audits.iter().map(audit_json).collect();
        let doc = serde_json::json!({
            "dir": dir.display().to_string(),
            "files": files,
            "totals": totals_json(&audits),
        });
        println!("{}", serde_json::to_string_pretty(&doc)?);
        return Ok(());
    }
    println!("{:<60} {:>17} {:>10} {:>12}", "file", "tool_result_lines", "bytes", "est_saved");
    for a in &audits {
        println!(
            "{:<60} {:>17} {:>10} {:>12.0}",
            a.file.display().to_string(),
            a.tool_result_lines,
            a.bytes,
            a.estimated_savings_bytes,
        );
    }
    let (files, lines, bytes, saved) = totals(&audits);
    println!(
        "total: {} file(s), {} tool_result line(s), {} bytes, est. savings {:.0} bytes ({}% heuristic on files > {} bytes)",
        files,
        lines,
        bytes,
        saved,
        (ESTIMATED_SAVINGS_SHARE * 100.0) as u64,
        LARGE_FILE_BYTES,
    );
    Ok(())
}

fn totals(audits: &[FileAudit]) -> (usize, u64, u64, f64) {
    (
        audits.len(),
        audits.iter().map(|a| a.tool_result_lines).sum(),
        audits.iter().map(|a| a.bytes).sum(),
        audits.iter().map(|a| a.estimated_savings_bytes).sum(),
    )
}

fn totals_json(audits: &[FileAudit]) -> serde_json::Value {
    let (files, lines, bytes, saved) = totals(audits);
    serde_json::json!({
        "files": files,
        "tool_result_lines": lines,
        "bytes": bytes,
        "estimated_savings_bytes": saved,
    })
}

fn audit_json(a: &FileAudit) -> serde_json::Value {
    serde_json::json!({
        "file": a.file.display().to_string(),
        "tool_result_lines": a.tool_result_lines,
        "bytes": a.bytes,
        "estimated_savings_bytes": a.estimated_savings_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("headroom-audit-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn audits_two_synthetic_files() {
        let dir = temp_dir("counts");
        let small_line = "{\"type\":\"user\",\"message\":\"hi\"}";
        let tool_line = "{\"type\":\"tool_result\",\"content\":\"...\"}";
        // File A: 3 lines, 2 tool results, padded past the 2000-byte cutoff.
        let pad = "x".repeat(2200);
        let a = format!("{tool_line}\n{tool_line}\n{small_line} {pad}\n");
        std::fs::write(dir.join("a.jsonl"), &a).unwrap();
        // File B: 2 lines, 0 tool results, below the cutoff.
        std::fs::write(dir.join("b.jsonl"), format!("{small_line}\n{small_line}\n")).unwrap();
        // Non-jsonl files are ignored.
        std::fs::write(dir.join("c.txt"), tool_line).unwrap();

        let audits = scan_dir(&dir);
        assert_eq!(audits.len(), 2);
        let a_audit = audits.iter().find(|x| x.file.ends_with("a.jsonl")).unwrap();
        assert_eq!(a_audit.tool_result_lines, 2);
        assert_eq!(a_audit.bytes, a.len() as u64);
        assert!((a_audit.estimated_savings_bytes - a.len() as f64 * 0.6).abs() < 1e-6);
        let b_audit = audits.iter().find(|x| x.file.ends_with("b.jsonl")).unwrap();
        assert_eq!(b_audit.tool_result_lines, 0);
        assert_eq!(b_audit.estimated_savings_bytes, 0.0); // below the cutoff
    }

    #[test]
    fn subdirectories_are_scanned() {
        let dir = temp_dir("nested");
        let nested = dir.join("proj").join("sub");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("t.jsonl"), "{\"type\":\"tool_result\"}\n").unwrap();
        let audits = scan_dir(&dir);
        assert_eq!(audits.len(), 1);
        assert_eq!(audits[0].tool_result_lines, 1);
    }

    #[test]
    fn missing_dir_is_empty() {
        let missing = temp_dir("missing").join("nope");
        assert!(scan_dir(&missing).is_empty());
    }
}
