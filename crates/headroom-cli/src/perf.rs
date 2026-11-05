//! headroom perf — aggregate proxy log statistics.
//!
//! Scans workspace/logs/proxy-*.log (JSONL, one JSON object per line) and
//! reports honest counts: only fields actually present in the logs are
//! aggregated, and averages are reported only when at least one sample
//! exists. The span is metadata about the analysis window; lines are not
//! filtered by age.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use clap::ValueEnum;

/// Output format for headroom perf.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PerfFormat {
    /// Human-readable aligned lines.
    Text,
    /// Single JSON object.
    Json,
}

/// Aggregated proxy-log statistics.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PerfReport {
    /// Analysis window in hours (metadata, not a line filter).
    pub span_hours: u64,
    pub files: u64,
    pub total_lines: u64,
    /// Lines that parsed as JSON.
    pub json_lines: u64,
    /// Lines with level == "error".
    pub errors: u64,
    /// Lines carrying ts_ms or ts.
    pub timestamped: u64,
    /// event value -> count (only when the field is present).
    pub events: BTreeMap<String, u64>,
    pub duration_samples: u64,
    pub duration_sum_ms: f64,
}

impl PerfReport {
    /// Mean duration_ms over the lines that carry one; None when none do.
    pub fn avg_duration_ms(&self) -> Option<f64> {
        if self.duration_samples == 0 {
            None
        } else {
            Some(self.duration_sum_ms / self.duration_samples as f64)
        }
    }

    /// Fold one JSONL line into the report. Blank lines are ignored;
    /// non-JSON lines count as total_lines only.
    pub fn add_line(&mut self, line: &str) {
        let line = line.trim();
        if line.is_empty() {
            return;
        }
        self.total_lines += 1;
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            return;
        };
        self.json_lines += 1;
        if v.get("level").and_then(|l| l.as_str()).is_some_and(|l| l.eq_ignore_ascii_case("error")) {
            self.errors += 1;
        }
        if let Some(event) = v.get("event").and_then(|e| e.as_str()) {
            *self.events.entry(event.to_string()).or_insert(0) += 1;
        }
        if v.get("ts_ms").is_some() || v.get("ts").is_some() {
            self.timestamped += 1;
        }
        if let Some(d) = v.get("duration_ms").and_then(|d| d.as_f64()) {
            self.duration_samples += 1;
            self.duration_sum_ms += d;
        }
    }
}

/// Logs dir: HEADROOM_WORKSPACE_DIR/logs or ~/.headroom/logs.
pub fn log_dir() -> PathBuf {
    let dir = std::env::var("HEADROOM_WORKSPACE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| headroom_store::ccr::default_workspace_dir());
    dir.join("logs")
}

/// proxy-*.log files in dir, sorted by name.
pub fn collect_log_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(entries) => entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("proxy-") && n.ends_with(".log"))
                    .unwrap_or(false)
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    files.sort();
    files
}

/// Aggregate every proxy-*.log under dir. Ok(None) when no log files exist.
/// Unreadable files are skipped with a warning (fail-open).
pub fn analyze_dir(dir: &Path, hours: u64) -> anyhow::Result<Option<PerfReport>> {
    let files = collect_log_files(dir);
    if files.is_empty() {
        return Ok(None);
    }
    let mut report =
        PerfReport { span_hours: hours, files: files.len() as u64, ..Default::default() };
    for file in &files {
        match std::fs::read_to_string(file) {
            Ok(content) => {
                for line in content.lines() {
                    report.add_line(line);
                }
            }
            Err(e) => eprintln!("warning: skipping {}: {e}", file.display()),
        }
    }
    Ok(Some(report))
}

/// Entry point for headroom perf.
pub fn run(hours: u64, format: PerfFormat) -> anyhow::Result<()> {
    let dir = log_dir();
    match analyze_dir(&dir, hours)? {
        None => println!("no proxy logs found in {}", dir.display()),
        Some(report) => match format {
            PerfFormat::Text => print_text(&report),
            PerfFormat::Json => {
                println!("{}", serde_json::to_string_pretty(&report_json(&report))?)
            }
        },
    }
    Ok(())
}

fn report_json(r: &PerfReport) -> serde_json::Value {
    serde_json::json!({
        "span_hours": r.span_hours,
        "files": r.files,
        "total_lines": r.total_lines,
        "json_lines": r.json_lines,
        "errors": r.errors,
        "timestamped": r.timestamped,
        "events": r.events,
        "duration_samples": r.duration_samples,
        "avg_duration_ms": r.avg_duration_ms(),
    })
}

fn print_text(r: &PerfReport) {
    println!("{:<16} last {}h", "window", r.span_hours);
    println!("{:<16} {}", "files", r.files);
    println!("{:<16} {}", "lines", r.total_lines);
    println!("{:<16} {}", "json lines", r.json_lines);
    println!("{:<16} {}", "errors", r.errors);
    println!("{:<16} {}", "timestamped", r.timestamped);
    if r.events.is_empty() {
        println!("{:<16} -", "events");
    } else {
        let joined: Vec<String> = r.events.iter().map(|(k, v)| format!("{k}={v}")).collect();
        println!("{:<16} {}", "events", joined.join(", "));
    }
    match r.avg_duration_ms() {
        Some(avg) => {
            println!("{:<16} {:.1} ms ({} samples)", "avg duration", avg, r.duration_samples)
        }
        None => println!("{:<16} -", "avg duration"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("headroom-perf-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn aggregates_three_lines() {
        let mut report = PerfReport { span_hours: 24, ..Default::default() };
        report
            .add_line(r#"{"level":"info","event":"compress","ts_ms":1793880000000,"duration_ms":10.0}"#);
        report.add_line(r#"{"level":"error","event":"upstream","duration_ms":30}"#);
        report.add_line("not json at all");
        assert_eq!(report.total_lines, 3);
        assert_eq!(report.json_lines, 2);
        assert_eq!(report.errors, 1);
        assert_eq!(report.timestamped, 1);
        assert_eq!(report.events.get("compress"), Some(&1));
        assert_eq!(report.events.get("upstream"), Some(&1));
        assert_eq!(report.duration_samples, 2);
        assert_eq!(report.avg_duration_ms(), Some(20.0));
    }

    #[test]
    fn avg_absent_without_durations() {
        let mut report = PerfReport::default();
        report.add_line(r#"{"level":"info"}"#);
        assert_eq!(report.avg_duration_ms(), None);
        assert_eq!(report.errors, 0);
        assert_eq!(report.total_lines, 1);
    }

    #[test]
    fn analyzes_temp_log_dir() {
        let logs = temp_dir("files").join("logs");
        std::fs::create_dir_all(&logs).unwrap();
        std::fs::write(
            logs.join("proxy-20261105.log"),
            "{\"event\":\"compress\",\"duration_ms\":4}\n{\"event\":\"retrieve\"}\n",
        )
        .unwrap();
        std::fs::write(logs.join("other.log"), "{\"event\":\"ignored\"}\n").unwrap();
        let report = analyze_dir(&logs, 24).unwrap().expect("report");
        assert_eq!(report.files, 1); // only proxy-*.log
        assert_eq!(report.total_lines, 2);
        assert_eq!(report.events.get("compress"), Some(&1));
        assert_eq!(report.events.get("retrieve"), Some(&1));
        assert_eq!(report.events.get("ignored"), None);
        assert_eq!(report.avg_duration_ms(), Some(4.0));
    }

    #[test]
    fn missing_log_dir_is_none() {
        let empty = temp_dir("empty");
        assert!(analyze_dir(&empty, 24).unwrap().is_none());
    }
}
