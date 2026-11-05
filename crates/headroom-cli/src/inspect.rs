//! headroom inspect — show recent proxy requests from logs/requests.jsonl.
//!
//! Each line of the request log is a JSON object with ts, path, before_bytes,
//! after_bytes, tokens_saved and strategies. Malformed lines are skipped.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

/// One request-log entry (logs/requests.jsonl).
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub ts: String,
    pub path: String,
    pub before_bytes: i64,
    pub after_bytes: i64,
    pub tokens_saved: i64,
    pub strategies: Vec<String>,
}

impl Entry {
    /// Parse one JSONL entry. Lenient: missing numeric fields default to 0,
    /// missing strategies to an empty list; None when the line is not a
    /// JSON object.
    pub fn parse(line: &str) -> Option<Entry> {
        let v: serde_json::Value = serde_json::from_str(line.trim()).ok()?;
        if !v.is_object() {
            return None;
        }
        let ts = match v.get("ts") {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => String::new(),
        };
        Some(Entry {
            ts,
            path: v.get("path").and_then(|p| p.as_str()).unwrap_or_default().to_string(),
            before_bytes: num(&v, "before_bytes"),
            after_bytes: num(&v, "after_bytes"),
            tokens_saved: num(&v, "tokens_saved"),
            strategies: v
                .get("strategies")
                .and_then(|s| s.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
                .unwrap_or_default(),
        })
    }
}

fn num(v: &serde_json::Value, key: &str) -> i64 {
    v.get(key).and_then(|x| x.as_i64()).unwrap_or(0)
}

/// Path of the proxy request log (HEADROOM_WORKSPACE_DIR or ~/.headroom).
pub fn requests_path() -> PathBuf {
    let dir = std::env::var("HEADROOM_WORKSPACE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| headroom_store::ccr::default_workspace_dir());
    dir.join("logs").join("requests.jsonl")
}

/// Read the last last entries from a requests JSONL file. Malformed lines
/// are skipped; a missing file yields an empty vec.
pub fn read_entries(path: &Path, last: usize) -> Vec<Entry> {
    if last == 0 {
        return Vec::new();
    }
    let Ok(content) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut ring: VecDeque<Entry> = VecDeque::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(e) = Entry::parse(line) {
            if ring.len() == last {
                ring.pop_front();
            }
            ring.push_back(e);
        }
    }
    ring.into_iter().collect()
}

/// Entry point for headroom inspect.
pub fn run(last: usize, json: bool) -> anyhow::Result<()> {
    let path = requests_path();
    if !path.exists() {
        println!("no request log found at {}", path.display());
        return Ok(());
    }
    let entries = read_entries(&path, last);
    if json {
        let arr: Vec<serde_json::Value> = entries.iter().map(entry_json).collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
        return Ok(());
    }
    if entries.is_empty() {
        println!("no entries in {}", path.display());
        return Ok(());
    }
    for e in &entries {
        let strategies =
            if e.strategies.is_empty() { "-".to_string() } else { e.strategies.join("+") };
        println!(
            "{} {} {}->{} (saved {} tokens, {})",
            e.ts, e.path, e.before_bytes, e.after_bytes, e.tokens_saved, strategies
        );
    }
    Ok(())
}

fn entry_json(e: &Entry) -> serde_json::Value {
    serde_json::json!({
        "ts": e.ts,
        "path": e.path,
        "before_bytes": e.before_bytes,
        "after_bytes": e.after_bytes,
        "tokens_saved": e.tokens_saved,
        "strategies": e.strategies,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(name: &str, body: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("headroom-inspect-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("requests.jsonl");
        std::fs::write(&path, body).unwrap();
        path
    }

    const TWO_LINES: &str = concat!(
        "{\"ts\":\"2026-11-05T10:00:00Z\",\"path\":\"/v1/messages\",\"before_bytes\":1000,",
        "\"after_bytes\":400,\"tokens_saved\":600,\"strategies\":[\"tool_schema\"]}\n",
        "{\"ts\":\"2026-11-05T11:00:00Z\",\"path\":\"/v1/messages\",\"before_bytes\":2000,",
        "\"after_bytes\":500,\"tokens_saved\":1500,\"strategies\":[\"ccr\",\"tool_schema\"]}\n",
    );

    #[test]
    fn last_n_selection() {
        let path = temp_file("last_n", TWO_LINES);
        let all = read_entries(&path, 10);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].tokens_saved, 600);
        assert_eq!(all[0].before_bytes, 1000);
        assert_eq!(all[0].after_bytes, 400);
        assert_eq!(all[0].strategies, vec!["tool_schema"]);
        let one = read_entries(&path, 1);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].ts, "2026-11-05T11:00:00Z");
        assert_eq!(one[0].tokens_saved, 1500);
        assert_eq!(read_entries(&path, 0).len(), 0);
    }

    #[test]
    fn malformed_lines_skipped() {
        let path = temp_file("malformed", "{\"ts\":\"t1\"}\nnot json\n{\"ts\":\"t2\"}\n");
        let entries = read_entries(&path, 10);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].ts, "t2");
        assert_eq!(entries[1].before_bytes, 0); // lenient defaults
        assert!(entries[1].strategies.is_empty());
    }

    #[test]
    fn missing_file_is_empty() {
        let missing = std::env::temp_dir().join("headroom-inspect-missing-file.jsonl");
        let _ = std::fs::remove_file(&missing);
        assert!(read_entries(&missing, 5).is_empty());
    }
}
