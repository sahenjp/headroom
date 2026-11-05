//! Savings ledger — durable per-request compression savings.
//!
//! Wire-compatible with the Python headroom's ~/.headroom/savings_events.jsonl:
//! append-only JSONL, schema v2 events, 30-day retention, 1 MB compaction.
//! Written by the proxy and the MCP tool; read by `headroom savings` and
//! the proxy `/stats` endpoint.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Fallback input price when the model is unknown or free: $3 per 1M tokens
/// (Python DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN).
pub const FALLBACK_INPUT_COST_PER_TOKEN: f64 = 3.0e-6;

/// Ledger retention window (Python MAX_RETENTION_DAYS).
pub const MAX_RETENTION_DAYS: i64 = 30;
/// Compact the ledger when it exceeds this size (Python compaction threshold).
pub const COMPACTION_BYTES: u64 = 1024 * 1024;

/// Cache mix counters attached to an event (Python CacheMix).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheMix {
    #[serde(default, skip_serializing_if = "is_zero")]
    pub cr: i64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub cw5: i64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub cw1h: i64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub cw: i64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub un: i64,
}

fn is_zero(v: &i64) -> bool {
    *v == 0
}

/// One recorded savings event — schema v2, field names match the Python
/// ledger exactly so both implementations read the same file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavingsEvent {
    pub v: i32,
    /// ISO-8601 UTC timestamp.
    pub ts: String,
    pub before: i64,
    pub after: i64,
    pub saved: i64,
    pub cost_usd: f64,
    pub model: String,
    pub client: String,
    /// "proxy" | "mcp" | other sources.
    pub source: String,
    pub pid: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_input: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deferred: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saved_tool_schema: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheMix>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_effective_usd: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub basis: Option<String>,
}

impl SavingsEvent {
    /// Build a proxy-source event for one compressed request.
    pub fn new_proxy(model: &str, client: &str, before: i64, after: i64) -> Self {
        let saved = (before - after).max(0);
        let cost_usd = saved as f64 * input_cost_per_token(model);
        Self {
            v: 2,
            ts: now_iso8601(),
            before,
            after,
            saved,
            cost_usd,
            model: model.to_string(),
            client: client.to_string(),
            source: "proxy".into(),
            pid: std::process::id() as i64,
            new_input: None,
            deferred: None,
            saved_tool_schema: None,
            cache: None,
            provider: None,
            cost_effective_usd: None,
            basis: None,
        }
    }
}

/// Rough input price per token for common models; falls back to
/// FALLBACK_INPUT_COST_PER_TOKEN. Mirrors the Python litellm lookup intent
/// without the dependency.
pub fn input_cost_per_token(model: &str) -> f64 {
    let m = model.to_lowercase();
    let per_m = |usd: f64| usd / 1_000_000.0;
    if m.contains("opus-4") {
        per_m(15.0)
    } else if m.contains("sonnet-4") {
        per_m(3.0)
    } else if m.contains("haiku") {
        per_m(0.8)
    } else if m.contains("gpt-5") || m.contains("gpt-6") {
        per_m(1.25)
    } else if m.contains("gpt-4o") {
        per_m(2.5)
    } else if m.contains("gpt-4") {
        per_m(10.0)
    } else if m.contains("o1") || m.contains("o3") || m.contains("o4") {
        per_m(15.0)
    } else if m.contains("deepseek") {
        per_m(0.27)
    } else if m.contains("glm-5") || m.contains("glm-4") {
        per_m(0.6)
    } else if m.contains("llama") || m.contains("qwen") || m.contains("mistral") {
        per_m(0.5)
    } else {
        FALLBACK_INPUT_COST_PER_TOKEN
    }
}

fn now_iso8601() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;
    let millis = now.subsec_millis();
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (h, mi, s) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    let (y, mo, d) = civil_from_days(days);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{millis:03}Z")
}

/// Days-since-epoch to civil date (Howard Hinnant's algorithm).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 - doe / 36524 + doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn parse_ts_ms(ts: &str) -> Option<i64> {
    let y: i64 = ts.get(0..4)?.parse().ok()?;
    let mo: i64 = ts.get(5..7)?.parse().ok()?;
    let d: i64 = ts.get(8..10)?.parse().ok()?;
    let h: i64 = ts.get(11..13)?.parse().ok()?;
    let mi: i64 = ts.get(14..16)?.parse().ok()?;
    let s: i64 = ts.get(17..19)?.parse().ok()?;
    let days = days_from_civil(y, mo as u32, d as u32);
    Some(days * 86_400_000 + (h * 3600 + mi * 60 + s) * 1000)
}

fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = if m > 2 { m - 3 } else { m + 9 } as i64;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Append-only JSONL ledger.
pub struct SavingsLedger {
    path: PathBuf,
}

impl SavingsLedger {
    /// Ledger at workspace/savings_events.jsonl (HEADROOM_SAVINGS_EVENTS_PATH,
    /// then HEADROOM_WORKSPACE_DIR, then ~/.headroom).
    pub fn default_path() -> PathBuf {
        if let Ok(p) = std::env::var("HEADROOM_SAVINGS_EVENTS_PATH") {
            if !p.is_empty() {
                return PathBuf::from(p);
            }
        }
        let dir = std::env::var("HEADROOM_WORKSPACE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| crate::ccr::default_workspace_dir());
        dir.join("savings_events.jsonl")
    }

    pub fn open_default() -> Self {
        Self { path: Self::default_path() }
    }

    pub fn open(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Append one event. Fail-open: ledger errors never break the caller.
    pub fn record(&self, event: &SavingsEvent) {
        if std::env::var("HEADROOM_STATELESS").is_ok() {
            return;
        }
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut line = serde_json::to_string(event).unwrap_or_default();
        line.push('\n');
        let _ = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .and_then(|mut f| f.write_all(line.as_bytes()));
        let _ = self.maybe_compact();
    }

    /// Read all events (skips malformed lines; drops events older than the
    /// retention window, like the Python read path).
    pub fn read_all(&self) -> Vec<SavingsEvent> {
        self.read_all_raw()
            .into_iter()
            .filter(|e| !older_than_retention(e))
            .collect()
    }

    fn read_all_raw(&self) -> Vec<SavingsEvent> {
        let Ok(content) = std::fs::read_to_string(&self.path) else {
            return Vec::new();
        };
        content.lines().filter_map(|l| serde_json::from_str(l).ok()).collect()
    }

    /// Reset (delete) the ledger.
    pub fn reset(&self) -> std::io::Result<()> {
        std::fs::remove_file(&self.path).or_else(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Ok(())
            } else {
                Err(e)
            }
        })
    }

    /// Compact when the file exceeds COMPACTION_BYTES: keep only events
    /// inside the retention window.
    fn maybe_compact(&self) -> std::io::Result<()> {
        let Ok(meta) = std::fs::metadata(&self.path) else {
            return Ok(());
        };
        if meta.len() <= COMPACTION_BYTES {
            return Ok(());
        }
        let kept: Vec<SavingsEvent> = self
            .read_all_raw()
            .into_iter()
            .filter(|e| !older_than_retention(e))
            .collect();
        let tmp = self.path.with_extension("jsonl.tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            for e in &kept {
                writeln!(f, "{}", serde_json::to_string(e).unwrap_or_default())?;
            }
        }
        std::fs::rename(&tmp, &self.path)
    }
}

fn older_than_retention(e: &SavingsEvent) -> bool {
    let Some(ts_ms) = parse_ts_ms(&e.ts) else {
        return false;
    };
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    now_ms - ts_ms > MAX_RETENTION_DAYS * 86_400_000
}

/// Aggregated bucket for one window.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SavingsBucket {
    pub tokens_saved: i64,
    pub tokens_before: i64,
    pub cost_usd: f64,
    pub calls: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub new_input_tokens: Option<i64>,
}

impl SavingsBucket {
    pub fn savings_percent(&self) -> f64 {
        if self.tokens_before > 0 {
            self.tokens_saved as f64 / self.tokens_before as f64 * 100.0
        } else {
            0.0
        }
    }
}

/// Aggregate events into today / last-7d / last-30d buckets plus per-model and
/// per-client cost rankings (mirrors the Python aggregate_savings shape).
pub fn aggregate(events: &[SavingsEvent]) -> serde_json::Value {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);

    let mut today = SavingsBucket::default();
    let mut week = SavingsBucket::default();
    let mut month = SavingsBucket::default();
    let mut lifetime = SavingsBucket::default();
    let mut by_model: BTreeMap<String, (f64, i64, i64)> = BTreeMap::new();
    let mut by_client: BTreeMap<String, (f64, i64, i64)> = BTreeMap::new();

    // "Today" uses a rolling 24h window (documented deviation: the Python
    // ledger aggregates from local midnight).
    for e in events {
        let Some(ts_ms) = parse_ts_ms(&e.ts) else { continue };
        let age = now_ms - ts_ms;
        let bucket_add = |b: &mut SavingsBucket| {
            b.tokens_saved += e.saved;
            b.tokens_before += e.before;
            b.cost_usd += e.cost_usd;
            b.calls += 1;
            if let Some(ni) = e.new_input {
                *b.new_input_tokens.get_or_insert(0) += ni;
            }
        };
        bucket_add(&mut lifetime);
        if age <= 86_400_000 {
            bucket_add(&mut today);
        }
        if age <= 7 * 86_400_000 {
            bucket_add(&mut week);
        }
        if age <= 30 * 86_400_000 {
            bucket_add(&mut month);
        }
        let m = by_model.entry(e.model.clone()).or_insert((0.0, 0, 0));
        m.0 += e.cost_usd;
        m.1 += e.saved;
        m.2 += 1;
        let c = by_client.entry(e.client.clone()).or_insert((0.0, 0, 0));
        c.0 += e.cost_usd;
        c.1 += e.saved;
        c.2 += 1;
    }

    let rank = |map: BTreeMap<String, (f64, i64, i64)>| {
        let mut v: Vec<(String, f64, i64, i64)> = map
            .into_iter()
            .map(|(k, (cost, saved, calls))| (k, cost, saved, calls))
            .collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v
    };

    serde_json::json!({
        "schema_version": 2,
        "lifetime": {
            "tokens_saved": lifetime.tokens_saved,
            "tokens_before": lifetime.tokens_before,
            "cost_usd": lifetime.cost_usd,
            "calls": lifetime.calls,
        },
        "windows": {
            "today": bucket_json(&today),
            "last_7_days": bucket_json(&week),
            "last_30_days": bucket_json(&month),
        },
        "by_model": rank(by_model).into_iter().map(|(m, cost, saved, calls)| serde_json::json!({
            "model": m, "cost_usd": cost, "tokens_saved": saved, "calls": calls,
        })).collect::<Vec<_>>(),
        "by_client": rank(by_client).into_iter().map(|(c, cost, saved, calls)| serde_json::json!({
            "client": c, "cost_usd": cost, "tokens_saved": saved, "calls": calls,
        })).collect::<Vec<_>>(),
    })
}

fn bucket_json(b: &SavingsBucket) -> serde_json::Value {
    let mut v = serde_json::json!({
        "tokens_saved": b.tokens_saved,
        "tokens_before": b.tokens_before,
        "cost_usd": b.cost_usd,
        "calls": b.calls,
        "savings_percent": b.savings_percent(),
    });
    if let Some(ni) = b.new_input_tokens {
        v["new_input_tokens"] = serde_json::json!(ni);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_v2_schema_roundtrip() {
        let dir = std::env::temp_dir().join(format!("headroom-ledger-{}", std::process::id()));
        let ledger = SavingsLedger::open(dir.join("savings_events.jsonl"));
        let ev = SavingsEvent::new_proxy("claude-sonnet-4-5", "codex", 10_000, 2_000);
        assert_eq!(ev.v, 2);
        assert_eq!(ev.saved, 8_000);
        ledger.record(&ev);
        let all = ledger.read_all();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].saved, 8_000);
        assert_eq!(all[0].source, "proxy");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn iso8601_roundtrip() {
        let ts = now_iso8601();
        assert!(ts.ends_with('Z'));
        assert!(parse_ts_ms(&ts).is_some());
        // 2026-11-05T12:00:20Z = 1793880020 s (verified with date -u); the
        // fractional part is truncated by the parser.
        assert_eq!(parse_ts_ms("2026-11-05T12:00:20.342Z"), Some(1_793_880_020_000));
    }

    #[test]
    fn pricing_fallbacks() {
        assert!(input_cost_per_token("claude-sonnet-4-5") > 0.0);
        assert_eq!(input_cost_per_token("some-free-model"), FALLBACK_INPUT_COST_PER_TOKEN);
    }

    #[test]
    fn aggregate_windows() {
        let ev = SavingsEvent::new_proxy("gpt-5", "codex", 1000, 400);
        let out = aggregate(&[ev]);
        assert_eq!(out["windows"]["today"]["tokens_saved"], 600);
        assert_eq!(out["by_model"][0]["model"], "gpt-5");
        assert_eq!(out["by_client"][0]["client"], "codex");
    }
}
