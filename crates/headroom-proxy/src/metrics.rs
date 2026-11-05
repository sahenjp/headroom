//! Prometheus text exposition for proxy counters.

use std::collections::BTreeMap;

/// Counter registry with label support (gauge/counter only; the proxy only
/// needs monotonic counters).
#[derive(Default)]
pub struct Metrics {
    counters: std::sync::Mutex<BTreeMap<String, u64>>,
}

impl Metrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment a counter series. Labels become part of the series key.
    pub fn inc(&self, name: &str, labels: &[(&str, &str)]) {
        let key = series_key(name, labels);
        let mut counters = self.counters.lock().unwrap();
        *counters.entry(key).or_insert(0) += 1;
    }

    /// Render the registry in Prometheus text format, grouped by metric
    /// name, sorted.
    pub fn render(&self) -> String {
        let counters = self.counters.lock().unwrap();
        let mut out = String::new();
        let mut last_base = String::new();
        for (key, value) in counters.iter() {
            let (base, labels) = key.split_once('|').unwrap_or((key, ""));
            if base != last_base {
                out.push_str(&format!("# TYPE {base} counter\n"));
                last_base = base.to_string();
            }
            if labels.is_empty() {
                out.push_str(&format!("{base} {value}\n"));
            } else {
                let rendered: Vec<String> = labels
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|pair| {
                        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
                        format!("{}=\"{}\"", k, escape(v))
                    })
                    .collect();
                out.push_str(&format!("{base}{{{}}} {value}\n", rendered.join(",")));
            }
        }
        out
    }
}

fn series_key(name: &str, labels: &[(&str, &str)]) -> String {
    if labels.is_empty() {
        name.to_string()
    } else {
        let joined: Vec<String> = labels.iter().map(|(k, v)| format!("{k}={v}")).collect();
        format!("{name}|{}", joined.join(","))
    }
}

fn escape(v: &str) -> String {
    v.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_format() {
        let m = Metrics::new();
        m.inc("headroom_requests_total", &[]);
        m.inc("headroom_requests_total", &[]);
        m.inc("headroom_savings_total", &[("client", "codex")]);
        let out = m.render();
        assert!(out.contains("# TYPE headroom_requests_total counter"));
        assert!(out.contains("headroom_requests_total 2"));
        assert!(out.contains("headroom_savings_total{client=\"codex\"} 1"));
    }

    #[test]
    fn label_escaping() {
        let m = Metrics::new();
        m.inc("m", &[("k", "a\"b\\c")]);
        let out = m.render();
        assert!(out.contains("m{k=\"a\\\"b\\\\c\"} 1"), "got: {out}");
    }
}
