//! Statistical crusher for JSON arrays.
//!
//! [`SmartCrusher::crush_array`] first tries the lossless `[N]{schema}` CSV
//! densification (see [`crate::compaction`]); when that does not save enough
//! it switches to a deterministic lossy strategy: every item receives a
//! keep-score built from anchor position, error mentions, numeric anomalies,
//! query relevance (BM25) and change points, identical duplicates are
//! dropped, and only the top [`SmartCrusherConfig::max_items_after_crush`]
//! items survive. Dropped rows are offloaded behind a CCR pointer and the
//! full original JSON is returned in [`CrushResult::ccr_original`] for the
//! caller to persist under the reported cache key.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value};

use crate::ccr;
use crate::compaction;

use super::{contains_any_ci, estimate_tokens, tokenize, Bm25Index};

/// Bonus for the first/last anchor items (always outranks other bonuses).
const ANCHOR_BONUS: f64 = 100.0;
/// Bonus for items that mention errors.
const ERROR_BONUS: f64 = 100.0;
/// Bonus for items at which a string field value changes vs the previous item.
const CHANGE_POINT_BONUS: f64 = 25.0;
/// Bonus per anomalous numeric field.
const ANOMALY_BONUS_PER_FIELD: f64 = 10.0;
/// Total anomaly bonus cap per item.
const ANOMALY_BONUS_MAX: f64 = 30.0;
/// Weight applied to the normalized BM25 relevance before the bias multiplier.
const RELEVANCE_WEIGHT: f64 = 30.0;
/// Minimum length (chars) of a string leaf before it can become an opaque blob.
const MIN_BLOB_CHARS: usize = 500;
/// Whitespace fraction below which a long string counts as minified.
const MINIFIED_WHITESPACE_RATIO: f64 = 0.02;

/// Item substrings (lowercase) that mark an item as error-ish.
const ERROR_NEEDLES: &[&str] = &["error", "failed", "exception"];

const STRATEGY_PASSTHROUGH: &str = "passthrough";
const STRATEGY_COMPACTION: &str = "compaction";
const STRATEGY_LOSSY: &str = "lossy";
const STRATEGY_BLOB: &str = "blob";

/// Configuration for [`SmartCrusher`].
#[derive(Debug, Clone, PartialEq)]
pub struct SmartCrusherConfig {
    /// Master switch; when `false` every array is passed through unchanged.
    pub enabled: bool,
    /// Arrays shorter than this are passed through without analysis.
    pub min_items_to_analyze: usize,
    /// Arrays whose estimated token count (~chars/4) is below this are passed
    /// through.
    pub min_tokens_to_crush: usize,
    /// Standard deviations above the per-field mean at which a numeric leaf
    /// counts as an anomaly.
    pub variance_threshold: f64,
    /// Reserved for near-duplicate item detection (unused in this port).
    pub uniqueness_threshold: f64,
    /// Reserved for near-duplicate item detection (unused in this port).
    pub similarity_threshold: f64,
    /// Maximum number of items kept by the lossy crush.
    pub max_items_after_crush: usize,
    /// Keep items where a string field value changes vs the previous item.
    pub preserve_change_points: bool,
    /// Reserved for the optional value-factoring path (unused in this port).
    pub factor_out_constants: bool,
    /// Reserved for the optional summary path (unused in this port).
    pub include_summaries: bool,
    /// Drop later duplicates of identical items, keeping the first.
    pub dedup_identical_items: bool,
    /// Fraction of `max_items_after_crush` used as the leading anchor window.
    pub first_fraction: f64,
    /// Fraction of `max_items_after_crush` used as the trailing anchor window.
    pub last_fraction: f64,
    /// Minimum byte savings ratio for the lossless compaction to win.
    pub lossless_min_savings_ratio: f64,
    /// Never drop rows: only the lossless compaction is allowed.
    pub lossless_only: bool,
    /// Reserved for the compaction codec (owned by `crate::compaction`).
    pub compaction_core_field_fraction: f64,
    /// Reserved for the compaction codec (owned by `crate::compaction`).
    pub compaction_heterogeneous_core_ratio: f64,
    /// Reserved for the compaction codec (owned by `crate::compaction`).
    pub compaction_max_flatten_inner_keys: usize,
    /// Reserved for the compaction codec (owned by `crate::compaction`).
    pub compaction_min_buckets: usize,
    /// Reserved for the compaction codec (owned by `crate::compaction`).
    pub compaction_max_buckets: usize,
}

impl Default for SmartCrusherConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_items_to_analyze: 5,
            min_tokens_to_crush: 200,
            variance_threshold: 2.0,
            uniqueness_threshold: 0.1,
            similarity_threshold: 0.8,
            max_items_after_crush: 15,
            preserve_change_points: true,
            factor_out_constants: false,
            include_summaries: false,
            dedup_identical_items: true,
            first_fraction: 0.3,
            last_fraction: 0.15,
            lossless_min_savings_ratio: 0.15,
            lossless_only: false,
            compaction_core_field_fraction: 0.8,
            compaction_heterogeneous_core_ratio: 0.6,
            compaction_max_flatten_inner_keys: 6,
            compaction_min_buckets: 2,
            compaction_max_buckets: 8,
        }
    }
}

/// Statistical crusher for JSON arrays.
#[derive(Debug, Clone, Default)]
pub struct SmartCrusher {
    /// Tuning knobs.
    pub config: SmartCrusherConfig,
}

/// Result of a [`SmartCrusher::crush_array`] run.
#[derive(Debug, Clone, PartialEq)]
pub struct CrushResult {
    /// Crushed payload: an array for the lossy strategies, the densified
    /// string for the `"compaction"` strategy.
    pub compressed: Value,
    /// Number of items in the input array.
    pub original_len: usize,
    /// Number of elements in `compressed` (1 for the compaction strategy).
    pub compressed_len: usize,
    /// Token estimate (~chars/4) of the bytes removed.
    pub tokens_saved: usize,
    /// Key under which the caller persists `ccr_original` (`None` when
    /// nothing was offloaded).
    pub cache_key: Option<String>,
    /// Strategy that produced the result: `"passthrough"`, `"compaction"`,
    /// `"lossy"` or `"blob"`.
    pub strategy: String,
    /// `true` when the compressed form differs from the input.
    pub was_modified: bool,
    /// The stashed original to persist under `cache_key`: the full original
    /// array JSON when rows were dropped, otherwise (blob-only strategy) a
    /// JSON object mapping each blob marker to its original leaf value.
    pub ccr_original: Option<String>,
}

impl CrushResult {
    fn passthrough(items: &[Value]) -> Self {
        Self {
            compressed: Value::Array(items.to_vec()),
            original_len: items.len(),
            compressed_len: items.len(),
            tokens_saved: 0,
            cache_key: None,
            strategy: STRATEGY_PASSTHROUGH.to_string(),
            was_modified: false,
            ccr_original: None,
        }
    }
}

impl SmartCrusher {
    /// Creates a crusher with the given configuration.
    pub fn new(config: SmartCrusherConfig) -> Self {
        Self { config }
    }

    /// Crushes `items` (a parsed JSON array) against `query`.
    ///
    /// `bias` scales the BM25 relevance term: higher values favour items that
    /// match the query. The crush is fully deterministic.
    pub fn crush_array(&self, items: &[Value], query: &str, bias: f64) -> CrushResult {
        let n = items.len();
        if n == 0 || !self.config.enabled || n < self.config.min_items_to_analyze {
            return CrushResult::passthrough(items);
        }
        let original_json = to_json_string(&Value::Array(items.to_vec()));
        if estimate_tokens(original_json.chars().count()) < self.config.min_tokens_to_crush {
            return CrushResult::passthrough(items);
        }
        if let Some(result) = self.try_compaction(items, &original_json) {
            return result;
        }
        if self.config.lossless_only {
            return CrushResult::passthrough(items);
        }
        self.lossy_crush(items, &original_json, query, bias)
    }

    /// Lossless-first: the `[N]{schema}` CSV densification when it saves enough.
    fn try_compaction(&self, items: &[Value], original_json: &str) -> Option<CrushResult> {
        if !items.iter().all(|v| v.is_object()) {
            return None;
        }
        let compact = compaction::encode_compacted(items)?;
        let savings = savings_ratio(original_json, &compact);
        if savings < self.config.lossless_min_savings_ratio {
            return None;
        }
        let tokens_saved = estimate_tokens(
            original_json
                .chars()
                .count()
                .saturating_sub(compact.chars().count()),
        );
        Some(CrushResult {
            compressed: Value::String(compact),
            original_len: items.len(),
            compressed_len: 1,
            tokens_saved,
            cache_key: None,
            strategy: STRATEGY_COMPACTION.to_string(),
            was_modified: true,
            ccr_original: None,
        })
    }

    /// Lossy keep/drop scoring and selection.
    fn lossy_crush(
        &self,
        items: &[Value],
        original_json: &str,
        query: &str,
        bias: f64,
    ) -> CrushResult {
        let n = items.len();
        let texts: Vec<String> = items.iter().map(to_json_string).collect();

        let query_terms = tokenize(query);
        let relevance = Bm25Index::new(texts.iter().map(|t| tokenize(t)).collect())
            .normalized_scores(&query_terms);
        let anomaly = numeric_anomaly_bonuses(items, self.config.variance_threshold);

        let mut seen: HashSet<String> = HashSet::new();
        let mut is_dup = vec![false; n];
        if self.config.dedup_identical_items {
            for (i, text) in texts.iter().enumerate() {
                if !seen.insert(text.clone()) {
                    is_dup[i] = true;
                }
            }
        }

        let first_anchor =
            (self.config.max_items_after_crush as f64 * self.config.first_fraction).ceil() as usize;
        let last_anchor =
            (self.config.max_items_after_crush as f64 * self.config.last_fraction).ceil() as usize;
        let string_maps: Vec<HashMap<String, String>> = items.iter().map(string_leaves).collect();

        let mut scored: Vec<(usize, f64)> = Vec::with_capacity(n);
        for i in 0..n {
            if is_dup[i] {
                continue;
            }
            let mut score = 0.0;
            if i < first_anchor || i + last_anchor >= n {
                score += ANCHOR_BONUS;
            }
            if contains_any_ci(&texts[i], ERROR_NEEDLES) {
                score += ERROR_BONUS;
            }
            score += anomaly[i];
            score += bias * RELEVANCE_WEIGHT * relevance[i];
            if self.config.preserve_change_points
                && i > 0
                && string_maps_differ(&string_maps[i - 1], &string_maps[i])
            {
                score += CHANGE_POINT_BONUS;
            }
            scored.push((i, score));
        }
        // Highest score first; ties resolved by the lower original index.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        let mut keep: Vec<usize> = scored
            .iter()
            .take(self.config.max_items_after_crush)
            .map(|&(i, _)| i)
            .collect();
        // Emit survivors in original array order.
        keep.sort_unstable();

        let mut kept_values: Vec<Value> = keep.iter().map(|&i| items[i].clone()).collect();
        let mut blob_stash = Map::new();
        let blobs_replaced = replace_opaque_blobs(&mut kept_values, &mut blob_stash);
        let dropped = n - keep.len();

        if dropped == 0 {
            if blobs_replaced == 0 {
                return CrushResult::passthrough(items);
            }
            // Only opaque blobs were replaced: stash marker -> original leaf.
            let compressed_json = to_json_string(&Value::Array(kept_values.clone()));
            return CrushResult {
                compressed: Value::Array(kept_values),
                original_len: n,
                compressed_len: n,
                tokens_saved: estimate_tokens(
                    original_json
                        .chars()
                        .count()
                        .saturating_sub(compressed_json.chars().count()),
                ),
                cache_key: Some(ccr::ccr_hash12(original_json)),
                strategy: STRATEGY_BLOB.to_string(),
                was_modified: true,
                ccr_original: Some(to_json_string(&Value::Object(blob_stash))),
            };
        }

        let hash = ccr::ccr_hash12(original_json);
        if items.iter().all(|v| v.is_object()) {
            let mut sentinel = Map::new();
            sentinel.insert(
                "_ccr_dropped".to_string(),
                Value::String(format!("<<ccr:{hash} {dropped}_rows_offloaded>>")),
            );
            kept_values.push(Value::Object(sentinel));
        } else {
            kept_values.push(Value::String(format!(
                "… {dropped} more items <<ccr:{hash} {dropped}_items_offloaded>>"
            )));
        }
        let compressed_json = to_json_string(&Value::Array(kept_values.clone()));
        CrushResult {
            compressed: Value::Array(kept_values),
            original_len: n,
            compressed_len: keep.len() + 1,
            tokens_saved: estimate_tokens(
                original_json
                    .chars()
                    .count()
                    .saturating_sub(compressed_json.chars().count()),
            ),
            cache_key: Some(hash),
            strategy: STRATEGY_LOSSY.to_string(),
            was_modified: true,
            ccr_original: Some(original_json.to_string()),
        }
    }
}

/// Serializes a `Value`; infallible for `serde_json::Value`.
fn to_json_string(value: &Value) -> String {
    serde_json::to_string(value).expect("serde_json::Value is always serializable")
}

/// Byte savings ratio of `compact` relative to `original`.
fn savings_ratio(original: &str, compact: &str) -> f64 {
    let original_chars = original.chars().count() as f64;
    if original_chars <= 0.0 {
        return 0.0;
    }
    1.0 - compact.chars().count() as f64 / original_chars
}

/// Collects the string leaves of `value` into a path -> value map.
fn string_leaves(value: &Value) -> HashMap<String, String> {
    let mut out = HashMap::new();
    collect_string_leaves(value, "", &mut out);
    out
}

fn collect_string_leaves(value: &Value, prefix: &str, out: &mut HashMap<String, String>) {
    match value {
        Value::String(s) => {
            out.insert(prefix.to_string(), s.clone());
        }
        Value::Array(a) => {
            for (i, v) in a.iter().enumerate() {
                collect_string_leaves(v, &format!("{prefix}[{i}]"), out);
            }
        }
        Value::Object(o) => {
            for (k, v) in o {
                collect_string_leaves(v, &child_prefix(prefix, k), out);
            }
        }
        _ => {}
    }
}

fn collect_numeric_leaves(value: &Value, prefix: &str, out: &mut HashMap<String, f64>) {
    match value {
        Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                out.insert(prefix.to_string(), f);
            }
        }
        Value::Array(a) => {
            for (i, v) in a.iter().enumerate() {
                collect_numeric_leaves(v, &format!("{prefix}[{i}]"), out);
            }
        }
        Value::Object(o) => {
            for (k, v) in o {
                collect_numeric_leaves(v, &child_prefix(prefix, k), out);
            }
        }
        _ => {}
    }
}

fn child_prefix(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        key.to_string()
    } else {
        format!("{prefix}.{key}")
    }
}

/// `true` when the two path -> value maps differ in any string field.
fn string_maps_differ(a: &HashMap<String, String>, b: &HashMap<String, String>) -> bool {
    if a.len() != b.len() {
        return true;
    }
    a.iter().any(|(k, v)| b.get(k) != Some(v))
}

/// Per-item bonus for numeric leaves above `mean + std_threshold * std` of
/// their field across the array.
fn numeric_anomaly_bonuses(items: &[Value], std_threshold: f64) -> Vec<f64> {
    let n = items.len();
    let mut per_path: HashMap<String, Vec<(usize, f64)>> = HashMap::new();
    for (i, item) in items.iter().enumerate() {
        let mut leaves = HashMap::new();
        collect_numeric_leaves(item, "", &mut leaves);
        for (path, v) in leaves {
            per_path.entry(path).or_default().push((i, v));
        }
    }
    let mut anomaly = vec![0.0_f64; n];
    for occurrences in per_path.values() {
        if occurrences.len() < 3 {
            continue;
        }
        let mean = occurrences.iter().map(|(_, v)| v).sum::<f64>() / occurrences.len() as f64;
        let variance = occurrences
            .iter()
            .map(|(_, v)| (v - mean).powi(2))
            .sum::<f64>()
            / occurrences.len() as f64;
        let limit = mean + std_threshold * variance.sqrt();
        for &(i, v) in occurrences {
            if v > limit {
                anomaly[i] = (anomaly[i] + ANOMALY_BONUS_PER_FIELD).min(ANOMALY_BONUS_MAX);
            }
        }
    }
    anomaly
}

/// Replaces opaque blob leaves (>= [`MIN_BLOB_CHARS`] chars that look like
/// base64 or minified data) with `<<ccr:HASH,KIND,SIZE>>` pointers, stashing
/// the original leaf under its marker. Returns the number of replacements.
fn replace_opaque_blobs(values: &mut [Value], stash: &mut Map<String, Value>) -> usize {
    let mut replaced = 0;
    for value in values.iter_mut() {
        replaced += replace_opaque_blobs_in_value(value, stash);
    }
    replaced
}

fn replace_opaque_blobs_in_value(value: &mut Value, stash: &mut Map<String, Value>) -> usize {
    match value {
        Value::String(s) => match opaque_blob_marker(s) {
            Some((marker, original)) => {
                stash.insert(marker.clone(), Value::String(original));
                *s = marker;
                1
            }
            None => 0,
        },
        Value::Array(a) => {
            let mut replaced = 0;
            for v in a.iter_mut() {
                replaced += replace_opaque_blobs_in_value(v, stash);
            }
            replaced
        }
        Value::Object(o) => {
            let mut replaced = 0;
            for (_, v) in o.iter_mut() {
                replaced += replace_opaque_blobs_in_value(v, stash);
            }
            replaced
        }
        _ => 0,
    }
}

/// Builds a CCR blob marker for a long base64/minified string leaf.
fn opaque_blob_marker(s: &str) -> Option<(String, String)> {
    let chars = s.chars().count();
    if chars < MIN_BLOB_CHARS || ccr::contains_ccr_marker(s) {
        return None;
    }
    let stripped: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    let kind = if !stripped.is_empty()
        && stripped
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=')
    {
        "base64"
    } else {
        let whitespace = s.chars().filter(|c| c.is_whitespace()).count() as f64;
        if whitespace / (chars as f64) < MINIFIED_WHITESPACE_RATIO {
            "blob"
        } else {
            return None;
        }
    };
    let marker = format!(
        "<<ccr:{},{},{}>>",
        ccr::ccr_hash12(s),
        kind,
        super::human_size(s.len())
    );
    Some((marker, s.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// ~50-char records so the default token gate (200 tokens ~ 800 chars) passes.
    fn scalar_records(n: usize) -> Vec<Value> {
        (0..n)
            .map(|i| {
                Value::String(format!(
                    "record {i:03} from the ingestion pipeline stage one"
                ))
            })
            .collect()
    }

    fn object_records(n: usize) -> Vec<Value> {
        (0..n)
            .map(|i| {
                json!({
                    "id": i,
                    "msg": format!("log record {i:03} from the ingestion pipeline stage one")
                })
            })
            .collect()
    }

    #[test]
    fn passthrough_below_min_items() {
        let items = vec![json!(1), json!(2), json!(3)];
        let crusher = SmartCrusher::default();
        let result = crusher.crush_array(&items, "", 1.0);
        assert_eq!(result.strategy, "passthrough");
        assert!(!result.was_modified);
        assert_eq!(result.compressed, Value::Array(items.clone()));
        assert_eq!(result.cache_key, None);
        assert_eq!(result.ccr_original, None);
    }

    #[test]
    fn anchors_are_kept_and_capped() {
        let items = scalar_records(30);
        let crusher = SmartCrusher::new(SmartCrusherConfig {
            max_items_after_crush: 6,
            first_fraction: 0.5,
            last_fraction: 0.5,
            ..Default::default()
        });
        let result = crusher.crush_array(&items, "", 1.0);
        assert_eq!(result.strategy, "lossy");
        assert!(result.was_modified);
        assert_eq!(result.original_len, 30);
        assert_eq!(result.compressed_len, 7); // 6 kept + scalar sentinel
        let arr = result.compressed.as_array().expect("array");
        assert_eq!(arr[0], items[0]);
        assert!(arr.contains(&items[29]));
        assert!(!arr.contains(&items[15]));
        // Scalar-array sentinel: "… N more items <<ccr:HASH N_items_offloaded>>"
        let sentinel = arr.last().unwrap().as_str().expect("string sentinel");
        assert!(
            sentinel.starts_with("… 24 more items <<ccr:"),
            "bad sentinel: {sentinel}"
        );
        assert!(sentinel.ends_with(" 24_items_offloaded>>"));
        let hash = ccr::ccr_hash12(&serde_json::to_string(&Value::Array(items.clone())).unwrap());
        assert_eq!(result.cache_key.as_deref(), Some(hash.as_str()));
        assert_eq!(
            result.ccr_original.as_deref(),
            Some(
                serde_json::to_string(&Value::Array(items))
                    .unwrap()
                    .as_str()
            )
        );
    }

    #[test]
    fn errors_are_kept_and_dict_sentinel_is_used() {
        let mut items = object_records(20);
        items[10] = json!({"id": 10, "error": "boom", "msg": "failed to connect upstream"});
        // Disable the lossless path so the lossy strategy is exercised.
        let crusher = SmartCrusher::new(SmartCrusherConfig {
            max_items_after_crush: 5,
            lossless_min_savings_ratio: 99.0,
            ..Default::default()
        });
        let result = crusher.crush_array(&items, "", 1.0);
        assert_eq!(result.strategy, "lossy");
        let arr = result.compressed.as_array().expect("array");
        assert!(arr.contains(&items[10]), "error item must be kept");
        assert!(arr.contains(&items[0]));
        assert!(arr.contains(&items[19]));
        assert_eq!(arr.len(), 6); // 5 kept + object sentinel
        let sentinel = arr.last().unwrap().as_object().expect("object sentinel");
        let dropped = sentinel.get("_ccr_dropped").unwrap().as_str().unwrap();
        assert!(dropped.starts_with("<<ccr:"));
        assert!(dropped.ends_with(" 15_rows_offloaded>>"));
    }

    #[test]
    fn identical_duplicates_are_deduplicated() {
        let mut items = scalar_records(30);
        for i in [10, 11, 12] {
            items[i] = items[9].clone();
        }
        let crusher = SmartCrusher::default();
        let result = crusher.crush_array(&items, "", 1.0);
        assert_eq!(result.strategy, "lossy");
        let arr = result.compressed.as_array().expect("array");
        assert_eq!(result.compressed_len, 16); // 15 kept + sentinel
        let dup_value = items[9].clone();
        assert_eq!(
            arr.iter().filter(|v| **v == dup_value).count(),
            1,
            "duplicate value must appear exactly once"
        );
    }

    #[test]
    fn max_cap_limits_kept_items() {
        let items = scalar_records(40);
        let crusher = SmartCrusher::default();
        let result = crusher.crush_array(&items, "", 1.0);
        assert_eq!(result.strategy, "lossy");
        assert_eq!(result.compressed_len, 16); // 15 kept + sentinel
        assert_eq!(result.original_len, 40);
    }

    #[test]
    fn query_relevant_items_are_kept() {
        let mut items = scalar_records(30);
        items[5] = Value::String(
            "record 005 from the ingestion pipeline stage one hit a timeout".to_string(),
        );
        items[15] = Value::String(
            "record 015 from the ingestion pipeline stage one hit a timeout".to_string(),
        );
        let crusher = SmartCrusher::default();
        let result = crusher.crush_array(&items, "timeout upstream", 3.0);
        let arr = result.compressed.as_array().expect("array");
        assert!(arr.contains(&items[5]), "relevant item 5 must be kept");
        assert!(arr.contains(&items[15]), "relevant item 15 must be kept");
        assert!(!arr.contains(&items[20]), "plain item 20 must be dropped");
    }

    #[test]
    fn numeric_anomalies_are_kept() {
        let mut items: Vec<Value> = (0..20)
            .map(|i| {
                json!({
                    "temp": (10 + i) as f64,
                    "note": "measurement recorded during the nightly cycle run"
                })
            })
            .collect();
        items[7] = json!({
            "temp": 999.9,
            "note": "measurement recorded during the nightly cycle run"
        });
        let crusher = SmartCrusher::new(SmartCrusherConfig {
            lossless_min_savings_ratio: 99.0,
            ..Default::default()
        });
        let result = crusher.crush_array(&items, "", 1.0);
        let arr = result.compressed.as_array().expect("array");
        assert!(arr.contains(&items[7]), "anomalous item must be kept");
        assert!(!arr.contains(&items[13]), "plain item must be dropped");
    }

    #[test]
    fn lossless_compaction_wins_for_uniform_object_arrays() {
        let items: Vec<Value> = (0..24)
            .map(|i| {
                json!({
                    "name": format!("user-{i:04}"),
                    "status": "active",
                    "score": i
                })
            })
            .collect();
        let crusher = SmartCrusher::default();
        let result = crusher.crush_array(&items, "", 1.0);
        assert_eq!(result.strategy, "compaction");
        assert!(result.was_modified);
        let compact = result.compressed.as_str().expect("densified string");
        assert!(!compact.contains("<<ccr:"), "compaction must not add CCR");
        assert_eq!(result.cache_key, None);
        assert_eq!(result.ccr_original, None);
        assert!(result.tokens_saved > 0);
    }

    #[test]
    fn lossless_only_never_drops() {
        let items = scalar_records(30);
        let crusher = SmartCrusher::new(SmartCrusherConfig {
            lossless_only: true,
            ..Default::default()
        });
        let result = crusher.crush_array(&items, "", 1.0);
        assert_eq!(result.strategy, "passthrough");
        assert!(!result.was_modified);
        assert_eq!(result.compressed, Value::Array(items));
        assert_eq!(result.cache_key, None);
        assert_eq!(result.ccr_original, None);
    }

    #[test]
    fn opaque_blobs_are_offloaded() {
        let blob: String = std::iter::repeat("Abc123+/").take(150).collect(); // 1200 chars
        let items = vec![
            json!({"id": 0, "msg": "plain object zero"}),
            json!({"id": 1, "msg": "plain object one"}),
            json!({"id": 2, "msg": "plain object two"}),
            json!({"id": 3, "msg": "plain object three"}),
            json!({"id": 4, "payload": blob.clone()}),
            Value::String("plain scalar for mixing".to_string()),
        ];
        let crusher = SmartCrusher::default();
        let result = crusher.crush_array(&items, "", 1.0);
        assert_eq!(result.strategy, "blob");
        assert!(result.was_modified);
        let arr = result.compressed.as_array().expect("array");
        assert_eq!(arr.len(), 6);
        let payload = arr[4].as_object().unwrap().get("payload").unwrap();
        let marker = payload.as_str().unwrap();
        assert!(marker.starts_with("<<ccr:"));
        assert!(marker.ends_with(",base64,1.2KB>>"), "bad marker: {marker}");
        // Stash maps the marker back to the original leaf.
        let stash: Value =
            serde_json::from_str(result.ccr_original.as_deref().expect("stash")).unwrap();
        assert_eq!(stash.get(marker), Some(&Value::String(blob)));
        assert!(result.cache_key.is_some());
    }

    #[test]
    fn deterministic_for_identical_calls() {
        let items = scalar_records(40);
        let crusher = SmartCrusher::default();
        let first = crusher.crush_array(&items, "record timeout", 2.0);
        let second = crusher.crush_array(&items, "record timeout", 2.0);
        assert_eq!(first, second);
    }
}
