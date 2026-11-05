//! CCR (Compress-Cache-Retrieve) store: the hash -> original map.
//!
//! Wire-compatible with the Python headroom's `~/.headroom/ccr_store.db`:
//! same SQLite schema, same JSON entry blob, same purge discipline, so a
//! store written by either implementation is readable by the other.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rusqlite::Connection;
use serde::{Deserialize, Serialize};

/// Default TTL: 30 minutes (session-scale), matches Python DEFAULT_CCR_TTL_SECONDS.
pub const DEFAULT_TTL_SECONDS: u64 = 1800;
/// Default capacity, matches Python CCRConfig.store_max_entries.
pub const DEFAULT_MAX_ENTRIES: usize = 1000;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS ccr_entries (
    hash TEXT PRIMARY KEY,
    entry_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    ttl INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ccr_expiry_deadline
    ON ccr_entries (created_at + ttl);
";

/// Error type for store operations. All operations are fail-open by design:
/// callers treat errors as "miss" and never propagate compression failures.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// A stored entry. Field names mirror the Python `CompressionEntry` dataclass
/// so the JSON blob in the DB is interchangeable.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CompressionEntry {
    pub hash: String,
    pub original_content: String,
    pub compressed_content: String,
    #[serde(default)]
    pub original_tokens: i64,
    #[serde(default)]
    pub compressed_tokens: i64,
    #[serde(default)]
    pub original_item_count: Option<i64>,
    #[serde(default)]
    pub compressed_item_count: Option<i64>,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub query_context: Option<String>,
    pub created_at: f64,
    #[serde(default = "default_ttl")]
    pub ttl: u64,
    #[serde(default)]
    pub tool_signature_hash: Option<String>,
    #[serde(default)]
    pub compression_strategy: Option<String>,
    #[serde(default)]
    pub retrieval_count: i64,
    #[serde(default)]
    pub search_queries: Vec<String>,
    #[serde(default)]
    pub last_accessed: Option<f64>,
}

fn default_ttl() -> u64 {
    DEFAULT_TTL_SECONDS
}

/// Where the store lives.
#[derive(Debug, Clone)]
pub struct StoreConfig {
    /// SQLite path. None => HEADROOM_CCR_SQLITE_PATH or ~/.headroom/ccr_store.db.
    pub path: Option<PathBuf>,
    pub ttl: u64,
    pub max_entries: usize,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self { path: None, ttl: DEFAULT_TTL_SECONDS, max_entries: DEFAULT_MAX_ENTRIES }
    }
}

impl StoreConfig {
    fn resolve_path(&self) -> PathBuf {
        if let Some(p) = &self.path {
            return p.clone();
        }
        if let Ok(p) = std::env::var("HEADROOM_CCR_SQLITE_PATH") {
            if !p.is_empty() {
                return PathBuf::from(p);
            }
        }
        let dir = std::env::var("HEADROOM_WORKSPACE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| default_workspace_dir());
        dir.join("ccr_store.db")
    }
}

/// ~/.headroom (Linux default workspace).
pub fn default_workspace_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Path::new(&home).join(".headroom")
}

fn now_unix() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// SQLite-backed CCR store. One connection guarded by a mutex; the Python
/// service runs WAL so cross-process concurrent access is safe.
pub struct CompressionStore {
    conn: std::sync::Mutex<Connection>,
    ttl: u64,
    max_entries: usize,
    #[allow(dead_code)]
    path: PathBuf,
}

impl CompressionStore {
    /// Open (creating if needed) the store. Applies WAL + busy_timeout, sets
    /// 0600 perms on a fresh file, purges expired rows opportunistically.
    pub fn open(config: &StoreConfig) -> Result<Self, StoreError> {
        let path = config.resolve_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let fresh = !path.exists();
        let conn = Connection::open(&path)?;
        conn.busy_timeout(Duration::from_millis(5000))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        if fresh {
            let _ = set_private_perms(&path);
        } else {
            // Keep perms tight even for pre-existing files (best effort).
            let _ = set_private_perms(&path);
        }
        conn.execute_batch(SCHEMA)?;
        let store = Self {
            conn: std::sync::Mutex::new(conn),
            ttl: config.ttl,
            max_entries: config.max_entries,
            path,
        };
        store.purge_expired()?;
        Ok(store)
    }

    /// In-memory store (HEADROOM_STATELESS / tests).
    pub fn open_memory() -> Result<Self, StoreError> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self {
            conn: std::sync::Mutex::new(conn),
            ttl: DEFAULT_TTL_SECONDS,
            max_entries: DEFAULT_MAX_ENTRIES,
            path: PathBuf::from(":memory:"),
        })
    }

    /// Store an original under `hash_key`. Refuses bare CCR markers (the
    /// Python store does the same — never stash a placeholder as an original).
    pub fn store(&self, entry: CompressionEntry) -> Result<String, StoreError> {
        if entry.original_content.contains("<<ccr:") && is_bare_marker(&entry.original_content) {
            return Ok(entry.hash);
        }
        let json = serde_json::to_string(&entry).map_err(|e| {
            StoreError::Io(std::io::Error::other(e))
        })?;
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO ccr_entries (hash, entry_json, created_at, ttl) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(hash) DO UPDATE SET entry_json=?2, created_at=?3, ttl=?4",
            rusqlite::params![entry.hash, json, entry.created_at, entry.ttl as i64],
        )?;
        self.enforce_capacity(&conn)?;
        Ok(entry.hash)
    }

    /// Convenience: store with defaults, computing the hash from the original.
    pub fn store_original(
        &self,
        original: &str,
        compressed: &str,
        original_tokens: i64,
        compressed_tokens: i64,
        tool_name: Option<&str>,
        strategy: Option<&str>,
    ) -> Result<String, StoreError> {
        let hash = crate::hash::ccr_hash(original);
        let entry = CompressionEntry {
            hash: hash.clone(),
            original_content: original.to_string(),
            compressed_content: compressed.to_string(),
            original_tokens,
            compressed_tokens,
            original_item_count: None,
            compressed_item_count: None,
            tool_name: tool_name.map(|s| s.to_string()),
            tool_call_id: None,
            query_context: None,
            created_at: now_unix(),
            ttl: self.ttl,
            tool_signature_hash: None,
            compression_strategy: strategy.map(|s| s.to_string()),
            retrieval_count: 0,
            search_queries: Vec::new(),
            last_accessed: None,
        };
        self.store(entry)?;
        Ok(hash)
    }

    /// Retrieve by hash. Expired => delete + miss. Records access stats.
    pub fn retrieve(&self, hash: &str) -> Option<CompressionEntry> {
        let conn = self.conn.lock().unwrap();
        let row = conn
            .query_row(
                "SELECT entry_json, created_at, ttl FROM ccr_entries WHERE hash = ?1",
                rusqlite::params![hash],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, f64>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                },
            )
            .ok()?;
        let (json, created_at, ttl) = row;
        if now_unix() - created_at > ttl as f64 {
            let _ = conn.execute("DELETE FROM ccr_entries WHERE hash = ?1", rusqlite::params![hash]);
            return None;
        }
        // Field-filtered deserialization: unknown keys dropped by serde default.
        let mut entry: CompressionEntry = serde_json::from_str(&json).ok()?;
        // Update access stats (best effort).
        entry.retrieval_count += 1;
        entry.last_accessed = Some(now_unix());
        if let Ok(updated) = serde_json::to_string(&entry) {
            let _ = conn.execute(
                "UPDATE ccr_entries SET entry_json = ?2 WHERE hash = ?1",
                rusqlite::params![hash, updated],
            );
        }
        Some(entry)
    }

    /// Delete expired rows. Called on open and opportunistically on writes.
    pub fn purge_expired(&self) -> Result<usize, StoreError> {
        let conn = self.conn.lock().unwrap();
        let n = conn.execute(
            "DELETE FROM ccr_entries WHERE created_at + ttl < ?1",
            rusqlite::params![now_unix()],
        )?;
        Ok(n)
    }

    /// Evict overflow: expired first (already purged), then oldest created_at.
    fn enforce_capacity(&self, conn: &Connection) -> Result<(), StoreError> {
        let count: i64 =
            conn.query_row("SELECT COUNT(*) FROM ccr_entries", [], |r| r.get(0))?;
        if count as usize <= self.max_entries {
            return Ok(());
        }
        let excess = count as usize - self.max_entries;
        conn.execute(
            "DELETE FROM ccr_entries WHERE hash IN (
                SELECT hash FROM ccr_entries ORDER BY created_at ASC LIMIT ?1
            )",
            rusqlite::params![excess as i64],
        )?;
        Ok(())
    }

    /// Number of live entries (diagnostics/tests).
    pub fn len(&self) -> Result<usize, StoreError> {
        let conn = self.conn.lock().unwrap();
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM ccr_entries", [], |r| r.get(0))?;
        Ok(n as usize)
    }

    pub fn is_empty(&self) -> Result<bool, StoreError> {
        Ok(self.len()? == 0)
    }
}

fn is_bare_marker(text: &str) -> bool {
    let t = text.trim();
    t.starts_with("<<ccr:") && t.ends_with(">>") && !t[5..t.len() - 2].contains(' ')
        || t.starts_with("<<ccr:")
            && t.ends_with(">>")
            && t.matches(':').count() >= 1
            && t.lines().count() == 1
            && t.len() < 128
}

#[cfg(unix)]
fn set_private_perms(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)?.permissions();
    perms.set_mode(0o600);
    std::fs::set_permissions(path, perms)
}

#[cfg(not(unix))]
fn set_private_perms(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(hash: &str, original: &str) -> CompressionEntry {
        CompressionEntry {
            hash: hash.to_string(),
            original_content: original.to_string(),
            compressed_content: "<<ccr:placeholder>>".to_string(),
            original_tokens: 100,
            compressed_tokens: 10,
            original_item_count: Some(10),
            compressed_item_count: Some(2),
            tool_name: Some("Bash".into()),
            tool_call_id: None,
            query_context: None,
            created_at: now_unix(),
            ttl: DEFAULT_TTL_SECONDS,
            tool_signature_hash: None,
            compression_strategy: Some("smart_crusher".into()),
            retrieval_count: 0,
            search_queries: vec![],
            last_accessed: None,
        }
    }

    #[test]
    fn store_and_retrieve_roundtrip() {
        let store = CompressionStore::open_memory().unwrap();
        let h = store
            .store_original("original text", "compressed", 10, 2, Some("Bash"), Some("log"))
            .unwrap();
        assert_eq!(h.len(), 24);
        let got = store.retrieve(&h).unwrap();
        assert_eq!(got.original_content, "original text");
        assert_eq!(got.retrieval_count, 1);
    }

    #[test]
    fn expired_entry_is_miss() {
        let store = CompressionStore::open_memory().unwrap();
        let mut e = entry("abc123", "soon-gone");
        e.created_at = now_unix() - (DEFAULT_TTL_SECONDS as f64 + 10.0);
        store.store(e).unwrap();
        assert!(store.retrieve("abc123").is_none());
    }

    #[test]
    fn bare_marker_refused() {
        let store = CompressionStore::open_memory().unwrap();
        let mut e = entry("marker1", "<<ccr:deadbeef1234>>");
        e.original_content = "<<ccr:deadbeef1234,base64,4.5KB>>".to_string();
        store.store(e).unwrap();
        // Not stored as an original.
        assert!(store.retrieve("marker1").is_none());
    }

    #[test]
    fn capacity_evicts_oldest() {
        let mut cfg = StoreConfig::default();
        cfg.max_entries = 3;
        let store = CompressionStore {
            conn: std::sync::Mutex::new(Connection::open_in_memory().unwrap()),
            ttl: cfg.ttl,
            max_entries: cfg.max_entries,
            path: PathBuf::from(":memory:"),
        };
        store.conn.lock().unwrap().execute_batch(SCHEMA).unwrap();
        for i in 0..5 {
            let mut e = entry(&format!("h{i}"), "content");
            e.created_at = now_unix() + i as f64; // strictly increasing
            store.store(e).unwrap();
        }
        assert_eq!(store.len().unwrap(), 3);
        // Oldest two evicted.
        assert!(store.retrieve("h0").is_none());
        assert!(store.retrieve("h4").is_some());
    }

    #[test]
    fn overwrite_in_place() {
        let store = CompressionStore::open_memory().unwrap();
        let mut e1 = entry("same", "v1");
        e1.created_at = now_unix();
        store.store(e1).unwrap();
        let e2 = entry("same", "v2");
        store.store(e2).unwrap();
        assert_eq!(store.len().unwrap(), 1);
        assert_eq!(store.retrieve("same").unwrap().original_content, "v2");
    }
}
