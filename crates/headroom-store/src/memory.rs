//! Memory store — durable agent memories in a single SQLite file.
//!
//! Simplified port of the Python headroom memory subsystem: one 'memories'
//! table (soft deletes, supersession, JSON-encoded tags) plus an optional
//! FTS5 index, stored as 'memory.db' under the workspace dir. Follows the
//! CCR store's conventions (WAL, busy_timeout, 0600 perms, fail-open index
//! handling).
//!
//! FTS5 note: the index is keyed by rowid = memories.id. When the SQLite
//! build lacks FTS5 the store still works — search degrades to a LIKE scan
//! and [MemoryStats::fts_enabled] reports false.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

pub use crate::ccr::StoreError;

/// Store result alias; errors are the shared [StoreError].
pub type Result<T> = std::result::Result<T, StoreError>;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'project',
    project TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    superseded_by INTEGER,
    deleted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_memories_list ON memories (deleted, created_at DESC);
";

/// Best-effort FTS5 index (rowid = memories.id, body = content). Creation
/// failure is not fatal: the store falls back to LIKE search.
const FTS_SCHEMA: &str = "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(body)";

const COLS: &str = "id, content, scope, project, tags, created_at, updated_at, superseded_by";

/// A new memory to store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewMemory {
    pub content: String,
    /// "project" | "user" | "global" — stored verbatim, not enforced.
    pub scope: String,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// One live memory (soft-deleted rows are invisible to all reads).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: i64,
    pub content: String,
    pub scope: String,
    pub project: Option<String>,
    pub tags: Vec<String>,
    /// Unix seconds.
    pub created_at: f64,
    /// Unix seconds.
    pub updated_at: f64,
    /// Id of the memory that replaced this one, if any.
    pub superseded_by: Option<i64>,
}

/// Aggregate counters for the memory stats command.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryStats {
    pub total: i64,
    pub by_scope: BTreeMap<String, i64>,
    /// False when the SQLite build lacks FTS5 and search uses LIKE.
    pub fts_enabled: bool,
}

/// SQLite-backed memory store. One connection behind a mutex, WAL mode,
/// same conventions as the CCR store.
pub struct MemoryDb {
    conn: Mutex<Connection>,
    fts_enabled: bool,
    #[allow(dead_code)]
    path: PathBuf,
}

impl MemoryDb {
    /// memory.db under the workspace dir (HEADROOM_MEMORY_DB_PATH, then
    /// HEADROOM_WORKSPACE_DIR, then ~/.headroom).
    pub fn default_path() -> PathBuf {
        if let Ok(p) = std::env::var("HEADROOM_MEMORY_DB_PATH") {
            if !p.is_empty() {
                return PathBuf::from(p);
            }
        }
        let dir = std::env::var("HEADROOM_WORKSPACE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| crate::ccr::default_workspace_dir());
        dir.join("memory.db")
    }

    /// Open (creating if needed) the default memory.db.
    pub fn open_default() -> Result<Self> {
        Self::open(&Self::default_path())
    }

    /// Open (creating if needed) the store at path. Applies WAL,
    /// busy_timeout 5000 and 0600 perms, creates the schema and tries to
    /// create the FTS5 index (an optional SQLite feature).
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let conn = Connection::open(path)?;
        conn.busy_timeout(Duration::from_millis(5000))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        // Best effort, like the CCR store: keep the db private.
        let _ = set_private_perms(path);
        conn.execute_batch(SCHEMA)?;
        let fts_enabled = conn.execute_batch(FTS_SCHEMA).is_ok();
        Ok(Self { conn: Mutex::new(conn), fts_enabled, path: path.to_path_buf() })
    }

    /// Whether the FTS5 index is in use (else search uses LIKE).
    pub fn fts_enabled(&self) -> bool {
        self.fts_enabled
    }

    /// Insert a memory; returns its id. The FTS row is best-effort — a
    /// missed index entry is healed by [Self::reindex].
    pub fn add(&self, m: &NewMemory) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let now = now_unix();
        let tags = serde_json::to_string(&m.tags).unwrap_or_else(|_| "[]".into());
        conn.execute(
            "INSERT INTO memories (content, scope, project, tags, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
            rusqlite::params![m.content, m.scope, m.project, tags, now],
        )?;
        let id = conn.last_insert_rowid();
        if self.fts_enabled {
            let _ = conn.execute(
                "INSERT INTO memory_fts(rowid, body) VALUES (?1, ?2)",
                rusqlite::params![id, m.content],
            );
        }
        Ok(id)
    }

    /// Fetch one live memory by id (None for missing or soft-deleted rows).
    pub fn get(&self, id: i64) -> Result<Option<MemoryRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&format!(
            "SELECT {} FROM memories WHERE id = ?1 AND deleted = 0",
            COLS
        ))?;
        let rec = stmt.query_row(rusqlite::params![id], row_to_record).optional()?;
        Ok(rec)
    }

    /// Replace a memory's content (bumps updated_at, refreshes the index).
    /// Returns false when the id is missing or soft-deleted.
    pub fn edit(&self, id: i64, content: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let n = conn.execute(
            "UPDATE memories SET content = ?2, updated_at = ?3 WHERE id = ?1 AND deleted = 0",
            rusqlite::params![id, content, now_unix()],
        )?;
        if n > 0 && self.fts_enabled {
            let _ = conn.execute("DELETE FROM memory_fts WHERE rowid = ?1", rusqlite::params![id]);
            let _ = conn.execute(
                "INSERT INTO memory_fts(rowid, body) VALUES (?1, ?2)",
                rusqlite::params![id, content],
            );
        }
        Ok(n > 0)
    }

    /// Soft delete (deleted = 1); the row stays until prune/purge.
    /// Returns false when the id is missing or already deleted.
    pub fn delete(&self, id: i64) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let n = conn.execute(
            "UPDATE memories SET deleted = 1, updated_at = ?2 WHERE id = ?1 AND deleted = 0",
            rusqlite::params![id, now_unix()],
        )?;
        if n > 0 && self.fts_enabled {
            let _ = conn.execute("DELETE FROM memory_fts WHERE rowid = ?1", rusqlite::params![id]);
        }
        Ok(n > 0)
    }

    /// List live memories, newest first. scope filters by scope,
    /// since_ms (unix milliseconds) by created_at.
    pub fn list(
        &self,
        limit: usize,
        scope: Option<&str>,
        since_ms: Option<i64>,
    ) -> Result<Vec<MemoryRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&format!(
            "SELECT {} FROM memories
             WHERE deleted = 0
               AND (?1 IS NULL OR scope = ?1)
               AND (?2 IS NULL OR created_at >= ?2 / 1000.0)
             ORDER BY created_at DESC, id DESC
             LIMIT ?3",
            COLS
        ))?;
        let rows = stmt.query_map(
            rusqlite::params![scope, since_ms, limit as i64],
            row_to_record,
        )?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// Search live memories: FTS5 MATCH when the index is available, else
    /// LIKE '%q%'. FTS5 query-syntax errors degrade to LIKE (fail-open).
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryRecord>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        if self.fts_enabled {
            if let Ok(rows) = self.search_fts(query, limit) {
                return Ok(rows);
            }
        }
        self.search_like(query, limit)
    }

    fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<MemoryRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT m.id, m.content, m.scope, m.project, m.tags, m.created_at, m.updated_at, m.superseded_by
             FROM memory_fts
             JOIN memories m ON m.id = memory_fts.rowid
             WHERE memory_fts MATCH ?1 AND m.deleted = 0
             ORDER BY bm25(memory_fts)
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(rusqlite::params![query, limit as i64], row_to_record)?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    fn search_like(&self, query: &str, limit: usize) -> Result<Vec<MemoryRecord>> {
        let conn = self.conn.lock().unwrap();
        let pattern = format!("%{}%", escape_like(query));
        let mut stmt = conn.prepare(&format!(
            "SELECT {} FROM memories
             WHERE deleted = 0 AND content LIKE ?1 ESCAPE '\\'
             ORDER BY created_at DESC, id DESC LIMIT ?2",
            COLS
        ))?;
        let rows = stmt.query_map(rusqlite::params![pattern, limit as i64], row_to_record)?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// Live-row counts, total and per scope.
    pub fn stats(&self) -> Result<MemoryStats> {
        let conn = self.conn.lock().unwrap();
        let total: i64 =
            conn.query_row("SELECT COUNT(*) FROM memories WHERE deleted = 0", [], |r| r.get(0))?;
        let mut stmt = conn.prepare(
            "SELECT scope, COUNT(*) FROM memories WHERE deleted = 0 GROUP BY scope ORDER BY scope",
        )?;
        let by_scope = stmt
            .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)))?
            .collect::<std::result::Result<BTreeMap<_, _>, _>>()?;
        Ok(MemoryStats { total, by_scope, fts_enabled: self.fts_enabled })
    }

    /// Hard-delete soft-deleted and superseded rows older than
    /// older_than_days (by updated_at). Returns the rows removed.
    pub fn prune(&self, older_than_days: f64) -> Result<usize> {
        let cutoff = now_unix() - older_than_days * 86_400.0;
        let conn = self.conn.lock().unwrap();
        let n = conn.execute(
            "DELETE FROM memories
             WHERE (deleted = 1 OR superseded_by IS NOT NULL) AND updated_at < ?1",
            rusqlite::params![cutoff],
        )?;
        if n > 0 && self.fts_enabled {
            sync_fts(&conn);
        }
        Ok(n)
    }

    /// Hard-delete ALL soft-deleted rows. Returns the rows removed.
    pub fn purge(&self) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let n = conn.execute("DELETE FROM memories WHERE deleted = 1", [])?;
        if n > 0 && self.fts_enabled {
            sync_fts(&conn);
        }
        Ok(n)
    }

    /// Mark old_id as replaced by new_id (sets superseded_by on the old
    /// row). Missing ids are a no-op.
    pub fn supersede(&self, old_id: i64, new_id: i64) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE memories SET superseded_by = ?2 WHERE id = ?1",
            rusqlite::params![old_id, new_id],
        )?;
        Ok(())
    }

    /// Export all live memories (id order) as a JSON array string.
    pub fn export_json(&self) -> Result<String> {
        let records = self.all_live()?;
        serde_json::to_string_pretty(&records)
            .map_err(|e| StoreError::Io(std::io::Error::other(e)))
    }

    /// Import a JSON array of records (as produced by [Self::export_json]).
    /// Lenient: missing fields get defaults, ids are preserved and replace
    /// conflicting rows. Returns the number of records imported.
    pub fn import_json(&self, json: &str) -> Result<usize> {
        let items: Vec<MemoryJson> =
            serde_json::from_str(json).map_err(|e| StoreError::Io(std::io::Error::other(e)))?;
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        let mut count = 0usize;
        for item in &items {
            let now = now_unix();
            let created = item.created_at.unwrap_or(now);
            let updated = item.updated_at.unwrap_or(created);
            let tags = serde_json::to_string(&item.tags).unwrap_or_else(|_| "[]".into());
            tx.execute(
                "INSERT OR REPLACE INTO memories
                 (id, content, scope, project, tags, created_at, updated_at, superseded_by, deleted)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0)",
                rusqlite::params![
                    item.id,
                    item.content,
                    item.scope,
                    item.project,
                    tags,
                    created,
                    updated,
                    item.superseded_by
                ],
            )?;
            if self.fts_enabled {
                let id = tx.last_insert_rowid();
                let _ = tx.execute("DELETE FROM memory_fts WHERE rowid = ?1", rusqlite::params![id]);
                let _ = tx.execute(
                    "INSERT INTO memory_fts(rowid, body) VALUES (?1, ?2)",
                    rusqlite::params![id, item.content],
                );
            }
            count += 1;
        }
        tx.commit()?;
        Ok(count)
    }

    /// Rebuild the FTS index from live memories. Returns the number of
    /// indexed rows, or Ok(0) when FTS5 is unavailable.
    pub fn reindex(&self) -> Result<usize> {
        if !self.fts_enabled {
            return Ok(0);
        }
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        tx.execute("DELETE FROM memory_fts", [])?;
        let n = tx.execute(
            "INSERT INTO memory_fts(rowid, body) SELECT id, content FROM memories WHERE deleted = 0",
            [],
        )?;
        tx.commit()?;
        Ok(n)
    }

    fn all_live(&self) -> Result<Vec<MemoryRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&format!(
            "SELECT {} FROM memories WHERE deleted = 0 ORDER BY id ASC",
            COLS
        ))?;
        let rows = stmt.query_map([], row_to_record)?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }
}

/// Drop FTS rows that no longer map to a live memory (best effort).
fn sync_fts(conn: &Connection) {
    let _ = conn.execute("DELETE FROM memory_fts WHERE rowid NOT IN (SELECT id FROM memories)", []);
}

fn row_to_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryRecord> {
    let tags_json: String = row.get("tags")?;
    Ok(MemoryRecord {
        id: row.get("id")?,
        content: row.get("content")?,
        scope: row.get("scope")?,
        project: row.get("project")?,
        tags: parse_tags(&tags_json),
        created_at: row.get("created_at")?,
        updated_at: row.get("updated_at")?,
        superseded_by: row.get("superseded_by")?,
    })
}

/// Lenient import record: missing fields get defaults, unknown fields are
/// ignored, so exports from the Python implementation import cleanly.
#[derive(Debug, Deserialize)]
struct MemoryJson {
    #[serde(default)]
    id: Option<i64>,
    content: String,
    #[serde(default = "default_scope")]
    scope: String,
    #[serde(default)]
    project: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    created_at: Option<f64>,
    #[serde(default)]
    updated_at: Option<f64>,
    #[serde(default)]
    superseded_by: Option<i64>,
}

fn default_scope() -> String {
    "project".to_string()
}

/// Parse a tags JSON blob; malformed tags degrade to an empty list.
fn parse_tags(json: &str) -> Vec<String> {
    serde_json::from_str(json).unwrap_or_default()
}

/// Escape LIKE wildcards in user input (paired with a backslash ESCAPE char).
fn escape_like(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c == '%' || c == '_' || c == '\\' {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

fn now_unix() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
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

    fn temp_db(name: &str) -> (PathBuf, MemoryDb) {
        let dir = std::env::temp_dir().join(format!("headroom-memory-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db = MemoryDb::open(&dir.join("memory.db")).expect("open temp memory db");
        (dir, db)
    }

    fn nm(content: &str) -> NewMemory {
        NewMemory { content: content.into(), scope: "project".into(), project: None, tags: vec![] }
    }

    /// Age a row (tests may touch the private connection directly).
    fn backdate(db: &MemoryDb, id: i64, days_ago: f64) {
        let conn = db.conn.lock().unwrap();
        conn.execute(
            "UPDATE memories SET created_at = ?2, updated_at = ?2 WHERE id = ?1",
            rusqlite::params![id, now_unix() - days_ago * 86_400.0],
        )
        .unwrap();
    }

    fn row_count_including_deleted(db: &MemoryDb) -> i64 {
        let conn = db.conn.lock().unwrap();
        conn.query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0)).unwrap()
    }

    #[test]
    fn add_get_roundtrip() {
        let (_dir, db) = temp_db("add_get");
        let id = db
            .add(&NewMemory {
                content: "hello world".into(),
                scope: "project".into(),
                project: Some("headroom".into()),
                tags: vec!["rust".into(), "port".into()],
            })
            .unwrap();
        let rec = db.get(id).unwrap().expect("row exists");
        assert_eq!(rec.id, id);
        assert_eq!(rec.content, "hello world");
        assert_eq!(rec.scope, "project");
        assert_eq!(rec.project.as_deref(), Some("headroom"));
        assert_eq!(rec.tags, vec!["rust", "port"]);
        assert!(rec.superseded_by.is_none());
        assert!(rec.created_at > 0.0 && rec.updated_at >= rec.created_at);
        assert!(db.get(999).unwrap().is_none());
    }

    #[test]
    fn edit_updates_content_and_timestamp() {
        let (_dir, db) = temp_db("edit");
        let id = db.add(&nm("v1")).unwrap();
        backdate(&db, id, 5.0);
        let before = db.get(id).unwrap().unwrap().updated_at;
        assert!(db.edit(id, "v2").unwrap());
        let rec = db.get(id).unwrap().unwrap();
        assert_eq!(rec.content, "v2");
        assert!(rec.updated_at > before);
        assert!(!db.edit(999, "nope").unwrap());
    }

    #[test]
    fn delete_is_soft() {
        let (_dir, db) = temp_db("delete");
        let id = db.add(&nm("gone soon")).unwrap();
        assert!(db.delete(id).unwrap());
        assert!(db.get(id).unwrap().is_none());
        assert!(db.list(10, None, None).unwrap().is_empty());
        assert!(!db.delete(id).unwrap()); // already deleted
        assert_eq!(row_count_including_deleted(&db), 1); // soft, not hard
    }

    #[test]
    fn list_filters_and_orders() {
        let (_dir, db) = temp_db("list");
        let a = db
            .add(&NewMemory { content: "old".into(), scope: "user".into(), project: None, tags: vec![] })
            .unwrap();
        let b = db.add(&nm("mid")).unwrap();
        let c = db.add(&nm("new")).unwrap();
        backdate(&db, a, 10.0);
        let all = db.list(10, None, None).unwrap();
        let ids: Vec<i64> = all.iter().map(|r| r.id).collect();
        assert_eq!(ids, vec![c, b, a]); // newest first
        assert_eq!(db.list(2, None, None).unwrap().len(), 2); // limit
        let users = db.list(10, Some("user"), None).unwrap();
        assert_eq!(users.len(), 1);
        assert_eq!(users[0].id, a);
        let now_ms = (now_unix() * 1000.0) as i64;
        assert_eq!(db.list(10, None, Some(now_ms + 60_000)).unwrap().len(), 0); // future cutoff
        assert_eq!(db.list(10, None, Some(0)).unwrap().len(), 3);
    }

    #[test]
    fn search_finds_and_falls_back() {
        let (_dir, db) = temp_db("search");
        db.add(&nm("the rust port of headroom compresses context")).unwrap();
        db.add(&nm("unrelated note about pasta")).unwrap();
        let hits = db.search("headroom", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].content.contains("headroom"));
        let id = db.add(&nm("secret launch codes")).unwrap();
        db.delete(id).unwrap();
        assert!(db.search("launch codes", 10).unwrap().is_empty()); // deleted invisible
        let _ = db.search("50%_off(\"", 10).unwrap(); // FTS-hostile query must not error
        db.add(&nm("headroom again")).unwrap();
        assert_eq!(db.search("headroom", 1).unwrap().len(), 1); // limit respected
        assert!(db.search("   ", 10).unwrap().is_empty()); // blank query
    }

    #[test]
    fn supersede_points_old_to_new() {
        let (_dir, db) = temp_db("supersede");
        let old = db.add(&nm("old fact")).unwrap();
        let new = db.add(&nm("new fact")).unwrap();
        db.supersede(old, new).unwrap();
        assert_eq!(db.get(old).unwrap().unwrap().superseded_by, Some(new));
        assert_eq!(db.get(new).unwrap().unwrap().superseded_by, None);
    }

    #[test]
    fn prune_removes_old_deleted_and_superseded() {
        let (_dir, db) = temp_db("prune");
        let dead = db.add(&nm("deleted long ago")).unwrap();
        let superseded = db.add(&nm("replaced long ago")).unwrap();
        let replacement = db.add(&nm("current")).unwrap();
        let fresh_deleted = db.add(&nm("deleted recently")).unwrap();
        db.delete(dead).unwrap();
        db.supersede(superseded, replacement).unwrap();
        db.delete(fresh_deleted).unwrap();
        backdate(&db, dead, 40.0);
        backdate(&db, superseded, 40.0);
        assert_eq!(db.prune(30.0).unwrap(), 2);
        assert!(db.get(dead).unwrap().is_none()); // hard-deleted
        assert!(db.get(superseded).unwrap().is_none()); // hard-deleted
        assert_eq!(row_count_including_deleted(&db), 2); // replacement + fresh soft-deleted
        assert_eq!(db.prune(30.0).unwrap(), 0); // nothing left to prune
    }

    #[test]
    fn purge_removes_all_soft_deleted() {
        let (_dir, db) = temp_db("purge");
        let a = db.add(&nm("a")).unwrap();
        let b = db.add(&nm("b")).unwrap();
        let c = db.add(&nm("c")).unwrap();
        db.delete(a).unwrap();
        db.delete(b).unwrap();
        assert_eq!(db.purge().unwrap(), 2);
        assert_eq!(row_count_including_deleted(&db), 1);
        assert_eq!(db.get(c).unwrap().unwrap().content, "c");
        assert_eq!(db.purge().unwrap(), 0);
    }

    #[test]
    fn export_import_roundtrip() {
        let (_dir, db) = temp_db("export");
        let a = db
            .add(&NewMemory {
                content: "alpha".into(),
                scope: "project".into(),
                project: Some("p".into()),
                tags: vec!["t1".into()],
            })
            .unwrap();
        let b = db
            .add(&NewMemory { content: "beta".into(), scope: "user".into(), project: None, tags: vec![] })
            .unwrap();
        db.delete(b).unwrap(); // deleted rows are not exported
        let json = db.export_json().unwrap();
        assert!(json.contains("alpha"));
        assert!(!json.contains("beta"));

        let (_dir2, db2) = temp_db("import");
        assert_eq!(db2.import_json(&json).unwrap(), 1);
        let rec = db2.get(a).unwrap().unwrap(); // id preserved
        assert_eq!(rec.content, "alpha");
        assert_eq!(rec.tags, vec!["t1"]);
        assert_eq!(rec.scope, "project");
        assert_eq!(rec.project.as_deref(), Some("p"));
        assert_eq!(db2.import_json(&json).unwrap(), 1); // re-import replaces by id
        assert_eq!(db2.stats().unwrap().total, 1);
    }

    #[test]
    fn import_rejects_garbage() {
        let (_dir, db) = temp_db("import_bad");
        assert!(db.import_json("not json").is_err());
        assert_eq!(db.import_json("[]").unwrap(), 0);
    }

    #[test]
    fn stats_counts_by_scope() {
        let (_dir, db) = temp_db("stats");
        db.add(&nm("p1")).unwrap();
        db.add(&nm("p2")).unwrap();
        db.add(&NewMemory { content: "u1".into(), scope: "user".into(), project: None, tags: vec![] })
            .unwrap();
        let dead = db.add(&nm("dead")).unwrap();
        db.delete(dead).unwrap();
        let s = db.stats().unwrap();
        assert_eq!(s.total, 3);
        assert_eq!(s.by_scope.get("project"), Some(&2));
        assert_eq!(s.by_scope.get("user"), Some(&1));
        assert_eq!(s.fts_enabled, db.fts_enabled());
    }

    #[test]
    fn reindex_rebuilds_fts() {
        let (_dir, db) = temp_db("reindex");
        db.add(&nm("index me")).unwrap();
        let total = db.stats().unwrap().total;
        let n = db.reindex().unwrap();
        if db.stats().unwrap().fts_enabled {
            assert_eq!(n, usize::try_from(total).unwrap_or(0));
            assert_eq!(db.search("index me", 10).unwrap().len(), 1);
        } else {
            assert_eq!(n, 0); // FTS unavailable: no-op
        }
    }
}
