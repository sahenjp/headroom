//! headroom memory — CLI for the durable memory store (headroom-store::memory).
//!
//! Wired into main.rs by the lead; until then this module compiles
//! standalone as part of the headroom-cli lib target.

#![allow(dead_code)]

use std::path::PathBuf;

use clap::Subcommand;
use headroom_store::memory::{MemoryDb, MemoryRecord, NewMemory};

/// headroom memory subcommands.
#[derive(Debug, Subcommand)]
pub enum MemoryCommand {
    /// List recent memories (newest first).
    List {
        /// Maximum number of memories to show.
        #[arg(long, default_value_t = 20)]
        limit: usize,
        /// Only memories with this scope (project|user|global).
        #[arg(long)]
        scope: Option<String>,
    },
    /// Show one memory in full.
    Show {
        /// Memory id.
        id: i64,
    },
    /// Show memory counts and search-index state.
    Stats,
    /// Add a memory.
    Add {
        /// Memory content.
        content: String,
        /// Scope: project, user or global.
        #[arg(long, default_value = "project")]
        scope: String,
        /// Comma-separated tags.
        #[arg(long)]
        tags: Option<String>,
    },
    /// Replace a memory's content.
    Edit {
        /// Memory id.
        id: i64,
        /// New content.
        content: String,
    },
    /// Soft-delete a memory (recoverable until prune/purge).
    Delete {
        /// Memory id.
        id: i64,
    },
    /// Hard-delete soft-deleted and superseded rows older than N days.
    Prune {
        /// Age cutoff in days.
        #[arg(long, default_value_t = 30.0)]
        older_than: f64,
    },
    /// Hard-delete ALL soft-deleted memories (destructive).
    Purge {
        /// Confirm the destructive purge.
        #[arg(long)]
        confirm: bool,
    },
    /// Export live memories as a JSON array.
    Export {
        /// Write to this file instead of stdout.
        #[arg(long)]
        path: Option<String>,
    },
    /// Import memories from a JSON export file.
    Import {
        /// Path to the JSON export.
        path: String,
    },
    /// Rebuild the FTS search index from live memories.
    Reindex,
    /// Full-text search across memories.
    Search {
        /// Query (FTS5 syntax when the index is available, else substring).
        query: String,
        /// Maximum number of hits.
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
}

/// Resolve the default memory db and dispatch.
pub fn run(cmd: MemoryCommand) -> anyhow::Result<()> {
    run_with_db(MemoryDb::default_path(), cmd)
}

/// Dispatch against an explicit db path (tests, custom workspaces).
pub fn run_with_db(db: PathBuf, cmd: MemoryCommand) -> anyhow::Result<()> {
    let db = MemoryDb::open(&db)
        .map_err(|e| anyhow::anyhow!("cannot open memory db {}: {e}", db.display()))?;
    dispatch(&db, cmd)
}

fn dispatch(db: &MemoryDb, cmd: MemoryCommand) -> anyhow::Result<()> {
    match cmd {
        MemoryCommand::List { limit, scope } => {
            print_rows(&db.list(limit, scope.as_deref(), None)?);
        }
        MemoryCommand::Show { id } => match db.get(id)? {
            Some(rec) => print_record(&rec),
            None => anyhow::bail!("memory {id} not found"),
        },
        MemoryCommand::Stats => {
            let s = db.stats()?;
            println!("total {}", s.total);
            println!(
                "fts    {}",
                if s.fts_enabled { "enabled" } else { "disabled (LIKE fallback)" }
            );
            for (scope, n) in &s.by_scope {
                println!("{scope:<7} {n}");
            }
        }
        MemoryCommand::Add { content, scope, tags } => {
            let mem = NewMemory {
                content,
                scope,
                project: None,
                tags: tags.map(|t| parse_tags(&t)).unwrap_or_default(),
            };
            let id = db.add(&mem)?;
            println!("added memory {id}");
        }
        MemoryCommand::Edit { id, content } => {
            if db.edit(id, &content)? {
                println!("updated memory {id}");
            } else {
                anyhow::bail!("memory {id} not found");
            }
        }
        MemoryCommand::Delete { id } => {
            if db.delete(id)? {
                println!("deleted memory {id}");
            } else {
                anyhow::bail!("memory {id} not found");
            }
        }
        MemoryCommand::Prune { older_than } => {
            let n = db.prune(older_than)?;
            println!("pruned {n} row(s)");
        }
        MemoryCommand::Purge { confirm } => {
            if !confirm {
                anyhow::bail!("purge hard-deletes ALL soft-deleted memories; re-run with --confirm");
            }
            let n = db.purge()?;
            println!("purged {n} row(s)");
        }
        MemoryCommand::Export { path } => {
            let json = db.export_json()?;
            match path {
                Some(p) => {
                    std::fs::write(&p, json)?;
                    println!("exported to {p}");
                }
                None => println!("{json}"),
            }
        }
        MemoryCommand::Import { path } => {
            let json = std::fs::read_to_string(&path)?;
            let n = db.import_json(&json)?;
            println!("imported {n} memory(ies)");
        }
        MemoryCommand::Reindex => {
            let n = db.reindex()?;
            if n == 0 && !db.fts_enabled() {
                println!("fts index unavailable; search uses the LIKE fallback");
            } else {
                println!("reindexed {n} row(s)");
            }
        }
        MemoryCommand::Search { query, limit } => {
            print_rows(&db.search(&query, limit)?);
        }
    }
    Ok(())
}

/// One line per memory: id  created  scope  tags  content[:80].
fn print_rows(rows: &[MemoryRecord]) {
    if rows.is_empty() {
        println!("no memories");
        return;
    }
    for r in rows {
        println!(
            "{:<5} {:<19} {:<7} {:<12} {}",
            r.id,
            format_ts(r.created_at),
            r.scope,
            if r.tags.is_empty() { "-".to_string() } else { r.tags.join(",") },
            truncate_chars(&r.content, 80),
        );
    }
}

fn print_record(r: &MemoryRecord) {
    println!("id      {}", r.id);
    println!("scope   {}", r.scope);
    println!("project {}", r.project.as_deref().unwrap_or("-"));
    println!("tags    {}", if r.tags.is_empty() { "-".to_string() } else { r.tags.join(", ") });
    println!("created {}", format_ts(r.created_at));
    println!("updated {}", format_ts(r.updated_at));
    if let Some(next) = r.superseded_by {
        println!("superseded by {next}");
    }
    println!("content:");
    println!("{}", r.content);
}

/// Parse a comma-separated tag list, trimming and dropping empties.
fn parse_tags(raw: &str) -> Vec<String> {
    raw.split(',').map(str::trim).filter(|t| !t.is_empty()).map(str::to_string).collect()
}

/// Truncate to at most max chars, on char boundaries.
fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}…")
    }
}

/// Unix seconds -> "YYYY-MM-DD HH:MM:SS" (UTC, no external deps).
fn format_ts(unix: f64) -> String {
    let secs = unix.floor() as i64;
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (y, mo, d) = civil_from_days(days);
    format!("{y:04}-{mo:02}-{d:02} {:02}:{:02}:{:02}", tod / 3600, (tod % 3600) / 60, tod % 60)
}

/// Days-since-epoch to civil date (Howard Hinnant's algorithm, as in ledger.rs).
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

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("headroom-memory-cmd-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("memory.db")
    }

    #[test]
    fn add_then_list() {
        let path = temp_db_path("add_list");
        run_with_db(
            path.clone(),
            MemoryCommand::Add {
                content: "cli smoke test memory".into(),
                scope: "project".into(),
                tags: Some("cli,test".into()),
            },
        )
        .unwrap();
        // The command printed the row; assert on the resulting db state.
        let db = MemoryDb::open(&path).unwrap();
        let rows = db.list(10, None, None).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].content, "cli smoke test memory");
        assert_eq!(rows[0].tags, vec!["cli", "test"]);
        run_with_db(path, MemoryCommand::List { limit: 10, scope: None }).unwrap();
    }

    #[test]
    fn edit_delete_and_show() {
        let path = temp_db_path("edit_delete");
        run_with_db(
            path.clone(),
            MemoryCommand::Add { content: "before".into(), scope: "project".into(), tags: None },
        )
        .unwrap();
        run_with_db(path.clone(), MemoryCommand::Edit { id: 1, content: "after".into() }).unwrap();
        run_with_db(path.clone(), MemoryCommand::Show { id: 1 }).unwrap();
        run_with_db(path.clone(), MemoryCommand::Delete { id: 1 }).unwrap();
        let db = MemoryDb::open(&path).unwrap();
        assert!(db.list(10, None, None).unwrap().is_empty());
        // Missing ids are errors.
        assert!(run_with_db(path.clone(), MemoryCommand::Show { id: 42 }).is_err());
        assert!(run_with_db(path, MemoryCommand::Edit { id: 42, content: "x".into() }).is_err());
    }

    #[test]
    fn purge_requires_confirm() {
        let path = temp_db_path("purge_confirm");
        assert!(run_with_db(path, MemoryCommand::Purge { confirm: false }).is_err());
    }

    #[test]
    fn search_stats_reindex_run() {
        let path = temp_db_path("search_stats");
        run_with_db(
            path.clone(),
            MemoryCommand::Add {
                content: "searchable haystack".into(),
                scope: "user".into(),
                tags: None,
            },
        )
        .unwrap();
        run_with_db(path.clone(), MemoryCommand::Search { query: "haystack".into(), limit: 5 })
            .unwrap();
        run_with_db(path.clone(), MemoryCommand::Stats).unwrap();
        run_with_db(path, MemoryCommand::Reindex).unwrap();
    }

    #[test]
    fn export_import_roundtrip() {
        let src = temp_db_path("export_src");
        let dst = temp_db_path("import_dst");
        run_with_db(
            src.clone(),
            MemoryCommand::Add {
                content: "roundtrip".into(),
                scope: "project".into(),
                tags: Some("x".into()),
            },
        )
        .unwrap();
        let export_file = src.parent().unwrap().join("export.json");
        run_with_db(
            src.clone(),
            MemoryCommand::Export { path: Some(export_file.to_string_lossy().into_owned()) },
        )
        .unwrap();
        run_with_db(
            dst.clone(),
            MemoryCommand::Import { path: export_file.to_string_lossy().into_owned() },
        )
        .unwrap();
        let db = MemoryDb::open(&dst).unwrap();
        assert_eq!(db.list(10, None, None).unwrap()[0].content, "roundtrip");
    }

    #[test]
    fn tags_parse() {
        assert_eq!(parse_tags("a, b,, c "), vec!["a", "b", "c"]);
        assert!(parse_tags("").is_empty());
    }

    #[test]
    fn ts_format() {
        // 2026-11-05T12:00:20Z (same epoch second verified in ledger.rs tests).
        assert_eq!(format_ts(1_793_880_020.0), "2026-11-05 12:00:20");
    }
}
