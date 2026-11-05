//! Cross-implementation golden fixtures: Rust-generated codec blocks, CCR
//! markers, and a CCR store row are written to /tmp/headroom-golden/ and
//! validated by the Python headroom (scripts/golden.py).
//! Run via: cargo test --test golden

use headroom_core::ccr;
use headroom_core::compaction::encode_compacted;
use headroom_store::ccr::{CompressionStore, StoreConfig};
use serde_json::json;

#[test]
fn write_codec_and_marker_fixtures() {
    // Uniform flat objects (the SmartCrusher compaction sweet spot).
    let items: Vec<serde_json::Value> = (0..24)
        .map(|i| {
            json!({
                "name": format!("test_case_{i}"),
                "status": if i % 7 == 3 { "failed" } else { "passed" },
                "duration_ms": 100 + i * 3,
            })
        })
        .collect();
    let densified = encode_compacted(&items).expect("uniform objects must densify");

    // Dotted flattening + nulls + empty strings + literal "null".
    let mixed = vec![
        json!({"path": "a.rs", "meta": {"owner": "alice"}, "note": null, "tag": ""}),
        json!({"path": "b.rs", "meta": {"owner": "bob"}, "note": "null", "tag": "x"}),
        json!({"path": "c.rs", "meta": {"owner": null}, "note": "comma, quote\" here", "tag": "y,z"}),
    ];
    let densified_mixed = encode_compacted(&mixed).expect("mixed objects must densify");

    let dir = std::env::temp_dir().join("headroom-golden");
    std::fs::create_dir_all(&dir).unwrap();
    // One JSON-encoded block per line (the densified text contains raw
    // newlines, so it must be JSON-escaped to survive the JSONL fixture).
    let fixture = format!(
        "{}\n{}\n",
        serde_json::to_string(&densified).unwrap(),
        serde_json::to_string(&densified_mixed).unwrap()
    );
    std::fs::write(dir.join("rust_blocks.jsonl"), fixture).unwrap();

    // Every marker grammar form the Python regexes must accept.
    let markers = vec![
        ccr::pointer_marker(&ccr::ccr_hash("x")),
        ccr::row_drop_marker(&ccr::ccr_hash12("y"), 100),
        ccr::blob_marker(&ccr::ccr_hash12("z"), "base64", "4.5KB"),
        ccr::scalar_sentinel(5, &ccr::ccr_hash12("w"), 5),
        ccr::items_marker(100, 15, "", &ccr::ccr_hash("v"), 30),
        ccr::elided_marker(42, "comment/blank lines", &ccr::ccr_hash("u")),
    ];
    std::fs::write(dir.join("rust_markers.txt"), markers.join("\n")).unwrap();
    println!("fixtures at {}", dir.display());
}

#[test]
fn write_store_fixture_for_python() {
    let db = std::env::temp_dir().join("headroom-golden").join("rust_ccr.db");
    let _ = std::fs::remove_file(&db);
    let store = CompressionStore::open(&StoreConfig {
        path: Some(db.clone()),
        ttl: 1800,
        max_entries: 100,
    })
    .expect("store must open");
    store
        .store_original(
            "original payload for golden test",
            "<<ccr:placeholder>>",
            42,
            3,
            Some("Bash"),
            Some("log_compressor"),
        )
        .expect("store must write");
    println!("store fixture at {}", db.display());
}
