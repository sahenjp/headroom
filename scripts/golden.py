#!/usr/bin/env python3
"""Golden test: the Rust port's output must be readable by the Python headroom.

Checks (against the installed headroom-ai 0.41.0):
1. compaction blocks written by Rust expand via Python expand_compacted().
2. CCR markers written by Rust match the Python marker regexes.
3. A CCR store row written by Rust is readable by Python sqlite3.
"""
import json
import os
import re
import sqlite3
import sys
import tempfile

sys.path.insert(0, "/home/sahen/.local/share/uv/tools/headroom-ai/lib/python3.12/site-packages")
from headroom.transforms.compaction_codec import expand_compacted, is_compacted  # noqa: E402

TMP = os.path.join(tempfile.gettempdir(), "headroom-golden")
failures = []

# --- 1. codec blocks ---
with open(os.path.join(TMP, "rust_blocks.jsonl")) as f:
    blocks = [l.strip() for l in f if l.strip()]
for i, block in enumerate(blocks):
    if not is_compacted(block):
        failures.append(f"block {i}: python is_compacted()=False")
        continue
    expanded = expand_compacted(block)
    if expanded is None:
        failures.append(f"block {i}: python expand_compacted()=None")
print(f"codec blocks: {len(blocks)} checked")

# --- 2. marker regexes (from transforms/compression_units.py + parser.py) ---
CCR_MARKER_RE = re.compile(r"(?m)^.*(?:Retrieve more: hash=|Retrieve original: hash=|<<ccr:[^>]+>>).*$")
with open(os.path.join(TMP, "rust_markers.txt")) as f:
    markers = [l.rstrip("\n") for l in f if l.strip()]
for i, m in enumerate(markers):
    if not CCR_MARKER_RE.search(m):
        failures.append(f"marker {i}: python CCR_MARKER_RE does not match: {m!r}")
print(f"markers: {len(markers)} checked")

# --- 3. CCR store cross-read (Rust wrote it with the store crate) ---
db = os.environ.get("GOLDEN_CCR_DB")
if db and os.path.exists(db):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT hash, entry_json, created_at, ttl FROM ccr_entries").fetchall()
        for hash_key, entry_json, created_at, ttl in rows:
            entry = json.loads(entry_json)
            for field in ("hash", "original_content", "compressed_content", "created_at", "ttl"):
                if field not in entry:
                    failures.append(f"store row {hash_key}: missing field {field}")
        print(f"store rows: {len(rows)} checked")
    finally:
        conn.close()
else:
    print("store rows: skipped (set GOLDEN_CCR_DB)")

if failures:
    print("GOLDEN FAILURES:")
    for f_ in failures:
        print(" -", f_)
    sys.exit(1)
print("GOLDEN OK: all cross-implementation checks passed")
