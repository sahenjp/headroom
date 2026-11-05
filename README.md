# headroom (Rust port)

Rust port of [headroom-ai](https://pypi.org/project/headroom-ai/) — the local
context-optimization layer for LLM applications. The Python original (v0.41.0)
is ~240k lines; this port re-implements the full product surface in Rust: the
compression transforms, the local optimization proxy, the CCR
(Compress-Cache-Retrieve) store, the savings ledger, the memory subsystem, the
MCP server, and the complete CLI.

## Scope

**headroom-core** — pure compression transforms (no IO):
- content-type detection (JSON array / search results / build logs / git
  diff / HTML / tabular / config / source code / plain text)
- SmartCrusher (statistical JSON-array crusher with lossless-first
  `[N]{schema}` CSV densification), search / log / diff compressors,
  TextCrusher (deterministic extractive, BM25)
- config compressor (YAML/TOML/INI folding), tabular compressor, HTML
  main-content extraction, dense-line elider, error detection, lossless
  folds (grep heading / ANSI strip / log runs / JSON minify),
  cross-turn dedup, Read lifecycle, spreadsheet (xlsx -> CSV)
- CCR marker grammar (`<<ccr:HASH>>`, `Retrieve more: hash=...`), compaction
  codec (encode + expand), tag protector, estimating token counter

**headroom-store** — durable state:
- CCR store: SQLite (`~/.headroom/ccr_store.db`), schema-compatible with
  the Python implementation (same table, same JSON entry blob)
- savings ledger: JSONL (`~/.headroom/savings_events.jsonl`), schema v2
  events compatible with the Python ledger (the Rust CLI reads the ledger
  your Python proxy already wrote)
- memory DB: SQLite with FTS5 search (`~/.headroom/memory.db`)

**headroom-proxy** — axum-based local proxy (default `127.0.0.1:8787`):
- `POST /v1/messages` (Anthropic), `POST /v1/chat/completions` (OpenAI),
  `POST /v1/responses` + `/v1/codex/responses` (OpenAI Responses API)
- compresses large tool outputs in request bodies; SSE responses stream
  through untouched (with usage observation); fail-open on any error
- rate limiting (RPM/TPM sliding window), USD budget check (hourly/daily/
  monthly), Prometheus `/metrics`, cache-mode session prefix freezing
- `GET /livez`, `GET /readyz`, `GET /stats`, `POST /v1/retrieve`,
  `GET /v1/retrieve/{hash}`, catch-all passthrough

**headroom-mcp** — stdio MCP server exposing `headroom_compress`,
`headroom_retrieve`, `headroom_stats`, `headroom_read` (same tool names +
schemas as the Python server).

**headroom-cli** — the `headroom` binary, all 28 commands:
proxy, doctor, savings, retrieve, mcp, memory (list/show/stats/add/edit/
delete/prune/purge/export/import/reindex/search), perf, inspect, audit-reads,
wrap, unwrap, install (apply/remove/status), init, diff, loc, sg, update,
rollout, telemetry, copilot-auth, capture, evals, learn, agent-savings,
output-savings, dashboard, deploy, recover, tools.

**Documented deviations** (deterministic Rust replacements for ML/network
features): Kompress (ModernBERT ONNX) -> TextCrusher; trafilatura HTML
extraction -> heuristic extraction; litellm pricing -> built-in price table
with the same $3/1M fallback; bundled tool binaries (difft/scc/ast-grep) ->
PATH passthrough; no telemetry beacon is sent.

## Design rules (ported from the Python original)

1. **Fail-open everywhere** — compression, store, and ledger failures never
   break a request; the original body is forwarded instead.
2. **Deterministic compression** — no randomness or wall-clock decisions in
   the pipeline, so provider prompt-cache prefixes stay byte-stable.
3. **Inflation guard** — if the compressed body is larger, the original is
   forwarded.
4. **Protection gates** — the last 4 messages, frozen prefix, excluded tools
   (Read/Glob/Grep/Write/Edit/WebSearch/WebFetch/...), and error outputs
   (<= 8000 chars) are never compressed.
5. **Hash discipline** — CCR hashes are SHA-256 truncated to 24 hex (Python
   store default) or 12 hex (native crusher paths); markers accept 12-24.

## Cross-implementation compatibility (verified)

- Rust-encoded densified blocks expand via the Python `expand_compacted()`
- Rust-generated CCR markers match the Python marker regexes
- Rust-written CCR store rows are readable by the Python sqlite3 client
- The Rust `headroom savings` reads the ledger your Python proxy already
  wrote (schema v2 events)

## Build

    cargo build --release
    # binary at target/release/headroom

## Run

    headroom proxy                     # listen on 127.0.0.1:8787
    headroom proxy --rpm 60 --budget 10 --mode cache
    headroom doctor                    # health check
    headroom savings                   # savings report (shared ledger)
    headroom memory add "lesson"       # memories (SQLite FTS5)
    headroom retrieve <hash>           # print an original from the CCR store
    headroom mcp                       # MCP stdio server

Client wiring (same as Python headroom):

    export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
    export OPENAI_BASE_URL=http://127.0.0.1:8787/v1

## Tests

    cargo test --workspace          # 272 unit tests
    python3 scripts/golden.py       # cross-implementation checks (after
                                    # cargo test -p headroom-core --test golden)

## Layout

    crates/headroom-core      transforms, detection, codec, markers, tokens
    crates/headroom-store     CCR SQLite store, savings ledger, memory DB
    crates/headroom-proxy     axum proxy: routes, compression, upstream,
                              rate limit, budget, metrics, sessions, SSE
    crates/headroom-mcp       stdio JSON-RPC MCP server
    crates/headroom-cli       the headroom binary (28 commands)
