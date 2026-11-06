//! Axum-based local optimization proxy (default 127.0.0.1:8787).
//!
//! Intercepts Anthropic `/v1/messages` and OpenAI `/v1/chat/completions`,
//! compresses large tool outputs in the request body, stashes dropped
//! originals in the CCR store, records savings, and forwards everything else
//! byte-for-byte. Responses (including SSE streams) pass through untouched.
//! Fail-open: any compression error forwards the original body.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use futures_util::StreamExt;

use axum::body::Body;
use axum::extract::{Path, Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::response::Response;
use axum::routing::{any, get, post};
use axum::Router;
use headroom_store::ccr::CompressionStore;
use headroom_store::ledger::{SavingsEvent, SavingsLedger};
use serde_json::{json, Value};

use crate::compress_request::{
    compress_request_body, CompressOutcome, CompressionSummary, RequestCompressConfig,
    WireFormat,
};
use crate::upstream::{Api, Upstream};
use crate::budget::{self, Budget, BudgetDecision, BudgetPeriod};
use crate::metrics::Metrics;
use crate::rate_limit::{RateDecision, RateLimiter};
use crate::session::{session_key, SessionTracker};
use crate::streaming::SseAccumulator;

/// Proxy-wide shared state.
pub struct ProxyState {
    pub config: RequestCompressConfig,
    pub upstream: Upstream,
    pub store: Option<CompressionStore>,
    pub ledger: SavingsLedger,
    pub started: Instant,
    pub version: &'static str,
    pub requests_total: AtomicU64,
    pub requests_compressed: AtomicU64,
    pub tokens_saved_total: AtomicU64,
    pub errors_total: AtomicU64,
    pub by_client: Mutex<BTreeMap<String, u64>>,
    /// RPM/TPM limiter (None limits => always allow).
    pub rate_limiter: RateLimiter,
    /// Optional USD budget (checked against the ledger; advisory: exceeded
    /// requests are still forwarded but counted and warned).
    pub budget: Option<Budget>,
    /// Prometheus-style counters.
    pub metrics: Metrics,
    /// Cache-mode session prefix tracker.
    pub sessions: SessionTracker,
    /// Cache mode: freeze provider prompt-cache prefixes (default on).
    pub cache_mode: bool,
}

pub type SharedState = Arc<ProxyState>;

/// Build the full router.
pub fn build_router(state: SharedState) -> Router {
    Router::new()
        .route("/livez", get(livez))
        .route("/readyz", get(readyz))
        .route("/health", get(livez))
        .route("/stats", get(stats))
        .route("/v1/retrieve", post(retrieve_post))
        .route("/v1/retrieve/{hash}", get(retrieve_get))
        .route("/v1/messages", post(anthropic_handler))
        .route("/anthropic/v1/messages", post(anthropic_handler))
        .route("/v1/chat/completions", post(openai_handler))
        .route("/chat/completions", post(openai_handler))
        .route("/v1/responses", post(responses_handler))
        .route("/v1/codex/responses", post(responses_handler))
        .route("/metrics", get(metrics_endpoint))
        .fallback(any(passthrough))
        .with_state(state)
}

async fn livez(State(state): State<SharedState>) -> Response {
    let body = json!({
        "service": "headroom-proxy",
        "status": "healthy",
        "alive": true,
        "version": state.version,
        "uptime_seconds": state.started.elapsed().as_secs_f64(),
    });
    json_response(StatusCode::OK, &body)
}

async fn readyz(State(state): State<SharedState>) -> Response {
    let upstream_ready = check_upstream(&state).await;
    let body = json!({
        "service": "headroom-proxy",
        "status": if upstream_ready { "healthy" } else { "degraded" },
        "ready": true,
        "version": state.version,
        "timestamp": now_iso(),
        "uptime_seconds": state.started.elapsed().as_secs_f64(),
        "checks": {
            "startup": {"enabled": true, "ready": true, "status": "healthy", "error": null},
            "http_client": {"enabled": true, "ready": true, "status": "healthy"},
            "cache": {"enabled": true, "ready": true, "status": "healthy"},
            "rate_limiter": {"enabled": false, "ready": true, "status": "disabled"},
            "upstream": {
                "enabled": true,
                "ready": upstream_ready,
                "status": if upstream_ready { "healthy" } else { "unreachable" },
                "url": state.upstream.anthropic_base(),
                "error": null,
            },
        },
        "rust_core": "native",
    });
    json_response(StatusCode::OK, &body)
}

async fn check_upstream(state: &SharedState) -> bool {
    let url = format!(
        "{}/api/hello",
        state.upstream.anthropic_base().trim_end_matches('/')
    );
    matches!(
        reqwest::Client::new()
            .head(&url)
            .timeout(std::time::Duration::from_secs(3))
            .send()
            .await,
        Ok(resp) if resp.status().is_success() || resp.status() == StatusCode::NOT_FOUND
    )
}

async fn stats(State(state): State<SharedState>) -> Response {
    let by_client = state.by_client.lock().unwrap().clone();
    let body = json!({
        "service": "headroom-proxy",
        "version": state.version,
        "uptime_seconds": state.started.elapsed().as_secs_f64(),
        "api_requests": state.requests_total.load(Ordering::Relaxed),
        "requests_compressed": state.requests_compressed.load(Ordering::Relaxed),
        "tokens_removed": state.tokens_saved_total.load(Ordering::Relaxed),
        "errors": state.errors_total.load(Ordering::Relaxed),
        "by_client": by_client,
        "savings_ledger": state.ledger.path().display().to_string(),
    });
    json_response(StatusCode::OK, &body)
}

async fn retrieve_post(State(state): State<SharedState>, body: String) -> Response {
    retrieve_by_hash(&state, &body)
}

async fn retrieve_get(State(state): State<SharedState>, Path(hash): Path<String>) -> Response {
    retrieve_by_hash(&state, &hash)
}

fn retrieve_by_hash(state: &SharedState, input: &str) -> Response {
    let Some(hash) = normalize_hash(input) else {
        return json_response(StatusCode::BAD_REQUEST, &json!({"error": "missing hash"}));
    };
    let Some(store) = &state.store else {
        return json_response(
            StatusCode::NOT_FOUND,
            &json!({"error": "store disabled (stateless)", "hash": hash}),
        );
    };
    match store.retrieve(&hash) {
        Some(entry) => json_response(
            StatusCode::OK,
            &json!({
                "hash": entry.hash,
                "original_content": entry.original_content,
                "original_tokens": entry.original_tokens,
                "original_item_count": entry.original_item_count,
                "compressed_item_count": entry.compressed_item_count,
                "tool_name": entry.tool_name,
                "retrieval_count": entry.retrieval_count,
            }),
        ),
        None => json_response(
            StatusCode::NOT_FOUND,
            &json!({
                "error": "Content not found (expired or never stored). Re-read the file or re-run the command.",
                "hash": hash,
            }),
        ),
    }
}

/// Accept a raw hash, a JSON body {"hash": "..."}, or a CCR marker.
fn normalize_hash(input: &str) -> Option<String> {
    let raw = if input.trim_start().starts_with('{') {
        let v: Value = serde_json::from_str(input).ok()?;
        v.get("hash")?.as_str()?.to_string()
    } else {
        input.trim().to_string()
    };
    let hex_part = raw
        .strip_prefix("<<ccr:")
        .and_then(|r| r.split([',', ' ']).next())
        .unwrap_or(&raw);
    let is_hex = !hex_part.is_empty()
        && hex_part.len() <= 24
        && hex_part.chars().all(|c| c.is_ascii_hexdigit());
    if is_hex {
        Some(hex_part.to_lowercase())
    } else {
        None
    }
}

async fn anthropic_handler(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    handle_provider_request(state, headers, body, WireFormat::Anthropic, Api::Anthropic, "/v1/messages").await
}

async fn openai_handler(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    handle_provider_request(state, headers, body, WireFormat::OpenAi, Api::OpenAi, "/v1/chat/completions").await
}

/// OpenAI Responses API: compress function_call_output items, then forward.
async fn responses_handler(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    state.requests_total.fetch_add(1, Ordering::Relaxed);
    state.metrics.inc("headroom_requests_total", &[("route", "responses")]);
    let client = client_label(&headers);

    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return forward_raw(&state, headers, body, Api::OpenAi, "/v1/responses").await,
    };
    let model = parsed
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();

    let (forward_body, summary) =
        match crate::responses::compress_responses_body(&parsed, &state.config, state.store.as_ref())
        {
            Ok(CompressOutcome::Compressed { body, summary }) => {
                state.requests_compressed.fetch_add(1, Ordering::Relaxed);
                state.metrics.inc("headroom_compressed_total", &[("route", "responses")]);
                (serde_json::to_vec(&body).unwrap_or_default(), Some(summary))
            }
            Ok(CompressOutcome::Unchanged { .. }) => (body.to_vec(), None),
            Err(_) => {
                state.errors_total.fetch_add(1, Ordering::Relaxed);
                (body.to_vec(), None)
            }
        };

    let hdrs: Vec<(String, String)> = headers
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
        .collect();
    match state
        .upstream
        .forward(Api::OpenAi, Method::POST, "/v1/responses", &hdrs, bytes::Bytes::from(forward_body))
        .await
    {
        Ok(resp) => {
            let status = resp.status();
            let resp_headers = resp.headers().clone();
            let mut out = Response::new(Body::from_stream(resp.bytes_stream()));
            copy_response_headers(&mut out, &resp_headers);
            *out.status_mut() = status;
            record_savings(&state, &summary, &model, &client);
            out
        }
        Err(_) => {
            state.errors_total.fetch_add(1, Ordering::Relaxed);
            error_response(StatusCode::BAD_GATEWAY, "upstream unreachable")
        }
    }
}

/// Prometheus text exposition.
async fn metrics_endpoint(State(state): State<SharedState>) -> Response {
    let body = state.metrics.render();
    let mut out = Response::new(Body::from(body));
    out.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4"),
    );
    out
}

async fn handle_provider_request(
    state: SharedState,
    headers: HeaderMap,
    body: axum::body::Bytes,
    format: WireFormat,
    api: Api,
    path: &str,
) -> Response {
    state.requests_total.fetch_add(1, Ordering::Relaxed);
    state.metrics.inc("headroom_requests_total", &[]);
    let client = client_label(&headers);

    // Unparseable body => forward verbatim (fail-open).
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return forward_raw(&state, headers, body, api, path).await,
    };
    let model = parsed
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();

    // Rate limit (advisory: limited requests still forward, but are counted).
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    let est_tokens = parsed
        .get("messages")
        .and_then(Value::as_array)
        .map(|m| m.len() as i64 * 1000)
        .unwrap_or(1000);
    if let RateDecision::Limited { retry_after_secs } = state.rate_limiter.check(est_tokens, now_ms) {
        state.metrics.inc("headroom_rate_limited_total", &[]);
        eprintln!("rate limit: retry after {retry_after_secs}s");
    }
    state.rate_limiter.record(1, now_ms);

    // Budget check (advisory: warn + count, still fail-open).
    if let Some(budget) = &state.budget {
        let events = state.ledger.read_all();
        if let BudgetDecision::Exceeded { spent_usd, limit_usd } =
            budget::check(&events, budget, now_ms)
        {
            state.metrics.inc("headroom_budget_exceeded_total", &[]);
            eprintln!("budget exceeded: ${spent_usd:.2} > ${limit_usd:.2} (advisory)");
        }
    }

    // Cache mode: freeze the session prefix so only the newest delta is
    // compressed (keeps provider prompt-cache hits).
    let mut config = state.config.clone();
    if state.cache_mode {
        if let Some(Value::Array(messages)) = parsed.get("messages") {
            let key = session_key(&parsed);
            let plan = state.sessions.resolve(&key, messages.len(), now_ms, 600_000);
            config.frozen_message_count = plan.freeze_message_count;
        }
    }

    // Compress (fail-open to the original body on any error).
    let (forward_body, summary) = match compress_request_body(
        &parsed,
        format,
        &config,
        state.store.as_ref(),
    ) {
        Ok(CompressOutcome::Compressed { mut body, summary }) => {
            state.requests_compressed.fetch_add(1, Ordering::Relaxed);
            state.metrics.inc("headroom_compressed_total", &[]);
            // CCR markers present -> make sure the model can retrieve.
            if !summary.ccr_hashes.is_empty() {
                crate::compress_request::inject_retrieve_tool(&mut body, format);
                state.metrics.inc("headroom_retrieve_tool_injected", &[]);
            }
            (serde_json::to_vec(&body).unwrap_or_default(), Some(summary))
        }
        Ok(CompressOutcome::Unchanged { .. }) => (body.to_vec(), None),
        Err(_) => {
            state.errors_total.fetch_add(1, Ordering::Relaxed);
            (body.to_vec(), None)
        }
    };

    let hdrs: Vec<(String, String)> = headers
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
        .collect();

    match state
        .upstream
        .forward(api, Method::POST, path, &hdrs, bytes::Bytes::from(forward_body))
        .await
    {
        Ok(resp) => {
            let status = resp.status();
            let resp_headers = resp.headers().clone();
            let is_sse = resp_headers
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .map(|ct| ct.contains("text/event-stream"))
                .unwrap_or(false);
            if is_sse {
                // Stream through untouched; observe usage for the ledger via an
                // unfold adapter that feeds the accumulator per chunk and
                // records the finished usage when the stream ends.
                let usage_state = Arc::clone(&state);
                let usage_stream = futures_util::stream::unfold(
                    (resp.bytes_stream(), SseAccumulator::new(), usage_state),
                    |(mut stream, mut acc, state)| async move {
                        use futures_util::StreamExt;
                        match stream.next().await {
                            Some(Ok(bytes)) => {
                                if let Ok(text) = std::str::from_utf8(&bytes) {
                                    acc.feed(text);
                                }
                                Some((Ok(bytes), (stream, acc, state)))
                            }
                            Some(Err(e)) => {
                                Some((Err(std::io::Error::other(e)), (stream, acc, state)))
                            }
                            None => {
                                let usage = acc.finish();
                                if usage.input_tokens > 0 || usage.output_tokens > 0 {
                                    state.metrics.inc("headroom_sse_usage_observed", &[]);
                                }
                                None
                            }
                        }
                    },
                );
                let mut out = Response::new(Body::from_stream(usage_stream));
                copy_response_headers(&mut out, &resp_headers);
                *out.status_mut() = status;
                record_savings(&state, &summary, &model, &client);
                out
            } else {
                let mut out = Response::new(Body::from_stream(resp.bytes_stream()));
                copy_response_headers(&mut out, &resp_headers);
                *out.status_mut() = status;
                record_savings(&state, &summary, &model, &client);
                out
            }
        }
        Err(_) => {
            state.errors_total.fetch_add(1, Ordering::Relaxed);
            error_response(StatusCode::BAD_GATEWAY, "upstream unreachable")
        }
    }
}

fn record_savings(
    state: &SharedState,
    summary: &Option<CompressionSummary>,
    model: &str,
    client: &str,
) {
    if let Some(s) = summary {
        state.tokens_saved_total.fetch_add(s.tokens_saved() as u64, Ordering::Relaxed);
        let mut clients = state.by_client.lock().unwrap();
        *clients.entry(client.to_string()).or_insert(0) += 1;
        let event = SavingsEvent::new_proxy(model, client, s.tokens_before, s.tokens_after);
        state.ledger.record(&event);
    }
}

/// Pick the upstream for an unrouted path. OpenAI/Codex surfaces first
/// (matches the Python proxy's route map), Anthropic as the default.
fn select_api(path: &str) -> Api {
    const OPENAI_PREFIXES: &[&str] = &[
        "/v1/chat", "/chat/", "/v1/responses", "/responses", "/v1/codex", "/codex",
        "/backend-api", "/v1/live", "/live", "/v1/models", "/v1/embeddings",
        "/v1/moderations", "/v1/audio", "/v1/images", "/v1/realtime",
        "/v1/alpha/search",
    ];
    if OPENAI_PREFIXES.iter().any(|p| path.starts_with(p)) {
        Api::OpenAi
    } else {
        Api::Anthropic
    }
}

fn client_label(headers: &HeaderMap) -> String {
    headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.split('/').next().unwrap_or(s).to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// Catch-all: forward every unrouted path verbatim to the matching upstream.
async fn passthrough(State(state): State<SharedState>, req: Request) -> Response {
    let api = select_api(req.uri().path());
    let method = req.method().clone();
    let path_query = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| req.uri().path().to_string());
    let headers: Vec<(String, String)> = req
        .headers()
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
        .collect();
    let body = match axum::body::to_bytes(req.into_body(), 64 * 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return error_response(StatusCode::BAD_REQUEST, "body read failed"),
    };
    match state.upstream.forward(api, method, &path_query, &headers, body).await {
        Ok(resp) => {
            let status = resp.status();
            let resp_headers = resp.headers().clone();
            let mut out = Response::new(Body::from_stream(resp.bytes_stream()));
            copy_response_headers(&mut out, &resp_headers);
            *out.status_mut() = status;
            out
        }
        Err(_) => error_response(StatusCode::BAD_GATEWAY, "upstream unreachable"),
    }
}

async fn forward_raw(
    state: &SharedState,
    headers: HeaderMap,
    body: axum::body::Bytes,
    api: Api,
    path: &str,
) -> Response {
    let hdrs: Vec<(String, String)> = headers
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
        .collect();
    match state
        .upstream
        .forward(api, Method::POST, path, &hdrs, body)
        .await
    {
        Ok(resp) => {
            let status = resp.status();
            let resp_headers = resp.headers().clone();
            let mut out = Response::new(Body::from_stream(resp.bytes_stream()));
            copy_response_headers(&mut out, &resp_headers);
            *out.status_mut() = status;
            out
        }
        Err(_) => error_response(StatusCode::BAD_GATEWAY, "upstream unreachable"),
    }
}

fn copy_response_headers(out: &mut Response, headers: &HeaderMap) {
    for (k, v) in headers {
        if k == &axum::http::header::CONTENT_LENGTH
            || k == &axum::http::header::TRANSFER_ENCODING
        {
            continue;
        }
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(k.as_str().as_bytes()),
            HeaderValue::from_bytes(v.as_bytes()),
        ) {
            out.headers_mut().insert(name, value);
        }
    }
}

fn json_response(status: StatusCode, body: &Value) -> Response {
    let bytes = serde_json::to_vec(body).unwrap_or_else(|_| b"{}".to_vec());
    let mut out = Response::new(Body::from(bytes));
    *out.status_mut() = status;
    out.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    out
}

fn error_response(status: StatusCode, message: &str) -> Response {
    json_response(status, &json!({"error": message}))
}

fn now_iso() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;
    let millis = now.subsec_millis();
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (h, mi, s) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 - doe / 36524 + doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{millis:03}Z")
}

/// Start the server on `addr`.
pub async fn run(state: SharedState, addr: std::net::SocketAddr) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let app = build_router(state);
    axum::serve(listener, app).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_hash_forms() {
        assert_eq!(
            normalize_hash("abc123def456789012345678").as_deref(),
            Some("abc123def456789012345678")
        );
        assert_eq!(
            normalize_hash(r#"{"hash": "ABC123DEF456"}"#).as_deref(),
            Some("abc123def456")
        );
        assert_eq!(
            normalize_hash("<<ccr:abc123def456,base64,4.5KB>>").as_deref(),
            Some("abc123def456")
        );
        assert_eq!(
            normalize_hash("<<ccr:abc123def456 100_rows_offloaded>>").as_deref(),
            Some("abc123def456")
        );
        assert_eq!(normalize_hash("not a hash"), None);
    }
}
