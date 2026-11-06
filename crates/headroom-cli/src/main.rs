//! headroom — Rust port of the headroom-ai CLI (core commands).

use std::net::SocketAddr;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand};
use headroom_proxy::compress_request::RequestCompressConfig;
use headroom_proxy::server::ProxyState;
use headroom_proxy::upstream::Upstream;
use headroom_store::ccr::{CompressionStore, StoreConfig};
use headroom_store::ledger::SavingsLedger;

/// Restore default SIGPIPE handling so `headroom savings | head` exits
/// quietly instead of panicking on a closed pipe (the ripgrep pattern).
#[cfg(unix)]
fn reset_sigpipe() {
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

#[cfg(not(unix))]
fn reset_sigpipe() {}

#[derive(Parser)]
#[command(name = "headroom", version = headroom_core::VERSION, about = "Headroom - The Context Optimization Layer for LLM Applications (Rust port)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the optimization proxy server.
    Proxy {
        /// Bind address.
        #[arg(long, default_value = "127.0.0.1", env = "HEADROOM_HOST")]
        host: String,
        /// Port.
        #[arg(short, long, default_value_t = 8787, env = "HEADROOM_PORT")]
        port: u16,
        /// Anthropic upstream base URL.
        #[arg(long, env = "ANTHROPIC_TARGET_API_URL", default_value = "https://api.anthropic.com")]
        anthropic_api_url: String,
        /// OpenAI upstream base URL.
        #[arg(long, env = "OPENAI_TARGET_API_URL", default_value = "https://api.openai.com")]
        openai_api_url: String,
        /// Disable the CCR store (stateless).
        #[arg(long, env = "HEADROOM_STATELESS")]
        stateless: bool,
        /// Requests per minute limit (advisory).
        #[arg(long, env = "HEADROOM_RPM")]
        rpm: Option<u32>,
        /// Tokens per minute limit (advisory).
        #[arg(long, env = "HEADROOM_TPM")]
        tpm: Option<u32>,
        /// USD spend budget (advisory).
        #[arg(long, env = "HEADROOM_BUDGET")]
        budget: Option<f64>,
        /// Budget period.
        #[arg(long, env = "HEADROOM_BUDGET_PERIOD", default_value = "daily")]
        budget_period: String,
        /// Mode: cache (freeze prefix) or token (compress all history).
        #[arg(long, env = "HEADROOM_MODE", default_value = "cache")]
        mode: String,
        /// No rate limiting (alias for --rpm/--tpm unlimited).
        #[arg(long)]
        no_rate_limit: bool,
    },
    /// Check that the proxy and client routing are healthy.
    Doctor {
        /// Port the proxy listens on.
        #[arg(short, long, default_value_t = 8787)]
        port: u16,
    },
    /// Show durable compression savings over time.
    Savings {
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
        /// Reset (delete) the ledger.
        #[arg(long)]
        reset: bool,
    },
    /// Retrieve original content by hash.
    Retrieve {
        /// The CCR hash (12-24 hex) or a full marker.
        hash: String,
    },
    /// Run the MCP stdio server (tools: compress, retrieve, stats, read).
    Mcp {
        /// Accept the Python-style "serve" word for drop-in compatibility.
        #[arg(default_value = "serve")]
        command: String,
        /// Proxy URL for proxy-store fallback.
        #[arg(long, env = "HEADROOM_PROXY_URL")]
        proxy_url: Option<String>,
    },
    /// Manage memories stored in Headroom.
    Memory {
        #[command(subcommand)]
        cmd: headroom_cli::memory_cmd::MemoryCommand,
    },
    /// Analyze proxy performance from logs.
    Perf {
        /// Hours of logs to analyze.
        #[arg(long, default_value_t = 168)]
        hours: u64,
        /// Output format.
        #[arg(long, value_enum, default_value_t = headroom_cli::perf::PerfFormat::Text)]
        format: headroom_cli::perf::PerfFormat,
    },
    /// Show original vs compressed content for recent proxy requests.
    Inspect {
        /// Number of recent requests.
        #[arg(long, default_value_t = 1)]
        last: usize,
        /// Emit JSON.
        #[arg(long)]
        json: bool,
    },
    /// Audit Read-tool traffic for compression opportunities.
    AuditReads {
        /// Transcript directory (default ~/.claude/projects).
        #[arg(long)]
        path: Option<String>,
        /// Emit JSON.
        #[arg(long)]
        json: bool,
    },
    /// Wrap a CLI tool to run through the proxy.
    Wrap {
        /// Tool to wrap (claude, codex, ...).
        tool: String,
        /// Proxy port.
        #[arg(short, long, default_value_t = 8787)]
        port: u16,
        /// Print env instead of exec.
        #[arg(long)]
        print_env: bool,
        /// Args passed to the tool.
        args: Vec<String>,
    },
    /// Undo durable Headroom wrapping.
    Unwrap {
        /// Tool to unwrap.
        #[arg(default_value = "claude")]
        tool: String,
    },
    /// Install and manage persistent proxy deployments.
    Install {
        #[command(subcommand)]
        cmd: InstallCommand,
    },
    /// Install durable integrations for supported clients.
    Init {
        /// Client (claude, codex).
        client: String,
        /// Proxy port.
        #[arg(short, long, default_value_t = 8787)]
        port: u16,
    },
    /// Run difftastic (structural diff).
    Diff {
        args: Vec<String>,
    },
    /// Run scc (lines-of-code probe).
    Loc {
        args: Vec<String>,
    },
    /// Run ast-grep (structural search).
    Sg {
        args: Vec<String>,
    },
    /// Update Headroom (source rebuild notice).
    Update,
    /// Inspect runtime feature-rollout policy.
    Rollout,
    /// Show what telemetry collects and how to opt out.
    Telemetry {
        /// Show the payload.
        #[arg(long)]
        show: bool,
    },
    /// Manage the GitHub Copilot OAuth token.
    CopilotAuth {
        #[command(subcommand)]
        cmd: CopilotAuthCommand,
    },
    /// Capture and compare network traffic.
    Capture {
        label: String,
    },
    /// Run compression evaluation fixtures.
    Evals,
    /// Learn from tool call failures (store a lesson).
    Learn {
        /// The lesson text.
        #[arg(long)]
        lesson: Option<String>,
    },
    /// Render a token-savings profile env.
    AgentSavings {
        /// Profile name (agent-90, balanced, coding).
        #[arg(default_value = "agent-90")]
        profile: String,
        /// Output format (shell, json).
        #[arg(long, default_value = "shell")]
        format: String,
    },
    /// Show output-token reduction report.
    OutputSavings,
    /// Open the savings dashboard URL.
    Dashboard {
        /// Proxy port.
        #[arg(short, long, default_value_t = 8787)]
        port: u16,
    },
    /// Deploy a turnkey local proxy and configure clients.
    Deploy {
        /// Proxy port.
        #[arg(short, long, default_value_t = 8787)]
        port: u16,
    },
    /// Recover agent state left in a temporary Headroom home.
    Recover,
    /// Manage bundled CLI tool binaries (ast-grep, difft, scc).
    Tools {
        /// Action: list, doctor, install.
        #[arg(default_value = "list")]
        action: String,
    },
}

#[derive(Subcommand)]
enum InstallCommand {
    /// Write a systemd user unit for the proxy.
    Apply {
        /// Deployment profile name.
        #[arg(long, default_value = "default")]
        profile: String,
        /// Proxy port.
        #[arg(short, long, default_value_t = 8787)]
        port: u16,
    },
    /// Remove the unit file.
    Remove {
        #[arg(long, default_value = "default")]
        profile: String,
    },
    /// Show unit status.
    Status {
        #[arg(long, default_value = "default")]
        profile: String,
    },
}

#[derive(Subcommand)]
enum CopilotAuthCommand {
    /// Store the token.
    Login {
        /// The token (omit to see usage).
        token: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    reset_sigpipe();
    let cli = Cli::parse();
    match cli.command {
        Commands::Proxy { host, port, anthropic_api_url, openai_api_url, stateless, rpm, tpm, budget, budget_period, mode, no_rate_limit } => {
            run_proxy(host, port, anthropic_api_url, openai_api_url, stateless, rpm, tpm, budget, budget_period, mode, no_rate_limit).await
        }
        Commands::Doctor { port } => doctor(port).await,
        Commands::Savings { json, reset } => savings(json, reset),
        Commands::Retrieve { hash } => retrieve(&hash),
        Commands::Mcp { command, proxy_url } => {
            if command != "serve" {
                anyhow::bail!("unknown mcp subcommand: {command} (only 'serve' is supported)");
            }
            run_mcp(proxy_url)
        }
        Commands::Memory { cmd } => headroom_cli::memory_cmd::run(cmd),
        Commands::Perf { hours, format } => headroom_cli::perf::run(hours, format),
        Commands::Inspect { last, json } => headroom_cli::inspect::run(last, json),
        Commands::AuditReads { path, json } => headroom_cli::audit_reads::run(path, json),
        Commands::Wrap { tool, port, print_env, args } => {
            headroom_cli::wrap::wrap(&tool, port, &args, print_env)
        }
        Commands::Unwrap { tool } => headroom_cli::wrap::unwrap(&tool),
        Commands::Install { cmd } => match cmd {
            InstallCommand::Apply { profile, port } => {
                let path = headroom_cli::install::write_unit(port, &profile)?;
                println!("unit written: {}", path.display());
                println!("start: systemctl --user start headroom-proxy-{profile}");
                Ok(())
            }
            InstallCommand::Remove { profile } => {
                let removed = headroom_cli::install::remove_unit(&profile)?;
                println!("{}", if removed { "unit removed" } else { "no unit found" });
                Ok(())
            }
            InstallCommand::Status { profile } => {
                println!("{}", headroom_cli::install::status(&profile));
                Ok(())
            }
        },
        Commands::Init { client, port } => {
            let note = headroom_cli::install::init_client(&client, port)?;
            println!("{note}");
            Ok(())
        }
        Commands::Diff { args } => std::process::exit(headroom_cli::passthrough::diff(&args)?),
        Commands::Loc { args } => std::process::exit(headroom_cli::passthrough::loc(&args)?),
        Commands::Sg { args } => std::process::exit(headroom_cli::passthrough::sg(&args)?),
        Commands::Update => headroom_cli::misc::update().await,
        Commands::Rollout => headroom_cli::misc::rollout(),
        Commands::Telemetry { show } => headroom_cli::misc::telemetry(show),
        Commands::CopilotAuth { cmd } => match cmd {
            CopilotAuthCommand::Login { token } => headroom_cli::misc::copilot_login(token.as_deref()),
        },
        Commands::Capture { label } => headroom_cli::misc::capture(&label),
        Commands::Evals => headroom_cli::misc::evals(),
        Commands::Learn { lesson } => match lesson {
            Some(l) => headroom_cli::misc::learn(&l),
            None => {
                let db = headroom_store::memory::MemoryDb::open_default()?;
                for m in db.list(10, None, None)? {
                    println!("[{}] {}", m.id, m.content);
                }
                Ok(())
            }
        },
        Commands::AgentSavings { profile, format } => {
            headroom_cli::misc::agent_savings(&profile, &format)
        }
        Commands::OutputSavings => headroom_cli::misc::output_savings(),
        Commands::Dashboard { port } => headroom_cli::misc::dashboard(port),
        Commands::Deploy { port } => headroom_cli::misc::deploy(port),
        Commands::Recover => headroom_cli::misc::recover(),
        Commands::Tools { action } => headroom_cli::misc::tools(&action),
    }
}

async fn run_proxy(
    host: String,
    port: u16,
    anthropic_api: String,
    openai_api: String,
    stateless: bool,
    rpm: Option<u32>,
    tpm: Option<u32>,
    budget: Option<f64>,
    budget_period: String,
    mode: String,
    no_rate_limit: bool,
) -> Result<()> {
    let upstream = Upstream::new(anthropic_api, openai_api)?;
    let store = if stateless {
        None
    } else {
        match CompressionStore::open(&StoreConfig::default()) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("warning: CCR store unavailable ({e}); continuing stateless");
                None
            }
        }
    };
    let (rpm, tpm) = if no_rate_limit { (None, None) } else { (rpm, tpm) };
    let budget = budget.map(|limit_usd| headroom_proxy::budget::Budget {
        limit_usd,
        period: headroom_proxy::budget::BudgetPeriod::parse(&budget_period)
            .unwrap_or(headroom_proxy::budget::BudgetPeriod::Daily),
    });
    let cache_mode = mode != "token";
    let state = Arc::new(ProxyState {
        config: RequestCompressConfig::default(),
        upstream,
        store,
        ledger: SavingsLedger::open_default(),
        started: Instant::now(),
        version: headroom_core::VERSION,
        requests_total: AtomicU64::new(0),
        requests_compressed: AtomicU64::new(0),
        tokens_saved_total: AtomicU64::new(0),
        errors_total: AtomicU64::new(0),
        by_client: Mutex::new(std::collections::BTreeMap::new()),
        rate_limiter: headroom_proxy::rate_limit::RateLimiter::new(rpm, tpm),
        budget,
        metrics: headroom_proxy::metrics::Metrics::new(),
        sessions: headroom_proxy::session::SessionTracker::new(),
        cache_mode,
    });
    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    eprintln!("headroom-proxy v{} listening on http://{addr} (mode: {mode})", headroom_core::VERSION);
    headroom_proxy::run(state, addr).await?;
    Ok(())
}

async fn doctor(port: u16) -> Result<()> {
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();
    let mut failures = 0u32;
    let mut warnings = 0u32;

    let live = client
        .get(format!("{base}/livez"))
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await;
    match live {
        Ok(r) if r.status().is_success() => {
            let v: serde_json::Value = r.json().await.unwrap_or_default();
            let ver = v["version"].as_str().unwrap_or("?");
            println!("| {:<12} | {:<6} | proxy running at {base} (v{ver}) |", "proxy", "pass");
        }
        _ => {
            println!("| {:<12} | {:<6} | proxy not reachable at {base} |", "proxy", "FAIL");
            failures += 1;
        }
    }

    let ready = client
        .get(format!("{base}/readyz"))
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await;
    match ready {
        Ok(r) if r.status().is_success() => {
            let v: serde_json::Value = r.json().await.unwrap_or_default();
            let is_ready = v["ready"].as_bool().unwrap_or(false);
            if is_ready {
                println!("| {:<12} | {:<6} | /readyz reports ready |", "readiness", "pass");
            } else {
                println!("| {:<12} | {:<6} | /readyz reports not ready |", "readiness", "FAIL");
                failures += 1;
            }
            let upstream_ok = v["checks"]["upstream"]["ready"].as_bool().unwrap_or(false);
            if upstream_ok {
                println!("| {:<12} | {:<6} | upstream reachable |", "upstream", "pass");
            } else {
                println!("| {:<12} | {:<6} | upstream unreachable |", "upstream", "warn");
                warnings += 1;
            }
        }
        _ => {
            println!("| {:<12} | {:<6} | /readyz unreachable |", "readiness", "FAIL");
            failures += 1;
        }
    }

    let routed = std::env::var("ANTHROPIC_BASE_URL")
        .map(|v| v.contains(&format!(":{port}")))
        .unwrap_or(false)
        || std::env::var("OPENAI_BASE_URL")
            .map(|v| v.contains(&format!(":{port}")))
            .unwrap_or(false);
    if routed {
        println!("| {:<12} | {:<6} | shell routes through the proxy |", "shell env", "pass");
    } else {
        println!("| {:<12} | {:<6} | ANTHROPIC_BASE_URL / OPENAI_BASE_URL unset |", "shell env", "warn");
        warnings += 1;
        println!("  fix: export ANTHROPIC_BASE_URL={base}");
    }

    println!();
    println!("{failures} failure(s), {warnings} warning(s)");
    std::process::exit(if failures > 0 { 2 } else if warnings > 0 { 1 } else { 0 });
}

fn savings(json_output: bool, reset: bool) -> Result<()> {
    let ledger = SavingsLedger::open_default();
    if reset {
        ledger.reset()?;
        println!("ledger reset");
        return Ok(());
    }
    let events = ledger.read_all();
    let aggregated = headroom_store::ledger::aggregate(&events);
    if json_output {
        println!("{}", serde_json::to_string_pretty(&aggregated)?);
        return Ok(());
    }
    let windows = &aggregated["windows"];
    let rows = [
        ("Today", "today"),
        ("Last 7 days", "last_7_days"),
        ("Last 30 days", "last_30_days"),
    ];
    for (label, key) in rows {
        let w = &windows[key];
        let saved = w["tokens_saved"].as_i64().unwrap_or(0);
        let before = w["tokens_before"].as_i64().unwrap_or(0);
        let pct = w["savings_percent"].as_f64().unwrap_or(0.0);
        let cost = w["cost_usd"].as_f64().unwrap_or(0.0);
        println!("{label:<12} {pct:.1}%  saved {saved} / {before} tokens  ${cost:.4}");
    }
    if let Some(models) = aggregated["by_model"].as_array() {
        if !models.is_empty() {
            println!();
            println!("Cost avoided per model:");
            for m in models.iter().take(8) {
                let name = m["model"].as_str().unwrap_or("?");
                let cost = m["cost_usd"].as_f64().unwrap_or(0.0);
                println!("  {name:<40} ${cost:.4}");
            }
        }
    }
    if let Some(clients) = aggregated["by_client"].as_array() {
        if !clients.is_empty() {
            println!();
            println!("Savings by client:");
            for c in clients.iter().take(8) {
                let name = c["client"].as_str().unwrap_or("?");
                let calls = c["calls"].as_i64().unwrap_or(0);
                let saved = c["tokens_saved"].as_i64().unwrap_or(0);
                println!("  {name:<20} {calls} calls - {saved} tokens saved");
            }
        }
    }
    Ok(())
}

fn retrieve(hash: &str) -> Result<()> {
    let store = CompressionStore::open(&StoreConfig::default())?;
    let normalized = hash
        .strip_prefix("<<ccr:")
        .and_then(|r| r.split([',', ' ']).next())
        .unwrap_or(hash)
        .to_lowercase();
    match store.retrieve(&normalized) {
        Some(entry) => {
            println!("{}", entry.original_content);
            Ok(())
        }
        None => {
            eprintln!("not found (expired or never stored): {normalized}");
            std::process::exit(1);
        }
    }
}

fn run_mcp(proxy_url: Option<String>) -> Result<()> {
    let store = match CompressionStore::open(&StoreConfig::default()) {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("warning: CCR store unavailable ({e})");
            None
        }
    };
    let ctx = headroom_mcp::server::McpContext {
        store,
        ledger: SavingsLedger::open_default(),
        proxy_url: proxy_url.unwrap_or_else(|| "http://127.0.0.1:8787".into()),
        read_cache: Default::default(),
    };
    headroom_mcp::server::serve_stdio(ctx)
}