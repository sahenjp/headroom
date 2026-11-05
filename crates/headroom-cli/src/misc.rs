//! Small wave-2 commands: update, rollout, telemetry, copilot-auth,
//! capture, evals, learn, agent-savings, output-savings, dashboard.

use anyhow::Result;

/// Print the current version (no release channel is configured; updates are
/// source rebuilds).
pub async fn update() -> Result<()> {
    println!("current: headroom v{}", headroom_core::VERSION);
    println!("update: rebuild from source (cargo build --release)");
    Ok(())
}

/// Show the runtime feature-rollout policy (static in the Rust port).
pub fn rollout() -> Result<()> {
    println!("rollout policy (static in the Rust port):");
    println!("  smart_crusher           enabled");
    println!("  kompress (ML)           disabled (deterministic TextCrusher fallback)");
    println!("  intercept_tool_results  canary-gated (off)");
    println!("  read_maturation         off");
    Ok(())
}

/// Show what telemetry collects and the opt-out location. Nothing is sent.
pub fn telemetry(show: bool) -> Result<()> {
    if show {
        println!("telemetry beacon payload (only when explicitly enabled):");
        println!("  version, uptime, aggregate counters (no content, no identifiers)");
    }
    let optout = headroom_store::ccr::default_workspace_dir().join("telemetry_optout");
    println!(
        "opt-out file: {} ({})",
        optout.display(),
        if optout.exists() { "present" } else { "absent" }
    );
    Ok(())
}

/// Copilot token storage (file-based, 0600; no keyring in the port).
pub fn copilot_login(token: Option<&str>) -> Result<()> {
    let dir = headroom_store::ccr::default_workspace_dir();
    let path = dir.join("copilot_token");
    match token {
        Some(t) => {
            std::fs::create_dir_all(&dir)?;
            std::fs::write(&path, t)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
            }
            println!("token stored at {}", path.display());
        }
        None => println!("usage: headroom copilot-auth login <token>"),
    }
    Ok(())
}

/// Capture: prepare a capture slot for a before/after network diff.
pub fn capture(label: &str) -> Result<()> {
    let dir = headroom_store::ccr::default_workspace_dir().join("captures");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{label}.txt"));
    println!("capture slot: {}", path.display());
    println!("(replay the request twice through the proxy and diff the upstream bodies)");
    Ok(())
}

/// Evals: run compression fixtures and print achieved ratios.
pub fn evals() -> Result<()> {
    let items: Vec<serde_json::Value> = (0..40)
        .map(|i| serde_json::json!({"id": i, "name": format!("item_{i}"), "status": "ok"}))
        .collect();
    let json_input = serde_json::to_string(&items)?;
    let log_input: String = (0..200)
        .map(|i| format!("warning: unused import in module_{i}.rs"))
        .collect::<Vec<_>>()
        .join("\n");

    let compacted = headroom_core::compaction::encode_compacted(&items)
        .map(|c| c.len())
        .unwrap_or(json_input.len());
    report("json_array", json_input.len(), compacted);

    let log_r = headroom_core::transforms::log_compressor::LogCompressor::default()
        .compress(&log_input, "", 1.0);
    report("build_log", log_input.len(), log_r.compressed.len());
    Ok(())
}

fn report(name: &str, before: usize, after: usize) {
    let pct = if before > 0 { 100.0 * (1.0 - after as f64 / before as f64) } else { 0.0 };
    println!("{name:<12} {before:>6} -> {after:>6} bytes  ({pct:.1}% saved)");
}

/// Learn: record a lesson into the memory DB (scope=user).
pub fn learn(lesson: &str) -> Result<()> {
    let db = headroom_store::memory::MemoryDb::open_default()?;
    let id = db.add(&headroom_store::memory::NewMemory {
        content: lesson.to_string(),
        scope: "user".into(),
        project: None,
        tags: vec!["lesson".into()],
    })?;
    println!("lesson stored (id {id})");
    Ok(())
}

/// Agent savings profile: render the env seeds for a profile.
pub fn agent_savings(profile: &str, format: &str) -> Result<()> {
    let (target, ratio, min_tokens) = match profile {
        "agent-90" => (0.90, 0.10, 120),
        "balanced" => (0.70, 0.30, 250),
        "coding" => (0.50, 0.0, 10),
        other => {
            return Err(anyhow::anyhow!(
                "unknown profile: {other} (agent-90|balanced|coding)"
            ))
        }
    };
    if format == "json" {
        println!(
            "{{\"profile\":\"{profile}\",\"target_savings\":{target},\"target_ratio\":{ratio},\"min_tokens\":{min_tokens}}}"
        );
    } else {
        println!("export HEADROOM_SAVINGS_PROFILE={profile}");
        println!("export HEADROOM_SAVINGS_TARGET={target}");
        println!("export HEADROOM_TARGET_RATIO={ratio}");
        println!("export HEADROOM_MIN_TOKENS={min_tokens}");
    }
    Ok(())
}

/// Output savings: point at the measured report.
pub fn output_savings() -> Result<()> {
    println!("output-shaper savings: not measured in the Rust port");
    println!("run 'headroom savings' for measured compression savings");
    Ok(())
}

/// Dashboard: print the proxy dashboard URL.
pub fn dashboard(port: u16) -> Result<()> {
    println!("dashboard: http://127.0.0.1:{port}/dashboard");
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn evals_runs() {
        super::evals().unwrap();
    }
}
/// Deploy: turnkey — write the unit, init claude, print next steps.
pub fn deploy(port: u16) -> Result<()> {
    let path = crate::install::write_unit(port, "deploy")?;
    println!("unit written: {}", path.display());
    let note = crate::install::init_client("claude", port)?;
    println!("claude init: {note}");
    println!("next: systemctl --user start headroom-proxy-deploy");
    Ok(())
}

/// Recover: scan for agent state left in temporary Headroom homes.
pub fn recover() -> Result<()> {
    let tmp = std::env::temp_dir();
    let mut found = 0;
    if let Ok(entries) = std::fs::read_dir(&tmp) {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with("headroom") && name.ends_with(".db") {
                found += 1;
                println!("found: {}", e.path().display());
            }
        }
    }
    if found == 0 {
        println!("no temporary headroom state found in {}", tmp.display());
    } else {
        println!("copy the files into ~/.headroom/ to recover, then delete the temp copies");
    }
    Ok(())
}

/// Tools: manage bundled CLI tool binaries (doctor/install/list).
pub fn tools(action: &str) -> Result<()> {
    let tools_list = [("ast-grep", "sg"), ("difft", "diff"), ("scc", "loc")];
    match action {
        "list" => {
            for (bin, cmd) in tools_list {
                let present = which(bin).is_some();
                println!("{bin:<10} {} (headroom {cmd})", if present { "present" } else { "missing" });
            }
        }
        "doctor" => {
            let mut missing = 0;
            for (bin, _) in tools_list {
                if which(bin).is_none() {
                    println!("missing: {bin}");
                    missing += 1;
                }
            }
            println!("{missing} missing (install: cargo install ast-grep difftastic scc)");
        }
        "install" => {
            println!("install with: cargo install ast-grep difftastic scc");
            println!("(the Rust port does not vendor binaries)");
        }
        other => return Err(anyhow::anyhow!("unknown action: {other} (list|doctor|install)")),
    }
    Ok(())
}

fn which(bin: &str) -> Option<std::path::PathBuf> {
    for dir in std::env::var("PATH").unwrap_or_default().split(':') {
        let candidate = std::path::Path::new(dir).join(bin);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

#[cfg(test)]
mod recover_tests {
    #[test]
    fn recover_runs() {
        super::recover().unwrap();
    }

    #[test]
    fn tools_list_runs() {
        super::tools("list").unwrap();
    }
}

