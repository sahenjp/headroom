//! install / deploy / init — durable integration file generation.
//!
//! Writes systemd user units, hook configs, and MCP registrations. This
//! module only WRITES files; starting/stopping services is left to the user
//! (the exact systemctl commands are printed at the end).

use anyhow::Result;

/// Write a systemd user unit for the proxy (never started automatically).
pub fn write_unit(port: u16, profile: &str) -> Result<std::path::PathBuf> {
    let dir = std::env::var("HOME")
        .ok()
        .map(|h| std::path::Path::new(&h).join(".config/systemd/user"))
        .ok_or_else(|| anyhow::anyhow!("HOME not set"))?;
    std::fs::create_dir_all(&dir)?;
    let bin = std::env::current_exe()?.display().to_string();
    let unit = format!(
        "[Unit]\nDescription=Headroom proxy ({profile})\nAfter=network-online.target\n\n[Service]\nType=simple\nExecStart={bin} proxy --port {port}\nRestart=on-failure\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n"
    );
    let path = dir.join(format!("headroom-proxy-{profile}.service"));
    std::fs::write(&path, unit)?;
    Ok(path)
}

/// Remove a written unit file. Returns whether it existed.
pub fn remove_unit(profile: &str) -> Result<bool> {
    let path = std::env::var("HOME").ok().map(|h| {
        std::path::Path::new(&h).join(format!(".config/systemd/user/headroom-proxy-{profile}.service"))
    });
    let Some(path) = path else {
        return Ok(false);
    };
    if path.exists() {
        std::fs::remove_file(&path)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Unit status (file existence + how-to-start hint).
pub fn status(profile: &str) -> String {
    let path = std::env::var("HOME").ok().map(|h| {
        std::path::Path::new(&h).join(format!(".config/systemd/user/headroom-proxy-{profile}.service"))
    });
    match path {
        Some(p) if p.exists() => format!(
            "installed: {}\nstart: systemctl --user start headroom-proxy-{profile}",
            p.display()
        ),
        _ => format!("not installed (run: headroom install apply --profile {profile})"),
    }
}

/// init: merge proxy env into a client config file (additive; existing
/// values win). Returns the touched path (or a manual-step note).
pub fn init_client(client: &str, port: u16) -> Result<String> {
    let home = std::env::var("HOME")?;
    match client {
        "claude" => {
            let path = std::path::Path::new(&home).join(".claude/settings.json");
            if !path.exists() {
                std::fs::create_dir_all(path.parent().unwrap())?;
                std::fs::write(&path, "{}\n")?;
            }
            let raw = std::fs::read_to_string(&path)?;
            let mut v: serde_json::Value =
                serde_json::from_str(&raw).unwrap_or(serde_json::json!({}));
            if let Some(obj) = v.as_object_mut() {
                let env = obj
                    .entry("env")
                    .or_insert_with(|| serde_json::json!({}));
                if let Some(envobj) = env.as_object_mut() {
                    envobj
                        .entry("ANTHROPIC_BASE_URL")
                        .or_insert_with(|| serde_json::json!(format!("http://127.0.0.1:{port}")));
                }
            }
            std::fs::write(&path, serde_json::to_string_pretty(&v)? + "\n")?;
            Ok(path.display().to_string())
        }
        "codex" => {
            let path = std::path::Path::new(&home).join(".codex/config.toml");
            Ok(format!(
                "manual step: add OPENAI_BASE_URL = 'http://127.0.0.1:{port}/v1' to {}",
                path.display()
            ))
        }
        other => Err(anyhow::anyhow!("unknown client: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_reports_missing_unit() {
        let s = status("nonexistent-test-profile");
        assert!(s.contains("not installed"));
    }
}
