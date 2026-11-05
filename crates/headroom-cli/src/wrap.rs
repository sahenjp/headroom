//! wrap / unwrap — launch a CLI tool through the proxy.
//!
//! `headroom wrap claude` execs the tool with ANTHROPIC_BASE_URL /
//! OPENAI_BASE_URL pointed at the proxy. `unwrap` prints the unset lines.

use anyhow::Result;

/// Tools whose API base env vars we know.
const KNOWN_TOOLS: &[(&str, &str, &str)] = &[
    ("claude", "ANTHROPIC_BASE_URL", "http://127.0.0.1:{port}"),
    ("codex", "OPENAI_BASE_URL", "http://127.0.0.1:{port}/v1"),
];

/// Print (or exec) the wrapped environment for a tool.
pub fn wrap(tool: &str, port: u16, args: &[String], print_env: bool) -> Result<()> {
    let tool_l = tool.to_lowercase();
    let known = KNOWN_TOOLS.iter().find(|(n, _, _)| *n == tool_l);
    let envs: Vec<(String, String)> = match known {
        Some((_, key, template)) => {
            vec![(key.to_string(), template.replace("{port}", &port.to_string()))]
        }
        None => vec![(
            "ANTHROPIC_BASE_URL".to_string(),
            format!("http://127.0.0.1:{port}"),
        )],
    };
    if print_env {
        for (k, v) in &envs {
            println!("export {k}={v}");
        }
        return Ok(());
    }
    let program = which_or_fail(tool)?;
    let err = std::os::unix::process::CommandExt::exec(
        std::process::Command::new(&program).args(args),
    );
    Err(anyhow::anyhow!("exec {program} failed: {err}"))
}

/// Remove proxy env overrides (prints the unset lines).
pub fn unwrap(_tool: &str) -> Result<()> {
    println!("unset ANTHROPIC_BASE_URL");
    println!("unset OPENAI_BASE_URL");
    Ok(())
}

fn which_or_fail(tool: &str) -> Result<String> {
    for dir in std::env::var("PATH").unwrap_or_default().split(':') {
        let candidate = std::path::Path::new(dir).join(tool);
        if candidate.is_file() {
            return Ok(candidate.display().to_string());
        }
    }
    Err(anyhow::anyhow!("tool not found in PATH: {tool}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unwrap_prints_unset() {
        unwrap("claude").unwrap();
    }
}
