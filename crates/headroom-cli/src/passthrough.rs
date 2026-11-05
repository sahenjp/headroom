//! Pass-through commands to bundled/external tools (diff, loc, sg).
//!
//! The Python headroom ships difftastic/scc/ast-grep binaries; the Rust port
//! shells out to whatever is on PATH and reports absence honestly.

use anyhow::Result;
use std::process::Command;

fn runtool(program: &str, args: &[String]) -> Result<i32> {
    let status = Command::new(program).args(args).status().map_err(|_| {
        anyhow::anyhow!("{program} not found in PATH (install: difftastic / scc / ast-grep)")
    })?;
    Ok(status.code().unwrap_or(1))
}

/// difftastic structural diff.
pub fn diff(args: &[String]) -> Result<i32> {
    runtool("difft", args)
}

/// scc lines-of-code probe.
pub fn loc(args: &[String]) -> Result<i32> {
    runtool("scc", args)
}

/// ast-grep structural search/replace.
pub fn sg(args: &[String]) -> Result<i32> {
    runtool("ast-grep", args)
}
