//! headroom-cli library: command handlers for the headroom binary.
//!
//! Wave-2 commands live here as modules; main.rs owns the clap parser and
//! dispatches into these handlers.

#![allow(dead_code)]

pub mod audit_reads;
pub mod inspect;
pub mod install;
pub mod memory_cmd;
pub mod misc;
pub mod passthrough;
pub mod perf;
pub mod wrap;
