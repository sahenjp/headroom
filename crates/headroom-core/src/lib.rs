//! headroom-core: pure compression transforms and pipeline (no IO).
//!
//! Rust port of the headroom-ai Python package's compression core.

pub mod ccr;
pub mod compaction;
pub mod content;
pub mod tag_protector;
pub mod tokens;
pub mod transforms;
pub mod pipeline;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
