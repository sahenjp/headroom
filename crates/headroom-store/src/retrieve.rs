//! Retrieve facade: hash-addressed retrieval of pre-compression originals.
//!
//! The one true map is the CCR store ([`crate::ccr::CompressionStore`]).
//! This module re-exports it under the name the MCP tools and the proxy
//! retrieve endpoint use.

pub use crate::ccr::{CompressionEntry, CompressionStore, StoreConfig, StoreError};

/// Outcome of a retrieval attempt.
#[derive(Debug)]
pub enum RetrieveOutcome {
    /// Found: the original content.
    Found(CompressionEntry),
    /// Expired or never stored. The caller should tell the model to re-run
    /// the tool / re-read the file (the Python CCR_MISS_MESSAGE behavior).
    Miss,
}

/// The miss message shown to the model when a hash cannot be resolved.
pub const CCR_MISS_MESSAGE: &str = "Original content not found (expired or never stored). Re-read the file or re-run the command to get fresh output.";
