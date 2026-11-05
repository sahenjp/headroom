//! headroom-proxy: local LLM optimization proxy.

pub mod budget;
pub mod compress_request;
pub mod metrics;
pub mod rate_limit;
pub mod responses;
pub mod server;
pub mod session;
pub mod streaming;
pub mod upstream;

pub use server::{run, ProxyState, SharedState};
