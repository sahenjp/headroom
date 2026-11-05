//! Upstream forwarding: one reqwest client, header hygiene, streaming.

use reqwest::{Client, Method, Response};

/// Hop-by-hop headers never forwarded across the proxy.
const HOP_BY_HOP: &[&str] = &[
    "host",
    "connection",
    "content-length",
    "transfer-encoding",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "upgrade",
    "expect",
];

/// A shareable upstream client. rustls only; no automatic decompression so
/// SSE bytes pass through untouched.
#[derive(Clone)]
pub struct Upstream {
    client: Client,
    anthropic_base: String,
    openai_base: String,
}

impl Upstream {
    pub fn new(anthropic_base: String, openai_base: String) -> anyhow::Result<Self> {
        let client = Client::builder()
            .pool_idle_timeout(std::time::Duration::from_secs(90))
            .build()?;
        Ok(Self { client, anthropic_base, openai_base })
    }

    pub fn anthropic_base(&self) -> &str {
        &self.anthropic_base
    }

    pub fn openai_base(&self) -> &str {
        &self.openai_base
    }

    /// Forward a request upstream. `path_and_query` keeps the original path
    /// plus query string (e.g. `/v1/messages?beta=true`).
    pub async fn forward(
        &self,
        api: Api,
        method: Method,
        path_and_query: &str,
        headers: &[(String, String)],
        body: bytes::Bytes,
    ) -> anyhow::Result<Response> {
        let base = match api {
            Api::Anthropic => &self.anthropic_base,
            Api::OpenAi => &self.openai_base,
        };
        let url = format!("{}{}", base.trim_end_matches('/'), path_and_query);
        let mut req = self.client.request(method, &url);
        for (name, value) in headers {
            if is_hop_by_hop(name) {
                continue;
            }
            req = req.header(name, value);
        }
        req = req.body(body);
        Ok(req.send().await?)
    }
}

/// Which upstream API a request targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Api {
    Anthropic,
    OpenAi,
}

fn is_hop_by_hop(name: &str) -> bool {
    HOP_BY_HOP.iter().any(|h| name.eq_ignore_ascii_case(h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hop_by_hop_filter() {
        assert!(is_hop_by_hop("Host"));
        assert!(is_hop_by_hop("content-length"));
        assert!(!is_hop_by_hop("x-api-key"));
        assert!(!is_hop_by_hop("authorization"));
    }
}
