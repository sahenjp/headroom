//! Cache-mode prefix tracking: freeze the provider prompt-cache prefix,
//! compress only the newest delta (the Python proxy's "cache" mode).

use std::collections::HashMap;

use serde_json::Value;

/// Derive a session key from the request's system prompt + first user
/// message text (sha-based; absent fields contribute empty strings).
pub fn session_key(body: &Value) -> String {
    let mut material = String::new();
    if let Some(sys) = body.get("system") {
        match sys {
            Value::String(s) => material.push_str(s),
            Value::Array(blocks) => {
                for b in blocks {
                    if let Some(t) = b.get("text").and_then(Value::as_str) {
                        material.push_str(t);
                    }
                }
            }
            _ => {}
        }
    }
    if let Some(Value::Array(messages)) = body.get("messages") {
        for m in messages {
            let is_user = m.get("role") == Some(&serde_json::json!("user"));
            if !is_user {
                continue;
            }
            match m.get("content") {
                Some(Value::String(s)) => {
                    material.push_str(s);
                    break;
                }
                Some(Value::Array(blocks)) => {
                    for b in blocks {
                        if let Some(t) = b.get("text").and_then(Value::as_str) {
                            material.push_str(t);
                        }
                    }
                    break;
                }
                _ => break,
            }
        }
    }
    headroom_core::ccr::ccr_hash(&material)
}

/// Per-session state.
#[derive(Debug, Clone)]
pub struct SessionState {
    pub frozen_messages: usize,
    pub last_seen_ms: i64,
}

/// The plan for one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrozenPlan {
    pub freeze_message_count: usize,
}

/// Session tracker with TTL-based expiry.
#[derive(Default)]
pub struct SessionTracker {
    sessions: std::sync::Mutex<HashMap<String, SessionState>>,
}

const MAX_SESSIONS: usize = 1000;

impl SessionTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve the freeze plan for a request. A live session freezes the
    /// prefix it saw last time; a new/expired session freezes nothing and
    /// records the current length.
    pub fn resolve(&self, key: &str, messages_len: usize, now_ms: i64, session_ttl_ms: i64) -> FrozenPlan {
        let mut sessions = self.sessions.lock().unwrap();
        // Evict expired overflow (oldest last_seen first).
        if sessions.len() > MAX_SESSIONS {
            let mut by_age: Vec<(String, i64)> = sessions
                .iter()
                .map(|(k, s)| (k.clone(), s.last_seen_ms))
                .collect();
            by_age.sort_by_key(|(_, t)| *t);
            for (k, _) in by_age.into_iter().take(sessions.len() - MAX_SESSIONS) {
                sessions.remove(&k);
            }
        }
        let plan = match sessions.get(key) {
            Some(state) if now_ms - state.last_seen_ms < session_ttl_ms => FrozenPlan {
                freeze_message_count: state.frozen_messages.min(messages_len),
            },
            _ => FrozenPlan { freeze_message_count: 0 },
        };
        sessions.insert(
            key.to_string(),
            SessionState { frozen_messages: messages_len, last_seen_ms: now_ms },
        );
        plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body() -> Value {
        serde_json::json!({
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        })
    }

    #[test]
    fn session_key_is_stable() {
        assert_eq!(session_key(&body()), session_key(&body()));
        assert_eq!(session_key(&body()).len(), 24);
    }

    #[test]
    fn repeat_request_freezes_prefix() {
        let t = SessionTracker::new();
        let k = session_key(&body());
        assert_eq!(t.resolve(&k, 2, 1_000, 600_000).freeze_message_count, 0);
        // Second turn: 4 messages; the first 2 freeze.
        assert_eq!(t.resolve(&k, 4, 2_000, 600_000).freeze_message_count, 2);
        // Third turn: 6 messages; still the first frozen prefix.
        assert_eq!(t.resolve(&k, 6, 3_000, 600_000).freeze_message_count, 4);
    }

    #[test]
    fn ttl_expiry_resets() {
        let t = SessionTracker::new();
        let k = session_key(&body());
        assert_eq!(t.resolve(&k, 2, 0, 600_000).freeze_message_count, 0);
        assert_eq!(t.resolve(&k, 4, 10, 600_000).freeze_message_count, 2);
        // After the TTL the session is gone: freeze resets to 0.
        assert_eq!(t.resolve(&k, 6, 700_000, 600_000).freeze_message_count, 0);
    }
}
