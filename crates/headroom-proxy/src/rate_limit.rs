//! RPM/TPM sliding-window rate limiter (fail-open: callers never reject
//! on limiter errors; a Limited decision is advisory).

use std::collections::VecDeque;

/// Decision for one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateDecision {
    Allow,
    Limited { retry_after_secs: u64 },
}

struct State {
    window: VecDeque<(i64, i64)>, // (now_ms, tokens)
}

/// Sliding 60s window limiter. `None` limits always allow.
pub struct RateLimiter {
    rpm: Option<u32>,
    tpm: Option<u32>,
    state: std::sync::Mutex<State>,
}

const WINDOW_MS: i64 = 60_000;

impl RateLimiter {
    pub fn new(rpm: Option<u32>, tpm: Option<u32>) -> Self {
        Self {
            rpm,
            tpm,
            state: std::sync::Mutex::new(State { window: VecDeque::new() }),
        }
    }

    /// Check whether a request with `est_tokens` fits the window. Does NOT
    /// record anything (call `record` after the request completes).
    pub fn check(&self, est_tokens: i64, now_ms: i64) -> RateDecision {
        let mut state = self.state.lock().unwrap();
        evict(&mut state.window, now_ms);
        let requests = state.window.len() as u64;
        let tokens: i64 = state.window.iter().map(|(_, t)| *t).sum();
        if let Some(rpm) = self.rpm {
            if requests >= rpm as u64 {
                return RateDecision::Limited { retry_after_secs: retry_secs(&state.window, now_ms) };
            }
        }
        if let Some(tpm) = self.tpm {
            if tokens + est_tokens.max(0) > tpm as i64 {
                return RateDecision::Limited { retry_after_secs: retry_secs(&state.window, now_ms) };
            }
        }
        RateDecision::Allow
    }

    /// Record an actual request (tokens actually used).
    pub fn record(&self, tokens: i64, now_ms: i64) {
        let mut state = self.state.lock().unwrap();
        evict(&mut state.window, now_ms);
        state.window.push_back((now_ms, tokens.max(0)));
    }
}

fn evict(window: &mut VecDeque<(i64, i64)>, now_ms: i64) {
    while let Some(&(t, _)) = window.front() {
        if now_ms - t >= WINDOW_MS {
            window.pop_front();
        } else {
            break;
        }
    }
}

fn retry_secs(window: &VecDeque<(i64, i64)>, now_ms: i64) -> u64 {
    match window.front() {
        Some(&(t, _)) => ((t + WINDOW_MS - now_ms).max(0) / 1000 + 1) as u64,
        None => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_limits_always_allow() {
        let rl = RateLimiter::new(None, None);
        assert_eq!(rl.check(1_000_000, 0), RateDecision::Allow);
    }

    #[test]
    fn rpm_exhaustion() {
        let rl = RateLimiter::new(Some(2), None);
        assert_eq!(rl.check(0, 0), RateDecision::Allow);
        rl.record(1, 0);
        assert_eq!(rl.check(0, 1), RateDecision::Allow);
        rl.record(1, 1);
        match rl.check(0, 2) {
            RateDecision::Limited { retry_after_secs } => assert!(retry_after_secs >= 1),
            RateDecision::Allow => panic!("should be limited"),
        }
    }

    #[test]
    fn tpm_exhaustion() {
        let rl = RateLimiter::new(None, Some(100));
        assert_eq!(rl.check(60, 0), RateDecision::Allow);
        rl.record(60, 0);
        assert!(matches!(rl.check(60, 1), RateDecision::Limited { .. }));
    }

    #[test]
    fn window_expiry() {
        let rl = RateLimiter::new(Some(1), None);
        rl.record(1, 0);
        assert!(matches!(rl.check(0, 1), RateDecision::Limited { .. }));
        // After the 60s window passes the entry is evicted.
        assert_eq!(rl.check(0, 61_000), RateDecision::Allow);
    }
}
