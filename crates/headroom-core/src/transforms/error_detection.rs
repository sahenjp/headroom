//! Keyword-scored line importance for error triage.
//!
//! Python headroom delegates this to a Rust `score_line` plus a keyword
//! registry snapshot; this port implements both natively with the same
//! score tiers. Scoring is case-insensitive and deterministic: the highest
//! matching keyword score wins and unmatched lines score `0.0`.

use std::collections::HashSet;
use std::sync::OnceLock;

use super::tokenize;

/// A scored keyword registry: lowercase keywords with their importance.
#[derive(Debug, Clone)]
pub struct KeywordRegistry {
    /// Lowercase keyword (a single word or a phrase) with its score in
    /// `0.0..=1.0`.
    pub keywords: Vec<(String, f64)>,
}

impl KeywordRegistry {
    /// The built-in registry, shared by every caller.
    ///
    /// Score tiers: outright failures (`error`, `failed`, `exception`,
    /// `panic`, `fatal`, `critical`, `traceback`) score `1.0`;
    /// cautionary lines (`warning`, `warn`, `deprecated`) score `0.6`;
    /// annotations (`todo`, `fixme`, `note`) score `0.4`; success
    /// lines (`test passed`, `ok`) score `0.2`.
    pub fn snapshot() -> &'static KeywordRegistry {
        static SNAPSHOT: OnceLock<KeywordRegistry> = OnceLock::new();
        SNAPSHOT.get_or_init(|| KeywordRegistry {
            keywords: [
                ("error", 1.0),
                ("failed", 1.0),
                ("exception", 1.0),
                ("panic", 1.0),
                ("fatal", 1.0),
                ("critical", 1.0),
                ("traceback", 1.0),
                ("warning", 0.6),
                ("warn", 0.6),
                ("deprecated", 0.6),
                ("todo", 0.4),
                ("fixme", 0.4),
                ("note", 0.4),
                ("test passed", 0.2),
                ("ok", 0.2),
            ]
            .iter()
            .map(|(keyword, score)| ((*keyword).to_string(), *score))
            .collect(),
        })
    }

    /// Scores `line` in `0.0..=1.0` by its most important keyword.
    ///
    /// Matching is case-insensitive. Multi-word keywords match as
    /// substrings; single words longer than three chars also match as
    /// substrings (so `errors` hits `error`), while very short words such
    /// as `ok` only match whole alphanumeric tokens to avoid false hits
    /// inside unrelated words (`token`, `broker`).
    pub fn score(&self, line: &str) -> f64 {
        let lower = line.to_lowercase();
        let tokens: HashSet<String> = tokenize(line).into_iter().collect();
        let mut best = 0.0;
        for (keyword, score) in &self.keywords {
            let hit = if keyword.contains(' ') || keyword.len() > 3 {
                lower.contains(keyword.as_str())
            } else {
                tokens.contains(keyword)
            };
            if hit && *score > best {
                best = *score;
            }
        }
        best
    }
}

/// Scores a line with the built-in `KeywordRegistry::snapshot`.
pub fn score_line(line: &str) -> f64 {
    KeywordRegistry::snapshot().score(line)
}

/// True when the line scores at least `0.9` (an outright error line).
pub fn is_error_line(line: &str) -> bool {
    score_line(line) >= 0.9
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_keywords_score_one() {
        for line in [
            "ERROR: disk full",
            "Build FAILED in 3s",
            "Unhandled exception: boom",
            "thread 'main' panicked at 'x'",
            "FATAL: cannot continue",
            "CRITICAL battery empty",
            "Traceback (most recent call last):",
        ] {
            assert_eq!(score_line(line), 1.0, "line: {line}");
            assert!(is_error_line(line));
        }
    }

    #[test]
    fn warning_keywords_score_six_tenths() {
        assert_eq!(score_line("warning: deprecated flag"), 0.6);
        assert_eq!(score_line("WARN  slow query"), 0.6);
        assert!(!is_error_line("warning: deprecated flag"));
    }

    #[test]
    fn annotation_keywords_score_four_tenths() {
        assert_eq!(score_line("TODO: refactor this"), 0.4);
        assert_eq!(score_line("FIXME later"), 0.4);
        assert_eq!(score_line("note: kept for review"), 0.4);
    }

    #[test]
    fn success_keywords_score_two_tenths() {
        assert_eq!(score_line("test passed"), 0.2);
        assert_eq!(score_line("ok"), 0.2);
        assert_eq!(score_line("OK (12 tests)"), 0.2);
    }

    #[test]
    fn unmatched_lines_score_zero() {
        assert_eq!(score_line("everything is fine"), 0.0);
        assert_eq!(score_line(""), 0.0);
        // 'ok' is token-matched: 'token' and 'broker' do not hit it
        assert_eq!(score_line("token bucket broker"), 0.0);
    }

    #[test]
    fn substrings_hit_for_long_words() {
        // plural and inflected forms still match through substring scoring
        assert_eq!(score_line("0 errors, all good"), 1.0);
        assert_eq!(score_line("2 warnings emitted"), 0.6);
    }

    #[test]
    fn highest_matching_tier_wins() {
        assert_eq!(score_line("error in warning about a todo"), 1.0);
        assert_eq!(score_line("warning about a todo"), 0.6);
    }

    #[test]
    fn snapshot_is_a_shared_static() {
        assert!(std::ptr::eq(KeywordRegistry::snapshot(), KeywordRegistry::snapshot()));
    }

    #[test]
    fn custom_registry_scores_with_its_own_keywords() {
        let registry = KeywordRegistry {
            keywords: vec![("boom".to_string(), 0.5)],
        };
        assert_eq!(registry.score("BOOM!"), 0.5);
        assert_eq!(registry.score("all quiet"), 0.0);
    }
}
