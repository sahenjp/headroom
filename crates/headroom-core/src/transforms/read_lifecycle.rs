//! Stale / superseded Read detection.
//!
//! Pure decision function: the proxy supplies the timestamps and this
//! module only classifies. A read is Stale when the file was modified
//! after the read happened (current mtime newer than the mtime recorded at
//! read time) and Superseded when a later read of the same file already
//! happened — the proxy encodes that by supplying a file_mtime_ms that
//! postdates read_time_ms, which is impossible for a single consistent
//! read. Disabled configs and reads below min_size_bytes degrade to Fresh
//! (fail-open).

/// Configuration for the Read lifecycle transform.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadLifecycleConfig {
    /// Master switch; when `false` everything classifies as fresh.
    pub enabled: bool,
    /// Compress reads whose file changed after the read happened.
    pub compress_stale: bool,
    /// Compress reads superseded by a later read of the same file.
    pub compress_superseded: bool,
    /// Reads below this size (chars) are never touched.
    pub min_size_bytes: usize,
}

impl Default for ReadLifecycleConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            compress_stale: true,
            compress_superseded: false,
            min_size_bytes: 512,
        }
    }
}

/// Lifecycle state of a stored Read result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadState {
    /// The read still matches the file; keep it as-is.
    Fresh,
    /// The file was modified after this read happened.
    Stale,
    /// A later read of the same file already replaced this one.
    Superseded,
}

/// Read lifecycle decision helper.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReadDecision;

impl ReadDecision {
    /// Classifies a stored read.
    ///
    /// * content_chars — length of the stored read content,
    /// * file_mtime_ms — file mtime recorded at read time,
    /// * read_time_ms — when the read happened,
    /// * file_mtime_now_ms — the file's current mtime.
    ///
    /// Superseded wins over stale when both apply and its compression is
    /// enabled; when compress_superseded is off the stale rule still
    /// applies, and both flags off degrade to Fresh.
    pub fn classify(
        content_chars: usize,
        file_mtime_ms: i64,
        read_time_ms: i64,
        file_mtime_now_ms: i64,
        config: &ReadLifecycleConfig,
    ) -> ReadState {
        if !config.enabled || content_chars < config.min_size_bytes {
            return ReadState::Fresh;
        }
        if read_time_ms < file_mtime_ms && config.compress_superseded {
            return ReadState::Superseded;
        }
        if file_mtime_now_ms > file_mtime_ms && config.compress_stale {
            return ReadState::Stale;
        }
        ReadState::Fresh
    }
}

/// Marker for a stale read; wording is load-bearing.
pub fn stale_marker(file: &str, hash: &str) -> String {
    format!(
        "[Read content stale: {file} was modified after this read — re-read the file for current content. Retrieve original: hash={hash}]"
    )
}

/// Marker for a superseded read; wording is load-bearing.
pub fn superseded_marker(file: &str, hash: &str) -> String {
    format!(
        "[Read content superseded: {file} was re-read later — re-read the file if needed. Retrieve original: hash={hash}]"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_when_file_unmodified() {
        // mtime at read time == current mtime, read after that mtime
        assert_eq!(
            ReadDecision::classify(1000, 100, 200, 100, &ReadLifecycleConfig::default()),
            ReadState::Fresh
        );
    }

    #[test]
    fn stale_when_file_modified_after_read() {
        assert_eq!(
            ReadDecision::classify(1000, 100, 200, 300, &ReadLifecycleConfig::default()),
            ReadState::Stale
        );
    }

    #[test]
    fn superseded_when_proxy_reports_a_later_read() {
        // proxy contract: file_mtime_ms postdates read_time_ms only when a
        // newer read of the same file already happened
        let config = ReadLifecycleConfig {
            compress_superseded: true,
            ..ReadLifecycleConfig::default()
        };
        assert_eq!(
            ReadDecision::classify(1000, 300, 200, 300, &config),
            ReadState::Superseded
        );
    }

    #[test]
    fn superseded_falls_back_to_stale_when_disabled() {
        let config = ReadLifecycleConfig::default(); // compress_superseded = false
        // still stale: the file changed again afterwards
        assert_eq!(
            ReadDecision::classify(1000, 300, 200, 400, &config),
            ReadState::Stale
        );
        // no further modification: nothing to report
        assert_eq!(
            ReadDecision::classify(1000, 300, 200, 300, &config),
            ReadState::Fresh
        );
    }

    #[test]
    fn min_size_reads_stay_fresh() {
        assert_eq!(
            ReadDecision::classify(511, 100, 200, 300, &ReadLifecycleConfig::default()),
            ReadState::Fresh
        );
        assert_eq!(
            ReadDecision::classify(512, 100, 200, 300, &ReadLifecycleConfig::default()),
            ReadState::Stale
        );
    }

    #[test]
    fn disabled_config_is_always_fresh() {
        let config = ReadLifecycleConfig {
            enabled: false,
            ..ReadLifecycleConfig::default()
        };
        assert_eq!(
            ReadDecision::classify(9999, 100, 200, 300, &config),
            ReadState::Fresh
        );
    }

    #[test]
    fn compress_stale_off_degrades_to_fresh() {
        let config = ReadLifecycleConfig {
            compress_stale: false,
            ..ReadLifecycleConfig::default()
        };
        assert_eq!(
            ReadDecision::classify(1000, 100, 200, 300, &config),
            ReadState::Fresh
        );
    }

    #[test]
    fn markers_use_exact_wording() {
        assert_eq!(
            stale_marker("src/a.rs", "abc123"),
            "[Read content stale: src/a.rs was modified after this read — re-read the file for current content. Retrieve original: hash=abc123]"
        );
        assert_eq!(
            superseded_marker("src/a.rs", "abc123"),
            "[Read content superseded: src/a.rs was re-read later — re-read the file if needed. Retrieve original: hash=abc123]"
        );
    }
}
