//! SSE usage extraction (read-only: responses still stream through
//! untouched; this only observes tokens for the savings ledger).

/// Accumulated usage from a streamed response.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SseUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_tokens: i64,
}

/// Incremental SSE accumulator. Feed raw chunks; events may split across
/// chunk boundaries (a trailing partial line is buffered).
#[derive(Debug, Default)]
pub struct SseAccumulator {
    pending: String,
    usage: SseUsage,
}

impl SseAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one raw chunk of the stream.
    pub fn feed(&mut self, chunk: &str) {
        self.pending.push_str(chunk);
        let pending = std::mem::take(&mut self.pending);
        let mut lines: Vec<&str> = pending.split('\n').collect();
        let trailing = lines.pop().unwrap_or("").to_string();
        self.pending = trailing;
        for line in lines {
            self.parse_line(line);
        }
    }

    fn parse_line(&mut self, line: &str) {
        let Some(data) = line.strip_prefix("data: ") else {
            return;
        };
        if data.trim() == "[DONE]" {
            return;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
            return;
        };
        // Anthropic: message_start carries usage.input_tokens (+cache).
        if v["type"] == "message_start" {
            let u = &v["message"]["usage"];
            if let Some(n) = u["input_tokens"].as_i64() {
                self.usage.input_tokens = self.usage.input_tokens.max(n);
            }
            if let Some(n) = u["cache_read_input_tokens"].as_i64() {
                self.usage.cache_read_tokens = self.usage.cache_read_tokens.max(n);
            }
        }
        // Anthropic: message_delta carries usage.output_tokens (cumulative).
        if v["type"] == "message_delta" {
            if let Some(n) = v["usage"]["output_tokens"].as_i64() {
                self.usage.output_tokens = self.usage.output_tokens.max(n);
            }
        }
        // OpenAI chat chunk with stream_options.include_usage.
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            if let Some(n) = u["prompt_tokens"].as_i64() {
                self.usage.input_tokens = self.usage.input_tokens.max(n);
            }
            if let Some(n) = u["completion_tokens"].as_i64() {
                self.usage.output_tokens = self.usage.output_tokens.max(n);
            }
            if let Some(n) = u["prompt_tokens_details"]["cached_tokens"].as_i64() {
                self.usage.cache_read_tokens = self.usage.cache_read_tokens.max(n);
            }
        }
    }

    /// Finish and return the accumulated usage.
    pub fn finish(mut self) -> SseUsage {
        // Flush any complete-but-unterminated final line.
        if !self.pending.is_empty() {
            let last = std::mem::take(&mut self.pending);
            self.parse_line(&last);
        }
        self.usage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ANTHROPIC_STREAM: &str = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":120,\"cache_read_input_tokens\":80}}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":42}}\n\ndata: [DONE]\n\n";

    #[test]
    fn anthropic_stream_usage() {
        let mut acc = SseAccumulator::new();
        acc.feed(ANTHROPIC_STREAM);
        let u = acc.finish();
        assert_eq!(u.input_tokens, 120);
        assert_eq!(u.cache_read_tokens, 80);
        assert_eq!(u.output_tokens, 42);
    }

    #[test]
    fn openai_stream_usage() {
        let stream = "data: {\"id\":\"1\",\"usage\":{\"prompt_tokens\":50,\"completion_tokens\":7,\"prompt_tokens_details\":{\"cached_tokens\":30}}}\n\ndata: [DONE]\n\n";
        let mut acc = SseAccumulator::new();
        acc.feed(stream);
        let u = acc.finish();
        assert_eq!(u.input_tokens, 50);
        assert_eq!(u.output_tokens, 7);
        assert_eq!(u.cache_read_tokens, 30);
    }

    #[test]
    fn event_split_across_chunks() {
        let mut acc = SseAccumulator::new();
        acc.feed("data: {\"type\":\"message_de");
        acc.feed("lta\",\"usage\":{\"output_tokens\":9}}\n");
        let u = acc.finish();
        assert_eq!(u.output_tokens, 9);
    }
}
