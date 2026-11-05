//! Heuristic main-content extraction for HTML pages.
//!
//! Deviation from Python headroom (documented): the Python port delegates
//! to `trafilatura`; this port uses a deterministic, dependency-free
//! heuristic — boilerplate-element removal, block-boundary newlines, tag
//! stripping, and largest-text-block selection. It is weaker than
//! trafilatura on adversarial pages but stable, allocation-light and
//! side-effect free.

use std::sync::LazyLock;

use regex::Regex;

use crate::ccr;

/// Minimum extracted length (chars) for an extraction to be usable.
const MIN_EXTRACT_CHARS: usize = 200;

/// HTML comments, removed with their content.
static COMMENT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<!--.*?-->").expect("static regex is valid"));

/// Boilerplate elements removed with their entire content (one regex per
/// element; the regex crate has no backreferences). An unclosed boilerplate
/// element swallows the rest of the document (fail-closed for script/style
/// noise, which is what an unclosed tag usually means).
static BOILERPLATE_RES: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    ["script", "style", "head", "nav", "footer", "header", "aside"]
        .iter()
        .map(|tag| {
            Regex::new(&format!(r"(?is)<{tag}\b[^>]*>.*?(?:</\s*{tag}\s*>|\z)"))
                .expect("static regex is valid")
        })
        .collect()
});

/// Block-level tags render as a newline so text blocks stay separated.
static BLOCK_TAG_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)</?(?:p|div|br|h[1-6]|li|tr|td|th|table|blockquote|pre|section|article|main|ul|ol|dl|dt|dd|figure|figcaption|hr|form|fieldset|caption)\b[^>]*>",
    )
    .expect("static regex is valid")
});

/// Any remaining tag is dropped. The leading character class requires a
/// tag-like start so prose such as `"I <3 rust"` survives.
static ANY_TAG_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"</?[a-zA-Z!/?][^>]*>").expect("static regex is valid"));

/// Extracts the main text content of an HTML page.
///
/// Strips comments, `script`/`style`/`head`/`nav`/`footer`/`header`/`aside`
/// blocks and all remaining tags (block-level tags become newlines, blank
/// lines collapse away), then returns the largest text block between
/// block-level boundaries. Returns `None` when the largest block is
/// shorter than 200 chars or not shorter than the input.
pub fn extract_main_content(html: &str) -> Option<String> {
    if html.is_empty() {
        return None;
    }
    let without_comments = COMMENT_RE.replace_all(html, "");
    let mut without_boilerplate = without_comments.into_owned();
    for boilerplate in BOILERPLATE_RES.iter() {
        without_boilerplate = boilerplate.replace_all(&without_boilerplate, " ").into_owned();
    }
    let with_breaks = BLOCK_TAG_RE.replace_all(&without_boilerplate, "\n");
    let text_only = ANY_TAG_RE.replace_all(&with_breaks, "");

    let best = text_only
        .lines()
        .filter(|line| !line.trim().is_empty())
        .max_by_key(|line| line.trim().chars().count())?;
    // Compare against the *untrimmed* line so a tagless input (where the
    // only change is trimming) is rejected as a no-op.
    if best.trim().chars().count() < MIN_EXTRACT_CHARS || best.len() >= html.len() {
        return None;
    }
    Some(best.trim().to_string())
}

/// Result of an `HtmlCompressor::compress` run.
#[derive(Debug, Clone, PartialEq)]
pub struct HtmlResult {
    /// Extracted main content with the appended retrieval marker.
    pub extracted: String,
    /// Length of the original HTML in chars.
    pub original_len: usize,
}

/// Heuristic HTML main-content extractor.
#[derive(Debug, Clone, Default)]
pub struct HtmlCompressor;

impl HtmlCompressor {
    /// Extracts the main content and appends the retrieval marker
    /// `"[HTML main content extracted ({extracted_len} of {original_len}
    /// chars). Retrieve original: hash=...]"`.
    ///
    /// Returns `None` when `extract_main_content` declines the page.
    pub fn compress(&self, html: &str) -> Option<HtmlResult> {
        let extracted = extract_main_content(html)?;
        let marker = format!(
            "\n[HTML main content extracted ({} of {} chars). Retrieve original: hash={}]",
            extracted.chars().count(),
            html.chars().count(),
            ccr::ccr_hash(html)
        );
        Some(HtmlResult {
            extracted: format!("{extracted}{marker}"),
            original_len: html.chars().count(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A page whose dominant block is one long paragraph.
    fn page_with_paragraph() -> String {
        let paragraph = "Alpha beta gamma delta epsilon zeta eta theta iota kappa. ".repeat(5);
        format!(
            "<html><head><title>Doc</title><style>body {{ color: red; }}</style></head>\n\
             <body>\n\
             <nav>Home Links About Contact</nav>\n\
             <script>var tracking = 1; function t() {{ return 2; }}</script>\n\
             <article><h1>Report</h1><p>{paragraph}</p><p>tiny</p></article>\n\
             <footer>Copyright 2026 Example Corp</footer>\n\
             </body></html>\n"
        )
    }

    #[test]
    fn extracts_largest_paragraph_and_strips_boilerplate() {
        let html = page_with_paragraph();
        let extracted = extract_main_content(&html).unwrap();
        let paragraph = "Alpha beta gamma delta epsilon zeta eta theta iota kappa. ".repeat(5);
        assert_eq!(extracted, paragraph.trim());
        assert!(!extracted.contains("var tracking"));
        assert!(!extracted.contains("Home Links"));
        assert!(!extracted.contains("Copyright"));
        assert!(!extracted.contains("color: red"));
        assert!(!extracted.contains("<p>"));
    }

    #[test]
    fn compress_appends_marker_with_counts() {
        let html = page_with_paragraph();
        let extracted = extract_main_content(&html).unwrap();
        let result = HtmlCompressor.compress(&html).unwrap();
        assert_eq!(result.original_len, html.chars().count());
        assert!(result.extracted.starts_with(extracted.as_str()));
        assert!(result.extracted.contains(&format!(
            "\n[HTML main content extracted ({} of {} chars). Retrieve original: hash={}]",
            extracted.chars().count(),
            html.chars().count(),
            ccr::ccr_hash(&html)
        )));
        assert!(ccr::contains_ccr_marker(&result.extracted));
    }

    #[test]
    fn tiny_html_passes_through() {
        assert!(extract_main_content("<p>hi</p>").is_none());
        assert!(HtmlCompressor.compress("<html><body>short</body></html>").is_none());
    }

    #[test]
    fn untagged_text_is_not_shorter_so_rejected() {
        let text = "plain prose without any tags at all, just a long line. ".repeat(6);
        assert!(extract_main_content(&text).is_none());
    }

    #[test]
    fn unclosed_script_swallows_the_rest() {
        let html = format!(
            "<html><body><script>evil = '{}';<p>legit but swallowed</p>",
            "x".repeat(300)
        );
        // the extraction declines rather than returning script noise
        assert!(extract_main_content(&html).is_none());
    }

    #[test]
    fn prose_angle_bracket_survives_tag_stripping() {
        let text = "I <3 rust and rust <3 me, this is a genuinely long line of prose. ".repeat(6);
        // no tag-like construct: extraction declines (not shorter than input)
        assert!(extract_main_content(&text).is_none());
    }
}
