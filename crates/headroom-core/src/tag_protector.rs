//! Protect custom (non-HTML) XML tags from text compression.
//!
//! Port of headroom/transforms/tag_protector.py, which delegates to the
//! single-pass Rust walker. Custom tags (workflow markers, tool fences,
//! anything not in the HTML5 list) are lifted out of the text and replaced
//! with numbered placeholders before a compressor runs; [restore_tags] swaps
//! them back afterwards.
//!
//! Walker properties (the bug fixes the Python shim documents):
//! - single pass, linear in input length (no scan-and-replace restarts);
//! - output is stitched by offset, so two identical tag blocks always get
//!   distinct placeholders;
//! - self-closing custom tags are handled in the same pass;
//! - if the input already contains a literal '{{HEADROOM_TAG_' substring the
//!   placeholder prefix is salted so it cannot collide with user text;
//! - [restore_tags] discards a block whose placeholder the compressor dropped
//!   instead of re-injecting orphan tags.

/// The HTML5 tag list: tags here are real markup and never protected.
pub fn known_html_tag_names() -> &'static [&'static str] {
    &[
        "a",
        "abbr",
        "address",
        "area",
        "article",
        "aside",
        "audio",
        "b",
        "base",
        "bdi",
        "bdo",
        "blockquote",
        "body",
        "br",
        "button",
        "canvas",
        "caption",
        "cite",
        "code",
        "col",
        "colgroup",
        "data",
        "datalist",
        "dd",
        "del",
        "details",
        "dfn",
        "dialog",
        "div",
        "dl",
        "dt",
        "em",
        "embed",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "head",
        "header",
        "hgroup",
        "hr",
        "html",
        "i",
        "iframe",
        "img",
        "input",
        "ins",
        "kbd",
        "label",
        "legend",
        "li",
        "link",
        "main",
        "map",
        "mark",
        "menu",
        "meta",
        "meter",
        "nav",
        "noscript",
        "object",
        "ol",
        "optgroup",
        "option",
        "output",
        "p",
        "param",
        "picture",
        "pre",
        "progress",
        "q",
        "rp",
        "rt",
        "ruby",
        "s",
        "samp",
        "script",
        "section",
        "select",
        "slot",
        "small",
        "source",
        "span",
        "strong",
        "style",
        "sub",
        "summary",
        "sup",
        "table",
        "tbody",
        "td",
        "template",
        "textarea",
        "tfoot",
        "th",
        "thead",
        "time",
        "title",
        "tr",
        "track",
        "u",
        "ul",
        "var",
        "video",
        "wbr",
    ]
}

/// Case-insensitive HTML5 tag membership check.
pub fn is_html_tag(name: &str) -> bool {
    known_html_tag_names()
        .iter()
        .any(|known| known.eq_ignore_ascii_case(name))
}

/// Protect custom XML tag blocks from compression.
///
/// Walks 'text' once and lifts every custom-tag region (tag name matches
/// '[A-Za-z][A-Za-z0-9_-]*' and is not an HTML5 tag) into a numbered
/// placeholder of the form '{{HEADROOM_TAG_<n>}}'. Returns the cleaned text
/// and the (placeholder, original) pairs, in order; hand the whole list to
/// [restore_tags] after the compressor has run.
///
/// - 'compress_tagged_content == false' (default): the ENTIRE
///   '<tag ...>...</tag>' block is replaced by one placeholder. A custom
///   opener with no matching closer is left verbatim.
/// - 'compress_tagged_content == true': only the opening and closing tag
///   markers are protected; the inner content stays exposed for compression.
///   Self-closing tags always collapse to one placeholder.
///
/// If the input already contains a literal '{{HEADROOM_TAG_' substring, the
/// placeholder prefix is salted with the input's byte length
/// ('{{HEADROOM_TAG_X<len>_<n>}}') so placeholders can never collide with
/// user text. The salt is derived from the input, not randomness, so the
/// function stays deterministic.
pub fn protect_tags(text: &str, compress_tagged_content: bool) -> (String, Vec<(String, String)>) {
    let bytes = text.as_bytes();
    let prefix = placeholder_prefix(text);
    let mut out = String::with_capacity(text.len() + 16);
    let mut blocks: Vec<(String, String)> = Vec::new();
    let mut next_id: usize = 1;
    let mut i = 0usize;

    while i < bytes.len() {
        if bytes[i] != b'<' {
            // Fast path: copy the run up to the next '<'.
            let start = i;
            i += 1;
            while i < bytes.len() && bytes[i] != b'<' {
                i += 1;
            }
            out.push_str(&text[start..i]);
            continue;
        }
        if let Some((name, tag_end)) = parse_open_tag(bytes, i) {
            if !is_html_tag(&name) {
                let self_closing = bytes[tag_end - 2] == b'/';
                if self_closing || compress_tagged_content {
                    let placeholder =
                        take_block(&mut blocks, &mut next_id, &prefix, &text[i..tag_end]);
                    out.push_str(&placeholder);
                    i = tag_end;
                    continue;
                }
                let close_needle = format!("</{}>", name);
                if let Some(rel) = text[tag_end..].find(close_needle.as_str()) {
                    let close_end = tag_end + rel + close_needle.len();
                    let placeholder =
                        take_block(&mut blocks, &mut next_id, &prefix, &text[i..close_end]);
                    out.push_str(&placeholder);
                    i = close_end;
                    continue;
                }
                // No matching closer: not a block; leave the opener verbatim.
            }
        }
        if let Some((name, close_end)) = parse_close_tag(bytes, i) {
            if !is_html_tag(&name) && compress_tagged_content {
                let placeholder =
                    take_block(&mut blocks, &mut next_id, &prefix, &text[i..close_end]);
                out.push_str(&placeholder);
                i = close_end;
                continue;
            }
        }
        out.push('<');
        i += 1;
    }

    (out, blocks)
}

/// Swap placeholders back to their original tag blocks.
///
/// Every occurrence of each placeholder is replaced. A placeholder missing
/// from the compressed text means the compressor dropped it: that block is
/// discarded and its original tag bytes are NOT re-injected anywhere.
pub fn restore_tags(text: &str, blocks: &[(String, String)]) -> String {
    let mut out = text.to_string();
    for (placeholder, original) in blocks {
        if out.contains(placeholder.as_str()) {
            out = out.replace(placeholder.as_str(), original);
        }
    }
    out
}

/// Placeholder prefix, salted with the input byte length when the input
/// already contains a literal placeholder prefix.
fn placeholder_prefix(text: &str) -> String {
    if text.contains("{{HEADROOM_TAG_") {
        format!("{{{{HEADROOM_TAG_X{}_", text.len())
    } else {
        "{{HEADROOM_TAG_".to_string()
    }
}

/// Record one protected region and return its placeholder.
fn take_block(
    blocks: &mut Vec<(String, String)>,
    next_id: &mut usize,
    prefix: &str,
    original: &str,
) -> String {
    let placeholder = format!("{}{}{}", prefix, next_id, "}}");
    blocks.push((placeholder.clone(), original.to_string()));
    *next_id += 1;
    placeholder
}

/// Parse '<name ...>' starting at 'start' (bytes[start] == '<').
///
/// Returns the tag name and the byte index just past the closing '>'.
/// Bails on a nested '<' before the tag closes (malformed markup).
fn parse_open_tag(bytes: &[u8], start: usize) -> Option<(String, usize)> {
    let mut j = start + 1;
    if j >= bytes.len() || !bytes[j].is_ascii_alphabetic() {
        return None;
    }
    let name_start = j;
    while j < bytes.len()
        && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_' || bytes[j] == b'-')
    {
        j += 1;
    }
    let name = std::str::from_utf8(&bytes[name_start..j]).ok()?.to_string();
    while j < bytes.len() && bytes[j] != b'>' {
        if bytes[j] == b'<' {
            return None;
        }
        j += 1;
    }
    if j >= bytes.len() {
        return None;
    }
    Some((name, j + 1))
}

/// Parse '</name>' starting at 'start' (bytes[start] == '<').
///
/// Returns the tag name and the byte index just past the '>'.
fn parse_close_tag(bytes: &[u8], start: usize) -> Option<(String, usize)> {
    if start + 1 >= bytes.len() || bytes[start + 1] != b'/' {
        return None;
    }
    let mut j = start + 2;
    if j >= bytes.len() || !bytes[j].is_ascii_alphabetic() {
        return None;
    }
    let name_start = j;
    while j < bytes.len()
        && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_' || bytes[j] == b'-')
    {
        j += 1;
    }
    if j >= bytes.len() || bytes[j] != b'>' {
        return None;
    }
    let name = std::str::from_utf8(&bytes[name_start..j]).ok()?.to_string();
    Some((name, j + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_tag_list_and_membership() {
        let names = known_html_tag_names();
        assert!(names.contains(&"div"));
        assert!(names.contains(&"template"));
        assert!(names.len() > 100);
        // case-insensitive membership
        assert!(is_html_tag("div"));
        assert!(is_html_tag("DIV"));
        assert!(is_html_tag("Template"));
        assert!(!is_html_tag("workflow"));
        assert!(!is_html_tag("my-tag"));
        assert!(!is_html_tag(""));
    }

    #[test]
    fn protects_simple_custom_block() {
        let (cleaned, blocks) = protect_tags("before <workflow>step one</workflow> after", false);
        assert_eq!(cleaned, "before {{HEADROOM_TAG_1}} after");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].0, "{{HEADROOM_TAG_1}}");
        assert_eq!(blocks[0].1, "<workflow>step one</workflow>");
        assert_eq!(
            restore_tags(&cleaned, &blocks),
            "before <workflow>step one</workflow> after"
        );
    }

    #[test]
    fn nested_custom_tags_full_block_mode() {
        let text = "<outer>keep <inner>this</inner> too</outer>";
        let (cleaned, blocks) = protect_tags(text, false);
        // the outermost block wins; the inner tag is part of its original
        assert_eq!(cleaned, "{{HEADROOM_TAG_1}}");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].1, text);
        assert_eq!(restore_tags(&cleaned, &blocks), text);
    }

    #[test]
    fn nested_custom_tags_marker_mode_exposes_content() {
        let text = "<outer>a <inner>b</inner> c</outer>";
        let (cleaned, blocks) = protect_tags(text, true);
        assert_eq!(
            cleaned,
            "{{HEADROOM_TAG_1}}a {{HEADROOM_TAG_2}}b{{HEADROOM_TAG_3}} c{{HEADROOM_TAG_4}}"
        );
        assert_eq!(
            blocks,
            vec![
                ("{{HEADROOM_TAG_1}}".to_string(), "<outer>".to_string()),
                ("{{HEADROOM_TAG_2}}".to_string(), "<inner>".to_string()),
                ("{{HEADROOM_TAG_3}}".to_string(), "</inner>".to_string()),
                ("{{HEADROOM_TAG_4}}".to_string(), "</outer>".to_string()),
            ]
        );
        assert_eq!(restore_tags(&cleaned, &blocks), text);
    }

    #[test]
    fn duplicate_identical_blocks_get_distinct_placeholders() {
        let text = "<t>x</t> mid <t>x</t>";
        let (cleaned, blocks) = protect_tags(text, false);
        assert_eq!(cleaned, "{{HEADROOM_TAG_1}} mid {{HEADROOM_TAG_2}}");
        assert_eq!(blocks.len(), 2);
        assert_ne!(blocks[0].0, blocks[1].0);
        assert_eq!(blocks[0].1, "<t>x</t>");
        assert_eq!(blocks[1].1, "<t>x</t>");
        assert_eq!(restore_tags(&cleaned, &blocks), text);
    }

    #[test]
    fn self_closing_custom_tags() {
        let (cleaned, blocks) = protect_tags("a <flag/> b <spacer attr=\"1\" /> c", true);
        assert_eq!(cleaned, "a {{HEADROOM_TAG_1}} b {{HEADROOM_TAG_2}} c");
        assert_eq!(blocks[0].1, "<flag/>");
        assert_eq!(blocks[1].1, "<spacer attr=\"1\" />");
        // full-block mode also collapses self-closers in the same pass
        let (cleaned, blocks) = protect_tags("a <flag/> b", false);
        assert_eq!(cleaned, "a {{HEADROOM_TAG_1}} b");
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn html_tags_are_left_alone() {
        let text = "<div class=\"x\">hello <span>world</span></div>";
        let (cleaned, blocks) = protect_tags(text, false);
        assert_eq!(cleaned, text);
        assert!(blocks.is_empty());
        // case-insensitive HTML stays untouched too
        let (cleaned, blocks) = protect_tags("<DIV>hi</DIV> <br/>", true);
        assert_eq!(cleaned, "<DIV>hi</DIV> <br/>");
        assert!(blocks.is_empty());
    }

    #[test]
    fn tag_name_grammar() {
        // must start with a letter; digits cannot open a tag
        let (cleaned, blocks) = protect_tags("<1bad> x", false);
        assert_eq!(cleaned, "<1bad> x");
        assert!(blocks.is_empty());
        // hyphens and underscores are part of the name
        let (cleaned, blocks) = protect_tags("<my-tag_2>x</my-tag_2>", false);
        assert_eq!(cleaned, "{{HEADROOM_TAG_1}}");
        assert_eq!(blocks[0].1, "<my-tag_2>x</my-tag_2>");
        // unclosed custom tag is left verbatim in full-block mode
        let (cleaned, blocks) = protect_tags("a <t>unclosed", false);
        assert_eq!(cleaned, "a <t>unclosed");
        assert!(blocks.is_empty());
        // ... but its opener is still protected in marker mode
        let (cleaned, blocks) = protect_tags("a <t>unclosed", true);
        assert_eq!(cleaned, "a {{HEADROOM_TAG_1}}unclosed");
        assert_eq!(blocks[0].1, "<t>");
    }

    #[test]
    fn placeholder_collision_is_salted() {
        let text = "literal {{HEADROOM_TAG_1}} here <t>x</t>";
        let (cleaned, blocks) = protect_tags(text, false);
        // the salted prefix embeds the input byte length
        let salted = format!("{{{{HEADROOM_TAG_X{}_", text.len());
        assert!(cleaned.starts_with("literal {{HEADROOM_TAG_1}} here "));
        assert!(cleaned.contains(&salted));
        assert!(!cleaned.contains("{{HEADROOM_TAG_1}}<"));
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].0.starts_with(&salted));
        assert_eq!(blocks[0].1, "<t>x</t>");
        // the user's literal placeholder is never touched
        assert_eq!(restore_tags(&cleaned, &blocks), text);
    }

    #[test]
    fn restore_discards_lost_placeholder() {
        let (cleaned, blocks) = protect_tags("<x>1</x> keep <y>2</y>", false);
        assert_eq!(cleaned, "{{HEADROOM_TAG_1}} keep {{HEADROOM_TAG_2}}");
        // a compressor that dropped the second placeholder
        let compressed = "{{HEADROOM_TAG_1}} keep".to_string();
        let restored = restore_tags(&compressed, &blocks);
        // the surviving block is restored; the lost one is NOT re-injected
        assert_eq!(restored, "<x>1</x> keep");
    }

    #[test]
    fn restore_replaces_every_occurrence() {
        let (_, blocks) = protect_tags("<t>x</t>", false);
        let compressed = "{{HEADROOM_TAG_1}} and {{HEADROOM_TAG_1}}";
        assert_eq!(restore_tags(compressed, &blocks), "<t>x</t> and <t>x</t>");
    }

    #[test]
    fn empty_and_plain_text() {
        let (cleaned, blocks) = protect_tags("", false);
        assert_eq!(cleaned, "");
        assert!(blocks.is_empty());
        let (cleaned, blocks) = protect_tags("no tags at all", true);
        assert_eq!(cleaned, "no tags at all");
        assert!(blocks.is_empty());
        assert_eq!(restore_tags("unchanged", &[]), "unchanged");
    }
}
