//! CSV-schema densification codec: encode and inverse-decode.
//!
//! Mirror of headroom/transforms/compaction_codec.py. SmartCrusher renders a
//! JSON array of objects into a compact '[N]{schema}' header followed by CSV
//! rows; this module provides both directions for the Rust port:
//! [encode_compacted] (the engine's encode-only side, reimplemented here) and
//! [expand_compacted] (the full inverse decoder), so callers can verify
//! losslessness and re-expand on demand.
//!
//! Wire format:
//!
//! ```text
//! [N]{name:type,name:type?,...} [__dict:K] [__dropped:k]
//! @col=["v1","v2",...]          (K value-factor legend lines)
//! cell,cell,...                 (N CSV rows, newline-terminated)
//! ```
//!
//! Column declarations split on the FINAL ':' so dotted names
//! ('meta.owner', from one-level flattening of uniform nested objects)
//! survive. Type tags: int, float, bool, string, json, null; an unknown tag
//! degrades to string; a trailing '?' marks a nullable column.
//!
//! Cell-encoding contract (encoder and decoder round-trip):
//!
//! | original value      | rendered cell        | decoded back to          |
//! |---------------------|----------------------|--------------------------|
//! | missing key         | '' (bare empty)      | key omitted from the map |
//! | JSON null           | 'null' (bare)        | null                     |
//! | '' (empty string)   | '""' (quoted)        | ''                       |
//! | literal 'null'      | '"null"' (quoted)   | 'null'                   |
//! | string with , " or newline | CSV-quoted    | the string               |
//! | other string        | bare                 | the string               |
//! | number / bool       | bare                | int / float / bool       |
//! | nested object/array | CSV-quoted JSON      | the value                |
//! | opaque blob         | '<<ccr:hash,...>>'  | NOT reversible from text |
//!
//! [expand_compacted] refuses (returns None) any block carrying a removal
//! marker ([contains_removal_marker]): a CCR pointer or dropped-row count
//! means the text alone cannot reconstruct the original.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};

use serde_json::{Map, Value};

/// Bare cell for an absent key. '' and 'null' are always CSV-quoted when
/// they are string *values*, so neither is ambiguous on the wire.
const MISSING_CELL: &str = "";
/// Bare cell for JSON null.
const NULL_TOKEN: &str = "null";
/// Maximum inner keys for one-level flattening of a uniform nested object.
const MAX_FLATTEN_INNER_KEYS: usize = 6;

/// Scalar type tags a schema header can declare. 'json' covers nested
/// objects/arrays and mixed-type columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColType {
    /// Integer column.
    Int,
    /// Floating-point column.
    Float,
    /// Boolean column.
    Bool,
    /// String column.
    String,
    /// Nested object / array / mixed-type column.
    Json,
    /// Column that only ever holds null.
    Null,
}

impl ColType {
    /// Parse a schema type tag; an unknown tag degrades to [ColType::String]
    /// so the decoder stays total on foreign input.
    pub fn from_tag(tag: &str) -> Self {
        match tag {
            "int" => ColType::Int,
            "float" => ColType::Float,
            "bool" => ColType::Bool,
            "string" => ColType::String,
            "json" => ColType::Json,
            "null" => ColType::Null,
            _ => ColType::String,
        }
    }

    /// The wire tag for this type.
    pub fn as_tag(self) -> &'static str {
        match self {
            ColType::Int => "int",
            ColType::Float => "float",
            ColType::Bool => "bool",
            ColType::String => "string",
            ColType::Json => "json",
            ColType::Null => "null",
        }
    }
}

impl std::fmt::Display for ColType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_tag())
    }
}

/// One schema column declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Column {
    /// Column name; dotted ('meta.owner') when the encoder flattened a
    /// uniform nested object one level.
    pub name: String,
    /// Declared scalar type.
    pub ty: ColType,
    /// True when the column contains nulls or missing keys ('?' on the wire).
    pub nullable: bool,
    /// The name split on '.'; re-nests one level on decode.
    pub path: Vec<String>,
}

impl Column {
    /// Parse a single 'name:type' / 'name:type?' declaration.
    ///
    /// Splits on the FINAL ':' so dotted names survive. A declaration with
    /// no ':' (or an empty name) names a string-typed column; an unknown
    /// type tag degrades to [ColType::String].
    pub fn parse(decl: &str) -> Column {
        let (name, type_tag) = match decl.rfind(':') {
            Some(i) => (&decl[..i], &decl[i + 1..]),
            None => ("", decl),
        };
        let (name, type_tag) = if name.is_empty() {
            (type_tag, "string")
        } else {
            (name, type_tag)
        };
        let nullable = type_tag.ends_with('?');
        let ty = ColType::from_tag(if nullable {
            &type_tag[..type_tag.len() - 1]
        } else {
            type_tag
        });
        let path: Vec<String> = name.split('.').map(str::to_string).collect();
        Column {
            name: name.to_string(),
            ty,
            nullable,
            path,
        }
    }
}

/// Strip a JSON-string wrapper around a densified block.
///
/// Tool results whose content was a JSON array are re-serialized as a JSON
/// string, so the stored content is '"[N]{...}\n..."' rather than the bare
/// block. Both forms decode identically.
fn peel(text: &str) -> Cow<'_, str> {
    if text.starts_with('"') {
        if let Ok(inner) = serde_json::from_str::<String>(text) {
            return Cow::Owned(inner);
        }
    }
    Cow::Borrowed(text)
}

/// True if 'text' looks like a '[N]{schema}' densified block.
///
/// Structural check, not a heuristic: the first non-empty line must open with
/// '[<digits>]{' and the brace section must close on that line. Accepts a
/// JSON-string-wrapped block.
pub fn is_compacted(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    let peeled = peel(text);
    let first = peeled.trim_start().split('\n').next().unwrap_or("");
    if !first.starts_with('[') {
        return false;
    }
    let rbracket = match first.find(']') {
        Some(i) => i,
        None => return false,
    };
    if rbracket < 2 {
        return false;
    }
    let digits = &first[1..rbracket];
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    first[rbracket + 1..].starts_with('{') && first[rbracket..].contains('}')
}

/// True if the densified text removed content (CCR marker or dropped rows).
///
/// Either signal means the text alone cannot reconstruct the original.
pub fn contains_removal_marker(text: &str) -> bool {
    text.contains("<<ccr:") || text.contains("__dropped:")
}

/// Parse the header line '[N]{c:t,c:t?}' with optional ' __dict:K' and
/// ' __dropped:k' suffixes.
///
/// Returns the columns, the declared row count, and the value-factor legend
/// count. None when the line is not a header.
pub fn parse_header(line: &str) -> Option<(Vec<Column>, usize, usize)> {
    if !line.starts_with('[') {
        return None;
    }
    let rb = line.find(']')?;
    let open_brace = line[rb..].find('{')? + rb;
    let close_brace = line[open_brace..].find('}')? + open_brace;
    let declared_rows = line[1..rb].parse::<usize>().ok()?;
    let body = &line[open_brace + 1..close_brace];
    let cols: Vec<Column> = if body.is_empty() {
        Vec::new()
    } else {
        body.split(',').map(Column::parse).collect()
    };
    let mut dict_count = 0usize;
    for token in line[close_brace + 1..].split_whitespace() {
        if let Some(rest) = token.strip_prefix("__dict:") {
            dict_count = rest.parse::<usize>().unwrap_or(0);
        }
    }
    Some((cols, declared_rows, dict_count))
}

/// Split a CSV blob into rows of (was_quoted, value) cells.
///
/// Hand-rolled rather than a CSV crate because the quoted-vs-bare distinction
/// is load-bearing: quoted 'null' is the string, bare 'null' is JSON null.
/// Quotes inside quoted fields are doubled; newlines inside quotes are
/// honored; CRLF row endings are tolerated.
fn iter_rows(blob: &str) -> Vec<Vec<(bool, String)>> {
    let chars: Vec<char> = blob.chars().collect();
    let n = chars.len();
    let mut rows: Vec<Vec<(bool, String)>> = Vec::new();
    let mut row: Vec<(bool, String)> = Vec::new();
    let mut i = 0usize;
    while i < n {
        if chars[i] == '"' {
            // quoted cell
            i += 1;
            let mut buf = String::new();
            while i < n {
                if chars[i] == '"' {
                    if i + 1 < n && chars[i + 1] == '"' {
                        buf.push('"');
                        i += 2;
                        continue;
                    }
                    i += 1;
                    break;
                }
                buf.push(chars[i]);
                i += 1;
            }
            row.push((true, buf));
            if i < n && chars[i] == '\r' {
                i += 1; // tolerate CRLF row endings
            }
            if i < n && chars[i] == ',' {
                i += 1;
            } else if i < n && chars[i] == '\n' {
                rows.push(std::mem::take(&mut row));
                i += 1;
            }
        } else {
            // bare cell up to ',' or newline
            let mut j = i;
            while j < n && chars[j] != ',' && chars[j] != '\n' {
                j += 1;
            }
            let mut cell: String = chars[i..j].iter().collect();
            if cell.ends_with('\r') {
                // a bare cell never legitimately holds '\r' (the encoder
                // quotes such values), so this is a CRLF artifact
                cell.pop();
            }
            row.push((false, cell));
            if j < n && chars[j] == '\n' {
                rows.push(std::mem::take(&mut row));
            }
            i = j + 1;
        }
    }
    if !row.is_empty() {
        rows.push(row);
    }
    rows
}

/// The CSV body's data rows, dropping only a trailing-newline artifact.
///
/// The encoder newline-terminates every row, so a single-column row with a
/// missing key legitimately renders as a bare empty line; only a lone empty
/// cell that EXCEEDS the declared row count is the artifact.
fn split_data_rows(blob: &str, declared: usize) -> Vec<Vec<(bool, String)>> {
    let mut rows: Vec<Vec<(bool, String)>> = iter_rows(blob)
        .into_iter()
        .filter(|row| !row.is_empty())
        .collect();
    if rows.len() == declared.saturating_add(1) {
        let is_artifact = rows
            .last()
            .map(|last| last.len() == 1 && !last[0].0 && last[0].1.is_empty())
            .unwrap_or(false);
        if is_artifact {
            rows.pop();
        }
    }
    rows
}

/// Decode one cell. Returns (present, value); present == false means the key
/// was absent (bare empty cell).
fn decode_cell(was_quoted: bool, value: &str, col: &Column) -> (bool, Value) {
    if was_quoted {
        if col.ty == ColType::Json {
            if let Ok(parsed) = serde_json::from_str::<Value>(value) {
                return (true, parsed);
            }
        }
        // quoted "" -> "", quoted "null" -> "null", etc.
        return (true, Value::String(value.to_string()));
    }
    if value == MISSING_CELL {
        return (false, Value::Null);
    }
    if value == NULL_TOKEN {
        return (true, Value::Null);
    }
    match col.ty {
        ColType::Int => {
            if let Ok(i) = value.parse::<i64>() {
                return (true, Value::from(i));
            }
            if let Ok(u) = value.parse::<u64>() {
                return (true, Value::from(u));
            }
            (true, Value::String(value.to_string()))
        }
        ColType::Float => {
            if let Ok(f) = value.parse::<f64>() {
                if f.is_finite() {
                    return (true, Value::from(f));
                }
            }
            (true, Value::String(value.to_string()))
        }
        ColType::Bool => (true, Value::Bool(value == "true")),
        ColType::Json => {
            if let Ok(parsed) = serde_json::from_str::<Value>(value) {
                return (true, parsed);
            }
            (true, Value::String(value.to_string()))
        }
        _ => (true, Value::String(value.to_string())),
    }
}

/// Reconstruct a cell's exact on-wire text from a parsed (quoted, value).
fn render_raw(was_quoted: bool, value: &str) -> String {
    if was_quoted {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

/// Set 'val' at a possibly dotted 'path', building nested maps on the way.
fn assign(record: &mut Map<String, Value>, path: &[String], val: Value) {
    let (last, prefix) = match path.split_last() {
        Some(split) => split,
        None => return,
    };
    let mut current = record;
    for key in prefix {
        let is_object = current.get(key).map_or(false, |v| v.is_object());
        if !is_object {
            current.insert(key.clone(), Value::Object(Map::new()));
        }
        current = current
            .get_mut(key)
            .and_then(Value::as_object_mut)
            .expect("entry was just inserted as an object");
    }
    current.insert(last.clone(), val);
}

/// Reconstruct the original array of objects from densified 'text'.
///
/// Peels a JSON-string wrapper, parses the header, consumes ' __dict:K'
/// value-factor legend lines, then decodes the CSV rows; dotted column names
/// re-nest one level. Returns None when 'text' is not a parseable densified
/// block or carries an irreversible removal marker (CCR / dropped rows).
pub fn expand_compacted(text: &str) -> Option<Vec<Value>> {
    if !is_compacted(text) || contains_removal_marker(text) {
        return None;
    }
    let peeled = peel(text);
    let (head, mut rest): (&str, &str) = match peeled.split_once('\n') {
        Some((head, rest)) => (head, rest),
        None => (peeled.as_ref(), ""),
    };
    let (cols, declared, dict_count) = parse_header(head.trim())?;

    // Value-factor legends: '__dict:K' declares K '@col=[...]' lines mapping
    // a low-cardinality column's integer indices back to values.
    let mut legends: HashMap<String, Vec<Value>> = HashMap::new();
    for _ in 0..dict_count {
        let (line, tail): (&str, &str) = match rest.split_once('\n') {
            Some((line, tail)) => (line, tail),
            None => (rest, ""),
        };
        rest = tail;
        let (name, payload) = line.split_once('=')?;
        if !name.starts_with('@') {
            return None;
        }
        let values: Vec<Value> = serde_json::from_str(payload).ok()?;
        legends.insert(name[1..].to_string(), values);
    }

    let mut out: Vec<Value> = Vec::new();
    for cells in split_data_rows(rest, declared) {
        if cells.is_empty() {
            continue;
        }
        let mut record = Map::new();
        for (idx, col) in cols.iter().enumerate() {
            if idx >= cells.len() {
                break;
            }
            let (was_quoted, value) = &cells[idx];
            if let Some(legend) = legends.get(&col.name) {
                if !*was_quoted {
                    // dict-encoded cell: empty -> missing, bare null -> null,
                    // else an index into the legend
                    if value.is_empty() {
                        continue;
                    }
                    if value == NULL_TOKEN {
                        assign(&mut record, &col.path, Value::Null);
                        continue;
                    }
                    let index: i64 = value.parse::<i64>().ok()?;
                    let len = legend.len() as i64;
                    let pos = if index >= 0 { index } else { len + index };
                    if pos < 0 || pos >= len {
                        return None;
                    }
                    let decoded = legend[pos as usize].clone();
                    assign(&mut record, &col.path, decoded);
                    continue;
                }
            }
            let (present, decoded) = decode_cell(*was_quoted, value, col);
            if present {
                assign(&mut record, &col.path, decoded);
            }
        }
        out.push(Value::Object(record));
    }
    Some(out)
}

/// Look up a (possibly dotted) column path in a row object.
fn lookup_path<'a>(row: &'a Map<String, Value>, path: &[String]) -> Option<&'a Value> {
    let mut current = row.get(&path[0])?;
    for key in &path[1..] {
        current = current.as_object()?.get(key)?;
    }
    Some(current)
}

/// Scalar classes used for column type inference.
#[derive(Clone, Copy, PartialEq)]
enum ScalarKind {
    Int,
    Float,
    Bool,
    Str,
    Json,
}

/// Classify a present value; nulls return None (they never force the type).
fn scalar_kind(value: &Value) -> Option<ScalarKind> {
    match value {
        Value::Null => None,
        Value::Bool(_) => Some(ScalarKind::Bool),
        Value::Number(n) => Some(if n.is_f64() {
            ScalarKind::Float
        } else {
            ScalarKind::Int
        }),
        Value::String(_) => Some(ScalarKind::Str),
        Value::Object(_) | Value::Array(_) => Some(ScalarKind::Json),
    }
}

/// Build one column from its per-row values (None = key missing in that row).
fn make_column(name: String, vals: &[Option<&Value>]) -> Column {
    let present: Vec<&Value> = vals.iter().copied().flatten().collect();
    let nullable = vals.iter().any(|v| v.is_none()) || present.iter().any(|v| v.is_null());
    let mut kinds: Vec<ScalarKind> = Vec::new();
    for value in &present {
        if let Some(kind) = scalar_kind(value) {
            if !kinds.contains(&kind) {
                kinds.push(kind);
            }
        }
    }
    // Mixed scalar types (int + float, int + string, ...) collapse to json
    // so every value round-trips exactly through its compact JSON form.
    let ty = if kinds.is_empty() {
        ColType::Null
    } else if kinds.len() == 1 {
        match kinds[0] {
            ScalarKind::Int => ColType::Int,
            ScalarKind::Float => ColType::Float,
            ScalarKind::Bool => ColType::Bool,
            ScalarKind::Str => ColType::String,
            ScalarKind::Json => ColType::Json,
        }
    } else {
        ColType::Json
    };
    let path: Vec<String> = name.split('.').map(str::to_string).collect();
    Column {
        name,
        ty,
        nullable,
        path,
    }
}

/// True when a top-level key should be flattened into dotted columns: the
/// key is present in every row, every value is a non-empty object, all rows
/// share the same inner key set, and it has at most 6 inner keys.
fn should_flatten(top: &[Option<&Value>]) -> bool {
    let mut first_keys: Option<Vec<&String>> = None;
    for value in top {
        let obj = match value {
            Some(Value::Object(obj)) if !obj.is_empty() => obj,
            _ => return false,
        };
        match &first_keys {
            None => first_keys = Some(obj.keys().collect()),
            Some(keys) => {
                if obj.len() != keys.len() || !keys.iter().all(|k| obj.contains_key(*k)) {
                    return false;
                }
            }
        }
    }
    matches!(&first_keys, Some(keys) if (1..=MAX_FLATTEN_INNER_KEYS).contains(&keys.len()))
}

/// Infer the column list from the rows: ordered union of keys, with uniform
/// one-level nested objects flattened into dotted columns.
fn infer_columns(rows: &[&Map<String, Value>]) -> Vec<Column> {
    let mut names: Vec<String> = Vec::new();
    for row in rows {
        for key in row.keys() {
            if !names.iter().any(|n| n == key) {
                names.push(key.clone());
            }
        }
    }
    let mut cols: Vec<Column> = Vec::new();
    for name in names {
        let top: Vec<Option<&Value>> = rows.iter().map(|row| row.get(&name)).collect();
        if should_flatten(&top) {
            let first_inner = top[0]
                .unwrap()
                .as_object()
                .expect("checked by should_flatten");
            for inner_key in first_inner.keys() {
                let dotted = format!("{}.{}", name, inner_key);
                let inner: Vec<Option<&Value>> = rows
                    .iter()
                    .map(|row| {
                        row.get(&name)
                            .and_then(Value::as_object)
                            .and_then(|obj| obj.get(inner_key))
                    })
                    .collect();
                cols.push(make_column(dotted, &inner));
            }
        } else {
            cols.push(make_column(name, &top));
        }
    }
    cols
}

/// A string cell must be quoted when it would otherwise read as a missing
/// key, as JSON null, or would break the CSV framing.
fn needs_csv_quote(s: &str) -> bool {
    s.is_empty()
        || s == NULL_TOKEN
        || s.contains(',')
        || s.contains('"')
        || s.contains('\n')
        || s.contains('\r')
}

/// CSV-quote a value, doubling embedded quotes.
fn csv_quote(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

/// Compact JSON for a nested value; keys keep their original order.
fn value_json(value: &Value) -> String {
    serde_json::to_string(value).expect("serde_json values are always serializable")
}

/// Render one cell per the encoding contract.
fn render_cell(value: Option<&Value>, col: &Column) -> String {
    let value = match value {
        Some(v) => v,
        None => return String::new(),
    };
    match value {
        Value::Null => NULL_TOKEN.to_string(),
        Value::Bool(b) => (if *b { "true" } else { "false" }).to_string(),
        // numbers and bools are bare even in json columns: json.loads("5") is 5
        Value::Number(n) => n.to_string(),
        Value::String(s) => {
            if col.ty == ColType::Json {
                csv_quote(&value_json(value))
            } else if needs_csv_quote(s) {
                csv_quote(s)
            } else {
                s.clone()
            }
        }
        Value::Object(_) | Value::Array(_) => csv_quote(&value_json(value)),
    }
}

/// Densify an array of objects into the '[N]{schema}' + CSV wire format.
///
/// Uniform one-level nested objects are flattened into dotted columns (only
/// when every row has the same non-empty inner key set of up to
/// MAX_FLATTEN_INNER_KEYS keys); column types are inferred per column
/// (int / float / bool / string / json / null, nullable when any null or
/// missing; mixed scalar types collapse to json). Returns None when 'items'
/// is not a non-empty array of objects.
pub fn encode_compacted(items: &[Value]) -> Option<String> {
    if items.is_empty() {
        return None;
    }
    let mut rows: Vec<&Map<String, Value>> = Vec::with_capacity(items.len());
    for item in items {
        rows.push(item.as_object()?);
    }
    let cols = infer_columns(&rows);
    let decls: Vec<String> = cols
        .iter()
        .map(|col| {
            if col.nullable {
                format!("{}:{}?", col.name, col.ty.as_tag())
            } else {
                format!("{}:{}", col.name, col.ty.as_tag())
            }
        })
        .collect();
    let mut out = format!("[{}]{{{}}}\n", rows.len(), decls.join(","));
    for row in &rows {
        let mut cells: Vec<String> = Vec::with_capacity(cols.len());
        for col in &cols {
            let value = lookup_path(row, &col.path);
            cells.push(render_cell(value, col));
        }
        out.push_str(&cells.join(","));
        out.push('\n');
    }
    Some(out)
}

/// Dictionary-encode low-cardinality string columns. Lossless and reversible.
///
/// Port of the Python factor_values: a string column whose values repeat is
/// replaced by integer indices into an '@col=[...]' legend, hoisted after the
/// header as ' __dict:K'. A column is factored only when it strictly saves
/// bytes (legend + indices < the repeated cells). Returns 'text' unchanged
/// when nothing benefits, when the input is already factored, or when it is
/// not a clean densified block. A JSON-string wrapper is preserved.
pub fn factor_values(text: &str) -> String {
    let wrapped = text.starts_with('"');
    let peeled = peel(text);
    let inner: &str = peeled.as_ref();
    if !is_compacted(inner) || contains_removal_marker(inner) {
        return text.to_string();
    }
    let (head, rest): (&str, &str) = match inner.split_once('\n') {
        Some((head, rest)) => (head, rest),
        None => (inner, ""),
    };
    let parsed = match parse_header(head.trim()) {
        Some(parsed) => parsed,
        None => return text.to_string(),
    };
    let (cols, declared, existing_dict) = parsed;
    if existing_dict > 0 {
        return text.to_string(); // already value-factored: idempotent
    }
    let rows = split_data_rows(rest, declared);
    if rows.is_empty() {
        return text.to_string();
    }

    // legends: column index -> (distinct values in order, value -> index)
    let mut legends: BTreeMap<usize, (Vec<Value>, HashMap<String, usize>)> = BTreeMap::new();
    for (j, col) in cols.iter().enumerate() {
        if col.ty != ColType::String {
            continue;
        }
        let mut order: Vec<Value> = Vec::new();
        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut raw_chars = 0usize;
        let mut index_chars = 0usize;
        for cells in &rows {
            if j >= cells.len() {
                continue;
            }
            let (was_quoted, value) = &cells[j];
            if !*was_quoted && (value == NULL_TOKEN || value.is_empty()) {
                continue; // null / missing stay inline
            }
            let decoded = value.clone();
            raw_chars += render_raw(*was_quoted, value).chars().count();
            let index = match seen.get(decoded.as_str()) {
                Some(&index) => index,
                None => {
                    seen.insert(decoded.clone(), order.len());
                    order.push(Value::String(decoded));
                    order.len() - 1
                }
            };
            index_chars += index.to_string().chars().count();
        }
        if order.is_empty() {
            continue;
        }
        let legend_json = serde_json::to_string(&Value::Array(order.clone()))
            .expect("string values are serializable");
        // '@' + name + '=' + legend json + newline
        let legend_chars = 1 + col.name.chars().count() + 1 + legend_json.chars().count() + 1;
        if legend_chars + index_chars < raw_chars {
            legends.insert(j, (order, seen));
        }
    }
    if legends.is_empty() {
        return text.to_string();
    }

    // Rebuild: header + __dict:K, then K legend lines, then re-rendered rows.
    let open = head.find('{').unwrap_or(0);
    let close = head[open..].find('}').map(|i| i + open).unwrap_or(0);
    let new_head = format!(
        "{} __dict:{}{}",
        &head[..close + 1],
        legends.len(),
        &head[close + 1..]
    );
    let mut lines: Vec<String> = vec![new_head];
    for (j, (order, _)) in &legends {
        let legend_json = serde_json::to_string(&Value::Array(order.clone()))
            .expect("string values are serializable");
        lines.push(format!("@{}={}", cols[*j].name, legend_json));
    }
    for cells in &rows {
        let mut rendered: Vec<String> = Vec::with_capacity(cols.len());
        for j in 0..cols.len() {
            if j >= cells.len() {
                break;
            }
            let (was_quoted, value) = &cells[j];
            let inline = !*was_quoted && (value == NULL_TOKEN || value.is_empty());
            if let Some((_, index_map)) = legends.get(&j) {
                if !inline {
                    if let Some(index) = index_map.get(value.as_str()) {
                        rendered.push(index.to_string());
                        continue;
                    }
                }
            }
            rendered.push(render_raw(*was_quoted, value));
        }
        lines.push(rendered.join(","));
    }
    let factored = lines.join("\n") + "\n";
    if wrapped {
        serde_json::to_string(&Value::String(factored)).expect("factored block is serializable")
    } else {
        factored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Recursively sort object keys so order-insensitive comparison works.
    fn sorted(v: &Value) -> Value {
        match v {
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                let mut out = Map::new();
                for key in keys {
                    out.insert(key.clone(), sorted(&map[key]));
                }
                Value::Object(out)
            }
            Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
            other => other.clone(),
        }
    }

    /// Canonical form: serde_json with sorted keys.
    fn canonical(v: &Value) -> String {
        serde_json::to_string(&sorted(v)).expect("serializable")
    }

    fn assert_round_trip(items: Value) {
        let array = items.as_array().expect("test input is an array");
        let encoded = encode_compacted(array).expect("encodable");
        assert!(
            is_compacted(&encoded),
            "encoded block must look compacted: {encoded}"
        );
        assert!(
            !contains_removal_marker(&encoded),
            "encode must not emit removal markers"
        );
        let expanded = expand_compacted(&encoded).expect("expandable");
        assert_eq!(
            canonical(&Value::Array(expanded)),
            canonical(&items),
            "round trip failed for block:\n{encoded}"
        );
    }

    #[test]
    fn round_trip_mixed_scalars() {
        assert_round_trip(json!([
            {"id": 1, "name": "alpha", "score": 1.5, "active": true},
            {"id": 2, "name": "beta", "score": 2.0, "active": false},
            {"id": 3, "name": "gamma", "score": 3.25, "active": true},
        ]));
    }

    #[test]
    fn round_trip_nulls_missing_and_special_strings() {
        assert_round_trip(json!([
            {"a": "", "b": "null", "c": 5},
            {"a": "x", "c": 6},
            {"a": null, "b": "", "c": 7, "d": "with,comma"},
            {"b": "quote\"inside", "c": 8, "d": "line\nbreak"},
            {"c": 9, "d": "carriage\rreturn"},
        ]));
    }

    #[test]
    fn round_trip_nested_uniform_flattens_to_dotted() {
        let items = json!([
            {"id": 1, "meta": {"owner": "amy", "repo": "core"}},
            {"id": 2, "meta": {"owner": "bob", "repo": "cli"}},
        ]);
        let encoded = encode_compacted(items.as_array().unwrap()).unwrap();
        assert!(encoded.starts_with("[2]{id:int,meta.owner:string,meta.repo:string}\n"));
        assert!(encoded.contains("\n1,amy,core\n2,bob,cli\n"));
        assert_round_trip(items);
    }

    #[test]
    fn round_trip_nested_value_inside_flattened_column() {
        assert_round_trip(json!([
            {"meta": {"owner": {"login": "amy"}, "stars": 3}},
            {"meta": {"owner": {"login": "bob"}, "stars": 7}},
        ]));
    }

    #[test]
    fn nested_nonuniform_or_missing_stays_json() {
        let items = json!([
            {"id": 1, "meta": {"a": 1}},
            {"id": 2, "meta": {"b": 2}},
        ]);
        let encoded = encode_compacted(items.as_array().unwrap()).unwrap();
        assert!(encoded.contains("meta:json"));
        assert_round_trip(items);
        // key missing in one row: not uniform, stays json, decodes to missing
        assert_round_trip(json!([
            {"id": 1, "meta": {"a": 1}},
            {"id": 2},
        ]));
        // empty nested objects are never flattened (that would drop the key)
        assert_round_trip(json!([{"m": {}}, {"m": {}}]));
    }

    #[test]
    fn nested_with_too_many_inner_keys_stays_json() {
        let items = json!([
            {"m": {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}},
            {"m": {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}},
        ]);
        let encoded = encode_compacted(items.as_array().unwrap()).unwrap();
        assert!(encoded.contains("m:json"));
        assert_round_trip(items);
        // exactly six inner keys still flattens
        let six = json!([
            {"m": {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}},
            {"m": {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}},
        ]);
        let encoded = encode_compacted(six.as_array().unwrap()).unwrap();
        assert!(encoded.contains("m.1:int"));
        assert_round_trip(six);
    }

    #[test]
    fn mixed_scalar_columns_collapse_to_json() {
        assert_round_trip(json!([{"v": 1}, {"v": "x"}]));
        // int + float are distinct scalar types: json keeps both exact
        assert_round_trip(json!([{"v": 1}, {"v": 2.5}]));
        assert_round_trip(json!([{"v": true}, {"v": "x"}]));
        assert_round_trip(json!([{"v": [1, 2]}, {"v": 3}]));
    }

    #[test]
    fn json_column_strings_and_scalars_round_trip() {
        // strings that would misread as numbers/nulls/bools stay quoted JSON
        assert_round_trip(json!([
            {"v": "5"}, {"v": 5}, {"v": "null"}, {"v": "true"}, {"v": true},
        ]));
    }

    #[test]
    fn round_trip_all_null_column() {
        let items = json!([{"n": null, "x": 1}, {"n": null, "x": 2}]);
        let encoded = encode_compacted(items.as_array().unwrap()).unwrap();
        assert!(encoded.contains("n:null?"));
        assert_round_trip(items);
    }

    #[test]
    fn round_trip_large_uint() {
        assert_round_trip(json!([{"big": 18446744073709551615u64}, {"big": 1}]));
    }

    #[test]
    fn encode_skips_non_object_arrays() {
        assert!(encode_compacted(&[]).is_none());
        assert!(encode_compacted(&[json!(1), json!(2)]).is_none());
        assert!(encode_compacted(&[json!({"a": 1}), json!("str")]).is_none());
        assert!(encode_compacted(&[json!([1, 2])]).is_none());
    }

    #[test]
    fn is_compacted_structural_cases() {
        assert!(is_compacted("[3]{a:int}\n1\n2\n3\n"));
        assert!(is_compacted("  [3]{a:int}\n1\n")); // leading whitespace
        assert!(is_compacted("[12]{a:int,b:string?}\n"));
        assert!(is_compacted("[3]{a:int} __dropped:2\n"));
        assert!(!is_compacted(""));
        assert!(!is_compacted("just plain text"));
        assert!(!is_compacted("[]{a:int}\n")); // no row-count digits
        assert!(!is_compacted("[x]{a:int}\n")); // non-digit count
        assert!(!is_compacted("[3]a:int\n")); // no brace after ']'
        assert!(!is_compacted("[3]{a:int\n")); // brace never closes on the line
    }

    #[test]
    fn is_compacted_accepts_json_string_wrapper() {
        let wrapped =
            serde_json::to_string(&"[3]{a:int}\n1\n2\n3\n".to_string()).expect("serializable");
        assert!(is_compacted(&wrapped));
        assert_eq!(
            expand_compacted(&wrapped).unwrap(),
            vec![json!({"a": 1}), json!({"a": 2}), json!({"a": 3})]
        );
    }

    #[test]
    fn contains_removal_marker_cases() {
        assert!(contains_removal_marker(
            "[2]{a:int}\n1,<<ccr:abc,base64,1KB>>\n"
        ));
        assert!(contains_removal_marker("[2]{a:int} __dropped:5\n"));
        assert!(!contains_removal_marker("[2]{a:int}\n1,2\n"));
        // removal markers make expansion refuse
        assert!(
            expand_compacted("[1]{a:string}\n<<ccr:abc123def456 5_rows_offloaded>>\n").is_none()
        );
        assert!(expand_compacted("[1]{a:int} __dropped:2\n7\n").is_none());
    }

    #[test]
    fn parse_header_cases() {
        let (cols, declared, dict_count) = parse_header("[3]{a:int,b:string?}").unwrap();
        assert_eq!(declared, 3);
        assert_eq!(dict_count, 0);
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].name, "a");
        assert_eq!(cols[0].ty, ColType::Int);
        assert!(!cols[0].nullable);
        assert_eq!(cols[0].path, vec!["a".to_string()]);
        assert_eq!(cols[1].name, "b");
        assert_eq!(cols[1].ty, ColType::String);
        assert!(cols[1].nullable);

        // dotted name splits on the final ':' only
        let (cols, _, _) = parse_header("[2]{meta.owner:string}").unwrap();
        assert_eq!(cols[0].name, "meta.owner");
        assert_eq!(cols[0].path, vec!["meta".to_string(), "owner".to_string()]);

        // __dict count is picked out of the trailing tokens
        let (_, declared, dict_count) = parse_header("[4]{a:string} __dict:2 __dropped:1").unwrap();
        assert_eq!(declared, 4);
        assert_eq!(dict_count, 2);

        // unknown type tag degrades to string; bare name is string-typed
        let (cols, _, _) = parse_header("[1]{a:weird,b}").unwrap();
        assert_eq!(cols[0].ty, ColType::String);
        assert_eq!(cols[1].name, "b");
        assert_eq!(cols[1].ty, ColType::String);

        // empty schema decodes to zero columns
        let (cols, declared, _) = parse_header("[0]{}").unwrap();
        assert!(cols.is_empty());
        assert_eq!(declared, 0);

        // malformed headers
        assert!(parse_header("no bracket").is_none());
        assert!(parse_header("[x]{a:int}").is_none());
        assert!(parse_header("[3]a:int").is_none());
        assert!(parse_header("[3]{a:int").is_none());
    }

    #[test]
    fn expand_rejects_unparseable_blocks() {
        assert!(expand_compacted("plain text").is_none());
        assert!(expand_compacted("").is_none());
        // brace section not closed on the first line
        assert!(expand_compacted("[2]{a:string\nrow\n").is_none());
        // a malformed legend line
        assert!(expand_compacted("[1]{a:string} __dict:1\nnot-a-legend\nx\n").is_none());
        // fewer rows than declared simply decode fewer records
        assert_eq!(
            expand_compacted("[2]{a:string}\nrow-one\n").unwrap(),
            vec![json!({"a": "row-one"})]
        );
    }

    #[test]
    fn expand_decodes_dict_encoded_legends() {
        let text = "[2]{kind:string} __dict:1\n@kind=[\"car\",\"bike\"]\n0\n1\n";
        assert_eq!(
            expand_compacted(text).unwrap(),
            vec![json!({"kind": "car"}), json!({"kind": "bike"})]
        );
        // bare null -> null, empty -> missing in a legend column;
        // string-typed columns keep bare cells as strings
        let text = "[3]{k:string?,v:string} __dict:1\n@k=[\"a\",\"b\"]\n0,x\nnull,1\n,2\n";
        assert_eq!(
            expand_compacted(text).unwrap(),
            vec![
                json!({"k": "a", "v": "x"}),
                json!({"k": null, "v": "1"}),
                json!({"v": "2"}),
            ]
        );
        // quoted cells bypass the legend and stay strings
        let text = "[2]{k:string,v:string} __dict:1\n@k=[\"a\"]\n\"a\",x\n\"null\",y\n";
        assert_eq!(
            expand_compacted(text).unwrap(),
            vec![json!({"k": "a", "v": "x"}), json!({"k": "null", "v": "y"})]
        );
        // an out-of-range legend index is unparseable
        assert!(expand_compacted("[1]{k:string} __dict:1\n@k=[\"a\"]\n5\n").is_none());
    }

    #[test]
    fn csv_splitter_preserves_quoted_vs_bare() {
        let rows = iter_rows("\"a\"\"b\",plain\n\"with,comma\",\"line\nbreak\"\n");
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0],
            vec![(true, "a\"b".to_string()), (false, "plain".to_string())]
        );
        assert_eq!(
            rows[1],
            vec![
                (true, "with,comma".to_string()),
                (true, "line\nbreak".to_string())
            ]
        );
        // CRLF row endings are tolerated
        let rows = iter_rows("a,b\r\nc,d\r\n");
        assert_eq!(
            rows,
            vec![
                vec![(false, "a".to_string()), (false, "b".to_string())],
                vec![(false, "c".to_string()), (false, "d".to_string())],
            ]
        );
    }

    #[test]
    fn split_data_rows_drops_only_the_trailing_artifact() {
        // a single-column missing-key row is a legitimate empty line, kept
        let rows = split_data_rows("x\n\n", 2);
        assert_eq!(rows.len(), 2);
        // a lone empty cell beyond the declared count is the artifact
        let rows = split_data_rows("x\nx\n\n", 2);
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn factor_values_round_trips_through_expand() {
        let items = json!([
            {"file": "src/very/long/path/to/module_a.rs", "line": 1},
            {"file": "src/very/long/path/to/module_a.rs", "line": 2},
            {"file": "src/very/long/path/to/module_a.rs", "line": 3},
            {"file": "src/very/long/path/to/module_a.rs", "line": 4},
        ]);
        let encoded = encode_compacted(items.as_array().unwrap()).unwrap();
        let factored = factor_values(&encoded);
        assert_ne!(factored, encoded);
        assert!(factored.contains(" __dict:1"));
        assert!(factored.contains("@file="));
        assert!(is_compacted(&factored));
        assert_eq!(
            canonical(&Value::Array(expand_compacted(&factored).unwrap())),
            canonical(&items)
        );
    }

    #[test]
    fn factor_values_is_idempotent_and_skips_unprofitable() {
        // already factored: unchanged
        let factored = "[1]{k:string} __dict:1\n@k=[\"a\"]\n0\n";
        assert_eq!(factor_values(factored), factored);
        // unique values in every row: no byte savings -> unchanged
        let encoded =
            encode_compacted(json!([{"k": "a"}, {"k": "b"}]).as_array().unwrap()).unwrap();
        assert_eq!(factor_values(&encoded), encoded);
        // not compacted at all: unchanged
        assert_eq!(factor_values("plain"), "plain");
        // removal markers block factoring
        let marked = "[1]{k:string} __dropped:1\n<<ccr:abc,base64,1KB>>\n";
        assert_eq!(factor_values(marked), marked);
    }

    #[test]
    fn factor_values_preserves_json_wrapper() {
        let encoded = encode_compacted(
            json!([
                {"file": "src/very/long/path/to/module_a.rs", "line": 1},
                {"file": "src/very/long/path/to/module_a.rs", "line": 2},
            ])
            .as_array()
            .unwrap(),
        )
        .unwrap();
        let wrapped = serde_json::to_string(&encoded).expect("serializable");
        let out = factor_values(&wrapped);
        assert!(out.starts_with('"'));
        let inner: String = serde_json::from_str(&out).unwrap();
        assert!(inner.contains(" __dict:1"));
        assert!(is_compacted(&inner));
    }
}
