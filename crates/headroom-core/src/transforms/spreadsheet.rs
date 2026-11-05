//! XLSX workbook ingestion: each sheet rendered to CSV text.
//!
//! Mirrors the Python spreadsheet_ingest helper: every sheet becomes one
//! (sheet name, csv text) pair, with cells quoted per the compaction cell
//! rules (a cell containing a comma, quote or newline is wrapped in quotes
//! with inner quotes doubled) and empty trailing cells trimmed per row.
//!
//! Parsing uses calamine and is strictly None-on-error: a workbook that
//! fails to open, or one without sheets, yields `None` instead of a panic
//! (fail-open). The xlsx fixture format is deliberately not synthesized in
//! tests — calamine is read-only and hand-rolling zip+xml fixtures is not
//! worth it — so the CSV rendering is tested directly and the calamine
//! path is exercised only through its None-on-error behavior.

use std::io::Cursor;

use calamine::{Data, Reader, Xlsx};

/// Renders sheet rows to CSV text.
///
/// Cells are quoted per the compaction cell rules (a cell containing a
/// comma, quote or newline is wrapped in quotes with inner quotes doubled)
/// and empty trailing cells are trimmed per row. Rows join with a newline;
/// there is no trailing newline.
pub fn render_sheet(rows: &[Vec<String>]) -> String {
    rows.iter()
        .map(|row| render_row(row))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Renders one row: trailing empty cells trimmed, cells quoted.
fn render_row(row: &[String]) -> String {
    let mut end = row.len();
    while end > 0 && row[end - 1].is_empty() {
        end -= 1;
    }
    (0..end)
        .map(|index| quote_cell(&row[index]))
        .collect::<Vec<_>>()
        .join(",")
}

/// Quotes one CSV cell per the compaction cell rules.
fn quote_cell(cell: &str) -> String {
    if cell.contains(',') || cell.contains('"') || cell.contains('\n') || cell.contains('\r') {
        format!("\"{}\"", cell.replace('"', "\"\""))
    } else {
        cell.to_string()
    }
}

/// Converts a calamine cell to its text form; empty cells become "".
fn cell_to_string(cell: &Data) -> String {
    match cell {
        Data::Empty => String::new(),
        other => other.to_string(),
    }
}

/// Converts every sheet of an in-memory `.xlsx` workbook to CSV text.
///
/// Returns one (sheet name, csv) pair per sheet in workbook order, or
/// `None` when the bytes are not a readable workbook or contain no
/// sheets. Any hard parse failure yields `None` (fail-open).
pub fn xlsx_to_csv_messages(bytes: &[u8]) -> Option<Vec<(String, String)>> {
    let mut workbook: Xlsx<Cursor<&[u8]>> =
        calamine::open_workbook_from_rs(Cursor::new(bytes)).ok()?;
    let names = workbook.sheet_names().to_vec();
    let mut out: Vec<(String, String)> = Vec::with_capacity(names.len());
    for name in names {
        let range = workbook.worksheet_range(&name).ok()?;
        let rows: Vec<Vec<String>> = range
            .rows()
            .map(|row| row.iter().map(cell_to_string).collect())
            .collect();
        out.push((name, render_sheet(&rows)));
    }
    (!out.is_empty()).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_sheet_quotes_special_cells() {
        let rows = vec![
            vec!["plain".to_string(), "has,comma".to_string()],
            vec!["has\"quote".to_string(), "line\nbreak".to_string()],
            vec!["cr\rcell".to_string()],
        ];
        assert_eq!(
            render_sheet(&rows),
            "plain,\"has,comma\"\n\"has\"\"quote\",\"line\nbreak\"\n\"cr\rcell\""
        );
    }

    #[test]
    fn render_sheet_trims_trailing_empty_cells() {
        let rows = vec![
            vec!["a".to_string(), String::new(), String::new()],
            vec!["b".to_string(), "c".to_string(), String::new()],
        ];
        assert_eq!(render_sheet(&rows), "a\nb,c");
    }

    #[test]
    fn render_sheet_keeps_empty_middle_cells() {
        let rows = vec![vec![
            "a".to_string(),
            String::new(),
            "c".to_string(),
        ]];
        assert_eq!(render_sheet(&rows), "a,,c");
    }

    #[test]
    fn render_sheet_handles_empty_rows() {
        assert_eq!(render_sheet(&[]), "");
        assert_eq!(render_sheet(&[vec![], vec![String::new()]]), "\n");
    }

    #[test]
    fn xlsx_parsing_fails_open() {
        assert!(xlsx_to_csv_messages(b"not an xlsx").is_none());
        assert!(xlsx_to_csv_messages(b"").is_none());
    }
}
