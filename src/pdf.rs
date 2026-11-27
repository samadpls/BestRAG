//! PDF text in reading order, including right-to-left scripts like Arabic.
//!
//! pdf-extract reports every glyph with its position. Its own text assembly
//! writes glyphs in file order and guesses spaces from glyph widths. For
//! Arabic that gives reversed text with words cut into pieces: many PDFs store
//! right-to-left text in visual order, and report zero width for joined
//! letters. So BestRAG groups glyphs into lines by position and orders each
//! line itself, one glyph at a time. A lam-alef ligature is one glyph with two
//! letters, so it stays in the right order.

use anyhow::{Context, Result};
use pdf_extract::{Document, MediaBox, OutputDev, OutputError, Transform};
use std::path::Path;
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::is_combining_mark;

/// The text of each page, in page order.
pub fn pages(path: &Path) -> Result<Vec<String>> {
    let mut doc = Document::load(path)
        .with_context(|| format!("could not open {} as a PDF", path.display()))?;
    if doc.is_encrypted() {
        doc.decrypt("")
            .context("the PDF is encrypted with a password")?;
    }
    let mut lines = Lines::default();
    for page in 1..=doc.get_pages().len() as u32 {
        // A page pdf-extract cannot read stays empty, so later page numbers
        // still match the PDF.
        if pdf_extract::output_doc_page(&doc, &mut lines, page).is_err() {
            lines.glyphs.clear();
            lines.marks.clear();
            lines.lines.clear();
        }
        lines.finish_page();
    }
    Ok(lines.pages)
}

#[derive(Debug, Clone, PartialEq)]
struct Glyph {
    x: f64,
    /// Where the glyph ends on the line, when the PDF gives its width.
    end: Option<f64>,
    text: String,
}

#[derive(Default)]
struct Lines {
    pages: Vec<String>,
    /// The current page's lines, each with its font size.
    lines: Vec<(Vec<Glyph>, f64)>,
    glyphs: Vec<Glyph>,
    /// Combining marks of the current line, with their x position.
    marks: Vec<(f64, String)>,
    line_y: f64,
    font_size: f64,
}

impl Lines {
    fn finish_line(&mut self) {
        // A mark like a damma is drawn on its letter, so it shares the
        // letter's position. Files put it before or after the letter
        // depending on the text direction, so position is the reliable guide.
        for (x, mark) in std::mem::take(&mut self.marks) {
            let base = self
                .glyphs
                .iter_mut()
                .filter(|g| g.x <= x + 0.5)
                .max_by(|a, b| a.x.total_cmp(&b.x));
            match base {
                Some(glyph) => glyph.text.push_str(&mark),
                None => {
                    if let Some(first) = self.glyphs.iter_mut().min_by(|a, b| a.x.total_cmp(&b.x)) {
                        first.text.push_str(&mark);
                    }
                }
            }
        }
        if !self.glyphs.is_empty() {
            self.lines
                .push((std::mem::take(&mut self.glyphs), self.font_size));
        }
    }

    fn finish_page(&mut self) {
        self.finish_line();
        // The page's main direction decides how a mixed line is read, so
        // "نستخدم Qdrant" on an Arabic page and "the word الإجازة" on an
        // English page both come out right.
        let (rtl, ltr) = letter_counts(
            self.lines
                .iter()
                .flat_map(|(glyphs, _)| glyphs.iter().map(|g| g.text.as_str())),
        );
        let mut page = String::new();
        for (glyphs, font_size) in std::mem::take(&mut self.lines) {
            page.push_str(line_text(&glyphs, font_size, rtl > ltr).trim());
            page.push('\n');
        }
        // NFKC turns presentation forms and ligature code points (ﻻ, ﬁ) into
        // the plain letters people type.
        self.pages.push(page.nfkc().collect());
    }
}

impl OutputDev for Lines {
    fn begin_page(
        &mut self,
        _: u32,
        _: &MediaBox,
        _: Option<(f64, f64, f64, f64)>,
    ) -> Result<(), OutputError> {
        Ok(())
    }

    fn end_page(&mut self) -> Result<(), OutputError> {
        self.finish_line();
        Ok(())
    }

    fn output_character(
        &mut self,
        trm: &Transform,
        width: f64,
        _spacing: f64,
        font_size: f64,
        text: &str,
    ) -> Result<(), OutputError> {
        if text.is_empty() {
            return Ok(());
        }
        let (sx, sy) = (
            font_size * (trm.m11 + trm.m21),
            font_size * (trm.m12 + trm.m22),
        );
        let size = (sx * sy).abs().sqrt();
        let (x, y) = (trm.m31, trm.m32);
        if text.chars().all(is_combining_mark) {
            self.marks.push((x, text.to_owned()));
            return Ok(());
        }
        if !self.glyphs.is_empty() && (y - self.line_y).abs() > size * 0.5 {
            self.finish_line();
        }
        self.line_y = y;
        self.font_size = size;
        self.glyphs.push(Glyph {
            x,
            end: (width > 0.0).then_some(x + width * size),
            text: text.to_owned(),
        });
        Ok(())
    }

    fn begin_word(&mut self) -> Result<(), OutputError> {
        Ok(())
    }

    fn end_word(&mut self) -> Result<(), OutputError> {
        Ok(())
    }

    fn end_line(&mut self) -> Result<(), OutputError> {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Direction {
    Right,
    Left,
    Neutral,
}

fn is_rtl(c: char) -> bool {
    matches!(c,
        '\u{0590}'..='\u{05FF}' | '\u{0600}'..='\u{065F}' | '\u{066A}'..='\u{06EF}'
        | '\u{06FA}'..='\u{08FF}' | '\u{FB1D}'..='\u{FDFF}' | '\u{FE70}'..='\u{FEFF}')
}

/// Whether the space between two glyphs, left to right, is wide enough to be
/// a space. Without a known width, a step of more than one and a half letters
/// is a gap.
fn is_gap(prev: &Glyph, next: &Glyph, font_size: f64) -> bool {
    match prev.end {
        Some(end) => next.x - end > font_size * 0.15,
        None => next.x - prev.x > font_size * 1.5,
    }
}

/// Letters written right to left, and letters or digits written left to right.
fn letter_counts<'a>(texts: impl Iterator<Item = &'a str>) -> (usize, usize) {
    texts.flat_map(str::chars).fold((0, 0), |(rtl, ltr), c| {
        if is_rtl(c) {
            (rtl + 1, ltr)
        } else if c.is_alphanumeric() {
            (rtl, ltr + 1)
        } else {
            (rtl, ltr)
        }
    })
}

fn direction(text: &str) -> Direction {
    for c in text.chars() {
        if is_rtl(c) {
            return Direction::Right;
        }
        if c.is_alphanumeric() {
            return Direction::Left;
        }
    }
    Direction::Neutral
}

/// A line's glyphs from left to right on the page. The file's order is kept
/// inside each run it wrote in one go, because pdf-extract's positions come
/// from font widths that are often defaults, so a kerned letter can sit a
/// fraction of a point left of the space before it. Runs are split where the
/// file jumps backwards, then ordered by position. This covers files that
/// write right-to-left text from either end, and list markers written out of
/// order.
fn left_to_right(glyphs: &[Glyph], font_size: f64) -> Vec<Vec<&Glyph>> {
    let mut runs: Vec<Vec<&Glyph>> = Vec::new();
    for glyph in glyphs {
        match runs.last_mut() {
            // Kerning moves a letter back a fraction of a point; a file that
            // writes right to left steps back a whole letter width.
            Some(run) if glyph.x >= run[run.len() - 1].x - font_size * 0.2 => run.push(glyph),
            _ => runs.push(vec![glyph]),
        }
    }
    runs.sort_by(|a, b| a[0].x.total_cmp(&b[0].x));
    runs
}

/// One line's glyphs as text in reading order. A line with a clear majority
/// of letters in one direction reads that way; a close call, like "نستخدم
/// Qdrant", follows the page's direction (`page_rtl`).
fn line_text(glyphs: &[Glyph], font_size: f64, page_rtl: bool) -> String {
    let runs = left_to_right(glyphs, font_size);
    // Inside a run, trust the PDF's own space glyphs when it has them.
    // Between runs, and in PDFs without space glyphs, a wide gap is a space.
    let has_spaces = glyphs.iter().any(|g| g.text.trim().is_empty());
    let mut items: Vec<&str> = Vec::with_capacity(glyphs.len() * 2);
    let mut prev: Option<&Glyph> = None;
    for run in &runs {
        for (i, &glyph) in run.iter().enumerate() {
            let starts_run = i == 0;
            if let Some(prev) = prev
                && (starts_run || !has_spaces)
                && is_gap(prev, glyph, font_size)
                && items.last().is_some_and(|last| !last.trim().is_empty())
            {
                items.push(" ");
            }
            items.push(&glyph.text);
            prev = Some(glyph);
        }
    }
    let directions: Vec<Direction> = items.iter().map(|t| direction(t)).collect();
    if !directions.contains(&Direction::Right) {
        return items.concat();
    }
    let (rtl_letters, ltr_letters) = letter_counts(items.iter().copied());
    let rtl = if rtl_letters >= 2 * ltr_letters {
        true
    } else if ltr_letters >= 2 * rtl_letters {
        false
    } else {
        page_rtl
    };

    // Items run left to right on the page. Text against the page's direction
    // (English inside Arabic, or Arabic inside English) forms runs; a space
    // or mark between two such items belongs to the run.
    let against = if rtl {
        Direction::Left
    } else {
        Direction::Right
    };
    let strong = |mut range: Box<dyn Iterator<Item = usize> + '_>| {
        range.find_map(|j| (directions[j] != Direction::Neutral).then_some(directions[j]))
    };
    let flipped: Vec<bool> = (0..items.len())
        .map(|i| match directions[i] {
            Direction::Neutral => {
                strong(Box::new((0..i).rev())) == Some(against)
                    && strong(Box::new(i + 1..items.len())) == Some(against)
            }
            d => d == against,
        })
        .collect();
    let mut runs: Vec<(bool, &[&str])> = Vec::new();
    let mut start = 0;
    for i in 1..=items.len() {
        if i == items.len() || flipped[i] != flipped[start] {
            runs.push((flipped[start], &items[start..i]));
            start = i;
        }
    }
    // A right-to-left page is read from its right end, and every
    // right-to-left run is read from its own right end.
    if rtl {
        runs.reverse();
    }
    let mut out = String::new();
    for (flip, run) in runs {
        if flip != rtl {
            out.extend(run.iter().rev().copied());
        } else {
            out.extend(run.iter().copied());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Glyphs laid out left to right at the given positions, with known widths.
    fn glyphs(texts: &[&str]) -> Vec<Glyph> {
        texts
            .iter()
            .enumerate()
            .map(|(i, t)| Glyph {
                x: i as f64 * 10.0,
                end: Some(i as f64 * 10.0 + 9.0),
                text: (*t).into(),
            })
            .collect()
    }

    #[test]
    fn left_to_right_lines_keep_their_order() {
        let line = glyphs(&["H", "i", " ", "y", "o", "u"]);
        assert_eq!(line_text(&line, 10.0, false), "Hi you");
    }

    #[test]
    fn arabic_in_visual_order_is_read_right_to_left_with_the_ligature_intact() {
        // "الإجازة" as a PDF stores it in visual order: ة ز ا ج لإ ا
        let line = glyphs(&["ة", "ز", "ا", "ج", "لإ", "ا"]);
        assert_eq!(line_text(&line, 10.0, true), "الإجازة");
    }

    #[test]
    fn numbers_and_full_stops_land_where_a_reader_expects() {
        // Visual order of "مدة 14 يوماً." : the full stop is at the far left.
        let line = glyphs(&[
            ".", "ا", "ً", "م", "و", "ي", " ", "1", "4", " ", "ة", "د", "م",
        ]);
        assert_eq!(line_text(&line, 10.0, true), "مدة 14 يومًا.");
    }

    #[test]
    fn english_words_inside_arabic_keep_their_order_and_spaces() {
        let line = glyphs(&[
            "Q", "d", "r", "a", "n", "t", " ", "D", "B", " ", "م", "د", "خ", "ت", "س", "ن",
        ]);
        assert_eq!(line_text(&line, 10.0, true), "نستخدم Qdrant DB");
    }

    #[test]
    fn arabic_words_inside_an_english_page_are_turned_around() {
        // "the word الإجازة" in visual order.
        let line = glyphs(&[
            "t", "h", "e", " ", "w", "o", "r", "d", " ", "ة", "ز", "ا", "ج", "لإ", "ا",
        ]);
        assert_eq!(line_text(&line, 10.0, false), "the word الإجازة");
    }

    #[test]
    fn an_english_line_on_an_arabic_page_stays_english() {
        // "The word الإجازة means leave" in visual order, on a right-to-left page.
        let line = glyphs(&[
            "T", "h", "e", " ", "w", "o", "r", "d", " ", "ة", "ز", "ا", "ج", "لإ", "ا", " ", "m",
            "e", "a", "n", "s", " ", "l", "e", "a", "v", "e",
        ]);
        assert_eq!(line_text(&line, 10.0, true), "The word الإجازة means leave");
    }

    #[test]
    fn a_kerned_letter_does_not_jump_over_a_space() {
        let line = vec![
            Glyph {
                x: 0.0,
                end: Some(6.0),
                text: ".".into(),
            },
            Glyph {
                x: 6.0,
                end: Some(12.0),
                text: " ".into(),
            },
            Glyph {
                x: 5.9,
                end: Some(13.0),
                text: "T".into(),
            },
        ];
        assert_eq!(line_text(&line, 12.0, false), ". T");
    }

    #[test]
    fn arabic_the_file_writes_right_to_left_is_kept_as_it_is() {
        // Logical order with falling x positions: "لا يجوز".
        let texts = ["لا", " ", "ي", "ج", "و", "ز"];
        let line: Vec<Glyph> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| Glyph {
                x: 100.0 - i as f64 * 10.0,
                end: None,
                text: (*t).into(),
            })
            .collect();
        assert_eq!(line_text(&line, 10.0, true), "لا يجوز");
    }

    #[test]
    fn a_list_marker_written_first_still_starts_the_sentence() {
        // "أ. يسمى" laid out left to right, but the file writes the marker
        // (far right) before the words.
        let at = |x: f64, t: &str| Glyph {
            x,
            end: None,
            text: t.into(),
        };
        let line = vec![
            at(50.0, "أ"),
            at(46.0, "."),
            at(0.0, "ى"),
            at(4.0, "م"),
            at(8.0, "س"),
            at(12.0, "ي"),
            at(30.0, " "),
        ];
        assert_eq!(line_text(&line, 10.0, true), "أ. يسمى");
    }

    #[test]
    fn the_right_hand_part_of_an_arabic_header_comes_first() {
        let at = |x: f64, t: &str| Glyph {
            x,
            end: None,
            text: t.into(),
        };
        // Left part "دليل" at x 0..12, right part "شركة" at x 100..112.
        let line = vec![
            at(0.0, "ل"),
            at(4.0, "ي"),
            at(8.0, "ل"),
            at(12.0, "د"),
            at(100.0, "ة"),
            at(104.0, "ك"),
            at(108.0, "ر"),
            at(112.0, "ش"),
        ];
        assert_eq!(line_text(&line, 10.0, true), "شركة دليل");
    }

    #[test]
    fn the_arabic_fixture_reads_back_exactly() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/arabic.pdf");
        let page = &pages(&path).unwrap()[0];
        let lines: Vec<&str> = page.lines().collect();
        assert_eq!(
            lines,
            [
                "يحق للموظف الحصول على الإجازة السنوية بعد إكمال سنة كاملة من الخدمة.",
                "لا يجوز للموظف العمل لدى جهة أخرى أثناء الإجازة.",
                "تُصرف مكافأة نهاية الخدمة خلال 14 يوماً من تاريخ انتهاء العقد.",
                "نستخدم Qdrant للبحث في الوثائق.",
                "The word الإجازة means annual leave.",
            ]
        );
    }

    #[test]
    fn missing_spaces_are_inferred_from_wide_gaps() {
        let line = vec![
            Glyph {
                x: 0.0,
                end: Some(5.0),
                text: "a".into(),
            },
            // 15 after a's known end: a gap.
            Glyph {
                x: 20.0,
                end: None,
                text: "b".into(),
            },
            // 10 after b's start, with b's width unknown: not a gap.
            Glyph {
                x: 30.0,
                end: None,
                text: "c".into(),
            },
        ];
        assert_eq!(line_text(&line, 10.0, false), "a bc");
    }
}
