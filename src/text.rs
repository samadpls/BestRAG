//! Turning documents into chunks small enough for the embedding model.

/// One piece of a document, small enough to embed whole.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub text: String,
    /// 1-based PDF page. `None` for plain text.
    pub page: Option<u32>,
}

// bge-small reads at most 512 tokens. A chunk has 200 words, or up to 240 when
// it is the last one; 240 English words are about 320 to 360 tokens, which
// leaves room for long words and other languages.
pub const CHUNK_WORDS: usize = 200;
pub const OVERLAP_WORDS: usize = 40;

/// Overlapping windows of `CHUNK_WORDS` words. Whitespace is normalised.
// shortcut: windows are counted in whitespace-separated words, so scripts
// written without spaces (Chinese, Japanese, Thai) become one long "word" and
// get cut at the model's token limit; split by tokens if those matter.
pub fn chunk_words(text: &str, page: Option<u32>) -> Vec<Chunk> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < words.len() {
        // A tail of a few new words would cost a whole extra embedding, so
        // the last window takes up to OVERLAP_WORDS extra words instead.
        let end = if words.len() <= start + CHUNK_WORDS + OVERLAP_WORDS {
            words.len()
        } else {
            start + CHUNK_WORDS
        };
        chunks.push(Chunk {
            text: words[start..end].join(" "),
            page,
        });
        if end == words.len() {
            break;
        }
        start += CHUNK_WORDS - OVERLAP_WORDS;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(n: usize) -> String {
        (0..n)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn empty_text_has_no_chunks() {
        assert!(chunk_words("  \n\t ", None).is_empty());
    }

    #[test]
    fn short_text_is_one_chunk_with_clean_spaces() {
        let chunks = chunk_words("a  b\n\nc", Some(3));
        assert_eq!(
            chunks,
            vec![Chunk {
                text: "a b c".into(),
                page: Some(3)
            }]
        );
    }

    #[test]
    fn long_text_overlaps_and_covers_every_word() {
        let chunks = chunk_words(&words(450), None);
        assert_eq!(chunks.len(), 3);
        assert!(
            chunks
                .iter()
                .all(|c| c.text.split(' ').count() <= CHUNK_WORDS + OVERLAP_WORDS)
        );
        assert!(chunks[0].text.ends_with("w199"));
        assert!(chunks[1].text.starts_with("w160"));
        assert!(chunks[2].text.starts_with("w320"));
        assert!(chunks[2].text.ends_with("w449"));
    }

    #[test]
    fn a_short_tail_joins_the_last_chunk() {
        assert_eq!(chunk_words(&words(CHUNK_WORDS), None).len(), 1);
        assert_eq!(
            chunk_words(&words(CHUNK_WORDS + OVERLAP_WORDS), None).len(),
            1
        );
        assert_eq!(
            chunk_words(&words(CHUNK_WORDS + OVERLAP_WORDS + 1), None).len(),
            2
        );
    }
}
