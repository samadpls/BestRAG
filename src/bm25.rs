//! BM25 as Qdrant sparse vectors. Qdrant applies the IDF part itself
//! (`Modifier::Idf` on the sparse index), so a document vector only carries
//! BM25's term-frequency part and a query vector only marks which terms it has.

use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::is_combining_mark;

const K1: f32 = 1.2;
const B: f32 = 0.75;
// BM25 wants the average document length, which Qdrant cannot give us.
// fastembed's BM25 uses the same fixed value.
const AVG_LEN: f32 = 256.0;

/// Lowercased words in any script. NFKC and Arabic normalisation run first,
/// so spelling variants of one word share a BM25 term. Combining marks stay
/// inside their word, so vowel signs do not split Hindi or Urdu words.
pub fn tokens(text: &str) -> Vec<String> {
    let normal: String = text.nfkc().filter_map(normalize_arabic).collect();
    normal
        .split(|c: char| !(c.is_alphanumeric() || is_combining_mark(c)))
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// The rules of Lucene's ArabicNormalizer: alef with hamza or madda becomes
/// a bare alef, alef maqsura becomes yeh, teh marbuta becomes heh, and tatweel
/// and the short-vowel marks are dropped.
fn normalize_arabic(c: char) -> Option<char> {
    match c {
        '\u{0622}' | '\u{0623}' | '\u{0625}' => Some('\u{0627}'),
        '\u{0649}' => Some('\u{064A}'),
        '\u{0629}' => Some('\u{0647}'),
        '\u{0640}' | '\u{064B}'..='\u{0652}' => None,
        c => Some(c),
    }
}

/// FNV-1a. Stored vectors depend on these ids, so the hash must never change
/// between builds, which std's `DefaultHasher` does not promise.
fn token_id(token: &str) -> u32 {
    token.bytes().fold(0x811c_9dc5, |hash, byte| {
        (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
    })
}

pub fn document_vector(text: &str) -> (Vec<u32>, Vec<f32>) {
    let tokens = tokens(text);
    let mut counts: HashMap<u32, f32> = HashMap::new();
    for token in &tokens {
        *counts.entry(token_id(token)).or_default() += 1.0;
    }
    let length_norm = K1 * (1.0 - B + B * tokens.len() as f32 / AVG_LEN);
    counts
        .into_iter()
        .map(|(id, tf)| (id, tf * (K1 + 1.0) / (tf + length_norm)))
        .unzip()
}

pub fn query_vector(text: &str) -> Vec<(u32, f32)> {
    let mut ids: Vec<u32> = tokens(text).iter().map(|t| token_id(t)).collect();
    ids.sort_unstable();
    ids.dedup();
    ids.into_iter().map(|id| (id, 1.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_ids_are_the_published_fnv1a_values() {
        assert_eq!(token_id(""), 0x811c_9dc5);
        assert_eq!(token_id("a"), 0xe40c_292c);
        assert_eq!(token_id("foobar"), 0xbf9c_f968);
    }

    #[test]
    fn tokens_split_on_punctuation_and_keep_any_script() {
        assert_eq!(
            tokens("Hello, WORLD! snake_case x2 کتاب"),
            vec!["hello", "world", "snake", "case", "x2", "کتاب"]
        );
    }

    #[test]
    fn arabic_spelling_variants_share_a_term() {
        assert_eq!(tokens("الإجازة"), tokens("الاجازة"));
        assert_eq!(tokens("تُصرف"), tokens("تصرف"));
        assert_eq!(tokens("مدرسة"), tokens("مدرسه"));
        assert_eq!(tokens("\u{FEFB}"), tokens("لا"));
    }

    #[test]
    fn vowel_signs_do_not_split_words() {
        assert_eq!(tokens("हिन्दी भाषा").len(), 2);
        assert_eq!(tokens("تُصرف").len(), 1);
    }

    #[test]
    fn repeated_terms_count_more_but_saturate() {
        let (ids, values) = document_vector("cat cat cat dog");
        let value = |word: &str| values[ids.iter().position(|&i| i == token_id(word)).unwrap()];
        assert!(value("cat") > value("dog"));
        assert!(value("cat") < K1 + 1.0);
    }

    #[test]
    fn query_vector_has_each_term_once() {
        let query = query_vector("Rust rust RUST python");
        assert_eq!(query.len(), 2);
        assert!(query.iter().all(|&(_, v)| v == 1.0));
    }
}
