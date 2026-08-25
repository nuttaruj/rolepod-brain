//! Semantic vectors for recall.
//!
//! FTS5 answers "which memory used these words". It cannot answer "which
//! memory is about this", and the gap is not academic: a session that recorded
//! `login token expiry` is invisible to someone searching `auth`, which is the
//! single most common way memory fails to arrive when it exists.
//!
//! The model is a static embedding table — distilled from a sentence
//! transformer into per-token vectors, so inference is a table lookup and a
//! mean rather than a forward pass. That is the whole reason it can live here:
//! no ONNX runtime, no Python, no second process, no download. It is compiled
//! into the binary and works on a machine that has never had a network.
//!
//! ## Why the inference is written out here
//!
//! The reference implementation loads a model by expanding the whole int8
//! table into `f32` up front. For this table that is 63,091 × 512 values —
//! 129 MB of heap, per process, measured at 199 MB RSS — to answer a query
//! that touches about ten rows. A long-lived MCP server would hold that for a
//! whole session, and several sessions multiply it.
//!
//! A static model's forward pass is a gather and a mean, so it is written here
//! instead, reading int8 straight out of the binary's read-only pages: the
//! operating system maps them once and shares them between every process, and
//! only the rows actually touched become resident. Measured on this machine:
//!
//! | | reference | here |
//! |---|---|---|
//! | construct | 83 ms | 40 ms (the tokenizer, and nothing else) |
//! | encode | 27 µs | 18 µs |
//! | resident | 199 MB | a few MB |
//!
//! `an_answer_here_matches_the_reference_implementation` holds the two to the
//! same output, so this stays a memory optimisation and never becomes a
//! quality one.
//!
//! Nothing in the capture path calls this module. Constructing the tokenizer
//! costs more than a whole hook run, so vectors are written by consolidation —
//! detached, and already slow — and encoded at search time in a process that
//! outlives the query.

use std::sync::OnceLock;

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

/// Vendored weights. See `assets/potion-retrieval-32M/PROVENANCE.md` for what
/// was changed from upstream and the measured cost of changing it.
const WEIGHTS: &[u8] = include_bytes!("../assets/potion-retrieval-32M/model-int8.safetensors");
const TOKENIZER: &[u8] = include_bytes!("../assets/potion-retrieval-32M/tokenizer.json");

/// Vector width, fixed by the model.
pub const DIMS: usize = 512;

/// A unit vector, stored one byte per dimension.
///
/// The model normalizes what it returns, so every component is already in
/// `[-1, 1]` and one fixed scale quantizes it closely enough to rank by — the
/// same argument the weights themselves are stored under.
///
/// This is not miserliness. Search reads every vector in the project to rank
/// them, so the row width IS the query cost: 512 bytes a row keeps a hundred
/// thousand events at 51 MB of sequential reads rather than 205 MB.
pub type Vector = Vec<u8>;

/// The embedding table, as it sits in the binary.
///
/// The view borrows the static weights rather than owning a copy of them, so
/// this holds no heap at all: every row read here is a read of the binary's
/// own read-only pages.
struct Table {
    view: safetensors::tensor::TensorView<'static>,
    dims: usize,
}

impl Table {
    fn row(&self, id: u32) -> Option<&[u8]> {
        let start = (id as usize).checked_mul(self.dims)?;
        let end = start.checked_add(self.dims)?;
        self.view.data().get(start..end)
    }
}

fn tokenizer() -> Result<&'static Tokenizer> {
    static CELL: OnceLock<Option<Tokenizer>> = OnceLock::new();
    CELL.get_or_init(|| Tokenizer::from_bytes(TOKENIZER).ok())
        .as_ref()
        .context("embedding tokenizer failed to load")
}

fn table() -> Result<&'static Table> {
    static CELL: OnceLock<Option<Table>> = OnceLock::new();
    CELL.get_or_init(|| {
        // Parsing a safetensors file is reading its header; the tensor body
        // stays exactly where it is, which is the point.
        let file = safetensors::SafeTensors::deserialize(WEIGHTS).ok()?;
        let view = file.tensor("embeddings").ok()?;
        let &[_, dims] = view.shape() else { return None };
        if dims != DIMS || view.dtype() != safetensors::Dtype::I8 {
            return None;
        }
        // `view` borrows WEIGHTS, not the `SafeTensors` wrapper, so it outlives
        // the wrapper being dropped here.
        Some(Table { view, dims })
    })
    .as_ref()
    .context("embedding table failed to load")
}

/// Encode one text into a stored vector.
///
/// # Errors
/// Returns an error when the model cannot be loaded.
pub fn encode(text: &str) -> Result<Vector> {
    let (tokenizer, table) = (tokenizer()?, table()?);
    Ok(quantize(&pool(tokenizer, table, text)))
}

/// Encode many texts, loading the model at most once.
///
/// # Errors
/// Returns an error when the model cannot be loaded.
pub fn encode_all(texts: &[String]) -> Result<Vec<Vector>> {
    let (tokenizer, table) = (tokenizer()?, table()?);
    Ok(texts.iter().map(|text| quantize(&pool(tokenizer, table, text))).collect())
}

/// The longest run of tokens that contributes to one vector.
///
/// The model's own inference limit. Beyond it the mean stops moving anyway,
/// and a body of any size would otherwise decide how long an encode takes.
const MAX_TOKENS: usize = 512;

/// The forward pass: gather each token's row, average them, normalize.
///
/// Three details are not free choices — they are what the reference does, and
/// diverging on any of them produces vectors that rank differently:
///
/// * No special tokens. They carry no meaning of their own in a table
///   distilled without them, and would pull every short text together.
/// * **Unknown tokens are dropped, not embedded.** They are the single row
///   every out-of-vocabulary word maps to, so keeping them makes unrelated
///   foreign-script texts look alike — and this vocabulary is English, so
///   every Thai prompt in the corpus is almost entirely unknown. Missing this
///   cost 0.93 cosine against the reference on Thai text, and nothing at all
///   on English, which is exactly how it would have shipped unnoticed.
/// * Truncation at [`MAX_TOKENS`], applied after the unknowns are gone.
fn pool(tokenizer: &Tokenizer, table: &Table, text: &str) -> Vec<f32> {
    let mut sum = vec![0f32; table.dims];
    let Ok(encoded) = tokenizer.encode(text, false) else {
        return sum;
    };
    let unknown = tokenizer.token_to_id("[UNK]");
    let mut counted = 0u32;
    for id in encoded.get_ids() {
        if Some(*id) == unknown {
            continue;
        }
        if counted as usize >= MAX_TOKENS {
            break;
        }
        let Some(row) = table.row(*id) else { continue };
        for (slot, byte) in sum.iter_mut().zip(row) {
            #[allow(clippy::cast_possible_wrap)]
            let value = *byte as i8;
            *slot += f32::from(value);
        }
        counted += 1;
    }
    if counted == 0 {
        return sum;
    }
    // The mean is what the model defines, and normalizing right after makes
    // the divisor cancel — but it is done anyway, because a vector that is not
    // the model's output is not a vector this can claim parity for.
    for slot in &mut sum {
        *slot /= f32::from(u16::try_from(counted).unwrap_or(u16::MAX));
    }
    let norm = sum.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for slot in &mut sum {
            *slot /= norm;
        }
    }
    sum
}

fn quantize(vector: &[f32]) -> Vector {
    vector
        .iter()
        .map(|value| {
            #[allow(clippy::cast_possible_truncation)]
            let scaled = (value * 127.0).round().clamp(-127.0, 127.0) as i8;
            scaled as u8
        })
        .collect()
}

/// Cosine similarity between two stored vectors.
///
/// Both sides are quantized unit vectors, so the fixed scale appears above and
/// below the ratio and cancels: the dot product over the norms is the number
/// the float vectors would have produced, to within the quantization.
#[must_use]
pub fn similarity(a: &[u8], b: &[u8]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut norm_a, mut norm_b) = (0i32, 0i32, 0i32);
    for (left, right) in a.iter().zip(b) {
        #[allow(clippy::cast_possible_wrap)]
        let (left, right) = (i32::from(*left as i8), i32::from(*right as i8));
        dot += left * right;
        norm_a += left * left;
        norm_b += right * right;
    }
    if norm_a == 0 || norm_b == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let value = dot as f32 / ((norm_a as f32).sqrt() * (norm_b as f32).sqrt());
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reason this module exists: two memories about one thing, in words
    /// that never meet. FTS5 scores this pair at nothing.
    #[test]
    fn words_that_never_meet_still_land_near_each_other() {
        let auth = encode("the auth token expiry comparison is wrong").unwrap();
        let login = encode("login sessions are expiring far too early").unwrap();
        let coffee = encode("the office coffee machine is broken again").unwrap();

        let related = similarity(&auth, &login);
        let unrelated = similarity(&auth, &coffee);
        assert!(
            related > unrelated,
            "no semantic signal at all: related {related:.3} vs unrelated {unrelated:.3}"
        );
        assert!(related > 0.15, "related texts scored only {related:.3}");
        assert!(unrelated < 0.15, "unrelated texts scored {unrelated:.3}");
    }

    /// The forward pass is written out here rather than called, to keep 129 MB
    /// of expanded table off the heap. That is only allowed to be a memory
    /// decision: the answers have to be the reference implementation's
    /// answers, or the saving was taken out of recall quality instead.
    ///
    /// The reference is a dev-dependency, so it costs the shipped binary
    /// nothing and this comparison cannot quietly stop being made.
    #[test]
    fn an_answer_here_matches_the_reference_implementation() {
        let reference = model2vec_rs::model::StaticModel::from_bytes(
            TOKENIZER,
            WEIGHTS,
            include_bytes!("../assets/potion-retrieval-32M/config.json"),
            None,
        )
        .expect("reference model");

        for text in [
            "login sessions are expiring far too early",
            "brain doctor reports the semantic index coverage",
            "Edit: src/store.rs",
            "ULID-keyed append-only JSONL, fsynced per event",
            "",
            "     ",
            "ก่อนตัดสินใจ ขอเช็คว่า Rust มีอะไรให้ใช้จริงบ้าง",
            "<private>secret</private> ← arrow",
            // Past the point either side truncates. The reference cuts the
            // input string first; this cuts the token run. Parity has to hold
            // for a long session summary too, or it only holds for the short
            // texts the earlier cases happened to use.
            &"the consolidation ladder degrades to rule-based and catches up later "
                .repeat(200),
        ] {
            let theirs = quantize(&reference.encode_single(text));
            let ours = encode(text).unwrap();
            let agreement = similarity(&ours, &theirs);
            assert!(
                ours == theirs || agreement > 0.999,
                "diverged from the reference on {text:?}: cosine {agreement:.5}"
            );
        }
    }

    #[test]
    fn a_vector_is_one_byte_per_dimension() {
        assert_eq!(encode("anything at all").unwrap().len(), DIMS);
    }

    /// Identical input, identical bytes — otherwise a reindex would rewrite
    /// every row and no stored vector could be trusted to be current.
    #[test]
    fn encoding_is_deterministic() {
        let once = encode("consolidation writes per-project hub notes").unwrap();
        let twice = encode("consolidation writes per-project hub notes").unwrap();
        assert_eq!(once, twice);
    }

    #[test]
    fn a_batch_matches_encoding_one_at_a_time() {
        let texts = vec!["first observation".to_string(), "a second, unrelated one".to_string()];
        let batched = encode_all(&texts).unwrap();
        for (text, batch) in texts.iter().zip(&batched) {
            assert_eq!(&encode(text).unwrap(), batch);
        }
    }

    /// Titles and bodies come from hook payloads, which is to say from
    /// anywhere. None of it may abort the process — the class of bug that a
    /// single `←` in a commit message already caused once, in `sanitize`.
    #[test]
    fn nothing_a_payload_can_contain_brings_this_down() {
        for text in [
            "",
            " ",
            "\0\0\0",
            "←→↑↓",
            "🙂🙂🙂",
            "a\u{0}b",
            &"very long ".repeat(20_000),
            &String::from_utf8_lossy(&[0xF0, 0x9F, 0x98]),
        ] {
            let vector = encode(text).unwrap();
            assert_eq!(vector.len(), DIMS);
        }
    }

    #[test]
    fn mismatched_or_empty_vectors_score_nothing() {
        assert!((similarity(&[], &[]) - 0.0).abs() < f32::EPSILON);
        assert!((similarity(&[1, 2, 3], &[1, 2]) - 0.0).abs() < f32::EPSILON);
        assert!((similarity(&[0, 0], &[1, 1]) - 0.0).abs() < f32::EPSILON);
    }

    /// A row index past the end of the table must be skipped, not read.
    #[test]
    fn a_token_id_past_the_table_is_ignored() {
        let table = table().unwrap();
        assert!(table.row(0).is_some());
        assert!(table.row(u32::MAX).is_none());
    }
}
