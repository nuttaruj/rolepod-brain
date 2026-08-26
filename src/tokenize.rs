//! The Unigram tokenizer this model needs, at a size that fits beside it.
//!
//! `tokenizers` is a fine library and this module exists for one reason: its
//! Unigram model builds a trie of one node per byte, each node owning a hash
//! map of its children. For the 63,091-entry WordPiece vocabulary of the
//! English model that replaced was 29 MB and nobody noticed. For a 500,353-
//! entry multilingual one it is **710 MB, resident for the life of the
//! process** — measured, and steady rather than transient. Search, `doctor`,
//! consolidation and a session-long MCP server would each hold it.
//!
//! That is the same trade [`crate::embed`] already refused once for the
//! weights, and the answer is the same: the algorithm is small, so write it
//! out and hold the data the compact way. The vocabulary lives here as one
//! string blob, an offset per entry, a score per entry, and a permutation of
//! the ids in byte order — about 13 MB, and prefix search is a binary search
//! that narrows as it goes rather than a pointer chase through allocated
//! nodes.
//!
//! Measured on this machine, same query, same corpus:
//!
//! | | `tokenizers` | here |
//! |---|---|---|
//! | `brain search` | 740 MB | 103 MB |
//! | `brain doctor` | 739 MB | 66 MB |
//! | `brain hook` | 10 MB | 10 MB (never loads the model) |
//!
//! Latency is unchanged — a search spends its time reading vectors out of
//! SQLite, not segmenting a query.
//!
//! What is NOT reimplemented: normalization. The model's normalizer is a
//! sentencepiece `Precompiled` charsmap, which is a compiled table rather
//! than an algorithm, and getting it subtly wrong would change what every
//! accented character embeds to. `NormalizerWrapper` deserializes straight
//! out of the same `tokenizer.json` and costs nothing to keep, so it is kept.
//!
//! ## What holds this to the library's answers
//!
//! Writing a tokenizer out by hand buys memory with correctness, and the
//! only honest way to spend that is to check it against the thing it
//! replaced rather than against a reading of the algorithm.
//!
//! `every_id_matches_the_real_tokenizer_on_text_nobody_chose` constructs the
//! real `tokenizers` stack — paying the 710 MB, once, in a test, which is
//! the right place to spend it — and compares TOKEN IDS on text assembled at
//! random from the pieces that break tokenizers: scripts without spaces,
//! combining marks, zero-width characters, paths, punctuation runs, and the
//! boundaries between them. Forty thousand such texts were run while writing
//! this and none diverged; three thousand run on every `cargo test`.
//!
//! `embed::tests::an_answer_here_matches_the_reference_implementation` then
//! checks the whole pipeline end-to-end through `model2vec-rs`, comparing
//! pooled vectors on texts chosen by hand.
//!
//! What neither can cover is a DIFFERENT `tokenizer.json` needing a rule this
//! does not implement, because the only thing standing between that file and
//! a wrong answer is [`Unigram::from_json`]. So every rule not implemented
//! here refuses to load rather than approximate: `byte_fallback`, a missing
//! or differently-configured pre-tokenizer, an added token with
//! `single_word`/`lstrip`/`rstrip`/`normalized` set or an id the vocabulary
//! contradicts, `unk_id` absent or out of range, `truncation`, `padding`, a
//! post-processor that is not the identity template, and a version other
//! than 1.0.
//!
//! That list is what three clean-room reviews across two model families
//! turned up — none of them found an input that diverges on the vendored
//! asset, and all three found places where a different asset would have been
//! accepted and then tokenized almost right. A guard with a hole shaped like
//! the thing it guards against is the failure this module is most exposed to.

use anyhow::{Context, Result};
use serde::Deserialize;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::{NormalizedString, Normalizer};

/// What an unknown character costs, relative to the rarest thing in the
/// vocabulary. Sentencepiece's constant, and the reference's: changing it
/// would change where the segmentation falls.
const UNK_PENALTY: f64 = 10.0;

/// The pre-tokenizer's replacement for a space, which is also how a word
/// boundary is spelled inside this vocabulary.
const METASPACE: &str = "\u{2581}";

/// Refuse a pre-tokenizer other than the one written out in `prepare`.
///
/// `prepare` implements exactly one configuration: Metaspace, replacing a
/// space with `▁`, prepending it always, and not splitting on it. Every other
/// setting of those three is a real configuration that some model uses, and
/// applying this one to it would not fail — it would tokenize almost right,
/// which is the failure mode this whole module has to keep refusing.
fn check_pre_tokenizer(value: Option<&serde_json::Value>) -> Result<()> {
    // Absent is not "the same as ours". The reference skips pre-tokenizing
    // entirely when the field is null, so the model sees `hello world` where
    // this would hand it `▁hello▁world` - every id different, and the guard
    // that exists to catch exactly that would have waved it through.
    let Some(value) = value else {
        anyhow::bail!(
            "this tokenizer declares no pre-tokenizer, and the reference then does \
             none; `prepare` here always applies Metaspace, so every id would differ"
        )
    };
    let expected = serde_json::json!({
        "type": "Metaspace",
        "replacement": METASPACE,
        "prepend_scheme": "always",
        "split": false,
    });
    if *value == expected {
        return Ok(());
    }
    anyhow::bail!(
        "this tokenizer declares a pre-tokenizer other than the one written out here \
         ({value}); applying the wrong one segments almost right, which nothing detects"
    )
}

/// Is this post-processor one that hands the ids back unchanged?
///
/// `TemplateProcessing` whose templates are the bare sequences and whose
/// special-token map is empty inserts nothing — which is what the vendored
/// asset declares, and the only shape that can be waved through.
fn is_identity_post_processor(name: &str, value: &serde_json::Value) -> bool {
    if name != "post_processor" || value["type"] != "TemplateProcessing" {
        return false;
    }
    let bare = |template: &serde_json::Value| {
        template.as_array().is_some_and(|parts| {
            parts.iter().all(|part| part.get("Sequence").is_some())
        })
    };
    bare(&value["single"])
        && bare(&value["pair"])
        && value["special_tokens"].as_object().is_some_and(serde_json::Map::is_empty)
}

/// The parts of `tokenizer.json` this needs. Deserialized into owned data
/// once and then compacted, rather than read through `serde_json::Value`,
/// which would build half a million `Value::Array`s to throw them away.
#[derive(Deserialize)]
struct File {
    /// The reference refuses anything but `1.0`; a different one means a
    /// file shape this was never read against.
    version: Option<String>,
    normalizer: Option<serde_json::Value>,
    /// Both apply to the ids AFTER the model has produced them - truncation
    /// cuts them, padding appends - so either one non-null changes the answer
    /// without touching anything this module implements.
    truncation: Option<serde_json::Value>,
    padding: Option<serde_json::Value>,
    /// The vendored asset's is an identity template. Another asset's could
    /// insert ids of its own.
    post_processor: Option<serde_json::Value>,
    /// Matched against the raw text BEFORE anything else touches it, which is
    /// what makes `[UNK]` written out in a sentence come back as the unknown
    /// id rather than as the six characters that spell it.
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
    /// Checked, not used: this pre-tokenizer is written out below, and a
    /// different configuration would be applied wrongly rather than refused.
    pre_tokenizer: Option<serde_json::Value>,
    model: Model,
}

/// The vocabulary, read straight into the flat form it is kept in.
///
/// Deserializing it as `Vec<(String, f64)>` first costs half a million
/// separate allocations that are immediately copied into one blob and thrown
/// away — measured at about 80 MB of peak resident for a table whose final
/// form is 13 MB, in a process that may be a session-long MCP server. The
/// visitor below appends each token's bytes as it reads them, so the peak is
/// the final size.
struct Vocab {
    blob: Vec<u8>,
    offsets: Vec<u32>,
    scores: Vec<f64>,
    longest: usize,
    min_score: f64,
}

impl<'de> Deserialize<'de> for Vocab {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Entries;

        impl<'de> serde::de::Visitor<'de> for Entries {
            type Value = Vocab;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a list of [token, score] pairs")
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Vocab, A::Error> {
                let mut vocab = Vocab {
                    blob: Vec::new(),
                    offsets: vec![0],
                    scores: Vec::new(),
                    longest: 0,
                    min_score: f64::MAX,
                };
                // One reused buffer, not one String per entry.
                while let Some(entry) = seq.next_element::<(std::borrow::Cow<'de, str>, f64)>()? {
                    let (token, score) = entry;
                    vocab.blob.extend_from_slice(token.as_bytes());
                    let end = u32::try_from(vocab.blob.len())
                        .map_err(|_| serde::de::Error::custom("vocabulary is too large to index"))?;
                    vocab.offsets.push(end);
                    vocab.scores.push(score);
                    vocab.longest = vocab.longest.max(token.len());
                    vocab.min_score = vocab.min_score.min(score);
                }
                Ok(vocab)
            }
        }

        deserializer.deserialize_seq(Entries)
    }
}

/// An added token, with every flag that changes how it matches.
///
/// The flags are read so that a non-default one is refused rather than
/// ignored: `rstrip` swallows the whitespace after a match, `single_word`
/// discards a match that has word characters beside it, `normalized` matches
/// against normalized text instead of raw. Each changes which ids come back,
/// and none of them is implemented here.
#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    single_word: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    normalized: bool,
}

#[derive(Deserialize)]
struct Model {
    vocab: Vocab,
    unk_id: Option<u32>,
    /// Read only so that turning it on cannot pass unnoticed. The vendored
    /// vocabulary has it off, so a character it cannot spell becomes one
    /// unknown and the caller drops it. A vocabulary with it ON spells that
    /// character as its bytes instead, which is a different segmentation and
    /// therefore a different vector — silently, since neither side errors.
    #[serde(default)]
    byte_fallback: bool,
}

/// A Unigram vocabulary, stored flat.
pub struct Unigram {
    /// Every token's bytes, concatenated.
    blob: Box<[u8]>,
    /// Where each id's token starts in `blob`; one longer than the vocabulary.
    offsets: Box<[u32]>,
    /// Log-probability per id. `f64` because the reference sums in `f64` and
    /// a tie broken the other way is a different segmentation.
    scores: Box<[f64]>,
    /// Ids in byte order, which is what makes prefix search a binary search.
    sorted: Box<[u32]>,
    /// Literal strings that stand for one id however they are spelled, longest
    /// first so that a token containing another is matched whole.
    added: Vec<(String, u32)>,
    unk_id: Option<u32>,
    min_score: f64,
    longest: usize,
    normalizer: Option<NormalizerWrapper>,
}

impl Unigram {
    /// Read a `tokenizer.json`.
    ///
    /// # Errors
    /// Returns an error when the file is not readable JSON, carries no
    /// vocabulary, or declares a normalizer this build cannot construct.
    #[cfg(test)]
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        Self::build(serde_json::from_slice(bytes).context("parse tokenizer.json")?)
    }

    /// Read a `tokenizer.json` without holding the whole file.
    ///
    /// The vendored one is 17.8 MB and its parsed form is 13 MB, so reading
    /// it into memory first doubles the peak of a process - a session-long
    /// MCP server among them - for the duration of a parse.
    ///
    /// # Errors
    /// Returns an error when the file cannot be read or is not a tokenizer
    /// this can serve.
    pub fn from_path(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("open {}", path.display()))?;
        let reader = std::io::BufReader::with_capacity(1 << 20, file);
        Self::build(
            serde_json::from_reader(reader)
                .with_context(|| format!("parse {}", path.display()))?,
        )
    }

    fn build(file: File) -> Result<Self> {
        let normalizer = file
            .normalizer
            .map(serde_json::from_value::<NormalizerWrapper>)
            .transpose()
            .context("build the model's normalizer")?;

        let vocab = file.model.vocab;
        if vocab.scores.is_empty() {
            anyhow::bail!("tokenizer.json carries an empty vocabulary");
        }
        if file.model.byte_fallback {
            anyhow::bail!(
                "this vocabulary declares byte_fallback, which is not implemented here: \
                 a character it cannot spell would be dropped as unknown where the \
                 reference spells it out as <0xNN> tokens, and the two would embed \
                 differently with nothing to say so"
            );
        }
        if file.model.unk_id.is_none() {
            // Without byte fallback and without an unknown id there is no edge
            // to lay across a character the vocabulary cannot spell, so the
            // path would simply stop there and the ids returned would be a
            // prefix of the answer. The reference refuses; so does this.
            anyhow::bail!(
                "this vocabulary declares no unk_id and no byte_fallback, so text it \
                 cannot spell has no representation at all — the segmentation would \
                 stop at the first such character and return a partial answer"
            );
        }
        check_pre_tokenizer(file.pre_tokenizer.as_ref())?;
        if let Some(version) = &file.version {
            if version != "1.0" {
                anyhow::bail!("tokenizer.json declares version {version}, not 1.0");
            }
        }
        for (name, section) in [
            ("truncation", &file.truncation),
            ("padding", &file.padding),
            ("post_processor", &file.post_processor),
        ] {
            // The identity template the vendored asset carries adds nothing;
            // anything else in these three changes the ids after the model is
            // done, which is a stage this module does not have.
            let Some(value) = section else { continue };
            if value.is_null() || is_identity_post_processor(name, value) {
                continue;
            }
            anyhow::bail!(
                "tokenizer.json declares a {name} section, which is applied to the ids \
                 after the model runs and is not implemented here: {value}"
            );
        }
        let Vocab { blob, offsets, scores, longest, min_score } = vocab;
        let count = u32::try_from(scores.len()).context("vocabulary is too large to index")?;
        let mut sorted: Vec<u32> = (0..count).collect();
        let (blob, offsets) = (blob.into_boxed_slice(), offsets.into_boxed_slice());
        let token = |id: u32| -> &[u8] {
            let (start, end) = (offsets[id as usize] as usize, offsets[id as usize + 1] as usize);
            &blob[start..end]
        };
        // Byte order, then id DESCENDING. Two entries spelled the same are a
        // malformed vocabulary, but the reference resolves them by keeping
        // the last one it inserted - so the tie is broken the same way here
        // rather than left to whichever the sort happened to move first.
        sorted.sort_unstable_by(|left, right| {
            token(*left).cmp(token(*right)).then(right.cmp(left))
        });

        let mut added: Vec<(String, u32)> = Vec::new();
        for token in file.added_tokens {
            if token.content.is_empty() {
                // `"".find` in the scan below is always Some(0) over a
                // zero-length match, so the cursor would never advance and
                // the loop would push that id until the process died. The
                // reference drops these at load; so does this.
                continue;
            }
            if token.single_word || token.lstrip || token.rstrip || token.normalized {
                anyhow::bail!(
                    "added token {:?} sets single_word/lstrip/rstrip/normalized, and each \
                     changes which text the token claims; none is implemented here",
                    token.content
                );
            }
            added.push((token.content, token.id));
        }
        // Longest first: `[UNK]` must win over `[` if a vocabulary declares
        // both, at whatever position they start.
        added.sort_by_key(|(content, _)| std::cmp::Reverse(content.len()));

        let unigram = Self {
            blob,
            offsets,
            scores: scores.into_boxed_slice(),
            sorted: sorted.into_boxed_slice(),
            added,
            unk_id: file.model.unk_id,
            min_score,
            longest,
            normalizer,
        };

        if unigram.unk_id.is_some_and(|id| id >= count) {
            anyhow::bail!(
                "unk_id points past the end of a {count}-entry vocabulary, so every \
                 unknown character would come back as an id nothing can look up"
            );
        }
        // The reference discards the id an added token declares and uses the
        // one its content has in the model vocabulary. Rather than silently
        // preferring either, disagreement is refused: a file where the two
        // differ was written against a different reader.
        for (content, id) in &unigram.added {
            let found = unigram.lookup(content.as_bytes());
            if found.is_some_and(|actual| actual != *id) {
                anyhow::bail!(
                    "added token {content:?} declares id {id} but the vocabulary spells \
                     it {}; the reference uses the vocabulary's",
                    found.unwrap_or_default()
                );
            }
        }
        Ok(unigram)
    }

    /// The id of an exact vocabulary entry, by binary search over `sorted`.
    fn lookup(&self, text: &[u8]) -> Option<u32> {
        let at = self.sorted.partition_point(|id| self.token(*id) < text);
        let id = *self.sorted.get(at)?;
        (self.token(id) == text).then_some(id)
    }

    /// The id reserved for text this vocabulary cannot spell.
    #[must_use]
    pub fn unk_id(&self) -> Option<u32> {
        self.unk_id
    }

    fn token(&self, id: u32) -> &[u8] {
        let (start, end) =
            (self.offsets[id as usize] as usize, self.offsets[id as usize + 1] as usize);
        &self.blob[start..end]
    }

    /// Every vocabulary entry that is a prefix of `text`, shortest first.
    ///
    /// The range `[lo, hi)` always holds the entries sharing the prefix
    /// examined so far, and each byte narrows it by two binary searches. A
    /// token shorter than the current depth sorts ahead of every longer token
    /// that shares its bytes, so the same comparison that orders the range
    /// also steps over the entries that already ended.
    fn prefix_matches(&self, text: &[u8], out: &mut Vec<(u32, usize)>) {
        out.clear();
        let (mut lo, mut hi) = (0usize, self.sorted.len());
        for depth in 0..text.len().min(self.longest) {
            let wanted = text[depth];
            let head = lo;
            lo = head
                + self.sorted[head..hi].partition_point(|id| {
                    let token = self.token(*id);
                    token.len() <= depth || token[depth] < wanted
                });
            hi = lo
                + self.sorted[lo..hi].partition_point(|id| {
                    let token = self.token(*id);
                    token.len() > depth && token[depth] == wanted
                });
            if lo == hi {
                break;
            }
            let first = self.sorted[lo];
            if self.token(first).len() == depth + 1 {
                out.push((first, depth + 1));
            }
        }
    }

    /// Normalize, then spell word boundaries the way this vocabulary does.
    ///
    /// Text that normalizes to nothing stays nothing. The reference drops
    /// empty pieces before the pre-tokenizer ever sees them, so prepending a
    /// word boundary here would hand back a vector for a text with no
    /// content — and it would be the same vector every time, which is the one
    /// thing a search must never be handed.
    fn prepare(&self, text: &str) -> String {
        let mut normalized = NormalizedString::from(text);
        if let Some(normalizer) = &self.normalizer {
            // A normalizer that fails leaves the string as it was, which is a
            // worse tokenization of real text - never a wrong answer about
            // different text.
            let _ = normalizer.normalize(&mut normalized);
        }
        if normalized.get().is_empty() {
            return String::new();
        }
        let mut staged = normalized.get().replace(' ', METASPACE);
        if !staged.starts_with(METASPACE) {
            staged.insert_str(0, METASPACE);
        }
        staged
    }

    /// The best segmentation of `text`, as vocabulary ids.
    ///
    /// Viterbi over byte positions, exactly as the reference computes it: at
    /// each character boundary every vocabulary entry starting there proposes
    /// an edge, and a character no entry covers alone gets one unknown edge
    /// priced just below the rarest token. The reference then fuses adjacent
    /// unknowns into one piece before looking their id up again; that is
    /// skipped here because this vocabulary has no byte fallback, so a fused
    /// run and its parts are all the same unknown id — and the caller drops
    /// unknowns before pooling either way.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if self.added.is_empty() {
            return self.encode_piece(text);
        }
        // Added tokens are matched against the RAW text, before the
        // normalizer or the pre-tokenizer see it, and each stretch between
        // them is then tokenized on its own - which is why `a[UNK]b` comes
        // back as three ids and not as the letters that spell the middle one.
        let mut ids = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let mut found: Option<(usize, usize, u32)> = None;
            for (content, id) in &self.added {
                let Some(at) = rest.find(content.as_str()) else { continue };
                if found.is_none_or(|(best, _, _)| at < best) {
                    found = Some((at, content.len(), *id));
                }
            }
            let Some((at, len, id)) = found else {
                ids.extend(self.encode_piece(rest));
                break;
            };
            if at > 0 {
                ids.extend(self.encode_piece(&rest[..at]));
            }
            ids.push(id);
            rest = &rest[at + len..];
        }
        ids
    }

    /// One stretch of text with no added token in it.
    fn encode_piece(&self, text: &str) -> Vec<u32> {
        let staged = self.prepare(text);
        let bytes = staged.as_bytes();
        let size = bytes.len();
        if size == 0 {
            return Vec::new();
        }
        let unk_score = self.min_score - UNK_PENALTY;
        let mut score = vec![0f64; size + 1];
        let mut back: Vec<Option<(usize, u32)>> = vec![None; size + 1];
        let mut matches: Vec<(u32, usize)> = Vec::new();

        let mut start = 0usize;
        while start < size {
            let here = score[start];
            let step = staged[start..].chars().next().map_or(1, char::len_utf8);
            self.prefix_matches(&bytes[start..], &mut matches);
            let mut covered = false;
            for &(id, len) in &matches {
                let end = start + len;
                let candidate = self.scores[id as usize] + here;
                if back[end].is_none() || candidate > score[end] {
                    score[end] = candidate;
                    back[end] = Some((start, id));
                }
                covered |= len == step;
            }
            if !covered {
                if let Some(unk) = self.unk_id {
                    let end = start + step;
                    let candidate = unk_score + here;
                    if back[end].is_none() || candidate > score[end] {
                        score[end] = candidate;
                        back[end] = Some((start, unk));
                    }
                }
            }
            start += step;
        }

        let mut ids: Vec<u32> = Vec::new();
        let mut end = size;
        while end > 0 {
            let Some((from, id)) = back[end] else { break };
            // Adjacent unknowns are ONE piece, not one each. The reference
            // joins the text they cover and looks the joined string up again,
            // which for a vocabulary without byte fallback resolves to a
            // single unknown id. Emitting one per character would agree on
            // the pooled vector - the caller drops unknowns - and disagree on
            // the ids, and the ids are what this claims to reproduce.
            if Some(id) != self.unk_id || ids.last().copied() != Some(id) {
                ids.push(id);
            }
            end = from;
        }
        ids.reverse();
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vendored tokenizer, read from the checkout rather than from
    /// wherever this machine happens to have installed one.
    fn vendored() -> Vec<u8> {
        std::fs::read(
            std::path::Path::new(crate::embed::tests::ASSETS).join(crate::embed::TOKENIZER_FILE),
        )
        .expect("the tokenizer in assets/")
    }

    /// Every token id this produces, against every token id the real
    /// `tokenizers` stack produces, over text nobody chose by hand.
    ///
    /// The parity test in [`crate::embed`] compares pooled VECTORS on nine
    /// texts someone wrote down. That is the weaker claim twice over: nine
    /// texts is nine, and two different segmentations can pool to vectors
    /// close enough to pass. This compares the ids themselves, on text
    /// assembled from the pieces that actually break tokenizers — scripts
    /// without spaces, combining marks, paths, punctuation runs, and the
    /// boundaries between them.
    ///
    /// It costs the 710 MB this module exists to avoid, once, in a test.
    /// That is the correct place to spend it.
    #[test]
    fn every_id_matches_the_real_tokenizer_on_text_nobody_chose() {
        crate::embed::tests::use_checkout_model();
        let bytes = vendored();
        let reference =
            tokenizers::Tokenizer::from_bytes(&bytes).expect("the reference tokenizer");
        let ours = Unigram::from_json(&bytes).expect("ours");

        // Deterministic, dependency-free, and seeded so a failure is
        // reproducible from the seed printed in the message.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        const PIECES: &[&str] = &[
            " ", "  ", "\n", "\t", "", "a", "the", "auth", "src/store.rs", "0x1F",
            "ก", "แก้", "ระบบค้นหา", "ภาษาไทย", "เ", "้", "๙",
            "[UNK]", "[PAD]", "\u{10FFFF}", "\u{10FFFE}", "\u{E0001}",
            "日本語", "中文", "한국어", "ひらがな",
            "é", "ü", "ß", "İ", "ﬁ", "①", "🙂", "\u{200B}", "\u{FEFF}",
            "-", "—", "…", "\"", "'", "(", ")", "/", "\\", "|", "::", "▁",
            "consolidation", "ULID", "JSONL", "fsync",
        ];

        for round in 0..3000 {
            let count = (next() % 24) as usize;
            let mut text = String::new();
            for _ in 0..count {
                text.push_str(PIECES[(next() % PIECES.len() as u64) as usize]);
            }
            let theirs: Vec<u32> = reference
                .encode(text.as_str(), false)
                .expect("reference encode")
                .get_ids()
                .to_vec();
            assert_eq!(
                ours.encode(&text),
                theirs,
                "round {round} diverged on {text:?}"
            );
        }
    }

    fn vocab() -> Unigram {
        // Scores are log-probabilities: less negative is more likely, so
        // `ab` beats `a` + `b` and the segmentation has to prefer it.
        let json = serde_json::json!({
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Metaspace", "replacement": "\u{2581}",
                "prepend_scheme": "always", "split": false
            },
            "model": {
                "type": "Unigram",
                "unk_id": 0,
                "vocab": [["[UNK]", -13.5], ["\u{2581}", -3.0], ["a", -5.0],
                          ["b", -5.0], ["ab", -2.0], ["\u{2581}ab", -1.0]]
            }
        });
        Unigram::from_json(json.to_string().as_bytes()).unwrap()
    }

    #[test]
    fn a_vocabulary_this_cannot_serve_is_refused_rather_than_approximated() {
        // The whole risk of writing a tokenizer out by hand: a future model
        // whose vocabulary needs a rule this does not implement would be
        // tokenized ALMOST right, and almost right is a different vector with
        // nothing to announce it. Refusing to load is the only honest answer
        // available to a module that cannot check its own output.
        let json = serde_json::json!({
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Metaspace", "replacement": "\u{2581}",
                "prepend_scheme": "always", "split": false
            },
            "model": {
                "type": "Unigram", "unk_id": 0, "byte_fallback": true,
                "vocab": [["[UNK]", -13.5], ["a", -5.0]]
            }
        });
        let Err(error) = Unigram::from_json(json.to_string().as_bytes()) else {
            panic!("a byte_fallback vocabulary was accepted");
        };
        assert!(format!("{error}").contains("byte_fallback"), "{error}");
    }

    fn vocab_with(extra: serde_json::Value) -> Result<Unigram> {
        let mut json = serde_json::json!({
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Metaspace", "replacement": "\u{2581}",
                "prepend_scheme": "always", "split": false
            },
            "model": {
                "type": "Unigram", "unk_id": 0,
                "vocab": [["[UNK]", -13.5], ["\u{2581}", -3.0], ["a", -5.0]]
            }
        });
        for (key, value) in extra.as_object().unwrap() {
            json[key] = value.clone();
        }
        Unigram::from_json(json.to_string().as_bytes())
    }

    #[test]
    fn an_added_token_that_matches_nothing_cannot_hang_the_process() {
        crate::embed::tests::use_checkout_model();
        // `"".find` is Some(0) over a zero-length match, so a cursor advanced
        // by the match length never moves: the scan would push that id until
        // the process died. Not a divergence - a hang, in a search.
        let unigram = vocab_with(serde_json::json!({
            "added_tokens": [{"id": 0, "content": ""}]
        }))
        .expect("an empty added token should be dropped, not refused");
        assert_eq!(unigram.encode("a"), vec![1, 2], "the empty token changed the answer");
    }

    #[test]
    fn an_added_token_this_cannot_match_the_reference_way_is_refused() {
        for flag in ["single_word", "lstrip", "rstrip", "normalized"] {
            let result = vocab_with(serde_json::json!({
                "added_tokens": [{"id": 0, "content": "[UNK]", flag: true}]
            }));
            assert!(result.is_err(), "{flag} was accepted and would be ignored");
        }
    }

    #[test]
    fn a_pre_tokenizer_this_does_not_implement_is_refused() {
        // Including its ABSENCE: the reference then pre-tokenizes not at all,
        // where `prepare` always applies Metaspace.
        for pre in [
            serde_json::Value::Null,
            serde_json::json!({"type": "Whitespace"}),
            serde_json::json!({
                "type": "Metaspace", "replacement": "\u{2581}",
                "prepend_scheme": "first", "split": false
            }),
        ] {
            let mut json = serde_json::json!({
                "normalizer": null,
                "model": {"type": "Unigram", "unk_id": 0, "vocab": [["[UNK]", -13.5]]}
            });
            json["pre_tokenizer"] = pre.clone();
            assert!(
                Unigram::from_json(json.to_string().as_bytes()).is_err(),
                "accepted pre_tokenizer {pre}"
            );
        }
    }

    #[test]
    fn an_id_the_vocabulary_disagrees_with_is_refused() {
        // The reference uses the vocabulary's id and only warns. Preferring
        // either silently would answer a question the file asked twice.
        assert!(
            vocab_with(serde_json::json!({
                "added_tokens": [{"id": 2, "content": "[UNK]"}]
            }))
            .is_err(),
            "a declared id that contradicts the vocabulary was accepted"
        );
        assert!(
            vocab_with(serde_json::json!({
                "added_tokens": [{"id": 0, "content": "[UNK]"}]
            }))
            .is_ok(),
            "an id that agrees with the vocabulary was refused"
        );
    }

    #[test]
    fn the_likeliest_segmentation_wins_over_the_greedy_one() {
        crate::embed::tests::use_checkout_model();
        let unigram = vocab();
        // "ab" as one piece scores -1.0 with the boundary attached; taking
        // the boundary and the letters separately scores -13.0.
        assert_eq!(unigram.encode("ab"), vec![5]);
    }

    #[test]
    fn a_character_the_vocabulary_cannot_spell_becomes_one_unknown() {
        crate::embed::tests::use_checkout_model();
        let unigram = vocab();
        let ids = unigram.encode("a€b");
        assert!(ids.contains(&0), "no unknown emitted for an absent character: {ids:?}");
        assert_eq!(ids.iter().filter(|id| **id == 0).count(), 1, "{ids:?}");
    }

    #[test]
    fn prefix_search_finds_every_entry_that_starts_the_text_and_no_others() {
        let unigram = vocab();
        let mut out = Vec::new();
        unigram.prefix_matches("ab".as_bytes(), &mut out);
        let found: Vec<usize> = out.iter().map(|(_, len)| *len).collect();
        assert_eq!(found, vec![1, 2], "expected `a` then `ab`: {out:?}");

        unigram.prefix_matches("zz".as_bytes(), &mut out);
        assert!(out.is_empty(), "matched something for text not in the vocabulary: {out:?}");
    }

    #[test]
    fn an_empty_text_encodes_to_nothing_rather_than_to_a_boundary() {
        crate::embed::tests::use_checkout_model();
        // The pre-tokenizer prepends a word boundary to everything, so an
        // empty string would otherwise come back as one `▁` - one fixed
        // vector, handed out for every text that had no content, ranking
        // against the corpus as if it were a question.
        let unigram = vocab();
        assert!(unigram.encode("").is_empty(), "empty text produced tokens");
        assert!(!unigram.encode(" ").is_empty(), "a space is content the model sees");
    }
}

