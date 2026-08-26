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
//! The table covers 101 languages, which is not a feature so much as the
//! absence of a defect: the English-only model this replaced pooled a Thai
//! sentence into noise and ranked the memory that answered it at chance.
//! `assets/potion-multilingual-128M/PROVENANCE.md` has the measurement.
//!
//! ## Why the inference is written out here
//!
//! The reference implementation loads a model by expanding the whole int8
//! table into `f32` up front. For this table that is 500,353 × 256 values —
//! 489 MB of heap, per process — to answer a query that touches about ten
//! rows. A long-lived MCP server would hold that for a whole session, and
//! several sessions multiply it.
//!
//! A static model's forward pass is a gather and a mean, so it is written here
//! instead, reading int8 straight out of the binary's read-only pages: the
//! operating system maps them once and shares them between every process, and
//! only the rows actually touched become resident. Measured on this machine:
//!
//! | | reference | here |
//! |---|---|---|
//! | resident | the whole expanded table | only the rows a query touched |
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

use anyhow::Result;
use crate::tokenize::Unigram;

/// Which weights this build reads, and the directory they live in.
///
/// See `assets/potion-multilingual-128M/PROVENANCE.md` for what was changed
/// from upstream, why this model and not the English one, and the measured
/// cost of both. The name is in the path so that a build expecting different
/// weights finds nothing rather than reading the wrong ones.
pub const MODEL: &str = "potion-multilingual-128M";

/// The weights file inside [`MODEL`]'s directory.
pub const WEIGHTS_FILE: &str = "model-int8.safetensors";
/// The tokenizer file inside [`MODEL`]'s directory.
pub const TOKENIZER_FILE: &str = "tokenizer.json";

/// Ceiling on the safetensors header, so a corrupt length cannot ask for an
/// allocation the size of the disk. The real one is a few hundred bytes.
const MAX_HEADER: u64 = 1 << 20;


/// Vector width, fixed by the model.
///
/// Changing this changes the width of every stored vector, and `similarity`
/// scores mismatched widths at 0.0 - so an existing index does not degrade,
/// it stops answering. `Store::events_missing_vectors` counts a row of the
/// wrong width as absent for exactly that reason, which is what turns a model
/// change into an ordinary backlog instead of a migration.
pub const DIMS: usize = 256;

/// A unit vector, stored one byte per dimension.
///
/// The model normalizes what it returns, so every component is already in
/// `[-1, 1]` and one fixed scale quantizes it closely enough to rank by — the
/// same argument the weights themselves are stored under.
///
/// This is not miserliness. Search reads every vector in the project to rank
/// them, so the row width IS the query cost: 256 bytes a row keeps a hundred
/// thousand events at 26 MB of sequential reads rather than 102 MB.
pub type Vector = Vec<u8>;

/// The embedding table, as it sits in the binary.
///
/// The view borrows the static weights rather than owning a copy of them, so
/// this holds no heap at all: every row read here is a read of the binary's
/// own read-only pages.
#[derive(Debug)]
struct Table {
    file: std::fs::File,
    /// Where the tensor body starts, past the safetensors header.
    body: u64,
    rows: usize,
    dims: usize,
}

impl Table {
    /// Read one token's row into `into`, or leave it alone and say no.
    ///
    /// A read rather than a mapping, and that is the whole memory story: a
    /// query touches on the order of ten rows of 256 bytes out of half a
    /// million, so the table never needs to be in this process at all. The
    /// operating system's page cache keeps the pages that get used, shares
    /// them between every brain process, and the heap here stays empty.
    /// `mmap` would do the same thing and needs `unsafe`; this does not.
    fn row(&self, id: u32, into: &mut [u8]) -> bool {
        use std::os::unix::fs::FileExt;
        let id = id as usize;
        if id >= self.rows || into.len() != self.dims {
            return false;
        }
        let at = self.body + (id * self.dims) as u64;
        self.file.read_exact_at(into, at).is_ok()
    }
}

/// Everything here loads from bytes compiled into the binary, so a failure is
/// not a runtime condition — it is a defect in what was vendored, and the only
/// way to reach it is to change `assets/` without changing this file. The
/// tests below are what stop that shipping; these messages are what someone
/// reads through `brain doctor` if one ever does.
///
/// The error text is kept, not the error: a `OnceLock` has to hand out the
/// same value to every caller, and an `anyhow::Error` cannot be shared. A
/// string can, and a string is what the reader wanted anyway.
fn tokenizer() -> Result<&'static Unigram> {
    static CELL: OnceLock<Result<Unigram, String>> = OnceLock::new();
    CELL.get_or_init(|| {
        let paths = crate::config::Paths::resolve().map_err(|error| error.to_string())?;
        let path = paths.model_dir().join(TOKENIZER_FILE);
        if !path.is_file() {
            return Err(format!(
                "{} is not here. The embedding model is fetched once, after \
                 install; `brain doctor` says how",
                path.display()
            ));
        }
        Unigram::from_path(&path).map_err(|error| error.to_string())
    })
    .as_ref()
    .map_err(|error| anyhow::anyhow!("{error}"))
}

fn table() -> Result<&'static Table> {
    static CELL: OnceLock<Result<Table, String>> = OnceLock::new();
    CELL.get_or_init(|| {
        let paths = crate::config::Paths::resolve().map_err(|error| error.to_string())?;
        load_table(&paths.model_dir().join(WEIGHTS_FILE))
    })
        .as_ref()
        .map_err(|error| anyhow::anyhow!("{error}"))
}

/// Read the weights, and say precisely what is wrong when they cannot be.
///
/// Separate from [`table`] so the failures can be tested. Three things can be
/// wrong with a re-vendored file and they need three different answers: one is
/// a corrupt download, one is a model saved at the wrong precision, and one is
/// a different model entirely — and a single "failed to load" sends whoever
/// reads it looking in the wrong place for all three.
fn load_table(path: &std::path::Path) -> Result<Table, String> {
    // A safetensors file is a length, a JSON header, and the tensor bodies -
    // so the header alone answers where the rows begin, and the 122 MB behind
    // it is never read into this process.
    let mut file = std::fs::File::open(path)
        .map_err(|error| format!("{} is not readable: {error}", path.display()))?;
    let mut length = [0u8; 8];
    std::io::Read::read_exact(&mut file, &mut length)
        .map_err(|error| format!("{} is too short to be safetensors: {error}", path.display()))?;
    let header_len = u64::from_le_bytes(length);
    if header_len > MAX_HEADER {
        // Not a size problem in practice - it is what any file that is not
        // safetensors looks like when its first eight bytes are read as a
        // length. Saying "not a readable safetensors file" points at the
        // actual mistake; saying "header too large" points at nothing.
        return Err(format!(
            "not a readable safetensors file: {} opens with a {header_len}-byte header length",
            path.display()
        ));
    }
    #[allow(clippy::cast_possible_truncation)]
    let mut header = vec![0u8; header_len as usize];
    std::io::Read::read_exact(&mut file, &mut header)
        .map_err(|error| format!("{} has a truncated header: {error}", path.display()))?;
    let header: serde_json::Value = serde_json::from_slice(&header).map_err(|error| {
        format!("not a readable safetensors file: {} ({error})", path.display())
    })?;

    let tensor = header.get("embeddings").ok_or_else(|| {
        let names: Vec<&String> = header.as_object().map(|m| m.keys().collect()).unwrap_or_default();
        format!("no `embeddings` tensor; found {names:?}")
    })?;
    if tensor["dtype"] != "I8" {
        return Err(format!(
            "expected int8 weights, got {} — see assets/{MODEL}/quantize.py",
            tensor["dtype"]
        ));
    }
    let shape: Vec<usize> = serde_json::from_value(tensor["shape"].clone())
        .map_err(|error| format!("unreadable tensor shape: {error}"))?;
    let &[rows, dims] = shape.as_slice() else {
        return Err(format!("expected a 2-D tensor, got shape {shape:?}"));
    };
    if dims != DIMS {
        return Err(format!(
            "model is {dims}-dimensional but this build stores {DIMS}-byte vectors; \
             every vector already in the index would score 0.0 against a new one"
        ));
    }
    let offsets: Vec<u64> = serde_json::from_value(tensor["data_offsets"].clone())
        .map_err(|error| format!("unreadable tensor offsets: {error}"))?;
    let &[start, end] = offsets.as_slice() else {
        return Err(format!("expected two data offsets, got {offsets:?}"));
    };
    if rows == 0 || end.saturating_sub(start) != (rows * dims) as u64 {
        return Err(format!(
            "tensor spans {} bytes, which is not {rows} rows of {dims}",
            end.saturating_sub(start)
        ));
    }
    Ok(Table { file, body: 8 + header_len + start, rows, dims })
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
fn pool(tokenizer: &Unigram, table: &Table, text: &str) -> Vec<f32> {
    let mut sum = vec![0f32; table.dims];
    let encoded = tokenizer.encode(text);
    let unknown = tokenizer.unk_id();
    let mut counted = 0u32;
    let mut row = vec![0u8; table.dims];
    for id in &encoded {
        if Some(*id) == unknown {
            continue;
        }
        if counted as usize >= MAX_TOKENS {
            break;
        }
        if !table.row(*id, &mut row) {
            continue;
        }
        for (slot, byte) in sum.iter_mut().zip(&row) {
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

/// Is the vendored model usable, and if not, why not?
///
/// Reported by `brain doctor`. The weights are compiled in, so this can only
/// answer "no" after somebody replaced them — which is exactly when a specific
/// reason is worth having, and exactly when nobody has one.
///
/// # Errors
/// Returns the reason the model cannot be loaded.
pub fn readiness() -> Result<()> {
    tokenizer()?;
    table()?;
    Ok(())
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
pub(crate) mod tests {
    use super::*;

    /// The model as it sits in a checkout, which is where tests read it from.
    ///
    /// Not the installed copy: a test must not depend on whether the machine
    /// running it has fetched anything, and must not read a different version
    /// than the source tree it is checking. `assets/` is git-ignored and
    /// fetched by `bootstrap.sh --model-only --into assets`.
    pub(crate) const ASSETS: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/assets/potion-multilingual-128M");

    /// Point this process's data directory at a tree whose model is the one in
    /// the checkout.
    ///
    /// The loaders resolve the model through `Paths`, and a test must not
    /// depend on whether the machine running it has ever installed anything.
    /// Called by every test that encodes; the work happens once because the
    /// loaders themselves cache.
    pub(crate) fn use_checkout_model() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let home = std::env::temp_dir().join("rolepod-brain-test-home");
            let models = home.join("models").join(MODEL);
            std::fs::create_dir_all(&models).expect("test model directory");
            for file in [WEIGHTS_FILE, TOKENIZER_FILE, "config.json"] {
                let target = models.join(file);
                if !target.exists() {
                    std::fs::copy(std::path::Path::new(ASSETS).join(file), &target)
                        .expect("stage the checkout model");
                }
            }
            // SAFETY-adjacent: single-threaded here by `Once`, and every
            // reader of this variable runs after it.
            std::env::set_var(crate::config::DATA_DIR_ENV, &home);
        });
    }

    /// The reason this module exists: two memories about one thing, in words
    /// that never meet. FTS5 scores every one of these pairs at nothing.
    ///
    /// Stated as an ordering rather than a threshold, because the absolute
    /// cosine scale belongs to whichever model is vendored - a different table
    /// moves every number here at once without the ranking changing at all,
    /// and a test that pins the numbers would fail on a model that got better.
    /// What must hold for any model worth shipping is that each pair about one
    /// thing outscores the same text against something unrelated.
    #[test]
    fn words_that_never_meet_still_land_near_each_other() {
        use_checkout_model();
        let unrelated = encode("the office coffee machine is broken again").unwrap();
        let pairs = [
            ("the auth token expiry comparison is wrong", "login sessions are expiring far too early"),
            ("the nightly job never wrote a backup", "yesterday's database snapshot is missing"),
            ("every page waits on one slow query", "the site takes eight seconds to open"),
        ];
        for (left, right) in pairs {
            let anchor = encode(left).unwrap();
            let near = similarity(&anchor, &encode(right).unwrap());
            let far = similarity(&anchor, &unrelated);
            assert!(
                near > far,
                "no semantic signal: {left:?} scored {near:.3} against its pair \
                 and {far:.3} against unrelated text"
            );
        }
    }

    /// A question in Thai, and the English memory that answers it. The model
    /// before this one scored these at chance - not badly, at chance - because
    /// its vocabulary held thirty Thai characters and no Thai words, so every
    /// Thai sentence pooled to the same noise.
    #[test]
    fn a_question_in_another_language_finds_the_memory_that_answers_it() {
        use_checkout_model();
        let answer = encode("Fixed the login page rejecting valid credentials").unwrap();
        let asked = encode("แก้บั๊กหน้าเข้าสู่ระบบที่ล็อกอินไม่ได้").unwrap();
        let other = encode("Nightly automated database backup job").unwrap();

        let hit = similarity(&asked, &answer);
        let miss = similarity(&asked, &other);
        assert!(
            hit > miss,
            "a Thai question ranked the wrong English memory first: \
             {hit:.3} for the answer, {miss:.3} for unrelated work"
        );
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
        use_checkout_model();
        let assets = std::path::Path::new(ASSETS);
        let reference = model2vec_rs::model::StaticModel::from_bytes(
            std::fs::read(assets.join(TOKENIZER_FILE)).expect("tokenizer"),
            std::fs::read(assets.join(WEIGHTS_FILE)).expect("weights"),
            std::fs::read(assets.join("config.json")).expect("config"),
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
        use_checkout_model();
        assert_eq!(encode("anything at all").unwrap().len(), DIMS);
    }

    /// Identical input, identical bytes — otherwise a reindex would rewrite
    /// every row and no stored vector could be trusted to be current.
    #[test]
    fn encoding_is_deterministic() {
        use_checkout_model();
        let once = encode("consolidation writes per-project hub notes").unwrap();
        let twice = encode("consolidation writes per-project hub notes").unwrap();
        assert_eq!(once, twice);
    }

    #[test]
    fn a_batch_matches_encoding_one_at_a_time() {
        use_checkout_model();
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
        use_checkout_model();
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

    /// Three ways a re-vendored file can be wrong, three different answers.
    ///
    /// This is the whole reason the load path returns a message rather than a
    /// bare failure. The weights arrive as a fetched file now, so a truncated
    /// download, a half-written rename, or a replaced `assets/` all land here
    /// — and "failed to load" would send the reader looking in the wrong
    /// place for every one of them.
    #[test]
    fn a_bad_vendored_file_says_which_way_it_is_bad() {
        use safetensors::tensor::TensorView;

        // Each case needs a file, because that is what the loader reads.
        let leak = |bytes: Vec<u8>| -> std::path::PathBuf {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            bytes.hash(&mut hasher);
            let path = std::env::temp_dir().join(format!("brain-model-{}.bin", hasher.finish()));
            std::fs::write(&path, bytes).expect("write the case to a file");
            path
        };
        let built = |dtype, shape: Vec<usize>, data: Vec<u8>| {
            let view = TensorView::new(dtype, shape, &data).expect("a valid view");
            safetensors::serialize([("embeddings", view)], None).expect("serialize")
        };

        let corrupt = load_table(&leak(b"this is not a safetensors file".to_vec()));
        assert!(
            corrupt.unwrap_err().contains("not a readable safetensors"),
            "a corrupt file should say so"
        );

        let misnamed = {
            let data = vec![0u8; DIMS];
            let view = TensorView::new(safetensors::Dtype::I8, vec![1, DIMS], &data).unwrap();
            safetensors::serialize([("weights", view)], None).unwrap()
        };
        let misnamed = load_table(&leak(misnamed));
        assert!(
            misnamed.unwrap_err().contains("no `embeddings` tensor"),
            "a different model should name what it did contain"
        );

        let wrong_precision =
            load_table(&leak(built(safetensors::Dtype::F32, vec![1, DIMS], vec![0u8; DIMS * 4])));
        let wrong_precision = wrong_precision.unwrap_err();
        assert!(wrong_precision.contains("expected int8"), "{wrong_precision}");
        assert!(
            wrong_precision.contains("quantize.py"),
            "the answer to wrong precision is a script we ship: {wrong_precision}"
        );

        let wrong_width =
            load_table(&leak(built(safetensors::Dtype::I8, vec![1, 64], vec![0u8; 64])));
        let wrong_width = wrong_width.unwrap_err();
        assert!(wrong_width.contains("64-dimensional"), "{wrong_width}");
        assert!(
            wrong_width.contains("score 0.0"),
            "a width change silently voids every stored vector; say so: {wrong_width}"
        );

        // And the file actually shipped is none of these.
        assert!(load_table(&std::path::Path::new(ASSETS).join(WEIGHTS_FILE)).is_ok());
    }

    /// A row index past the end of the table must be skipped, not read.
    #[test]
    fn a_token_id_past_the_table_is_ignored() {
        let table = load_table(&std::path::Path::new(ASSETS).join(WEIGHTS_FILE)).unwrap();
        let mut row = vec![0u8; DIMS];
        assert!(table.row(0, &mut row));
        assert!(!table.row(u32::MAX, &mut row));
    }
}
