//! Reranking without leaving the process.
//!
//! A cross-encoder is not a language model. It reads one (question, entry)
//! pair and returns one number: how well the second answers the first. It
//! generates nothing, so there is no prompt to write, no answer to parse, and
//! no vocabulary an agent has to be taught. Thirty pairs take under two
//! seconds on a laptop CPU.
//!
//! That is the whole reason this exists. Reranking through a host CLI is
//! measured at a median of 12.2s on a real brain — worth waiting for once,
//! not worth waiting for often. The same work here costs 1.6s, needs no
//! subscription, no credential, and no process to spawn.
//!
//! Bounded the same way everything else here is bounded: absent until asked
//! for, and absent again the moment anything goes wrong. A missing model, a
//! corrupt file, a runtime that will not load - each returns `None` and the
//! caller falls through to the CLI it would have used anyway.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Value;
use tokenizers::Tokenizer;

pub const WEIGHTS_FILE: &str = "model.onnx";
/// The tokenizer inside [`MODEL`]'s directory.
pub const TOKENIZER_FILE: &str = "tokenizer.json";

/// ONNX Runtime itself, downloaded beside the weights.
///
/// Nothing links it at build time, which is what lets every target carry this
/// feature. The crate's own prebuilt runtime is what demanded glibc 2.38 and
/// what has no Intel macOS build; Microsoft publishes one per platform that
/// needs glibc 2.27, and 2.27 is old enough to be everywhere.
///
/// The name has no version in it on purpose. Which ONNX Runtime a machine
/// receives is a question for whatever downloads it - 1.28 where that exists,
/// 1.23 on Intel macOS, where 1.23 is the last one Microsoft built - and the
/// loader should not have to know which answer it got.
#[cfg(target_os = "macos")]
pub const RUNTIME_FILE: &str = "libonnxruntime.dylib";
#[cfg(windows)]
pub const RUNTIME_FILE: &str = "onnxruntime.dll";
#[cfg(not(any(target_os = "macos", windows)))]
pub const RUNTIME_FILE: &str = "libonnxruntime.so";

/// Tokens kept from one (query, entry) pair.
///
/// The model would take 512. Entries here are a title and the first 160 bytes
/// of a body, so 320 covers them with room for a long query, and every token
/// past what the text actually holds is padding that costs time to multiply
/// by zero.
const MAX_TOKENS: usize = 320;

/// Loaded once per process, kept for the life of it.
///
/// The MCP server lives exactly as long as one session, which is what makes a
/// local model affordable without a daemon: the first search of a session pays
/// the one-second load, every search after it pays nothing. A short-lived hook
/// process never reaches this code at all.
///
/// The error is kept as text rather than as an error, because a `OnceLock` has
/// to hand out the same value to every later caller and most error types are
/// not `Clone`.
static CELL: OnceLock<Result<Reranker, String>> = OnceLock::new();

struct Reranker {
    /// `Session::run` needs `&mut`, and this lives in a `OnceLock` that hands
    /// out shared references. One lock per search, uncontended in practice:
    /// the MCP server answers one call at a time.
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

/// Score `entries` against `query`, best first, or `None` if this build has no
/// reranker to hand.
///
/// `None` is not a failure to report. It is the ordinary state of a machine
/// whose model has not downloaded yet, or whose target `ort` publishes no
/// binaries for. The caller treats it as "not available" and uses the path it
/// would have used anyway.
#[must_use]
pub fn rerank(model_dir: &Path, query: &str, entries: &[String]) -> Option<Vec<usize>> {
    if entries.len() < 2 {
        return None;
    }
    let reranker = load(model_dir).as_ref().ok()?;
    match reranker.score(query, entries) {
        Ok(scores) => {
            let mut order: Vec<usize> = (0..scores.len()).collect();
            // Descending by score, ties broken by the order the caller gave -
            // which is the index's own ranking, and a better tiebreak than
            // whatever `sort_by` would otherwise do.
            order.sort_by(|a, b| {
                scores[*b].total_cmp(&scores[*a]).then_with(|| a.cmp(b))
            });
            Some(order)
        }
        Err(_) => None,
    }
}

fn load(model_dir: &Path) -> &'static Result<Reranker, String> {
    CELL.get_or_init(|| open(model_dir).map_err(|error| format!("{error:#}")))
}

fn open(model_dir: &Path) -> Result<Reranker> {
    let weights = model_dir.join(WEIGHTS_FILE);
    let tokenizer = model_dir.join(TOKENIZER_FILE);
    let runtime = model_dir.join(RUNTIME_FILE);
    anyhow::ensure!(weights.is_file(), "no reranker at {}", weights.display());
    anyhow::ensure!(tokenizer.is_file(), "no tokenizer at {}", tokenizer.display());
    anyhow::ensure!(runtime.is_file(), "no onnx runtime at {}", runtime.display());

    // The one call in this file that must not be allowed to panic. Every other
    // `ort` entry point loads the dylib on first use and aborts the process if
    // it cannot - wrong architecture, missing symbol, a runtime older than the
    // API floor - and this binary aborts rather than unwinds. `init_from` is
    // the fallible door: it performs exactly those checks and hands back an
    // error, which becomes a `None` upstream and a fall through to the host
    // CLI, which is where a machine that cannot run this model belongs anyway.
    //
    // `commit` returning false means another caller configured the environment
    // first. Nothing else in this binary touches `ort`, so that can only
    // happen if this ran twice, and the dylib is already the one we asked for.
    ort::init_from(&runtime)
        .map_err(|error| anyhow::anyhow!("{error}"))
        .with_context(|| format!("load onnx runtime from {}", runtime.display()))?
        .commit();

    let mut tokenizer = Tokenizer::from_file(&tokenizer)
        .map_err(|error| anyhow::anyhow!("{error}"))
        .context("read reranker tokenizer")?;
    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: MAX_TOKENS,
            ..Default::default()
        }))
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    tokenizer.with_padding(Some(tokenizers::PaddingParams::default()));

    let session = Session::builder()
        .context("build onnx session")?
        .commit_from_file(&weights)
        .context("load reranker weights")?;
    Ok(Reranker { session: Mutex::new(session), tokenizer })
}

impl Reranker {
    fn score(&self, query: &str, entries: &[String]) -> Result<Vec<f32>> {
        let pairs: Vec<(String, String)> =
            entries.iter().map(|entry| (query.to_string(), entry.clone())).collect();
        let encoded = self
            .tokenizer
            .encode_batch(pairs, true)
            .map_err(|error| anyhow::anyhow!("{error}"))
            .context("tokenize pairs")?;

        let rows = encoded.len();
        let width = encoded.first().map_or(0, |first| first.get_ids().len());
        anyhow::ensure!(width > 0, "tokenizer produced no tokens");

        let mut ids = Vec::with_capacity(rows * width);
        let mut mask = Vec::with_capacity(rows * width);
        for item in &encoded {
            ids.extend(item.get_ids().iter().map(|id| i64::from(*id)));
            mask.extend(item.get_attention_mask().iter().map(|bit| i64::from(*bit)));
        }

        let shape = [rows, width];
        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("reranker lock poisoned"))?;
        let outputs = session
            .run(ort::inputs![
                "input_ids" => Value::from_array((shape, ids))?,
                "attention_mask" => Value::from_array((shape, mask))?,
            ])
            .context("run reranker")?;
        let (_, scores) = outputs[0].try_extract_tensor::<f32>().context("read scores")?;
        anyhow::ensure!(
            scores.len() == rows,
            "reranker returned {} scores for {rows} entries",
            scores.len()
        );
        Ok(scores.to_vec())
    }
}

#[cfg(test)]
mod tests {
    /// Does the ONNX Runtime this platform is given actually load?
    ///
    /// Ignored by default because it needs a runtime file, which is a download
    /// rather than part of the checkout. Run it where the answer is not
    /// already known - a platform whose library nobody here can execute:
    ///
    /// ```text
    /// BRAIN_TEST_RUNTIME=path/to/onnxruntime.dll \
    ///     cargo test --features local-rerank -- --ignored runtime_loads
    /// ```
    ///
    /// It deliberately stops short of the model. What is in doubt for a new
    /// platform is the three things `init_from` checks - that the library
    /// opens, that it exports `OrtGetApiBase`, and that its API is not older
    /// than this build's floor. A 568 MB download would not tell us more
    /// about any of them.
    #[test]
    #[ignore = "needs a runtime file; set BRAIN_TEST_RUNTIME"]
    fn runtime_loads() {
        let path = std::env::var("BRAIN_TEST_RUNTIME")
            .expect("set BRAIN_TEST_RUNTIME to an onnxruntime library");
        let path = std::path::Path::new(&path);
        assert!(path.is_file(), "no runtime at {}", path.display());
        match ort::init_from(path) {
            Ok(builder) => {
                builder.commit();
            }
            Err(error) => panic!("{} did not load: {error}", path.display()),
        }
    }
}
