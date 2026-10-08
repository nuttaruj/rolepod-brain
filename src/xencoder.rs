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
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Value;
use tokenizers::Tokenizer;

use crate::onnx_external;

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

/// How long a loaded model may sit unused before it is let go.
///
/// Measured on a real brain: a process that has reranked holds about 1.7 GB,
/// and one that has not, or that has let go, about 115 MB. The price is the
/// next reranked search after an idle spell, which pays the load again: +1.5 to
/// 1.9 s. Ninety seconds keeps a burst of searches warm and returns the memory
/// of a session that has moved on to something else.
const IDLE_WINDOW: Duration = Duration::from_secs(90);

/// Loaded on the first rerank, dropped again after [`IDLE_WINDOW`] of none.
///
/// The MCP server lives as long as one session, which is what makes a local
/// model affordable without a daemon, but a session can sit idle for hours
/// holding 1.7 GB. The release needs a thread: the server spends its idle time
/// blocked reading stdin, so nothing would run to check the clock if the check
/// waited for the next request. One short-lived thread per load, which exits
/// the moment it has let go of the model.
///
/// A failed load is kept as text and kept for good, as it always was: the
/// model files do not appear mid-session, and trying again would cost every
/// search the attempt. Most error types are not `Clone`, hence the text.
static CELL: Idle<Reranker> = Idle::new(IDLE_WINDOW);

/// Has this process paid the model load yet, and not let go of it since? The
/// first rerank carries it - about a second - and the difference is the
/// cold/warm split `brain stats` reports, which a single latency figure would
/// blur.
#[must_use]
pub fn is_loaded() -> bool {
    CELL.is_loaded()
}

enum Slot<T> {
    Empty,
    Held { value: Arc<T>, last_used: Instant },
    Failed(String),
}

/// A value built on demand and dropped after a spell of disuse.
///
/// The clock is a parameter everywhere so a test can step it. A search in
/// progress holds its own `Arc`, so a release never pulls the model out from
/// under it: the slot lets go, the search finishes, and the last `Arc` frees it.
struct Idle<T> {
    window: Duration,
    slot: Mutex<Slot<T>>,
}

impl<T> Idle<T> {
    const fn new(window: Duration) -> Self {
        Self { window, slot: Mutex::new(Slot::Empty) }
    }

    /// A panic mid-load leaves the slot in a state every arm handles, so a
    /// poisoned lock is safe to take over.
    fn lock(&self) -> std::sync::MutexGuard<'_, Slot<T>> {
        self.slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The value, built by `load` if there is none, with whether this call built it.
    ///
    /// The lock is held through the load, so two callers that arrive together
    /// load once.
    fn get(
        &self,
        now: Instant,
        load: impl FnOnce() -> Result<T, String>,
    ) -> Result<(Arc<T>, bool), String> {
        let mut slot = self.lock();
        match &mut *slot {
            Slot::Held { value, last_used } => {
                *last_used = now;
                return Ok((Arc::clone(value), false));
            }
            Slot::Failed(why) => return Err(why.clone()),
            Slot::Empty => {}
        }
        match load() {
            Ok(value) => {
                let value = Arc::new(value);
                *slot = Slot::Held { value: Arc::clone(&value), last_used: now };
                Ok((value, true))
            }
            Err(why) => {
                *slot = Slot::Failed(why.clone());
                Err(why)
            }
        }
    }

    fn is_loaded(&self) -> bool {
        matches!(*self.lock(), Slot::Held { .. })
    }

    /// Restart the idle clock of a held value at `now`.
    fn touch(&self, now: Instant) {
        if let Slot::Held { last_used, .. } = &mut *self.lock() {
            *last_used = now;
        }
    }

    /// Let go of the value if it has been idle for the whole window. Returns
    /// how long until it would be, or `None` when nothing is held any more.
    fn release_if_idle(&self, now: Instant) -> Option<Duration> {
        let mut slot = self.lock();
        let Slot::Held { last_used, .. } = &*slot else {
            return None;
        };
        let idle = now.saturating_duration_since(*last_used);
        if idle >= self.window {
            *slot = Slot::Empty;
            return None;
        }
        Some(self.window - idle)
    }
}

impl Idle<Reranker> {
    /// Sleep out the window, release, and exit. A rerank in between pushes the
    /// deadline back, and the thread sleeps again for what is left. A failed
    /// spawn is ignored on purpose: the model then stays loaded, as it did
    /// before this holder existed.
    fn watch(&'static self) {
        let _ = std::thread::Builder::new().name("rerank-idle".into()).spawn(move || {
            while let Some(left) = self.release_if_idle(Instant::now()) {
                std::thread::sleep(left);
            }
        });
    }
}

struct Reranker {
    /// `Session::run` needs `&mut`, and this is shared through an `Arc`.
    /// One lock per search, uncontended in practice: the MCP server answers
    /// one call at a time.
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
    let reranker = load(model_dir).ok()?;
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

fn load(model_dir: &Path) -> Result<Arc<Reranker>, String> {
    let (reranker, fresh) =
        CELL.get(Instant::now(), || open(model_dir).map_err(|error| format!("{error:#}")))?;
    if fresh {
        // The clock handed to `get` was read before the load, and a first load
        // that converts the model takes tens of seconds: the idle window starts
        // when the model is ready, not when it was asked for.
        CELL.touch(Instant::now());
        CELL.watch();
    }
    Ok(reranker)
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

    let session = settle(&weights, |weights| load_weights(weights, &tokenizer))?;
    Ok(Reranker { session: Mutex::new(session), tokenizer })
}

/// Load the weights, and if they are a converted graph that cannot be loaded -
/// its sidecar is gone, is the wrong size, or the runtime refuses the pair -
/// remove the pair rather than keep it.
///
/// A converted `model.onnx` is a small file, and the installer skips a
/// `model.onnx` that is not empty; left in place it would hold the reranker
/// broken for good while everything reported it ready. With it gone the pair is
/// not ready, and the next rerank that falls to the CLI starts the download
/// again. An embedded model is never removed: it is the only copy there is.
fn settle<S>(weights: &Path, load: impl FnOnce(&Path) -> Result<S>) -> Result<S> {
    let loaded = onnx_external::require_sidecar(weights).and_then(|()| load(weights));
    match loaded {
        Err(error) if onnx_external::forget_converted(weights) => {
            Err(error.context("removed the unusable converted reranker, the next search fetches it again"))
        }
        other => other,
    }
}

fn build_session(weights: &Path) -> Result<Session> {
    Session::builder()
        .context("build onnx session")?
        .commit_from_file(weights)
        .context("load reranker weights")
}

/// The session for `weights`, with the weights moved out of the graph file the
/// first time they are found inside it.
///
/// Embedded, the model costs 1.46 GB to load; with its weights in a sidecar the
/// runtime maps them instead, for about a quarter of that. The conversion is
/// local and one-time, and it earns its place the hard way: both models score
/// the same probe pairs and the copy replaces the original only if every score
/// agrees to the bit. Anything else - a claim held by another process, a write
/// that fails, a score that differs - leaves the original in place, loaded as
/// it always was, and is remembered, so the next load does not try again.
fn load_weights(weights: &Path, tokenizer: &Tokenizer) -> Result<Session> {
    convert_and_load(
        weights,
        onnx_external::stage,
        build_session,
        |session| probe_scores(session, tokenizer),
    )
}

/// [`load_weights`] with its parts passed in, so the decisions can be tested
/// without a 543 MB model.
///
/// The models are never alive together. The original is scored and let go
/// before the converted copy is built, so the peak is one model, as it was
/// before conversion existed (about 1.5 GB), not two. If the copy is rejected
/// the original is built again, once, which is the price of a conversion that
/// does not agree and is paid a single time.
fn convert_and_load<S>(
    weights: &Path,
    stage: impl FnOnce(&Path) -> Result<Option<onnx_external::Staged>>,
    build: impl Fn(&Path) -> Result<S>,
    probe: impl Fn(&mut S) -> Result<Vec<Vec<u32>>>,
) -> Result<S> {
    let Some(staged) = stage(weights).ok().flatten() else {
        return build(weights);
    };
    let reference = match build(weights) {
        Ok(mut original) => probe(&mut original).ok(),
        Err(error) => {
            staged.discard();
            return Err(error);
        }
    };
    staged.touch();
    if let Some(reference) = reference {
        if let Ok(mut converted) = build(&staged.graph) {
            staged.touch();
            if probe(&mut converted).is_ok_and(|scores| scores == reference)
                && staged.publish().is_ok()
            {
                return Ok(converted);
            }
        }
    }
    staged.discard();
    build(weights)
}

/// Pairs the two models must score identically before one replaces the other.
///
/// The pool sizes matter as much as the text: this model's int8 quantization is
/// dynamic, so a pair's score depends on what it is batched with, and a
/// conversion that changed the arithmetic would show itself differently at
/// one, seven and thirty rows. Thai and CJK are here because they are where a
/// tokenizer or a weight read going wrong would show first.
const PROBE_POOLS: [(&str, usize); 3] = [
    ("how do I undo the last commit but keep my changes", 1),
    ("设计决策的原因是什么 どうして この 設計 に した の か", 7),
    ("ทำไมเราถึงเลือกใช้ฐานข้อมูลนี้แทนตัวเดิม", 30),
];

const PROBE_ENTRIES: [&str; 30] = [
    "Use git reset --soft HEAD~1 to undo the last commit and keep the changes staged.",
    "We chose SQLite because the whole store has to travel as a single file.",
    "ตัดสินใจใช้ SQLite เพราะต้องย้ายข้อมูลทั้งหมดเป็นไฟล์เดียวและไม่ต้องมีเซิร์ฟเวอร์",
    "เปลี่ยนจาก Postgres มาเป็น SQLite หลังจากวัดความเร็วแล้วไม่ต่างกัน",
    "我们选择 SQLite，因为整个存储必须作为单个文件迁移。",
    "决定放弃常驻进程，所有后台工作都在钩子触发时完成。",
    "設計判断: 常駐プロセスを持たず、フックから起動する方式にした。",
    "データベースは SQLite を採用。ファイル一つで持ち運べるため。",
    "결정: 상주 프로세스 없이 훅이 실행될 때만 작업한다.",
    "The release build failed because the linker ran out of memory on CI.",
    "fn main() { let store = Store::open(&home)?; store.consolidate()?; }",
    "SELECT id, body FROM events WHERE kind = 'observation' AND ts < ?1 ORDER BY ts",
    "Rotate the API token every ninety days and never log it.",
    "วิธีย้อน commit ล่าสุดโดยเก็บการแก้ไขไว้: git reset --soft HEAD~1",
    "The hook must finish inside five seconds or the agent stops waiting for it.",
    "ปัญหา hook ช้าเกิดจากการสแกนฐานข้อมูลทั้งก้อนตอนเปิด",
    "Weekly planning notes: ship the importer, defer the sync encryption review.",
    "钩子必须在五秒内完成，否则代理会停止等待。",
    "How to bake sourdough: feed the starter, fold the dough every thirty minutes.",
    "git checkout -- . discards every uncommitted change in the working tree.",
    "メモリ使用量はモデルを読み込んだ直後に最大になる。",
    "Embedding model: 256 dimensions, multilingual, 26 MB on disk.",
    "สรุปการประชุม: เลื่อนการปล่อยเวอร์ชันใหม่ไปสัปดาห์หน้า",
    "The reranker reads one query and one entry and returns a single number.",
    "Kubernetes pods restart when the liveness probe fails three times.",
    "我们把所有的密钥都放在环境变量里，而不是配置文件。",
    "Dockerfile: COPY the lockfile first so the dependency layer stays cached.",
    "ฐานข้อมูลถูกเลือกเพราะอ่านเร็วและไม่ต้องติดตั้งอะไรเพิ่ม",
    "回滚上一次提交但保留修改：git reset --soft HEAD~1",
    "Cache invalidation is hard; name the key after the thing it depends on.",
];

/// What `session` gives every probe pair, as the scores' bits: two models agree
/// to the last bit when these are equal.
fn probe_scores(session: &mut Session, tokenizer: &Tokenizer) -> Result<Vec<Vec<u32>>> {
    PROBE_POOLS
        .iter()
        .map(|(query, pool)| {
            let entries: Vec<String> =
                PROBE_ENTRIES.iter().take(*pool).map(|entry| (*entry).to_string()).collect();
            let scores = score_pairs(session, tokenizer, query, &entries)?;
            Ok(scores.iter().map(|score| score.to_bits()).collect())
        })
        .collect()
}

impl Reranker {
    fn score(&self, query: &str, entries: &[String]) -> Result<Vec<f32>> {
        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("reranker lock poisoned"))?;
        score_pairs(&mut session, &self.tokenizer, query, entries)
    }
}

fn score_pairs(
    session: &mut Session,
    tokenizer: &Tokenizer,
    query: &str,
    entries: &[String],
) -> Result<Vec<f32>> {
    let pairs: Vec<(String, String)> =
        entries.iter().map(|entry| (query.to_string(), entry.clone())).collect();
    let encoded = tokenizer
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

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::path::PathBuf;
    use std::rc::Rc;

    use super::*;

    const WINDOW: Duration = Duration::from_secs(90);

    #[test]
    fn an_idle_model_is_dropped_and_the_next_call_reloads_it() {
        let idle: Idle<String> = Idle::new(WINDOW);
        let t0 = Instant::now();
        let loads = std::cell::Cell::new(0);
        let load = || {
            loads.set(loads.get() + 1);
            Ok("model".to_string())
        };

        let (model, fresh) = idle.get(t0, load).expect("load");
        assert!(fresh && idle.is_loaded());
        let weak = Arc::downgrade(&model);
        drop(model);

        // Inside the window: kept, not reloaded, and the clock restarts.
        let (_, fresh) = idle.get(t0 + Duration::from_secs(60), load).expect("warm");
        assert!(!fresh && loads.get() == 1);
        assert_eq!(idle.release_if_idle(t0 + Duration::from_secs(100)), Some(Duration::from_secs(50)));
        assert!(idle.is_loaded() && weak.strong_count() == 1);

        // Past the window since the last use: dropped.
        assert_eq!(idle.release_if_idle(t0 + Duration::from_secs(150)), None);
        assert!(!idle.is_loaded());
        assert_eq!(weak.strong_count(), 0);

        let (_, fresh) = idle.get(t0 + Duration::from_secs(151), load).expect("reload");
        assert!(fresh && loads.get() == 2 && idle.is_loaded());
    }

    #[test]
    fn a_release_leaves_a_search_in_progress_its_model() {
        let idle: Idle<String> = Idle::new(WINDOW);
        let t0 = Instant::now();
        let (held, _) = idle.get(t0, || Ok("model".to_string())).expect("load");
        assert_eq!(idle.release_if_idle(t0 + WINDOW), None);
        assert!(!idle.is_loaded());
        assert_eq!(*held, "model");
        assert_eq!(Arc::strong_count(&held), 1);
    }

    #[test]
    fn a_failed_load_is_remembered_rather_than_retried() {
        let idle: Idle<String> = Idle::new(WINDOW);
        let t0 = Instant::now();
        assert_eq!(idle.get(t0, || Err("no model".to_string())).err().as_deref(), Some("no model"));
        let again = idle.get(t0, || panic!("must not retry"));
        assert_eq!(again.err().as_deref(), Some("no model"));
        assert!(!idle.is_loaded());
    }

    #[test]
    fn touching_a_held_value_restarts_its_idle_clock() {
        let idle: Idle<String> = Idle::new(WINDOW);
        let t0 = Instant::now();
        idle.get(t0, || Ok("model".to_string())).expect("load");
        // A load that took 80 s is not 80 s of the window already spent.
        idle.touch(t0 + Duration::from_secs(80));
        assert_eq!(
            idle.release_if_idle(t0 + Duration::from_secs(100)),
            Some(Duration::from_secs(70))
        );
        let empty: Idle<String> = Idle::new(WINDOW);
        empty.touch(t0);
        assert!(!empty.is_loaded(), "touching nothing does not create it");
    }

    /// A stand-in for a session that records when it is built and let go.
    struct Fake {
        name: &'static str,
        log: Rc<RefCell<Vec<String>>>,
    }

    impl Drop for Fake {
        fn drop(&mut self) {
            self.log.borrow_mut().push(format!("drop {}", self.name));
        }
    }

    fn which(path: &Path) -> &'static str {
        if path.to_string_lossy().ends_with(".new") { "converted" } else { "original" }
    }

    /// One load of `dir`'s model through the real conversion, with sessions that
    /// score `agree`-ably or not.
    fn load_fake(
        dir: &Path,
        log: &Rc<RefCell<Vec<String>>>,
        agree: bool,
        original_loads: bool,
    ) -> Result<Fake> {
        convert_and_load(
            &dir.join(WEIGHTS_FILE),
            |path| onnx_external::stage_from(path, 0),
            |path| {
                let name = which(path);
                log.borrow_mut().push(format!("build {name}"));
                anyhow::ensure!(original_loads || name != "original", "will not load");
                Ok(Fake { name, log: Rc::clone(log) })
            },
            |session| {
                log.borrow_mut().push(format!("score {}", session.name));
                let differs = session.name == "converted" && !agree;
                Ok(vec![vec![if differs { 2 } else { 1 }]])
            },
        )
    }

    fn model_dir(tag: &str) -> (PathBuf, Vec<u8>) {
        let dir = onnx_external::tests::scratch(tag);
        let source = onnx_external::tests::sample_model();
        std::fs::write(dir.join(WEIGHTS_FILE), &source).expect("write");
        (dir, source)
    }

    #[test]
    fn a_conversion_that_scores_the_same_replaces_the_model_and_the_models_are_never_alive_together() {
        let (dir, source) = model_dir("agree");
        let log = Rc::new(RefCell::new(Vec::new()));
        let loaded = load_fake(&dir, &log, true, true).expect("load");
        assert_eq!(loaded.name, "converted");
        assert_eq!(
            *log.borrow(),
            ["build original", "score original", "drop original", "build converted", "score converted"],
            "the original is gone before the copy is built"
        );
        let graph = std::fs::read(dir.join(WEIGHTS_FILE)).expect("model");
        assert!(graph.len() < source.len(), "the published model is the small graph");
        onnx_external::require_sidecar(&dir.join(WEIGHTS_FILE)).expect("its sidecar is whole");
        assert!(onnx_external::tests::sidecar_of(&dir).is_file());
        drop(loaded);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_conversion_that_scores_differently_keeps_the_original_and_is_not_tried_again() {
        let (dir, source) = model_dir("differ");
        let log = Rc::new(RefCell::new(Vec::new()));
        let loaded = load_fake(&dir, &log, false, true).expect("the original still loads");
        assert_eq!(loaded.name, "original");
        assert_eq!(std::fs::read(dir.join(WEIGHTS_FILE)).expect("model"), source);
        assert_eq!(onnx_external::tests::names(&dir), ["model.onnx", "model.onnx.noconvert"]);
        drop(loaded);

        log.borrow_mut().clear();
        let again = load_fake(&dir, &log, false, true).expect("load again");
        assert_eq!(*log.borrow(), ["build original"], "no second conversion, no probe");
        drop(again);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn an_original_that_will_not_load_is_an_error_that_leaves_the_model_and_no_staging() {
        let (dir, source) = model_dir("noload");
        let log = Rc::new(RefCell::new(Vec::new()));
        assert!(load_fake(&dir, &log, true, false).is_err());
        assert_eq!(std::fs::read(dir.join(WEIGHTS_FILE)).expect("model"), source);
        assert_eq!(onnx_external::tests::names(&dir), ["model.onnx", "model.onnx.noconvert"]);
        std::fs::remove_dir_all(dir).ok();
    }

    /// A model directory whose reranker is complete: a converted graph with its
    /// sidecar, a tokenizer and a runtime.
    fn ready_pair(tag: &str) -> PathBuf {
        let (dir, _) = model_dir(tag);
        let staged = onnx_external::stage_from(&dir.join(WEIGHTS_FILE), 0)
            .expect("stage")
            .expect("staged");
        staged.publish().expect("publish");
        drop(staged);
        for file in [TOKENIZER_FILE, RUNTIME_FILE] {
            std::fs::write(dir.join(file), b"x").expect("write");
        }
        assert!(crate::rerank::local_is_ready(&dir));
        dir
    }

    #[test]
    fn a_converted_model_whose_sidecar_is_gone_is_removed_so_it_is_fetched_again() {
        let dir = ready_pair("nosidecar");
        std::fs::remove_file(onnx_external::tests::sidecar_of(&dir)).expect("lose the weights");

        let weights = dir.join(WEIGHTS_FILE);
        let error = settle::<()>(&weights, |_| panic!("must not load a graph without weights"))
            .expect_err("unusable");
        assert!(format!("{error:#}").contains("missing"), "{error:#}");
        assert!(!crate::rerank::local_is_ready(&dir), "the pair is not ready");
        assert!(!weights.exists(), "the small graph is gone, so the installer downloads it again");
        assert!(dir.join(TOKENIZER_FILE).is_file() && dir.join(RUNTIME_FILE).is_file());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_converted_model_with_a_short_sidecar_is_removed_with_what_is_left_of_it() {
        let dir = ready_pair("short");
        let sidecar = onnx_external::tests::sidecar_of(&dir);
        let data = std::fs::read(&sidecar).expect("sidecar");
        std::fs::write(&sidecar, &data[..data.len() / 2]).expect("truncate");

        let weights = dir.join(WEIGHTS_FILE);
        settle::<()>(&weights, |_| panic!("must not load a short sidecar")).expect_err("unusable");
        assert!(!crate::rerank::local_is_ready(&dir));
        assert!(!weights.exists() && !sidecar.exists());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_converted_model_that_will_not_load_is_removed_and_one_that_does_is_kept() {
        let dir = ready_pair("noload-pair");
        let weights = dir.join(WEIGHTS_FILE);
        assert!(settle(&weights, |_| Ok(())).is_ok());
        assert!(crate::rerank::local_is_ready(&dir), "a pair that loads is left alone");

        let error = settle::<()>(&weights, |_| Err(anyhow::anyhow!("the runtime refuses it")))
            .expect_err("refused");
        let text = format!("{error:#}");
        assert!(text.contains("the runtime refuses it") && text.contains("fetches it again"), "{text}");
        assert!(!crate::rerank::local_is_ready(&dir));
        assert!(!weights.exists() && !dir.join("model.onnx.noconvert").exists());
        assert!(
            !onnx_external::tests::names(&dir).iter().any(|name| name.ends_with(".data")),
            "no partial sidecar is left behind"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn an_original_model_that_will_not_load_is_never_removed() {
        let (dir, source) = model_dir("original-stays");
        let weights = dir.join(WEIGHTS_FILE);
        settle::<()>(&weights, |_| Err(anyhow::anyhow!("no memory"))).expect_err("refused");
        assert_eq!(std::fs::read(&weights).expect("model"), source, "the only copy stays");
        std::fs::remove_dir_all(dir).ok();
    }

    /// The real model, converted on a copy: the graph file shrinks, the sidecar
    /// appears, nothing is left behind, and a second load of the converted copy
    /// scores the probe pairs exactly as the original did.
    ///
    /// Ignored because it needs the 568 MB download. `BRAIN_TEST_MODEL_DIR`
    /// is the folder that holds the weights, tokenizer and runtime; it is
    /// copied, never converted in place.
    ///
    /// ```text
    /// BRAIN_TEST_MODEL_DIR=~/.rolepod-brain/models/bge-reranker-v2-m3-int8 \
    ///     cargo test --features local-rerank -- --ignored converts_the_real_model
    /// ```
    #[test]
    #[ignore = "needs the reranker download; set BRAIN_TEST_MODEL_DIR"]
    fn converts_the_real_model_and_scores_the_same_after() {
        let source = std::env::var("BRAIN_TEST_MODEL_DIR").expect("set BRAIN_TEST_MODEL_DIR");
        let dir = std::env::temp_dir().join(format!("brain-real-model-{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&dir).expect("create");
        for file in [WEIGHTS_FILE, TOKENIZER_FILE, RUNTIME_FILE] {
            std::fs::copy(Path::new(&source).join(file), dir.join(file)).expect("copy");
        }
        let before = std::fs::metadata(dir.join(WEIGHTS_FILE)).expect("model").len();

        // The reference: the original model, loaded as it always was, before
        // anything converts it. Its runtime has to be up for that.
        let tokenizer = {
            let runtime = dir.join(RUNTIME_FILE);
            ort::init_from(&runtime).expect("runtime").commit();
            let mut tokenizer = Tokenizer::from_file(dir.join(TOKENIZER_FILE)).expect("tokenizer");
            tokenizer
                .with_truncation(Some(tokenizers::TruncationParams {
                    max_length: MAX_TOKENS,
                    ..Default::default()
                }))
                .expect("truncation");
            tokenizer.with_padding(Some(tokenizers::PaddingParams::default()));
            tokenizer
        };
        let reference = {
            let mut original = build_session(&dir.join(WEIGHTS_FILE)).expect("original loads");
            probe_scores(&mut original, &tokenizer).expect("original scores")
        };

        let started = Instant::now();
        let first = open(&dir).expect("first load converts");
        eprintln!("first load, with conversion: {:?}", started.elapsed());
        let graph = std::fs::metadata(dir.join(WEIGHTS_FILE)).expect("graph").len();
        let sidecar = onnx_external::tests::sidecar_of(&dir);
        let data = std::fs::metadata(&sidecar).expect("sidecar").len();
        eprintln!("model.onnx {before} -> {graph} + {data} sidecar");
        assert!(graph < before / 100, "the weights are still inside the graph file");
        assert!(graph + data <= before + before / 100, "the conversion grew the model on disk");
        let mut expected: Vec<String> = [RUNTIME_FILE, WEIGHTS_FILE, TOKENIZER_FILE]
            .map(String::from)
            .into();
        expected.push(sidecar.file_name().expect("name").to_string_lossy().into_owned());
        expected.sort();
        assert_eq!(
            onnx_external::tests::names(&dir),
            expected,
            "a conversion leaves only the model, its sidecar and the rest"
        );

        let second = open(&dir).expect("converted copy loads");
        for (index, (query, pool)) in PROBE_POOLS.into_iter().enumerate() {
            let entries: Vec<String> =
                PROBE_ENTRIES.iter().take(pool).map(|entry| (*entry).to_string()).collect();
            for loaded in [&first, &second] {
                let bits: Vec<u32> =
                    loaded.score(query, &entries).expect("score").iter().map(|s| s.to_bits()).collect();
                assert_eq!(bits, reference[index], "pool of {pool} differs from the original");
            }
        }
        std::fs::remove_dir_all(dir).ok();
    }

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
