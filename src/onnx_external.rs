//! Moving an ONNX model's weights out of the graph file, without Python.
//!
//! The reranker ships as one `model.onnx` with 569 MB of weights inside it.
//! ONNX Runtime reads such a file whole, keeps the parsed copy, and then makes
//! a second, tensor copy of every weight: about 1.46 GB for a model that is
//! 543 MB on disk. The same graph with its weights in a sidecar file is mapped
//! instead of copied, and measured +375 MB to load.
//!
//! The format is plain protobuf, so the conversion is a rewrite of a few
//! length-prefixed fields rather than a dependency:
//!
//! ```text
//! ModelProto.graph (7) -> GraphProto.initializer (5) -> TensorProto
//!     raw_data (9)        moved out to the sidecar
//!     external_data (13)  location, offset, length
//!     data_location (14)  EXTERNAL
//! ```
//!
//! Everything else in the file is copied byte for byte. Only the graph's own
//! initializers move: tensors inside a sub-graph, sparse tensors and tensors
//! that use the typed `float_data` style fields stay where they are, which is
//! correct and merely leaves those weights embedded. Nothing here decides
//! whether the result is trustworthy - the caller scores the same pairs
//! through both models and only publishes a conversion that agrees to the bit.
//!
//! # Safety of the files
//!
//! Every name an attempt creates carries a tag that no other attempt shares, so
//! two processes converting at once cannot truncate, publish over or delete
//! each other's files, and a rejected attempt removes only what it made. The
//! claim below keeps them from doing the work twice; nothing depends on it for
//! correctness. The model is replaced only by a rename, after the new graph and
//! the sidecar are on disk.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, ensure, Context, Result};

/// A tensor smaller than this stays where it is, as ONNX's own converter does.
/// Biases and shape constants are not worth a file offset.
const THRESHOLD: usize = 1024;

/// Offsets in the sidecar start on a 4 KiB boundary. The runtime rounds an
/// offset down to its own page size (16 KiB on Apple Silicon) before it maps,
/// so this is not a promise about pages; it keeps every tensor aligned for the
/// reader and costs at most 4 KiB a tensor.
const ALIGN: u64 = 4096;

/// A graph file this large cannot be only a graph. Below it, a model is read as
/// already converted without opening it, which is what every load after the
/// first one does.
const MIN_EMBEDDED: u64 = 64 * 1024 * 1024;

/// How long a claim on the conversion outlives its owner. The owner touches it
/// between the steps, none of which takes anywhere near this long; a claim
/// older than this belongs to a process that died.
const STALE_CLAIM: Duration = Duration::from_secs(180);

/// How long a failed or pointless conversion is remembered. A model file that
/// changes is looked at again at once; one that does not is retried weekly, so
/// a full disk that has since been cleaned is not held against it forever.
const DECLINE_TTL: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// What every file of one conversion attempt is named after: `model.onnx.cvt-...`.
const ATTEMPT: &str = "cvt-";

const MODEL_GRAPH: u32 = 7;
const GRAPH_INITIALIZER: u32 = 5;
const TENSOR_RAW_DATA: u32 = 9;
const TENSOR_EXTERNAL_DATA: u32 = 13;
const TENSOR_DATA_LOCATION: u32 = 14;
const LOCATION_EXTERNAL: u64 = 1;

/// One field of a protobuf message: its number, its bytes as written, and for a
/// length-delimited field the payload alone.
struct Field<'a> {
    number: u32,
    whole: &'a [u8],
    body: Option<&'a [u8]>,
    varint: u64,
}

fn varint(buf: &[u8], pos: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *buf.get(*pos).context("protobuf ends inside a number")?;
        *pos += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    bail!("protobuf number is longer than 64 bits")
}

fn fields(buf: &[u8]) -> impl Iterator<Item = Result<Field<'_>>> {
    let mut pos = 0;
    std::iter::from_fn(move || {
        if pos >= buf.len() {
            return None;
        }
        let start = pos;
        let next = (|| {
            let tag = varint(buf, &mut pos)?;
            let number = u32::try_from(tag >> 3).context("protobuf field number")?;
            let mut value = 0;
            let mut body = None;
            match tag & 7 {
                0 => value = varint(buf, &mut pos)?,
                1 => pos += 8,
                5 => pos += 4,
                2 => {
                    let len = usize::try_from(varint(buf, &mut pos)?).context("field length")?;
                    let end = pos.checked_add(len).context("field length")?;
                    body = Some(buf.get(pos..end).context("protobuf field runs past its message")?);
                    pos = end;
                }
                wire => bail!("protobuf wire type {wire} is not one an ONNX model uses"),
            }
            let whole = buf.get(start..pos).context("protobuf ends inside a field")?;
            Ok(Field { number, whole, body, varint: value })
        })();
        if next.is_err() {
            pos = buf.len();
        }
        Some(next)
    })
}

fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value & 0x7f) as u8 | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn put_bytes(out: &mut Vec<u8>, number: u32, body: &[u8]) {
    put_varint(out, u64::from(number) << 3 | 2);
    put_varint(out, body.len() as u64);
    out.extend_from_slice(body);
}

fn put_number(out: &mut Vec<u8>, number: u32, value: u64) {
    put_varint(out, u64::from(number) << 3);
    put_varint(out, value);
}

/// Where tensors go: the sidecar file, and how far into it we are.
struct Sink<W: Write> {
    out: W,
    pos: u64,
}

impl<W: Write> Sink<W> {
    /// The offset the bytes landed at.
    fn append(&mut self, bytes: &[u8]) -> io::Result<u64> {
        const ZEROS: [u8; ALIGN as usize] = [0; ALIGN as usize];
        let gap = (ALIGN - self.pos % ALIGN) % ALIGN;
        self.out.write_all(&ZEROS[..gap as usize])?;
        self.pos += gap;
        let at = self.pos;
        self.out.write_all(bytes)?;
        self.pos += bytes.len() as u64;
        Ok(at)
    }
}

/// One tensor with its weights moved to the sidecar, or `None` if it keeps them.
fn tensor<W: Write>(
    message: &[u8],
    location: &str,
    sink: &mut Sink<W>,
) -> Result<Option<Vec<u8>>> {
    let mut raw = None;
    for field in fields(message) {
        let field = field?;
        match field.number {
            TENSOR_RAW_DATA => raw = field.body,
            // Already external, wholly or in part: leave it alone.
            TENSOR_EXTERNAL_DATA => return Ok(None),
            TENSOR_DATA_LOCATION if field.varint == LOCATION_EXTERNAL => return Ok(None),
            _ => {}
        }
    }
    let Some(raw) = raw.filter(|raw| raw.len() >= THRESHOLD) else {
        return Ok(None);
    };
    let offset = sink.append(raw)?;

    let mut out = Vec::with_capacity(message.len() + 96);
    for field in fields(message) {
        let field = field?;
        // The weights go, and so does an explicit `data_location = DEFAULT`:
        // the message gets exactly one such field, the EXTERNAL one below.
        if field.number != TENSOR_RAW_DATA && field.number != TENSOR_DATA_LOCATION {
            out.extend_from_slice(field.whole);
        }
    }
    for (key, value) in [
        ("location", location.to_string()),
        ("offset", offset.to_string()),
        ("length", raw.len().to_string()),
    ] {
        let mut entry = Vec::new();
        put_bytes(&mut entry, 1, key.as_bytes());
        put_bytes(&mut entry, 2, value.as_bytes());
        put_bytes(&mut out, TENSOR_EXTERNAL_DATA, &entry);
    }
    put_number(&mut out, TENSOR_DATA_LOCATION, LOCATION_EXTERNAL);
    Ok(Some(out))
}

/// `model` with its large initializers written to `sink` and replaced by
/// references to `location`, with how many it moved, or `None` if there was
/// nothing to move.
fn externalize<W: Write>(
    model: &[u8],
    location: &str,
    sink: &mut Sink<W>,
) -> Result<Option<(Vec<u8>, usize)>> {
    let mut moved = 0usize;
    let mut out = Vec::new();
    for field in fields(model) {
        let field = field?;
        let (MODEL_GRAPH, Some(graph)) = (field.number, field.body) else {
            out.extend_from_slice(field.whole);
            continue;
        };
        let mut rewritten = Vec::with_capacity(graph.len().min(1 << 20));
        for inner in fields(graph) {
            let inner = inner?;
            let moved_tensor = match (inner.number, inner.body) {
                (GRAPH_INITIALIZER, Some(message)) => tensor(message, location, sink)?,
                _ => None,
            };
            match moved_tensor {
                Some(message) => {
                    moved += 1;
                    put_bytes(&mut rewritten, GRAPH_INITIALIZER, &message);
                }
                None => rewritten.extend_from_slice(inner.whole),
            }
        }
        put_bytes(&mut out, MODEL_GRAPH, &rewritten);
    }
    Ok((moved > 0).then_some((out, moved)))
}

/// Where each external file a graph names ends: the largest offset + length
/// among the tensors that live in it.
fn sidecar_ends(model: &[u8]) -> Result<BTreeMap<String, u64>> {
    let mut ends = BTreeMap::new();
    for field in fields(model) {
        let field = field?;
        let (MODEL_GRAPH, Some(graph)) = (field.number, field.body) else {
            continue;
        };
        for inner in fields(graph) {
            let inner = inner?;
            let (GRAPH_INITIALIZER, Some(message)) = (inner.number, inner.body) else {
                continue;
            };
            let (mut location, mut offset, mut length) = (None, 0u64, 0u64);
            for field in fields(message) {
                let field = field?;
                let (TENSOR_EXTERNAL_DATA, Some(entry)) = (field.number, field.body) else {
                    continue;
                };
                let (mut key, mut value) = (&[][..], &[][..]);
                for part in fields(entry) {
                    let part = part?;
                    match part.number {
                        1 => key = part.body.unwrap_or_default(),
                        2 => value = part.body.unwrap_or_default(),
                        _ => {}
                    }
                }
                let text = std::str::from_utf8(value).context("external_data value")?;
                match key {
                    b"location" => location = Some(text.to_string()),
                    b"offset" => offset = text.parse().context("external_data offset")?,
                    b"length" => length = text.parse().context("external_data length")?,
                    _ => {}
                }
            }
            if let Some(location) = location {
                let end = offset.saturating_add(length);
                let slot = ends.entry(location).or_insert(0);
                *slot = (*slot).max(end);
            }
        }
    }
    Ok(ends)
}

/// The tag that names one attempt and its claim.
fn new_tag() -> String {
    format!("{ATTEMPT}{}-{}", std::process::id(), ulid::Ulid::new())
}

/// A claim on converting one model, so two processes do not both write 569 MB.
/// Released when dropped. It is advisory: see the module notes.
pub struct Claim {
    path: PathBuf,
    tag: String,
}

impl Claim {
    fn take(path: PathBuf) -> Option<Self> {
        let tag = new_tag();
        for _ in 0..2 {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    let _ = file.write_all(tag.as_bytes());
                    return Some(Self { path, tag });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    let age = fs::metadata(&path)
                        .and_then(|meta| meta.modified())
                        .ok()
                        .and_then(|modified| SystemTime::now().duration_since(modified).ok());
                    // An unreadable stamp, or one in the future, is a live
                    // claim: the clock moved, not the owner.
                    if !age.is_some_and(|age| age >= STALE_CLAIM) {
                        return None;
                    }
                    // A rename can succeed for one taker only, which a
                    // check-then-remove cannot promise.
                    let mut grave = path.clone().into_os_string();
                    grave.push(format!(".stale-{tag}"));
                    let grave = PathBuf::from(grave);
                    if fs::rename(&path, &grave).is_err() {
                        return None;
                    }
                    let _ = fs::remove_file(&grave);
                }
                Err(_) => return None,
            }
        }
        None
    }

    /// Say the owner is alive.
    fn touch(&self) {
        let _ = OpenOptions::new()
            .write(true)
            .open(&self.path)
            .and_then(|file| file.set_modified(SystemTime::now()));
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        // Only our own: a claim taken over from us while we ran is not ours.
        if fs::read_to_string(&self.path).is_ok_and(|text| text == self.tag) {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// A model file's size and modification time, as text. A conversion is
/// published only over the file it read, and remembered as declined only for
/// that file.
fn stamp(meta: &fs::Metadata) -> String {
    let nanos = meta
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |since| since.as_nanos());
    format!("{} {nanos}", meta.len())
}

fn declined(marker: &Path, stamp: &str) -> bool {
    let fresh = fs::metadata(marker)
        .and_then(|meta| meta.modified())
        .is_ok_and(|modified| modified.elapsed().map_or(true, |age| age < DECLINE_TTL));
    fresh && fs::read_to_string(marker).is_ok_and(|text| text == stamp)
}

fn write_synced(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

/// Make the renames in `dir` durable. Best effort, and only where a directory
/// can be opened.
fn sync_dir(dir: &Path) {
    if cfg!(unix) {
        if let Ok(handle) = File::open(dir) {
            let _ = handle.sync_all();
        }
    }
}

/// Leftovers of attempts that died, by their names.
///
/// Called only while the claim is held and the model is still embedded, so no
/// graph in use refers to any of them. A file younger than the claim's lifetime
/// is left, in case its owner is only slow.
fn sweep(dir: &Path, name: &str) {
    let prefix = format!("{name}.{ATTEMPT}");
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let file = entry.file_name();
        if !file.to_str().is_some_and(|file| file.starts_with(&prefix)) {
            continue;
        }
        let old = entry
            .metadata()
            .and_then(|meta| meta.modified())
            .ok()
            .and_then(|modified| SystemTime::now().duration_since(modified).ok())
            .is_some_and(|age| age >= STALE_CLAIM);
        if old {
            let _ = fs::remove_file(entry.path());
        }
    }
}

/// A converted copy written beside the model and not yet in its place.
///
/// The sidecar already has the name the graph gives it - a graph has to find
/// it to load - but nothing refers to that name until the graph is published,
/// and the sidecar is written once and never touched again, since a file a
/// process has mapped must not change under it.
pub struct Staged {
    model: PathBuf,
    /// The converted graph, under a name the model's own path is not.
    pub graph: PathBuf,
    data: PathBuf,
    marker: PathBuf,
    stamp: String,
    published: Cell<bool>,
    claim: Claim,
}

impl Staged {
    /// The converted graph takes the model's name, atomically - unless the
    /// model is no longer the file this conversion was made from.
    pub fn publish(&self) -> io::Result<()> {
        if stamp(&fs::metadata(&self.model)?) != self.stamp {
            return Err(io::Error::other("the model changed while it was being converted"));
        }
        fs::rename(&self.graph, &self.model)?;
        self.published.set(true);
        if let Some(dir) = self.model.parent() {
            sync_dir(dir);
        }
        Ok(())
    }

    /// Say the owner is still working.
    pub fn touch(&self) {
        self.claim.touch();
    }

    /// Give this attempt up: remove what it made, and remember that this model
    /// was not worth converting, so the next load does not try again. A
    /// published conversion is not taken back.
    pub fn discard(&self) {
        if self.published.get() {
            return;
        }
        let _ = fs::remove_file(&self.graph);
        let _ = fs::remove_file(&self.data);
        let _ = fs::write(&self.marker, &self.stamp);
    }
}

/// Convert `model` into a staged graph and sidecar, or `None` when it is
/// already converted, is being converted by someone else, was declined before,
/// or has nothing to move.
pub fn stage(model: &Path) -> Result<Option<Staged>> {
    stage_from(model, MIN_EMBEDDED)
}

pub(crate) fn stage_from(model: &Path, min_len: u64) -> Result<Option<Staged>> {
    // The common case, every load after the first: a small graph file, with no
    // claim taken and nothing read.
    if fs::metadata(model)?.len() < min_len {
        return Ok(None);
    }
    let name = model.file_name().and_then(|name| name.to_str()).context("model file name")?;
    let dir = model.parent().context("model directory")?;
    let marker = dir.join(format!("{name}.noconvert"));
    let Some(claim) = Claim::take(dir.join(format!("{name}.converting"))) else {
        return Ok(None);
    };
    // Looked at again now that nobody else is converting: the claim before ours
    // may have just published, and a published graph is not ours to sweep.
    let meta = fs::metadata(model)?;
    if meta.len() < min_len {
        return Ok(None);
    }
    let stamp = stamp(&meta);
    if declined(&marker, &stamp) {
        return Ok(None);
    }
    sweep(dir, name);

    let data_name = format!("{name}.{}.data", claim.tag);
    let data = dir.join(&data_name);
    let data_tmp = dir.join(format!("{data_name}.tmp"));
    let graph = dir.join(format!("{name}.{}.new", claim.tag));

    let written = write_copy(model, &claim, dir, &data_name, &data_tmp, &data, &graph);
    if !matches!(written, Ok(true)) {
        for path in [&data_tmp, &data, &graph] {
            let _ = fs::remove_file(path);
        }
        let _ = fs::write(&marker, &stamp);
        return written.map(|_| None);
    }
    Ok(Some(Staged {
        model: model.to_path_buf(),
        graph,
        data,
        marker,
        stamp,
        published: Cell::new(false),
        claim,
    }))
}

/// Write the sidecar and the converted graph. `false`: nothing to move.
fn write_copy(
    model: &Path,
    claim: &Claim,
    dir: &Path,
    data_name: &str,
    data_tmp: &Path,
    data: &Path,
    graph: &Path,
) -> Result<bool> {
    let bytes = fs::read(model).with_context(|| format!("read {}", model.display()))?;
    claim.touch();
    let mut sink = Sink { out: BufWriter::new(File::create(data_tmp)?), pos: 0 };
    let converted = externalize(&bytes, data_name, &mut sink);
    drop(bytes);
    let Some((converted, _)) = converted? else {
        return Ok(false);
    };
    sink.out.flush()?;
    let file = sink.out.get_ref();
    file.sync_all()?;
    let on_disk = file.metadata()?.len();
    ensure!(on_disk == sink.pos, "the sidecar holds {on_disk} bytes of the {} written", sink.pos);
    claim.touch();

    fs::rename(data_tmp, data)?;
    write_synced(graph, &converted)?;
    sync_dir(dir);
    claim.touch();
    Ok(true)
}

/// A graph whose weights live in a sidecar of ours.
#[derive(Debug)]
pub struct Converted {
    /// The sidecar files the graph names.
    sidecars: Vec<PathBuf>,
    /// What is wrong with the pair, if anything: the sidecar is gone, or is not
    /// the size the graph says it is.
    flaw: Option<String>,
}

/// Is `model` a small converted graph, and is its sidecar there in full?
///
/// `None` for anything that is not one: an embedded model, or a graph that
/// names no sidecar of ours. The reranker comes with one file, and when the
/// weights are inside it that file is the original and nothing here may
/// touch it.
fn inspect(model: &Path) -> Result<Option<Converted>> {
    let len = fs::metadata(model)?.len();
    if len >= MIN_EMBEDDED {
        return Ok(None);
    }
    if len == 0 {
        return Ok(Some(Converted { sidecars: Vec::new(), flaw: Some("the file is empty".into()) }));
    }
    let name = model.file_name().and_then(|name| name.to_str()).context("model file name")?;
    let dir = model.parent().context("model directory")?;
    let bytes = fs::read(model)?;
    // A file that is not protobuf is not a graph this module wrote.
    let Ok(ends) = sidecar_ends(&bytes) else {
        return Ok(None);
    };
    let ours = |location: &str| {
        location.starts_with(&format!("{name}."))
            && location.ends_with(".data")
            && !location.contains(['/', '\\'])
            && !location.contains("..")
    };
    if ends.is_empty() || !ends.keys().all(|location| ours(location)) {
        return Ok(None);
    }
    let mut flaw = None;
    let mut sidecars = Vec::new();
    for (location, end) in &ends {
        let path = dir.join(location);
        match fs::metadata(&path) {
            Err(_) => flaw = Some(format!("the weights file {location} is missing")),
            Ok(meta) if meta.len() != *end => {
                flaw = Some(format!("{location} holds {} bytes and the graph needs {end}", meta.len()));
            }
            Ok(_) => {}
        }
        sidecars.push(path);
    }
    Ok(Some(Converted { sidecars, flaw }))
}

/// `Err` when `model` is a converted graph whose sidecar is missing or the wrong
/// size, which no amount of trying to load it will fix.
pub fn require_sidecar(model: &Path) -> Result<()> {
    match inspect(model)? {
        Some(Converted { flaw: Some(flaw), .. }) => bail!("{flaw}"),
        _ => Ok(()),
    }
}

/// Remove a converted graph and the sidecars it names, so that the next fetch
/// finds no `model.onnx` and downloads it again. Returns whether it did.
///
/// A model that is not a converted graph of ours is never removed: that is the
/// original, and the only copy there is.
pub fn forget_converted(model: &Path) -> bool {
    let Ok(Some(converted)) = inspect(model) else {
        return false;
    };
    // The graph first, so the pair stops looking ready before anything else goes.
    if fs::remove_file(model).is_err() {
        return false;
    }
    for sidecar in &converted.sidecars {
        let _ = fs::remove_file(sidecar);
    }
    true
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn tensor_message(name: &str, raw: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        put_number(&mut out, 1, 2);
        put_number(&mut out, 1, 3);
        put_number(&mut out, 2, 3);
        put_bytes(&mut out, 8, name.as_bytes());
        put_bytes(&mut out, TENSOR_RAW_DATA, raw);
        // The real model writes both: an explicit DEFAULT location and a doc string.
        put_number(&mut out, TENSOR_DATA_LOCATION, 0);
        put_bytes(&mut out, 12, b"a doc string");
        out
    }

    /// A model in the shape the real one has: header fields, a graph holding a
    /// node, large and small initializers, and a trailing model field.
    fn model(tensors: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut graph = Vec::new();
        put_bytes(&mut graph, 1, b"a node the converter must not touch");
        for (name, raw) in tensors {
            put_bytes(&mut graph, GRAPH_INITIALIZER, &tensor_message(name, raw));
        }
        put_bytes(&mut graph, 2, b"graph name");
        let mut out = Vec::new();
        put_number(&mut out, 1, 8);
        put_bytes(&mut out, 2, b"producer");
        put_bytes(&mut out, MODEL_GRAPH, &graph);
        put_bytes(&mut out, 14, b"metadata after the graph");
        out
    }

    /// An embedded model for a test elsewhere to convert.
    pub(crate) fn sample_model() -> Vec<u8> {
        model(&[("w", pattern(9000, 4)), ("b", pattern(4, 5))])
    }

    fn pattern(len: usize, seed: u8) -> Vec<u8> {
        (0..len).map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed)).collect()
    }

    pub(crate) fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-onnx-{tag}-{}", ulid::Ulid::new()));
        fs::create_dir_all(&dir).expect("create");
        dir
    }

    pub(crate) fn names(dir: &Path) -> Vec<String> {
        let mut left: Vec<_> = fs::read_dir(dir)
            .expect("dir")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        left
    }

    /// The only `.data` file in `dir`.
    pub(crate) fn sidecar_of(dir: &Path) -> PathBuf {
        let found: Vec<_> = names(dir).into_iter().filter(|n| n.ends_with(".data")).collect();
        assert_eq!(found.len(), 1, "expected one sidecar in {found:?}");
        dir.join(&found[0])
    }

    /// The external_data entries of a tensor as (key, value) text, in order.
    fn references(tensor: &[u8]) -> Vec<(String, String)> {
        let text = |bytes: &[u8]| String::from_utf8(bytes.to_vec()).expect("utf8");
        fields(tensor)
            .map(|f| f.expect("field"))
            .filter(|f| f.number == TENSOR_EXTERNAL_DATA)
            .map(|f| {
                let mut entry = fields(f.body.expect("entry")).map(|e| e.expect("entry field"));
                let key = text(entry.next().expect("key").body.expect("key text"));
                let value = text(entry.next().expect("value").body.expect("value text"));
                (key, value)
            })
            .collect()
    }

    fn initializers(model: &[u8]) -> Vec<Vec<u8>> {
        let graph = fields(model)
            .map(|f| f.expect("field"))
            .find(|f| f.number == MODEL_GRAPH)
            .and_then(|f| f.body)
            .expect("graph");
        fields(graph)
            .map(|f| f.expect("field"))
            .filter(|f| f.number == GRAPH_INITIALIZER)
            .map(|f| f.body.expect("tensor").to_vec())
            .collect()
    }

    /// A converted pair on disk: the graph published as `model.onnx`.
    fn converted_pair(dir: &Path) -> PathBuf {
        let path = dir.join("model.onnx");
        fs::write(&path, sample_model()).expect("write");
        let staged = stage_from(&path, 0).expect("stage").expect("staged");
        staged.publish().expect("publish");
        path
    }

    #[test]
    fn large_weights_move_to_the_sidecar_and_everything_else_stays_byte_for_byte() {
        let (big, huge, small) = (pattern(5000, 1), pattern(2048, 2), pattern(8, 3));
        let source = model(&[("w1", big.clone()), ("bias", small.clone()), ("w2", huge.clone())]);

        let mut sink = Sink { out: Vec::new(), pos: 0 };
        let (converted, moved) =
            externalize(&source, "model.onnx.data", &mut sink).expect("convert").expect("moved");
        assert_eq!(moved, 2);

        let before = initializers(&source);
        let after = initializers(&converted);
        assert_eq!(after.len(), 3);
        assert_eq!(after[1], before[1], "a small tensor is copied untouched");

        for (index, raw) in [(0, &big), (2, &huge)] {
            let tensor = &after[index];
            assert!(
                fields(tensor).all(|f| f.expect("field").number != TENSOR_RAW_DATA),
                "tensor {index} still carries its weights"
            );
            let locations: Vec<_> = fields(tensor)
                .map(|f| f.expect("field"))
                .filter(|f| f.number == TENSOR_DATA_LOCATION)
                .collect();
            assert_eq!(locations.len(), 1, "tensor {index} has {} data_location fields", locations.len());
            assert_eq!(locations[0].varint, LOCATION_EXTERNAL, "tensor {index} is not marked external");
            let refs = references(tensor);
            assert_eq!(refs[0], ("location".into(), "model.onnx.data".into()));
            let offset: usize = refs[1].1.parse().expect("offset");
            let length: usize = refs[2].1.parse().expect("length");
            assert_eq!((refs[1].0.as_str(), refs[2].0.as_str()), ("offset", "length"));
            assert_eq!(offset % ALIGN as usize, 0, "offset {offset} is not aligned");
            assert_eq!(&sink.out[offset..offset + length], raw.as_slice());
            // Name, dims and the doc string survive: only the weights and the
            // location bookkeeping differ.
            let kept: Vec<_> = fields(&before[index])
                .map(|f| f.expect("field"))
                .filter(|f| f.number != TENSOR_RAW_DATA && f.number != TENSOR_DATA_LOCATION)
                .map(|f| f.whole.to_vec())
                .collect();
            let head: Vec<_> = fields(tensor)
                .map(|f| f.expect("field"))
                .take(kept.len())
                .map(|f| f.whole.to_vec())
                .collect();
            assert_eq!(head, kept);
            assert!(kept.iter().any(|f| f.starts_with(&[12 << 3 | 2])), "the fixture carries a doc string");
        }

        // Header, node, graph name and the field after the graph are unchanged.
        let strip = |m: &[u8]| -> Vec<Vec<u8>> {
            fields(m).map(|f| f.expect("field")).filter(|f| f.number != MODEL_GRAPH).map(|f| f.whole.to_vec()).collect()
        };
        assert_eq!(strip(&converted), strip(&source));
        let graph_rest = |m: &[u8]| -> Vec<Vec<u8>> {
            let graph = fields(m).map(|f| f.expect("field")).find(|f| f.number == MODEL_GRAPH).and_then(|f| f.body).expect("graph");
            fields(graph).map(|f| f.expect("field")).filter(|f| f.number != GRAPH_INITIALIZER).map(|f| f.whole.to_vec()).collect()
        };
        assert_eq!(graph_rest(&converted), graph_rest(&source));
    }

    #[test]
    fn a_converted_model_has_nothing_left_to_move() {
        let source = model(&[("w", pattern(4096, 9))]);
        let mut sink = Sink { out: Vec::new(), pos: 0 };
        let (converted, _) = externalize(&source, "d", &mut sink).expect("convert").expect("moved");
        let mut again = Sink { out: Vec::new(), pos: 0 };
        assert!(externalize(&converted, "d", &mut again).expect("second pass").is_none());
        assert!(again.out.is_empty());
    }

    #[test]
    fn a_model_that_is_not_protobuf_is_an_error_and_is_not_tried_again() {
        let dir = scratch("bad");
        let path = dir.join("model.onnx");
        let mut broken = model(&[("w", pattern(4096, 9))]);
        broken.truncate(broken.len() - 20);
        fs::write(&path, &broken).expect("write");

        assert!(stage_from(&path, 0).is_err());
        assert_eq!(fs::read(&path).expect("read"), broken);
        assert_eq!(
            names(&dir),
            ["model.onnx", "model.onnx.noconvert"],
            "a failed conversion leaves the model and a note that it failed"
        );
        assert!(stage_from(&path, 0).expect("remembered").is_none(), "the failure is not retried");
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_model_with_nothing_to_move_is_not_read_again() {
        let dir = scratch("nothing");
        let path = dir.join("model.onnx");
        fs::write(&path, model(&[("tiny", pattern(8, 1))])).expect("write");
        assert!(stage_from(&path, 0).expect("stage").is_none());
        assert_eq!(names(&dir), ["model.onnx", "model.onnx.noconvert"]);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_note_about_a_model_does_not_outlive_a_change_to_it() {
        let dir = scratch("changed");
        let path = dir.join("model.onnx");
        fs::write(&path, model(&[("tiny", pattern(8, 1))])).expect("write");
        assert!(stage_from(&path, 0).expect("declined").is_none());

        fs::write(&path, sample_model()).expect("a new model arrives");
        let staged = stage_from(&path, 0).expect("stage").expect("a different file is looked at again");
        staged.discard();
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn staging_leaves_the_model_alone_until_it_is_published() {
        let dir = scratch("stage");
        let path = dir.join("model.onnx");
        let source = sample_model();
        fs::write(&path, &source).expect("write");

        let staged = stage_from(&path, 0).expect("stage").expect("staged");
        assert_eq!(fs::read(&path).expect("read"), source, "staging must not touch the model");
        assert!(sidecar_of(&dir).is_file());
        assert!(dir.join("model.onnx.converting").exists(), "the claim is held while staged");
        assert!(!names(&dir).iter().any(|n| n.ends_with(".tmp")));
        let graph = fs::read(&staged.graph).expect("graph");
        assert!(graph.len() < source.len(), "the weights left the graph");

        staged.publish().expect("publish");
        drop(staged);
        assert_eq!(fs::read(&path).expect("read"), graph);
        assert!(!dir.join("model.onnx.converting").exists(), "the claim is released");
        assert_eq!(names(&dir).len(), 2, "the model and its sidecar: {:?}", names(&dir));

        // The next load finds nothing to do, and does not even take a claim.
        assert!(stage(&path).expect("restage").is_none());
        assert_eq!(names(&dir).len(), 2);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_published_conversion_is_not_undone_by_a_late_discard() {
        let dir = scratch("late");
        let path = dir.join("model.onnx");
        fs::write(&path, sample_model()).expect("write");
        let staged = stage_from(&path, 0).expect("stage").expect("staged");
        staged.publish().expect("publish");
        staged.discard();
        assert!(sidecar_of(&dir).is_file(), "a published conversion keeps its sidecar");
        assert!(inspect(&path).expect("inspect").is_some_and(|c| c.flaw.is_none()));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn two_conversions_at_once_never_touch_each_others_files() {
        let dir = scratch("overlap");
        let path = dir.join("model.onnx");
        fs::write(&path, sample_model()).expect("write");
        // A second converter that missed the claim (it was judged dead).
        let first = stage_from(&path, 0).expect("stage").expect("staged");
        fs::remove_file(dir.join("model.onnx.converting")).expect("lose the claim");
        let second = stage_from(&path, 0).expect("stage").expect("staged");
        assert_ne!(first.graph, second.graph);
        assert_ne!(first.data, second.data);

        first.publish().expect("the first publishes");
        second.publish().expect_err("the second finds the model already replaced");
        second.discard();
        assert!(first.data.is_file(), "the loser must not delete the winner's sidecar");
        assert!(!second.data.exists() && !second.graph.exists());
        assert!(inspect(&path).expect("inspect").is_some_and(|c| c.flaw.is_none()));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn discarding_a_staged_copy_restores_the_directory() {
        let dir = scratch("discard");
        let path = dir.join("model.onnx");
        let source = sample_model();
        fs::write(&path, &source).expect("write");

        let staged = stage_from(&path, 0).expect("stage").expect("staged");
        staged.discard();
        drop(staged);
        assert_eq!(names(&dir), ["model.onnx", "model.onnx.noconvert"]);
        assert_eq!(fs::read(&path).expect("read"), source);
        assert!(stage_from(&path, 0).expect("remembered").is_none(), "a discarded model is not retried");
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_small_graph_file_is_not_opened() {
        let dir = scratch("small");
        let path = dir.join("model.onnx");
        fs::write(&path, b"not even protobuf").expect("write");
        assert!(stage(&path).expect("size gate").is_none());
        assert_eq!(names(&dir), ["model.onnx"]);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn one_process_converts_at_a_time_and_a_dead_claim_is_taken_over() {
        let dir = scratch("claim");
        let path = dir.join("model.onnx");
        fs::write(&path, sample_model()).expect("write");
        let claim = dir.join("model.onnx.converting");

        fs::write(&claim, b"someone else").expect("claim");
        assert!(stage_from(&path, 0).expect("held").is_none(), "a fresh claim blocks a second conversion");
        assert!(claim.exists(), "someone else's claim is not ours to remove");

        // A clock set back makes a live claim look like it is from the future.
        let future = SystemTime::now() + Duration::from_secs(3600);
        File::options().write(true).open(&claim).expect("open").set_modified(future).expect("age");
        assert!(stage_from(&path, 0).expect("future").is_none(), "an odd stamp is a live claim");

        let old = SystemTime::now() - STALE_CLAIM - Duration::from_secs(60);
        File::options().write(true).open(&claim).expect("open").set_modified(old).expect("age");
        let staged = stage_from(&path, 0).expect("stale").expect("a dead claim does not block");
        staged.discard();
        drop(staged);
        assert!(!claim.exists(), "the taker releases the claim it took");
        assert!(!names(&dir).iter().any(|n| n.contains("stale-")), "the dead claim is gone");
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_claim_taken_over_from_us_is_not_released_by_us() {
        let dir = scratch("takeover");
        let claim_path = dir.join("model.onnx.converting");
        let ours = Claim::take(claim_path.clone()).expect("claim");
        fs::write(&claim_path, b"the new owner").expect("taken over");
        drop(ours);
        assert!(claim_path.exists());
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn leftovers_of_a_dead_attempt_are_swept_but_a_recent_one_is_left() {
        let dir = scratch("sweep");
        let path = dir.join("model.onnx");
        fs::write(&path, sample_model()).expect("write");
        let dead = dir.join("model.onnx.cvt-1-OLD.data.tmp");
        let slow = dir.join("model.onnx.cvt-2-NEW.data");
        let other = dir.join("model.onnx.data");
        for file in [&dead, &slow, &other] {
            fs::write(file, b"x").expect("write");
        }
        let old = SystemTime::now() - STALE_CLAIM - Duration::from_secs(60);
        File::options().write(true).open(&dead).expect("open").set_modified(old).expect("age");

        let staged = stage_from(&path, 0).expect("stage").expect("staged");
        assert!(!dead.exists(), "a dead attempt's leftover is removed");
        assert!(slow.exists(), "a recent one may belong to a slow converter");
        assert!(other.exists(), "a file that is not an attempt's is never swept");
        staged.discard();
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_whole_converted_pair_is_recognised_and_left_alone() {
        let dir = scratch("whole");
        let path = converted_pair(&dir);
        let found = inspect(&path).expect("inspect").expect("a converted graph");
        assert!(found.flaw.is_none());
        require_sidecar(&path).expect("whole");
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_converted_graph_without_its_sidecar_is_forgotten_whole() {
        let dir = scratch("gone");
        let path = converted_pair(&dir);
        let sidecar = sidecar_of(&dir);
        fs::remove_file(&sidecar).expect("lose the weights");

        let why = require_sidecar(&path).expect_err("not usable").to_string();
        assert!(why.contains("missing"), "{why}");
        assert!(forget_converted(&path));
        assert!(!path.exists(), "the small graph is gone, so the next fetch downloads it");
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_converted_graph_with_a_short_sidecar_is_forgotten_with_it() {
        let dir = scratch("short");
        let path = converted_pair(&dir);
        let sidecar = sidecar_of(&dir);
        let data = fs::read(&sidecar).expect("sidecar");
        fs::write(&sidecar, &data[..data.len() - 1]).expect("truncate");

        let why = require_sidecar(&path).expect_err("not usable").to_string();
        assert!(why.contains("holds"), "{why}");
        assert!(forget_converted(&path));
        assert!(!path.exists() && !sidecar.exists(), "{:?}", names(&dir));
        assert!(names(&dir).is_empty() || names(&dir) == ["model.onnx.noconvert"]);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn an_embedded_model_is_never_forgotten() {
        let dir = scratch("original");
        let path = dir.join("model.onnx");
        let source = sample_model();
        fs::write(&path, &source).expect("write");
        assert!(inspect(&path).expect("inspect").is_none());
        assert!(!forget_converted(&path));
        assert_eq!(fs::read(&path).expect("read"), source);

        // Nor a file that is not a model at all, nor a graph whose external
        // files are not ours: those are someone else's to judge.
        fs::write(&path, b"not protobuf").expect("write");
        assert!(!forget_converted(&path));
        let mut sink = Sink { out: Vec::new(), pos: 0 };
        for location in ["elsewhere.bin", "../model.onnx.data", "model.onnx.sub/x.data"] {
            let (foreign, _) = externalize(&source, location, &mut sink).expect("convert").expect("moved");
            fs::write(&path, &foreign).expect("write");
            assert!(inspect(&path).expect("inspect").is_none(), "{location}");
            assert!(!forget_converted(&path), "{location}");
            assert!(path.exists());
        }
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn an_empty_model_file_is_forgotten() {
        let dir = scratch("empty");
        let path = dir.join("model.onnx");
        fs::write(&path, b"").expect("write");
        assert!(forget_converted(&path));
        assert!(!path.exists());
        fs::remove_dir_all(dir).ok();
    }
}
