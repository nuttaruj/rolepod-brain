//! The wire between a brain process and the hub: newline-delimited JSON, one
//! message per line, a line never longer than [`MAX_LINE`].
//!
//! Everything read off the socket is data. A line is read through a bounded
//! reader, parsed, and only then looked at; nothing from it is used to index,
//! allocate or open a path before the caps in this file have passed.

use std::io::{self, BufRead, Read, Write};

use serde::{Deserialize, Serialize};

/// The protocol this build speaks. A hub also answers [`PROTO`] - 1.
pub const PROTO: u32 = 1;
/// The longest line, either way: 1 MiB, newline included.
pub const MAX_LINE: usize = 1 << 20;
/// The longest `build` a Hello may carry.
pub const MAX_BUILD: usize = 128;
/// Caps on one rerank request. Over any of them the answer is `Unavailable`.
pub const MAX_ENTRIES: usize = 64;
pub const MAX_ENTRY_BYTES: usize = 1024;
pub const MAX_QUERY_BYTES: usize = 4096;

/// What a client sends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    /// `client` is the asking process's pid, for the hub's session count only.
    /// It is data from the wire: never signalled, never opened.
    Hello {
        proto: u32,
        build: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        client: Option<u32>,
    },
    Rerank { id: u64, query: String, entries: Vec<String>, deadline_ms: u64 },
    Status,
    /// Ask an older hub to finish its work and go. Only a strictly newer
    /// build is obeyed.
    Retire,
    Stop,
}

/// What the hub answers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Reply {
    Welcome { proto: u32, build: String, pid: u32 },
    Refused { reason: String },
    /// Indexes into the request's entries, best first.
    Order { id: u64, indices: Vec<usize> },
    /// The request could not start before its deadline.
    Busy { id: u64 },
    Unavailable { id: u64 },
    Status {
        pid: u32,
        build: String,
        proto: u32,
        uptime_ms: u64,
        /// Resident memory in KiB, `None` when it could not be sampled.
        footprint_kb: Option<u64>,
        clients: usize,
        /// Distinct client pids that said Hello in the last ten minutes.
        #[serde(default)]
        sessions: u32,
        queued: usize,
        model_loaded: bool,
        retiring: bool,
    },
    Ack,
}

/// Why a line could not be read.
#[derive(Debug)]
pub enum LineError {
    TooLong,
    NotUtf8,
    /// A read error that is not a timeout.
    Io,
    /// The read timeout ran out before a line arrived.
    TimedOut,
}

/// The next line, without its newline. `Ok(None)` at end of stream, and for a
/// last line that never got its newline: half a message is not a message.
///
/// At most [`MAX_LINE`] + 1 bytes are ever buffered.
///
/// # Errors
/// [`LineError::TooLong`] past the cap, [`LineError::NotUtf8`] for bytes that
/// are not text, [`LineError::Io`] for a read error, [`LineError::TimedOut`] for a timeout.
pub fn read_line<R: BufRead>(reader: &mut R) -> Result<Option<String>, LineError> {
    let mut buf = Vec::new();
    let limit = u64::try_from(MAX_LINE).unwrap_or(u64::MAX) + 1;
    reader.by_ref().take(limit).read_until(b'\n', &mut buf).map_err(|error| match error.kind() {
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock => LineError::TimedOut,
        _ => LineError::Io,
    })?;
    if buf.is_empty() {
        return Ok(None);
    }
    if buf.last() != Some(&b'\n') {
        return if buf.len() > MAX_LINE { Err(LineError::TooLong) } else { Ok(None) };
    }
    if buf.len() > MAX_LINE {
        return Err(LineError::TooLong);
    }
    buf.pop();
    String::from_utf8(buf).map(Some).map_err(|_| LineError::NotUtf8)
}

/// Write one message and its newline.
///
/// # Errors
/// A write error, or a message over the line cap (which is never sent).
pub fn write_line<W: Write, T: Serialize>(writer: &mut W, message: &T) -> io::Result<()> {
    let mut bytes = serde_json::to_vec(message).map_err(io::Error::other)?;
    if bytes.len() >= MAX_LINE {
        return Err(io::Error::other("message over the line cap"));
    }
    bytes.push(b'\n');
    writer.write_all(&bytes)?;
    writer.flush()
}

/// Does a hub whose newest protocol is `current` answer a client at `proto`?
#[must_use]
pub fn accepts(proto: u32, current: u32) -> bool {
    proto >= 1 && proto <= current && proto + 1 >= current
}

/// A build string is data: bounded, printable.
#[must_use]
pub fn build_ok(build: &str) -> bool {
    !build.is_empty() && build.len() <= MAX_BUILD && build.chars().all(|c| !c.is_control())
}

/// `major.minor.patch` of a build, for ordering. Anything else is `None`.
fn version(build: &str) -> Option<(u64, u64, u64)> {
    let core = build.split(['-', '+']).next()?;
    let mut parts = core.split('.').map(str::parse::<u64>);
    let triple = (parts.next()?.ok()?, parts.next()?.ok()?, parts.next()?.ok()?);
    parts.next().is_none().then_some(triple)
}

/// Is `theirs` strictly newer than `ours`? An unreadable build is never newer.
#[must_use]
pub fn newer_build(theirs: &str, ours: &str) -> bool {
    match (version(theirs), version(ours)) {
        (Some(a), Some(b)) => a > b,
        _ => false,
    }
}

/// Is a request within the caps? Over any one of them it is not run.
#[must_use]
pub fn within_caps(query: &str, entries: &[String]) -> bool {
    query.len() <= MAX_QUERY_BYTES
        && entries.len() <= MAX_ENTRIES
        && entries.iter().all(|entry| entry.len() <= MAX_ENTRY_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn proto_table() {
        // The current protocol and the one before it; nothing newer, nothing 0.
        assert!(accepts(1, 1));
        assert!(!accepts(2, 1));
        assert!(!accepts(0, 1));
        assert!(accepts(2, 2) && accepts(1, 2));
        assert!(!accepts(1, 3) && !accepts(4, 3));
    }

    #[test]
    fn line_over_1mib_rejected_bounded_alloc() {
        let mut data = vec![b'x'; MAX_LINE + 10];
        data.push(b'\n');
        let mut reader = Cursor::new(data);
        assert!(matches!(read_line(&mut reader), Err(LineError::TooLong)));
        // Only the cap + 1 was consumed: the rest was never buffered.
        assert_eq!(reader.position(), (MAX_LINE + 1) as u64);
        // Exactly at the cap, newline included, is fine.
        let mut ok = vec![b'x'; MAX_LINE - 1];
        ok.push(b'\n');
        assert!(matches!(read_line(&mut Cursor::new(ok)), Ok(Some(line)) if line.len() == MAX_LINE - 1));
    }

    #[test]
    fn invalid_json_rejected() {
        assert!(serde_json::from_str::<Request>("{not json").is_err());
        let mut bad = Cursor::new(vec![0xff, 0xfe, b'\n']);
        assert!(matches!(read_line(&mut bad), Err(LineError::NotUtf8)));
    }

    #[test]
    fn no_newline_eof_discarded() {
        assert!(matches!(read_line(&mut Cursor::new(b"{\"type\":\"status\"}".to_vec())), Ok(None)));
        assert!(matches!(read_line(&mut Cursor::new(Vec::new())), Ok(None)));
    }

    #[test]
    fn build_string_bounded() {
        assert!(build_ok("0.68.0"));
        assert!(!build_ok(""));
        assert!(!build_ok(&"9".repeat(MAX_BUILD + 1)));
        assert!(!build_ok("0.1.0\n../../etc"));
    }

    #[test]
    fn unknown_variant_no_panic() {
        assert!(serde_json::from_str::<Request>(r#"{"type":"format_disk"}"#).is_err());
        assert!(serde_json::from_str::<Request>(r#"{"nope":1}"#).is_err());
        // Unknown fields on a known variant are ignored.
        let hello: Request = serde_json::from_str(r#"{"type":"hello","proto":1,"build":"x","extra":[1]}"#).unwrap();
        assert_eq!(hello, Request::Hello { proto: 1, build: "x".into(), client: None });
    }

    #[test]
    fn hello_client_is_optional_both_ways() {
        let new: Request = serde_json::from_str(r#"{"type":"hello","proto":1,"build":"x","client":42}"#).unwrap();
        assert_eq!(new, Request::Hello { proto: 1, build: "x".into(), client: Some(42) });
        // An old client sends none; a new client without a pid sends none either,
        // so an old hub never sees a field it does not know.
        let sent = serde_json::to_string(&Request::Hello { proto: 1, build: "x".into(), client: None }).unwrap();
        assert!(!sent.contains("client"), "{sent}");
    }

    #[test]
    fn status_sessions_is_optional() {
        let old = r#"{"type":"status","pid":1,"build":"x","proto":1,"uptime_ms":0,"footprint_kb":null,"clients":2,"queued":0,"model_loaded":false,"retiring":false}"#;
        let Reply::Status { sessions, clients, .. } = serde_json::from_str::<Reply>(old).unwrap() else { panic!("not a status") };
        assert_eq!((sessions, clients), (0, 2));
        let new = old.replace(r#""clients":2"#, r#""clients":2,"sessions":3"#);
        let Reply::Status { sessions, .. } = serde_json::from_str::<Reply>(&new).unwrap() else { panic!("not a status") };
        assert_eq!(sessions, 3);
    }

    #[test]
    fn newer_build_is_strict() {
        assert!(newer_build("0.68.1", "0.68.0"));
        assert!(newer_build("1.0.0", "0.99.9"));
        assert!(!newer_build("0.68.0", "0.68.0"));
        assert!(!newer_build("0.67.9", "0.68.0"));
        assert!(!newer_build("garbage", "0.68.0"));
        assert!(!newer_build("0.69.0", "garbage"));
    }

    #[test]
    fn caps() {
        assert!(within_caps("q", &["a".into(), "b".into()]));
        assert!(!within_caps(&"q".repeat(MAX_QUERY_BYTES + 1), &[]));
        assert!(!within_caps("q", &vec!["a".to_string(); MAX_ENTRIES + 1]));
        assert!(!within_caps("q", &["a".repeat(MAX_ENTRY_BYTES + 1)]));
    }

    #[test]
    fn round_trip() {
        let mut out = Vec::new();
        write_line(&mut out, &Reply::Order { id: 7, indices: vec![1, 0] }).unwrap();
        let line = read_line(&mut Cursor::new(out)).unwrap().unwrap();
        assert_eq!(serde_json::from_str::<Reply>(&line).unwrap(), Reply::Order { id: 7, indices: vec![1, 0] });
    }
}
