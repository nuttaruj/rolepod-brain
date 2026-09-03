//! Reading the host CLI's transcript at consolidation time.
//!
//! The richest material a session produces is the model's own prose — the
//! reasoning it wrote, the decision it explained, the dead end it described.
//! Lifecycle hooks never see any of it; they see tool calls and prompts.
//!
//! The transcript already exists on disk, written by the host CLI for its own
//! purposes. So we read it when we consolidate, feed it to the summarizer
//! beside the captured events, and persist only the summary.
//!
//! One exception, added deliberately: [`last_answer`] keeps the model's most
//! recent reply, bounded at [`ANSWER_MAX_BYTES`] and scrubbed. A session that
//! ends mid-task leaves its tool calls and the user's questions in the log,
//! and what is missing is what the model already said - which is what makes
//! the next session ask the same question again. Nothing else here is stored:
//! not the user's turns, which capture already has verbatim, and not the
//! history, which consolidation reads when it needs it.
//!
//! Everything read is sanitized before it reaches the summarizer, because a
//! transcript contains whatever the user typed — including the secrets our
//! capture path would have scrubbed on the way in.

use std::path::Path;

use crate::sanitize::Sanitizer;

/// Most transcript text to feed one consolidation.
///
/// Transcripts are large — a working day on this machine produced 6 MB — and
/// the prompt budget is a fraction of that. We take the tail, because the tail
/// is what has not been consolidated yet.
pub const SPAN_MAX_BYTES: usize = 12 * 1024;

/// How much of a transcript's end to read when looking for the last answer.
///
/// One turn's worth, generously: an assistant message is a few KB and the
/// tool-call lines around it a few more. A turn whose answer does not fit is
/// one this declines to store rather than one it reads a whole file for -
/// the hook it runs in is measured in milliseconds.
const TAIL_BYTES: u64 = 256 * 1024;

/// Read the last `bytes` of a file as UTF-8, dropping the line the window cut.
///
/// Only when it cut one. A file shorter than the window was read whole and
/// its first line is a whole line - dropping it there loses the only content
/// a short transcript has, which is what a test caught.
fn read_tail(path: &Path, bytes: u64) -> Option<String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    let from = len.saturating_sub(bytes);
    file.seek(SeekFrom::Start(from)).ok()?;
    let mut raw = Vec::with_capacity(usize::try_from(len - from).unwrap_or(0));
    file.read_to_end(&mut raw).ok()?;
    // A window into the middle of a file lands mid-character as easily as
    // mid-line; both are the same repair.
    let text = String::from_utf8_lossy(&raw).into_owned();
    if from == 0 {
        return Some(text);
    }
    text.find('\n').map(|at| text[at + 1..].to_string())
}

/// Most of the model's last answer to keep as a memory.
///
/// Small on purpose, and a different budget from [`SPAN_MAX_BYTES`]: that one
/// feeds a summarizer once and is thrown away, this one is stored and spends
/// primer bytes in every session that follows. Enough to recognise an answer
/// already given; not enough to re-read the conversation.
pub const ANSWER_MAX_BYTES: usize = 600;

/// The model's most recent answer, sanitized and bounded.
///
/// The one part of a transcript worth keeping rather than reading once. A
/// session that ends mid-task leaves its tool calls and the user's questions
/// in the log, and the thing that is missing is what the model already said -
/// which is exactly what makes the next session ask again.
///
/// Everything else in the transcript stays unread and unstored: not the
/// user's turns, which capture already has verbatim, and not the history,
/// which consolidation reads when it needs it.
///
/// Returns `None` when there is no transcript, no assistant prose in it, or
/// nothing left after sanitizing.
#[must_use]
pub fn last_answer(path: &Path, cli: &str, sanitizer: &Sanitizer) -> Option<String> {
    // The tail, not the file. `stop` fires once a turn and is a path someone
    // waits on, and a transcript grows all session: this one reached 43 MB in
    // a day, and reading it whole to find its last line cost 80ms where the
    // rest of the hook costs 10. The tail is a fixed read no matter how long
    // the session runs.
    let raw = read_tail(path, TAIL_BYTES)?;
    let mut lines: Vec<Line> = Vec::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        match cli {
            "codex" => extract_codex(&value, &mut lines),
            _ => extract_claude(&value, &mut lines),
        }
    }
    let answer = lines.iter().rev().find(|line| line.speaker == "assistant")?;
    let text = sanitizer.scrub(answer.text.trim());
    let text = crate::sanitize::truncate(text.trim(), ANSWER_MAX_BYTES);
    (!text.is_empty()).then_some(text)
}

/// One line of extracted prose.
struct Line {
    speaker: &'static str,
    text: String,
}

/// Read the most recent prose from a session transcript.
///
/// Returns `None` when the file is missing, unreadable, or contains nothing we
/// recognize. Every one of those is expected rather than exceptional: host CLIs
/// clean transcripts up on their own schedule (Claude Code defaults to 30 days),
/// and three of the five CLIs we support write no transcript at all.
#[must_use]
pub fn read_span(path: &Path, cli: &str, sanitizer: &Sanitizer) -> Option<String> {
    let raw = std::fs::read_to_string(path).ok()?;

    let mut lines: Vec<Line> = Vec::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        match cli {
            "codex" => extract_codex(&value, &mut lines),
            _ => extract_claude(&value, &mut lines),
        }
    }
    if lines.is_empty() {
        return None;
    }

    // Walk backwards so the tail survives the budget, then restore order.
    let mut kept: Vec<String> = Vec::new();
    let mut size = 0usize;
    for line in lines.iter().rev() {
        let rendered = format!("{}: {}\n", line.speaker, line.text.trim());
        if size + rendered.len() > SPAN_MAX_BYTES {
            // The newest line alone can be over budget - an 8k-character
            // answer in a three-byte script is 13 KB - and it is the line the
            // span exists for. Its head is kept, cut to the budget, rather
            // than the whole span given up.
            if kept.is_empty() {
                kept.push(crate::sanitize::truncate(&rendered, SPAN_MAX_BYTES));
            }
            break;
        }
        size += rendered.len();
        kept.push(rendered);
    }
    if kept.is_empty() {
        return None;
    }
    kept.reverse();

    // Sanitize once, over the assembled span: a transcript holds whatever the
    // user typed, and none of it went through our capture-time scrub.
    Some(sanitizer.scrub(&kept.concat()))
}

/// Claude Code: `{"type": "assistant"|"user", "message": {"content": [...]}}`.
///
/// The bulk of a Claude Code transcript is `attachment` lines — 1,573 of them
/// against 747 assistant turns in a real day's file — which carry file contents
/// the events already reference. Only prose is worth the prompt budget.
fn extract_claude(value: &serde_json::Value, out: &mut Vec<Line>) {
    // Cursor writes the same `message.content[]` blocks but names the speaker
    // `role`; Claude Code calls it `type`. One extractor covers both.
    let speaker = match value
        .get("type")
        .or_else(|| value.get("role"))
        .and_then(serde_json::Value::as_str)
    {
        Some("assistant") => "assistant",
        Some("user") => "user",
        _ => return,
    };

    let content = &value["message"]["content"];
    if let Some(text) = content.as_str() {
        push_line(out, speaker, text);
        return;
    }
    for block in content.as_array().into_iter().flatten() {
        if block.get("type").and_then(serde_json::Value::as_str) == Some("text") {
            if let Some(text) = block.get("text").and_then(serde_json::Value::as_str) {
                push_line(out, speaker, text);
            }
        }
    }
}

/// Codex: `{"payload": {"type": "message"|"agent_message"|"reasoning", …}}`.
///
/// `reasoning` is the reason this is worth doing at all — it is the model's
/// own account of why it did something, which no lifecycle hook can observe.
fn extract_codex(value: &serde_json::Value, out: &mut Vec<Line>) {
    let payload = &value["payload"];
    let speaker = match payload.get("type").and_then(serde_json::Value::as_str) {
        // A `message` is whoever its `role` says. The rollout Codex writes
        // from its app server (VS Code, Codex Desktop) carries the model's
        // answers only this way - no `agent_message` event at all, 0 in a 9 MB
        // session - so reading every message as the user's left those
        // sessions with no assistant prose: every `stop` a bare "Turn
        // finished", and the final verdict of a comparison missing from the
        // span that was meant to carry it. `developer` messages are the host's
        // own injections (a memory block, hook output), not the session.
        Some("message") => match payload.get("role").and_then(serde_json::Value::as_str) {
            Some("assistant") => "assistant",
            Some("developer" | "system") => return,
            _ => "user",
        },
        Some("agent_message") => "assistant",
        Some("reasoning") => "reasoning",
        _ => return,
    };

    for key in ["text", "message", "content", "summary"] {
        match &payload[key] {
            serde_json::Value::String(text) => push_line(out, speaker, text),
            serde_json::Value::Array(blocks) => {
                for block in blocks {
                    if let Some(text) = block.as_str() {
                        push_line(out, speaker, text);
                    } else if let Some(text) = block.get("text").and_then(serde_json::Value::as_str)
                    {
                        push_line(out, speaker, text);
                    }
                }
            }
            _ => {}
        }
    }
}

fn push_line(out: &mut Vec<Line>, speaker: &'static str, text: &str) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    // `codex exec` writes each answer twice - once as the `message` item and
    // once as the `agent_message` event - and both now read as the assistant.
    if out.last().is_some_and(|last| last.speaker == speaker && last.text == text) {
        return;
    }
    out.push(Line { speaker, text: text.to_string() });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(name: &str, body: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-transcript-{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn claude_prose_is_extracted_and_attachments_ignored() {
        let path = write(
            "t.jsonl",
            concat!(
                r#"{"type":"attachment","content":"a whole file nobody needs in a prompt"}"#,
                "\n",
                r#"{"type":"user","message":{"content":[{"type":"text","text":"why is auth failing?"}]}}"#,
                "\n",
                r#"{"type":"assistant","message":{"content":[{"type":"text","text":"Because expiry uses < instead of <=."}]}}"#,
                "\n"
            ),
        );
        let span = read_span(&path, "claude-code", &Sanitizer::builtin()).unwrap();
        assert!(span.contains("why is auth failing?"));
        assert!(span.contains("expiry uses < instead of <="));
        assert!(!span.contains("nobody needs"), "attachments should not reach the prompt");
        assert!(span.starts_with("user:"), "order should be chronological");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn cursor_transcripts_read_with_the_same_extractor() {
        let path = write(
            "t.jsonl",
            concat!(
                r#"{"role":"user","message":{"content":[{"type":"text","text":"why is auth failing?"}]}}"#,
                "\n",
                r#"{"role":"assistant","message":{"content":[{"type":"text","text":"Expiry used < instead of <=."},{"type":"tool_use","name":"Shell"}]}}"#,
                "\n"
            ),
        );
        let span = read_span(&path, "cursor", &Sanitizer::builtin()).unwrap();
        assert!(span.contains("user: why is auth failing?"));
        assert!(span.contains("assistant: Expiry used"));
        assert!(!span.contains("tool_use"), "only prose belongs in the prompt");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn codex_reasoning_is_extracted() {
        let path = write(
            "t.jsonl",
            concat!(
                r#"{"payload":{"type":"session_meta","id":"x"},"type":"x"}"#,
                "\n",
                r#"{"payload":{"type":"reasoning","summary":["Chose SQLite because nothing may run resident."]}}"#,
                "\n",
                r#"{"payload":{"type":"agent_message","message":"Done."}}"#,
                "\n"
            ),
        );
        let span = read_span(&path, "codex", &Sanitizer::builtin()).unwrap();
        assert!(span.contains("reasoning: Chose SQLite"), "reasoning is the point: {span}");
        assert!(span.contains("assistant: Done."));
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// The app-server rollout (VS Code, Codex Desktop) has no `agent_message`
    /// event; the answer is a `message` item whose role is `assistant`.
    #[test]
    fn codex_app_server_answers_are_the_assistants_not_the_users() {
        let path = write(
            "app.jsonl",
            concat!(
                r#"{"type":"response_item","payload":{"type":"message","role":"developer","content":[{"type":"input_text","text":"MEMORY BLOCK"}]}}"#,
                "\n",
                r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"compare them"}]}}"#,
                "\n",
                r#"{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Verdict: keep local."}]}}"#,
                "\n"
            ),
        );
        let sanitizer = Sanitizer::builtin();
        let span = read_span(&path, "codex", &sanitizer).unwrap();
        assert!(span.contains("user: compare them"), "{span}");
        assert!(span.contains("assistant: Verdict: keep local."), "{span}");
        assert!(!span.contains("MEMORY BLOCK"), "host injections are not the session: {span}");
        assert_eq!(last_answer(&path, "codex", &sanitizer).as_deref(), Some("Verdict: keep local."));
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// `codex exec` writes the same answer as a `message` item and an
    /// `agent_message` event. One line, not two.
    #[test]
    fn a_codex_exec_answer_written_twice_is_read_once() {
        let path = write(
            "exec.jsonl",
            concat!(
                r#"{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Done."}]}}"#,
                "\n",
                r#"{"type":"event_msg","payload":{"type":"agent_message","message":"Done."}}"#,
                "\n"
            ),
        );
        let span = read_span(&path, "codex", &Sanitizer::builtin()).unwrap();
        assert_eq!(span.matches("assistant: Done.").count(), 1, "{span}");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// A final answer larger than the whole span budget is the case the span
    /// exists for; it is kept cut, never dropped with everything before it.
    #[test]
    fn an_oversized_newest_line_is_kept_cut_rather_than_dropping_the_span() {
        let long = "ก".repeat(SPAN_MAX_BYTES); // three bytes each: 3x the budget
        let path = write(
            "big.jsonl",
            &format!(
                concat!(
                    r#"{{"type":"assistant","message":{{"content":[{{"type":"text","text":"Earlier."}}]}}}}"#,
                    "\n",
                    r#"{{"type":"assistant","message":{{"content":[{{"type":"text","text":"Verdict {}"}}]}}}}"#,
                    "\n"
                ),
                long
            ),
        );
        let span = read_span(&path, "claude-code", &Sanitizer::builtin()).unwrap();
        assert!(span.len() <= SPAN_MAX_BYTES, "{} bytes", span.len());
        assert!(span.starts_with("assistant: Verdict"), "the newest line's head survives: {}", &span[..40]);
        assert!(!span.contains("Earlier."), "there was no room left for older lines");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn the_span_is_capped_and_keeps_the_tail() {
        let mut body = String::new();
        for index in 0..4000 {
            body.push_str(&format!(
                r#"{{"type":"assistant","message":{{"content":[{{"type":"text","text":"line {index} {}"}}]}}}}"#,
                "x".repeat(60)
            ));
            body.push('\n');
        }
        let path = write("t.jsonl", &body);
        let span = read_span(&path, "claude-code", &Sanitizer::builtin()).unwrap();
        assert!(span.len() <= SPAN_MAX_BYTES, "span was {} bytes", span.len());
        assert!(span.contains("line 3999"), "the tail is the unconsolidated part");
        assert!(!span.contains("line 0 "), "the head should have been dropped");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn a_transcript_is_sanitized_before_it_reaches_the_summarizer() {
        // Nothing in a transcript went through capture-time scrubbing.
        let path = write(
            "t.jsonl",
            concat!(
                r#"{"type":"user","message":{"content":[{"type":"text","text":"deploy with ghp_abcdefghijklmnopqrstuvwxyz0123"}]}}"#,
                "\n",
                r#"{"type":"user","message":{"content":[{"type":"text","text":"and <private>Acme Holdings</private>"}]}}"#,
                "\n"
            ),
        );
        let span = read_span(&path, "claude-code", &Sanitizer::builtin()).unwrap();
        assert!(!span.contains("ghp_abcdefghijklmnopqrstuvwxyz0123"));
        assert!(!span.contains("Acme Holdings"));
        assert!(span.contains("[REDACTED]") && span.contains("[PRIVATE]"));
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn the_last_answer_is_the_models_own_and_nothing_else() {
        let sanitizer = Sanitizer::builtin();
        let path = write(
            "turns.jsonl",
            concat!(
                r#"{"type":"user","message":{"content":[{"type":"text","text":"why is it slow"}]}}"#, "\n",
                r#"{"type":"assistant","message":{"content":[{"type":"text","text":"An older answer."}]}}"#, "\n",
                r#"{"type":"user","message":{"content":[{"type":"text","text":"and the index?"}]}}"#, "\n",
                r#"{"type":"assistant","message":{"content":[{"type":"text","text":"The index is unused: the query filters on a column it does not cover."}]}}"#, "\n",
            ),
        );
        let answer = last_answer(&path, "claude-code", &sanitizer).expect("an answer");
        assert!(answer.contains("index is unused"), "took the wrong turn: {answer}");
        assert!(!answer.contains("An older answer"), "took more than the last: {answer}");
        assert!(!answer.contains("and the index?"), "took the user's turn too: {answer}");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn a_stored_answer_is_bounded_and_scrubbed() {
        // It is stored, unlike everything else read here, so the two things
        // that make storing text dangerous have to be handled at the door:
        // an unbounded transcript, and a secret the user typed into one.
        let sanitizer = Sanitizer::builtin();
        let long = "x".repeat(ANSWER_MAX_BYTES * 3);
        let path = write(
            "big.jsonl",
            &format!(
                r#"{{"type":"assistant","message":{{"content":[{{"type":"text","text":"{long} sk-ant-api03-SECRETVALUE"}}]}}}}"#
            ),
        );
        let answer = last_answer(&path, "claude-code", &sanitizer).expect("an answer");
        assert!(answer.len() <= ANSWER_MAX_BYTES, "stored {} bytes", answer.len());
        assert!(!answer.contains("SECRETVALUE"), "a secret survived into storage");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn a_transcript_shorter_than_the_tail_window_is_read_whole() {
        // The window drops the line it cut in half. A file it did not cut has
        // no half line, and dropping the first one there threw away the only
        // turn a short transcript has.
        let sanitizer = Sanitizer::builtin();
        let path = write(
            "short.jsonl",
            concat!(
                r#"{"type":"assistant","message":{"content":[{"type":"text","text":"The only answer here."}]}}"#,
                "\n"
            ),
        );
        let answer = last_answer(&path, "claude-code", &sanitizer);
        assert_eq!(answer.as_deref(), Some("The only answer here."));
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn a_missing_or_empty_transcript_is_not_an_error() {
        let sanitizer = Sanitizer::builtin();
        assert!(read_span(Path::new("/nonexistent/x.jsonl"), "claude-code", &sanitizer).is_none());
        let path = write("empty.jsonl", "");
        assert!(read_span(&path, "claude-code", &sanitizer).is_none());
        let path2 = write("junk.jsonl", "not json\n{\"type\":\"attachment\"}\n");
        assert!(read_span(&path2, "claude-code", &sanitizer).is_none());
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
        std::fs::remove_dir_all(path2.parent().unwrap()).ok();
    }
}
