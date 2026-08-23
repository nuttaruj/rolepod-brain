//! Reading the host CLI's transcript at consolidation time.
//!
//! The richest material a session produces is the model's own prose — the
//! reasoning it wrote, the decision it explained, the dead end it described.
//! Lifecycle hooks never see any of it; they see tool calls and prompts.
//!
//! The transcript already exists on disk, written by the host CLI for its own
//! purposes. So we read it when we consolidate, feed it to the summarizer
//! beside the captured events, and **persist only the summary**. Nothing here
//! is ever copied into our storage: the log records that a transcript was read
//! and how much, never a line of it.
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
        Some("message") => "user",
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
    if !text.is_empty() {
        out.push(Line { speaker, text: text.to_string() });
    }
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
