//! Automatic context injection — pointers only, never content.
//!
//! The invariant: full-content auto-injection is 0, at every layer, always. What makes a memory system heavy is never the
//! memory — it is dumping content nobody asked for into every session. So we
//! push titles and ids, and the agent pulls bodies through MCP when the task
//! actually needs them.
//!
//! Budgets are in bytes, not counts. "50 observations" is an approximation:
//! fifty long lines and fifty short lines are not the same spend. Truncation
//! drops whole lines from the bottom of the ranking, never clips one, because
//! half an id is worse than no id.
//!
//! Two layers live here:
//!
//! - **Layer 1**, at session start: the project primer, one line per memory.
//! - **Layer 3**, after a file tool: 1-3 pointers for that exact file.
//!
//! Layer 2 is the MCP surface in [`crate::mcp`] and has no budget at all,
//! because the agent asked for it.

use std::fmt::Write as _;

use anyhow::Result;

use crate::config::InjectionConfig;
use crate::store::{Pointer, Store};

/// Most pointers one file-keyed injection may carry.
const MICRO_MAX_POINTERS: usize = 3;

/// Rendered injection plus the ids it spent, so the caller can record them.
#[derive(Debug, Default)]
pub struct Injection {
    pub text: String,
    pub ids: Vec<String>,
}

impl Injection {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

/// Build the session-start primer.
///
/// # Errors
/// Returns an error when the index cannot be queried.
pub fn primer(store: &Store, project: &str, config: &InjectionConfig) -> Result<Injection> {
    let pointers = store.primer_pointers(project, 200)?;
    if pointers.is_empty() {
        return Ok(Injection::default());
    }

    let header = "# Project memory\n\nPrior sessions in this project, most useful first. \
                  These are pointers, not content — call `brain_get` with an id, or \
                  `brain_search`, to read any of them. DEC decision, FND finding, \
                  FIX bugfix, NEW feature, CFG config, TST test, KNW durable knowledge, \
                  SUM session summary, NTE note; lowercase `raw` has not been consolidated yet.\n\n";

    // The primer is the higher-value spend, but it is still spend: it can
    // never exceed what the whole session is allowed.
    let budget = config.primer_budget.min(config.session_budget);

    let mut text = String::with_capacity(budget);
    text.push_str(header);
    let mut ids = Vec::new();

    for pointer in &pointers {
        if !worth_injecting(pointer) {
            continue;
        }
        let line = render_line(pointer);
        // Whole lines only: a clipped line costs the id that made it useful.
        if text.len() + line.len() > budget {
            break;
        }
        text.push_str(&line);
        ids.push(pointer.id.clone());
    }

    if ids.is_empty() {
        return Ok(Injection::default());
    }
    Ok(Injection { text, ids })
}

/// Build a file-keyed injection for a file just read or edited.
///
/// `exclude` is the event being captured right now. Without it the hook hands
/// the agent back a description of the action it just took, which costs budget
/// to say nothing.
///
/// Returns nothing when the file has no memory, when this file was already
/// covered this session, or when the session's injection budget is spent.
///
/// # Errors
/// Returns an error when the index cannot be queried.
pub fn for_file(
    store: &Store,
    project: &str,
    session: &str,
    path: &str,
    exclude: &str,
    config: &InjectionConfig,
) -> Result<Injection> {
    // Once per file per session. Repeating the same three pointers every time
    // a file is touched is how a "lightweight" system becomes noise.
    if store.file_already_injected(session, path)? {
        return Ok(Injection::default());
    }

    let spent = store.session_injected_bytes(session)?;
    if spent >= config.session_budget {
        // Layer 1 outranks layer 3 by design: when the budget runs out, the
        // file pointers go quiet and the primer keeps its spend.
        return Ok(Injection::default());
    }

    let pointers = store.pointers_for_file(project, path, MICRO_MAX_POINTERS * 3)?;
    if pointers.is_empty() {
        return Ok(Injection::default());
    }

    // The header is built first so it is counted against the budget. Adding it
    // afterwards silently overspent by its own length on every injection,
    // which is exactly how byte budgets stop meaning anything.
    let mut text = format!("Memory for `{path}`:\n");
    let mut ids = Vec::new();
    let remaining = config.session_budget - spent;

    for pointer in pointers {
        if ids.len() >= MICRO_MAX_POINTERS {
            break;
        }
        if pointer.id == exclude {
            continue;
        }
        // An id injected anywhere this session is never injected again.
        if store.already_injected(session, &pointer.id)? {
            continue;
        }
        let line = render_line(&pointer);
        if text.len() + line.len() > remaining {
            break;
        }
        text.push_str(&line);
        ids.push(pointer.id);
    }

    if ids.is_empty() {
        return Ok(Injection::default());
    }
    Ok(Injection { text, ids })
}

/// One pointer line: `id  time  TAG  title`.
fn render_line(pointer: &Pointer) -> String {
    let mut line = String::with_capacity(96);
    let _ = writeln!(
        line,
        "{}  {}  {}  {}",
        pointer.id,
        &pointer.ts[..pointer.ts.len().min(16)],
        tag(pointer),
        pointer.title.replace('\n', " ")
    );
    line
}

/// A three-character type column, so forty lines can be skimmed for the
/// decisions without reading the config noise.
///
/// Uppercase means something classified it; lowercase `raw` means nothing has
/// yet. That case difference is deliberate — it tells the reader at a glance
/// how much of this primer has been through consolidation. Plain ASCII, since
/// this lands in a terminal whose font we do not control.
fn tag(pointer: &Pointer) -> &'static str {
    if let Some(topic) = pointer.topic.as_deref() {
        return match topic {
            "decision" => "DEC",
            "bugfix" => "FIX",
            "feature" => "NEW",
            "discovery" => "FND",
            "config" => "CFG",
            "test" => "TST",
            _ => "---",
        };
    }
    match pointer.kind.as_str() {
        "knowledge" => "KNW",
        "session_summary" => "SUM",
        "note" => "NTE",
        "page_update" => "---",
        _ => "raw",
    }
}

/// Is this pointer worth spending primer bytes on?
///
/// The budget is a ceiling, not a quota: a short primer of real signal beats a
/// full one padded with noise.
///
/// The test is structural, never a match on the title's wording. A title is
/// the one thing here most likely to change — improving the rule-based titler
/// would silently disable a floor that keyed off `"Ran: "` — so the floor asks
/// what the event *is*, not how it happens to read:
///
/// - something classified it, so a model judged it worth recalling;
/// - it is not a raw capture at all (a summary, a note, a rewritten title);
/// - it touched a file, so it is findable and probably consequential;
/// - or a human typed it, which is signal even with nothing attached.
///
/// What is left is a tool call that changed no file and that nothing thought
/// worth naming. That is the padding.
fn worth_injecting(pointer: &Pointer) -> bool {
    pointer.topic.is_some()
        || pointer.kind != "observation"
        || pointer.has_files
        || pointer.hook == "user_prompt_submit"
}

/// Wrap an injection in the JSON a Claude Code hook must print.
///
/// Verified against the installed binary's own hook documentation:
/// `hookSpecificOutput.additionalContext` is the field that reaches the model.
#[must_use]
pub fn as_hook_output(hook_event_name: &str, injection: &Injection) -> String {
    if injection.is_empty() {
        return "{}".to_string();
    }
    serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": hook_event_name,
            "additionalContext": injection.text,
        }
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{Event, EventKind, Source};
    use uuid::Uuid;

    fn pointer(kind: &str, topic: Option<&str>, title: &str, has_files: bool) -> Pointer {
        Pointer {
            id: "01TEST".to_string(),
            ts: "2026-08-23T00:00:00Z".to_string(),
            kind: kind.to_string(),
            title: title.to_string(),
            topic: topic.map(str::to_string),
            has_files,
            hook: "post_tool_use".to_string(),
        }
    }

    fn store_with(project: Uuid, count: usize) -> Store {
        let store = Store::open_memory().unwrap();
        for index in 0..count {
            let mut event = Event::new(
                Uuid::nil(),
                project,
                Uuid::nil(),
                Source { cli: "claude-code".into(), hook: "post_tool_use".into() },
                EventKind::Observation,
                format!("Observation number {index} with a reasonably long title"),
                "body that must never be injected".repeat(50),
            );
            event.id = format!("01TEST{index:020}");
            event.files = vec!["src/auth.rs".to_string()];
            store.index(&event).unwrap();
        }
        store
    }

    #[test]
    fn the_primer_respects_its_byte_budget_exactly() {
        let project = Uuid::new_v4();
        let store = store_with(project, 200);
        let config = InjectionConfig { primer_budget: 1024, session_budget: 8192 };
        let injection = primer(&store, &project.to_string(), &config).unwrap();
        assert!(!injection.is_empty());
        assert!(
            injection.text.len() <= config.primer_budget,
            "primer was {} bytes, over its {}-byte budget",
            injection.text.len(),
            config.primer_budget
        );
    }

    #[test]
    fn the_primer_never_clips_a_line() {
        let project = Uuid::new_v4();
        let store = store_with(project, 200);
        let config = InjectionConfig { primer_budget: 700, session_budget: 8192 };
        let injection = primer(&store, &project.to_string(), &config).unwrap();
        for line in injection.text.lines().filter(|line| line.starts_with("01TEST")) {
            assert!(line.len() > 30, "a pointer line was clipped: {line:?}");
            assert_eq!(
                line.split_whitespace().next().unwrap().len(),
                26,
                "the id must survive intact or the pointer is useless"
            );
        }
    }

    #[test]
    fn the_primer_carries_no_bodies() {
        let project = Uuid::new_v4();
        let store = store_with(project, 20);
        let config = InjectionConfig::default();
        let injection = primer(&store, &project.to_string(), &config).unwrap();
        assert!(
            !injection.text.contains("body that must never be injected"),
            "full-content auto-injection must be 0"
        );
    }

    #[test]
    fn an_empty_project_injects_nothing_at_all() {
        let store = Store::open_memory().unwrap();
        let injection = primer(&store, &Uuid::new_v4().to_string(), &InjectionConfig::default())
            .unwrap();
        assert!(injection.is_empty());
        assert_eq!(as_hook_output("SessionStart", &injection), "{}");
    }

    #[test]
    fn file_injection_is_capped_and_deduplicated() {
        let project = Uuid::new_v4();
        let store = store_with(project, 20);
        let config = InjectionConfig::default();
        let session = "s1";

        let first =
            for_file(&store, &project.to_string(), session, "src/auth.rs", "", &config).unwrap();
        assert!(!first.is_empty());
        assert!(first.ids.len() <= MICRO_MAX_POINTERS);
        store.record_injected(session, &first.ids, first.text.len()).unwrap();
        store.record_injected_file(session, "src/auth.rs").unwrap();

        // Same file again in the same session: silence.
        let second =
            for_file(&store, &project.to_string(), session, "src/auth.rs", "", &config).unwrap();
        assert!(second.is_empty(), "a file must be injected once per session");
    }

    #[test]
    fn an_id_injected_once_is_never_injected_again() {
        let project = Uuid::new_v4();
        let store = store_with(project, 20);
        let config = InjectionConfig::default();

        let first = for_file(&store, &project.to_string(), "s1", "src/auth.rs", "", &config).unwrap();
        store.record_injected("s1", &first.ids, first.text.len()).unwrap();

        // A different file path that happens to share the same events.
        let again = for_file(&store, &project.to_string(), "s1", "src/auth.rs", "", &config).unwrap();
        for id in &again.ids {
            assert!(!first.ids.contains(id), "id {id} was injected twice");
        }
    }

    #[test]
    fn a_file_injection_counts_its_own_header() {
        let project = Uuid::new_v4();
        let store = store_with(project, 20);
        // Just enough for the header and nothing else.
        let config = InjectionConfig { primer_budget: 4096, session_budget: 30 };
        let injection =
            for_file(&store, &project.to_string(), "s1", "src/auth.rs", "", &config).unwrap();
        assert!(
            injection.text.len() <= config.session_budget,
            "injection was {} bytes against a {}-byte remaining budget",
            injection.text.len(),
            config.session_budget
        );
    }

    #[test]
    fn the_primer_cannot_outspend_the_whole_session() {
        let project = Uuid::new_v4();
        let store = store_with(project, 200);
        // A misconfiguration: primer budget larger than the session cap.
        let config = InjectionConfig { primer_budget: 100_000, session_budget: 2048 };
        let injection = primer(&store, &project.to_string(), &config).unwrap();
        assert!(injection.text.len() <= config.session_budget);
    }

    #[test]
    fn layer_three_goes_quiet_when_the_session_budget_is_spent() {
        let project = Uuid::new_v4();
        let store = store_with(project, 20);
        let config = InjectionConfig { primer_budget: 4096, session_budget: 100 };
        store.record_injected("s1", &[], 100).unwrap();
        let injection =
            for_file(&store, &project.to_string(), "s1", "src/auth.rs", "", &config).unwrap();
        assert!(injection.is_empty(), "the primer keeps the spend, not layer 3");
    }

    #[test]
    fn the_event_being_captured_is_not_handed_back_to_the_agent() {
        let project = Uuid::new_v4();
        let store = store_with(project, 3);
        let current = "01TEST".to_string() + &format!("{:020}", 2);
        let injection = for_file(
            &store,
            &project.to_string(),
            "s1",
            "src/auth.rs",
            &current,
            &InjectionConfig::default(),
        )
        .unwrap();
        assert!(!injection.ids.contains(&current), "told the agent what it just did");
    }

    #[test]
    fn a_file_with_no_memory_injects_nothing() {
        let project = Uuid::new_v4();
        let store = store_with(project, 5);
        let injection = for_file(
            &store,
            &project.to_string(),
            "s1",
            "src/never/touched.rs",
            "",
            &InjectionConfig::default(),
        )
        .unwrap();
        assert!(injection.is_empty());
    }

    #[test]
    fn hook_output_uses_the_field_claude_code_actually_reads() {
        let injection = Injection { text: "hello".into(), ids: vec!["01A".into()] };
        let output = as_hook_output("SessionStart", &injection);
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed["hookSpecificOutput"]["hookEventName"], "SessionStart");
        assert_eq!(parsed["hookSpecificOutput"]["additionalContext"], "hello");
    }

    #[test]
    fn every_topic_gets_a_distinct_tag() {
        let mut seen = std::collections::HashSet::new();
        for topic in crate::event::TOPICS {
            let tag = tag(&pointer("page_update", Some(topic), "t", true));
            assert_ne!(tag, "---", "{topic} has no tag");
            assert!(seen.insert(tag), "two topics share the tag {tag}");
        }
        assert_eq!(tag(&pointer("observation", None, "t", true)), "raw");
        assert_eq!(tag(&pointer("session_summary", None, "t", false)), "SUM");
        // Knowledge is untyped by the topic taxonomy - it is a different axis
        // - so it must be tagged by kind rather than falling through to `raw`.
        assert_eq!(tag(&pointer("knowledge", None, "t", false)), "KNW");
    }

    #[test]
    fn a_headless_runs_observations_rank_below_a_persons() {
        let project = Uuid::new_v4();
        let store = Store::open_memory().unwrap();
        for (index, headless) in [(0usize, true), (1, false)] {
            let mut event = crate::event::Event::new(
                Uuid::nil(),
                project,
                Uuid::nil(),
                crate::event::Source { cli: "claude-code".into(), hook: "post_tool_use".into() },
                crate::event::EventKind::Observation,
                if headless { "from a one-shot run".into() } else { "from a person".into() },
                String::new(),
            );
            event.id = format!("01RANK{index:020}");
            event.files = vec!["src/main.rs".to_string()];
            if headless {
                event.extra.insert(
                    "invocation".to_string(),
                    serde_json::Value::String("headless".to_string()),
                );
            }
            store.index(&event).unwrap();
        }
        let injection =
            primer(&store, &project.to_string(), &InjectionConfig::default()).unwrap();
        let person = injection.text.find("from a person").unwrap();
        let robot = injection.text.find("from a one-shot run").unwrap();
        assert!(person < robot, "a headless run outranked a person's session");
    }

    #[test]
    fn the_floor_drops_bare_commands_and_keeps_everything_earned() {
        // Dropped: nothing classified it, it touched no file, and its title is
        // the command that produced it.
        assert!(!worth_injecting(&pointer("observation", None, "Ran: cargo build", false)));

        // Kept, each for its own reason.
        assert!(worth_injecting(&pointer("observation", None, "Ran: cargo build", true)));
        assert!(worth_injecting(&pointer("page_update", Some("decision"), "Chose X", false)));
        assert!(worth_injecting(&pointer("session_summary", None, "Ran: anything", false)));
        assert!(worth_injecting(&pointer("note", None, "Ran: anything", false)));
        assert!(worth_injecting(&Pointer {
            hook: "user_prompt_submit".to_string(),
            ..pointer("observation", None, "Asked: why?", false)
        }));

        // And the floor must not depend on how a title happens to read: the
        // same event with a nicer title is still padding.
        assert!(!worth_injecting(&pointer("observation", None, "build: workspace", false)));
    }

    #[test]
    fn a_noisy_project_produces_a_short_primer_not_a_padded_one() {
        let project = Uuid::new_v4();
        let store = Store::open_memory().unwrap();
        for index in 0..40 {
            let mut event = crate::event::Event::new(
                Uuid::nil(),
                project,
                Uuid::nil(),
                crate::event::Source {
                    cli: "claude-code".into(),
                    hook: "post_tool_use".into(),
                },
                crate::event::EventKind::Observation,
                format!("Ran: echo noise {index}"),
                String::new(),
            );
            event.id = format!("01NOISE{index:019}");
            store.index(&event).unwrap();
        }
        let mut signal = crate::event::Event::new(
            Uuid::nil(),
            project,
            Uuid::nil(),
            crate::event::Source { cli: "claude-code".into(), hook: "consolidate".into() },
            crate::event::EventKind::PageUpdate,
            "Chose spawn-on-demand over a resident worker".into(),
            String::new(),
        );
        signal.id = "01SIGNAL0000000000000000".to_string();
        signal.topic = Some("decision".to_string());
        store.index(&signal).unwrap();

        let injection =
            primer(&store, &project.to_string(), &InjectionConfig::default()).unwrap();
        assert!(injection.text.contains("Chose spawn-on-demand"), "the signal was cut");
        assert!(!injection.text.contains("echo noise"), "noise was injected");
        assert_eq!(injection.ids.len(), 1, "only the earned line should appear");
        assert!(
            injection.text.len() < 800,
            "a noisy project should yield a SHORT primer, got {} bytes",
            injection.text.len()
        );
    }
}
