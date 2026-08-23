//! Privacy strip — redacts secrets before anything reaches durable storage.
//!
//! Two things matter about where this runs. It runs in the hook worker,
//! *before* the event is written, so a secret never enters the log at all —
//! there is no server-side second chance in this design. And redaction is
//! irreversible: an over-broad pattern destroys text with no error and no
//! recovery, which is why the patterns are anchored rather than greedy.

use std::sync::Arc;

use regex::Regex;

/// Ceiling for a stored event body, after redaction.
pub const BODY_MAX_BYTES: usize = 16 * 1024;

/// Credential shapes we redact, most specific first.
///
/// The strings are public facts - each vendor documents the format of its own
/// tokens - and the ordering matters only in that a narrow pattern should get
/// the first look at a match.
///
/// Two rules shaped every entry. **Anchor rather than widen:** redaction runs
/// before storage and cannot be undone, so an over-broad pattern destroys text
/// with no error and no way to recover it. And **accept false positives over
/// misses:** redacting a stray hash costs a little recall, while leaking a key
/// costs the user something they cannot take back.
///
/// Deliberately absent: bare high-entropy strings. A 32-character hex blob is
/// indistinguishable from a commit hash, a checksum or an id, and matching it
/// would quietly eat half of every real transcript. Operators who want that
/// can add it through `extra_patterns`.
const BUILTIN_PATTERN_STRS: &[&str] = &[
    // `Authorization: Bearer ...`, any casing.
    r#"(?i)bearer\s+[A-Za-z0-9._\-+/=]{16,}"#,
    // OpenAI-style `sk-` keys, and Stripe live keys. `rk_live_` is scoped
    // rather than full-access, but operators routinely scope it to charges and
    // refunds, so it is not meaningfully safer.
    r"sk-[A-Za-z0-9_\-]{16,}",
    r"(?:sk|rk)_live_[A-Za-z0-9_\-]{16,}",
    // Every GitHub token prefix, not only personal-access: p=personal,
    // o=OAuth (what `gh auth login` writes), u=user-to-server,
    // s=server-to-server and Actions, r=refresh. Plus fine-grained PATs.
    r"gh[pousr]_[A-Za-z0-9]{20,}",
    r"github_pat_[A-Za-z0-9_]{20,}",
    // AWS key ids, long-lived and STS-temporary. Anchored to the published
    // length - a four-character prefix plus sixteen - and bounded by word
    // breaks, because `ASIA` is also an English word: an open-ended tail
    // silently destroys ordinary uppercase text such as ASIAPACIFICREGION.
    r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    // Google API keys, then Google OAuth refresh tokens. The refresh token is
    // the more dangerous of the two: it mints access tokens until revoked, so
    // it outlives the session that leaked it.
    r"AIza[A-Za-z0-9_\-]{30,}",
    r"1//[0-9A-Za-z_\-]{20,}",
    // Meta and Facebook Graph access tokens - pages, ad accounts, business
    // administration.
    r"EAA[A-Za-z0-9]{20,}",
    // Telegram bot tokens, `<bot-id>:<secret>`, which grant full control of a
    // bot including everything it can read. Two branches: the `AA` form every
    // issued token has used, and the shorter documented form. Both are
    // length-anchored, because a bare digits-colon-word shape also matches
    // timestamps, `host:port` pairs, subtitle cues and `<sha>:<hex>` pairs.
    r"\b\d{6,10}:(?:AA[A-Za-z0-9_\-]{30,}|[A-Za-z0-9_\-]{34,35})\b",
    // GoHighLevel private integration tokens, which never expire on their own.
    // Anchored to the UUID tail rather than a permissive one, because `pit-`
    // is an English fragment and a loose tail would redact "pit-stop-analysis".
    r"pit-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
    // Slack bot, user, admin, app-level and refresh tokens.
    r"xox[abprs]-[A-Za-z0-9\-]{10,}",
    r"xapp-[A-Za-z0-9\-]{10,}",
    // JWTs: three base64url segments separated by dots.
    r"eyJ[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}",
    // PEM-bracketed private keys, spanning lines, matched lazily so that two
    // keys in one file do not collapse into a single match.
    r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    // Credentials embedded in a URL, as in `postgres://user:pass@host`.
    r"[a-zA-Z][a-zA-Z0-9+\-.]*://[^:/\s]+:[^@\s]+@[^\s]+",
    // Named provider variables, listed explicitly so a bare
    // `OPENAI_API_KEY=anything` is caught even when the value has no
    // recognizable shape of its own.
    r#"(?i)(ANTHROPIC_API_KEY|OPENAI_API_KEY|OPENROUTER_API_KEY|VOYAGE_API_KEY|MISTRAL_API_KEY|GROQ_API_KEY|HF_TOKEN|HUGGINGFACE_TOKEN|AWS_(SECRET_)?ACCESS_KEY[A-Z_]*|GITHUB_TOKEN|GH_TOKEN|GITLAB_TOKEN|GOOGLE_API_KEY|GEMINI_API_KEY|OLLAMA_API_KEY)\s*[=:]\s*\S+"#,
    // Bare credential words with no prefix. The general shape below requires an
    // underscore-separated prefix, so a plain `TOKEN=...` slipped past it. The
    // value class excludes `$` deliberately: `TOKEN=$(cat file)` and
    // `TOKEN=$OTHER` name a variable rather than hold a secret, and destroying
    // them costs recall while protecting nothing.
    r#"(?i)\b(TOKEN|SECRET|PASSWORD|PASSWD|APIKEY|API_KEY|CREDENTIAL)\s*[=:]\s*[A-Za-z0-9._\-+/]{8,}"#,
    // The general shape: any `*_KEY` / `*_TOKEN` / `*_SECRET` / `*_PASSWORD` /
    // `*_CREDENTIAL[S]` / `*_PRIVATE_KEY` assignment.
    r#"(?i)\b[A-Z][A-Z0-9_]*_(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|CREDENTIALS|PRIVATE_KEY)\s*[=:]\s*\S+"#,
    // Paths whose contents are credentials by convention. The path itself is
    // rarely secret; what it tells us is that the surrounding text concerns
    // one, and the surrounding text is what gets stored.
    r"(?:/[^/\s]+)*/\.ssh(?:/[^\s]+)?",
    r"(?:/[^/\s]+)*/\.aws(?:/[^\s]+)?",
    r"(?:/[^/\s]+)*/\.kube(?:/[^\s]+)?",
    r"(?:/[^/\s]+)*/\.config/gcloud(?:/[^\s]+)?",
    r"(?:/[^/\s]+)*/\.gnupg(?:/[^\s]+)?",
];

/// Compiled redaction patterns. Cheap to clone.
#[derive(Clone)]
pub struct Sanitizer {
    inner: Arc<SanitizerInner>,
}

struct SanitizerInner {
    patterns: Vec<Regex>,
    allowlist: Vec<String>,
}

impl std::fmt::Debug for Sanitizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sanitizer")
            .field("patterns", &self.inner.patterns.len())
            .field("allowlist", &self.inner.allowlist.len())
            .finish()
    }
}

/// Operator-tunable settings, mirroring `[sanitize]` in `config.toml`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SanitizeConfig {
    /// Extra regexes to redact. An invalid pattern fails loudly at startup
    /// rather than silently disabling itself.
    pub extra_patterns: Vec<String>,
    /// Substrings that survive redaction even when a pattern matches them —
    /// the escape hatch for a project codename that collides with the generic
    /// `*_TOKEN=` catch-all.
    pub allowlist: Vec<String>,
}

impl Sanitizer {
    /// Build from the built-in corpus plus operator extras.
    ///
    /// # Errors
    /// Returns [`regex::Error`] when an entry in `extra_patterns` is invalid.
    pub fn new(config: &SanitizeConfig) -> Result<Self, regex::Error> {
        let mut patterns =
            Vec::with_capacity(BUILTIN_PATTERN_STRS.len() + config.extra_patterns.len());
        for pattern in BUILTIN_PATTERN_STRS {
            patterns.push(Regex::new(pattern)?);
        }
        for pattern in &config.extra_patterns {
            patterns.push(Regex::new(pattern)?);
        }
        Ok(Self {
            inner: Arc::new(SanitizerInner {
                patterns,
                allowlist: config.allowlist.clone(),
            }),
        })
    }

    /// Built-in patterns only.
    ///
    /// # Panics
    /// Panics if the built-in corpus fails to compile, which is a programming
    /// error caught by the test below on every build.
    #[must_use]
    pub fn builtin() -> Self {
        Self::new(&SanitizeConfig::default()).expect("built-in patterns compile")
    }

    /// Replace every match with `[REDACTED]`, honouring the allowlist per match.
    ///
    /// Private regions are removed first, before any pattern runs: they mark
    /// things no pattern could recognize — a client's name, a figure from a
    /// contract — so the only safe treatment is to delete them, not to try to
    /// match them.
    #[must_use]
    pub fn scrub(&self, input: &str) -> String {
        let mut out = strip_private(input);
        for pattern in &self.inner.patterns {
            out = pattern
                .replace_all(&out, |caps: &regex::Captures<'_>| {
                    let matched = caps.get(0).map_or("", |m| m.as_str());
                    if self.inner.allowlist.iter().any(|allowed| matched.contains(allowed)) {
                        matched.to_string()
                    } else {
                        "[REDACTED]".to_string()
                    }
                })
                .into_owned();
        }
        out
    }

    /// Scrub a body, then clamp it to [`BODY_MAX_BYTES`].
    ///
    /// Order matters: clamping first could split a secret across the boundary
    /// and leave half of it stored.
    #[must_use]
    pub fn scrub_body(&self, input: &str) -> String {
        truncate_head_tail(&self.scrub(input), BODY_MAX_BYTES)
    }
}

impl Default for Sanitizer {
    fn default() -> Self {
        Self::builtin()
    }
}

/// Opening marker for a region the user wants forgotten.
const PRIVATE_OPEN: &str = "<private>";
/// Closing marker.
const PRIVATE_CLOSE: &str = "</private>";

/// Remove everything between `<private>` and `</private>`.
///
/// The escape hatch for secrets no pattern can catch. Two properties matter
/// more than elegance here:
///
/// - An UNCLOSED opening tag drops everything after it. Failing closed is the
///   only defensible reading: someone typed the tag because what follows must
///   not be stored, and a missing closer is far more likely a typo than an
///   invitation to keep the rest.
/// - It runs on the way IN, before storage, and again over model output. A
///   summarizer handed a private region it should never have seen is a bug in
///   the caller, but the marker still holds at the write boundary.
#[must_use]
pub fn strip_private(input: &str) -> String {
    if !input.contains(PRIVATE_OPEN) {
        return input.to_string();
    }
    let mut out = String::with_capacity(input.len());
    let mut rest = input;

    while let Some(start) = rest.find(PRIVATE_OPEN) {
        out.push_str(&rest[..start]);
        let after = &rest[start + PRIVATE_OPEN.len()..];
        match after.find(PRIVATE_CLOSE) {
            Some(end) => {
                out.push_str("[PRIVATE]");
                rest = &after[end + PRIVATE_CLOSE.len()..];
            }
            None => {
                // Unclosed: nothing after this point is safe to keep.
                out.push_str("[PRIVATE]");
                return out;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Truncate to at most `max` UTF-8 bytes, keeping the head, never splitting a
/// code point.
#[must_use]
pub fn truncate(input: &str, max: usize) -> String {
    if input.len() <= max {
        return input.to_string();
    }
    if max < '\u{2026}'.len_utf8() {
        return String::new();
    }
    let limit = max - '\u{2026}'.len_utf8();
    let mut end = 0;
    for (index, character) in input.char_indices() {
        let next = index + character.len_utf8();
        if next > limit {
            break;
        }
        end = next;
    }
    let mut out = String::with_capacity(max);
    out.push_str(&input[..end]);
    out.push('\u{2026}');
    out
}

/// Truncate to at most `max` UTF-8 bytes keeping **both** ends and eliding the
/// middle.
///
/// Head-only truncation loses the tail of a long tool output, which is usually
/// where the result is. Below 64 bytes the marker would outweigh the content,
/// so that case falls back to [`truncate`].
#[must_use]
pub fn truncate_head_tail(input: &str, max: usize) -> String {
    if input.len() <= max {
        return input.to_string();
    }
    if max < 64 {
        return truncate(input, max);
    }
    const MARKER_RESERVE: usize = 48;
    let half = max.saturating_sub(MARKER_RESERVE) / 2;

    let mut head_end = 0;
    for (index, character) in input.char_indices() {
        let next = index + character.len_utf8();
        if next > half {
            break;
        }
        head_end = next;
    }

    let tail_target = input.len().saturating_sub(half);
    let mut tail_start = input.len();
    for (index, _) in input.char_indices().rev() {
        tail_start = index;
        if index <= tail_target {
            break;
        }
    }
    if tail_start < head_end {
        tail_start = head_end;
    }

    let omitted = tail_start - head_end;
    let mut out = String::with_capacity(max);
    out.push_str(&input[..head_end]);
    out.push_str(&format!("\n...[truncated {omitted} bytes]...\n"));
    out.push_str(&input[tail_start..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_corpus_compiles() {
        let _ = Sanitizer::builtin();
    }

    #[test]
    fn redacts_common_credential_shapes() {
        let s = Sanitizer::builtin();
        for secret in [
            "sk-abcdefghijklmnopqrstuvwx",
            "ghp_abcdefghijklmnopqrstuvwxyz0123",
            "AKIAIOSFODNN7EXAMPLE",
            "xoxb-1234567890-abcdefghij",
            "ANTHROPIC_API_KEY=whatever-value",
            "postgres://user:hunter2@db.internal:5432/app",
        ] {
            let out = s.scrub(&format!("prefix {secret} suffix"));
            assert!(out.contains("[REDACTED]"), "not redacted: {secret}");
            assert!(!out.contains(secret), "secret survived: {secret}");
        }
    }

    #[test]
    fn redacts_bare_credential_names_without_a_prefix() {
        let s = Sanitizer::builtin();
        for secret in ["TOKEN=abc123def456", "secret: hunter2hunter2", "PASSWORD=correct-horse"] {
            let out = s.scrub(&format!("prefix {secret} suffix"));
            assert!(out.contains("[REDACTED]"), "not redacted: {secret}");
        }
    }

    #[test]
    fn keeps_variable_references_readable() {
        // Redaction is irreversible; destroying `$(cat file)` protects nothing
        // and costs the recall value of the command.
        let s = Sanitizer::builtin();
        for benign in ["TOKEN=$(cat /tmp/t.txt)", "TOKEN=$OTHER_VAR"] {
            assert_eq!(s.scrub(benign), benign, "over-redacted: {benign}");
        }
    }

    #[test]
    fn private_regions_are_removed_entirely() {
        let s = Sanitizer::builtin();
        let out = s.scrub("before <private>ACME Corp, 4.2M contract</private> after");
        assert_eq!(out, "before [PRIVATE] after");
        assert!(!out.contains("ACME"));
        assert!(!out.contains("4.2M"));
    }

    #[test]
    fn an_unclosed_private_tag_fails_closed() {
        // Someone typed the tag because what follows must not be stored. A
        // missing closer is a typo, not permission to keep the rest.
        let s = Sanitizer::builtin();
        let out = s.scrub("keep this <private>and never this, nor this");
        assert_eq!(out, "keep this [PRIVATE]");
    }

    #[test]
    fn several_private_regions_are_each_removed() {
        let out = strip_private("a <private>x</private> b <private>y</private> c");
        assert_eq!(out, "a [PRIVATE] b [PRIVATE] c");
    }

    #[test]
    fn text_without_the_tag_is_untouched() {
        let text = "a perfectly ordinary sentence about private matters";
        assert_eq!(strip_private(text), text);
    }

    #[test]
    fn a_private_region_containing_a_secret_leaves_nothing_behind() {
        let s = Sanitizer::builtin();
        let out = s.scrub("<private>token ghp_abcdefghijklmnopqrstuvwxyz0123</private>");
        assert!(!out.contains("ghp_"));
        assert_eq!(out, "[PRIVATE]");
    }

    #[test]
    fn leaves_ordinary_prose_alone() {
        let s = Sanitizer::builtin();
        let text = "Refactored the ASIAPACIFICREGION constant and ran the pit-stop analysis.";
        assert_eq!(s.scrub(text), text);
    }

    #[test]
    fn allowlist_survives_a_matching_pattern() {
        let s = Sanitizer::new(&SanitizeConfig {
            extra_patterns: vec![],
            allowlist: vec!["PROJECT_TOKEN".to_string()],
        })
        .unwrap();
        let out = s.scrub("PROJECT_TOKEN=visible OPENAI_API_KEY=hidden");
        assert!(out.contains("PROJECT_TOKEN=visible"));
        assert!(out.contains("[REDACTED]"));
    }

    #[test]
    fn invalid_extra_pattern_is_an_error_not_a_silent_skip() {
        let result = Sanitizer::new(&SanitizeConfig {
            extra_patterns: vec!["([unclosed".to_string()],
            allowlist: vec![],
        });
        assert!(result.is_err());
    }

    #[test]
    fn body_is_scrubbed_before_it_is_clamped() {
        let s = Sanitizer::builtin();
        let secret = "ghp_abcdefghijklmnopqrstuvwxyz0123";
        let body = format!("{}{secret}{}", "a".repeat(BODY_MAX_BYTES), "b".repeat(1000));
        let out = s.scrub_body(&body);
        assert!(out.len() <= BODY_MAX_BYTES);
        assert!(!out.contains(secret));
    }

    #[test]
    fn truncation_never_splits_a_code_point() {
        let text = "\u{e0aa}\u{e2b8}\u{e2ad}".repeat(100);
        let out = truncate(&text, 50);
        assert!(out.len() <= 50);
        assert!(std::str::from_utf8(out.as_bytes()).is_ok());
    }
}
