#!/usr/bin/env python3
"""
export_claude_chats.py -- collate Claude Code / Cowork session transcripts
into readable Markdown and HTML.

Both Claude Code and Cowork's local sessions save each session as a JSONL file.
This script finds those files, keeps the sessions that mention your keywords,
and writes one Markdown and/or HTML file per session plus an index.

Usage (on your Mac, no dependencies beyond Python 3.8+):

    python3 export_claude_chats.py -k virgil virgil apep -o ~/Desktop/virgil_chats

    # everything since a date, HTML only, include full tool output
    python3 export_claude_chats.py -k virgil --since 2026-06-01 --format html --max-tool-lines 0

    # just list the matching sessions without writing anything
    python3 export_claude_chats.py -k virgil --list

Default search locations (add more with --root):
    ~/.claude/projects                          (Claude Code CLI / desktop Code tab)
    ~/Library/Application Support/Claude        (Claude desktop app, incl. local Cowork sessions)

Note: Claude Code deletes transcripts older than ~30 days by default. To keep
them, set "cleanupPeriodDays": 100000 in ~/.claude/settings.json.
Cloud sessions (claude.ai/code/... links) are not stored on your Mac and
cannot be exported with this script.
"""

import argparse
import datetime as dt
import html
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

HOME = str(Path.home())
DEFAULT_ROOTS = [
    Path(HOME) / ".claude" / "projects",
    Path(HOME) / "Library" / "Application Support" / "Claude",
]

# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #
SECRET_PATTERNS = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"), "[REDACTED_ANTHROPIC_KEY]"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"), "[REDACTED_SLACK_TOKEN]"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
     "[REDACTED_PRIVATE_KEY]"),
    (re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key)\b(\s*[:=]\s*)(['\"]?)[^\s'\"]{6,}\3"),
     r"\1\2[REDACTED]"),
]
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Harness-injected blocks that are noise for a human reader.
NOISE_TAGS = ["system-reminder", "user_memory_snapshot", "command-message",
              "local-command-caveat"]
NOISE_RE = re.compile(r"<(%s)>.*?</\1>" % "|".join(NOISE_TAGS), re.S)


class Redactor:
    def __init__(self, paths=True, secrets=True, emails=False, extra=()):
        self.paths, self.secrets, self.emails = paths, secrets, emails
        self.extra = [re.compile(p) for p in extra]

    def __call__(self, s):
        if not s:
            return s
        if self.secrets:
            for pat, rep in SECRET_PATTERNS:
                s = pat.sub(rep, s)
        if self.paths:
            s = s.replace(HOME, "~")
            s = re.sub(r"/Users/[^/\s]+", "~", s)
        if self.emails:
            s = EMAIL_RE.sub("[EMAIL]", s)
        for pat in self.extra:
            s = pat.sub("[REDACTED]", s)
        return s


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def iter_jsonl_files(roots):
    seen = set()
    for root in roots:
        root = Path(root).expanduser()
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            # skip big irrelevant app caches
            dirnames[:] = [d for d in dirnames if d not in
                           ("Cache", "Code Cache", "GPUCache", "node_modules", "blob_storage")]
            for fn in filenames:
                if fn.endswith(".jsonl"):
                    p = Path(dirpath) / fn
                    try:
                        key = p.resolve()
                    except OSError:
                        continue
                    if key not in seen:
                        seen.add(key)
                        yield p


def load_records(path):
    recs = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(d, dict):
                    recs.append(d)
    except OSError:
        pass
    return recs


def looks_like_transcript(recs):
    return any(r.get("type") in ("user", "assistant") and isinstance(r.get("message"), dict)
               for r in recs[:200])


def blocks_of(message):
    c = message.get("content")
    if isinstance(c, str):
        return [{"type": "text", "text": c}]
    if isinstance(c, list):
        return [b for b in c if isinstance(b, dict)]
    return []


def tool_result_text(block):
    c = block.get("content")
    if isinstance(c, str):
        return c
    out = []
    if isinstance(c, list):
        for b in c:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                out.append(b.get("text", ""))
            elif b.get("type") == "image":
                out.append("[image]")
            elif b.get("type") == "tool_reference":
                out.append("[loaded tool: %s]" % b.get("tool_name", "?"))
    return "\n".join(out)


def clean_user_text(s):
    s = NOISE_RE.sub("", s)
    # slash-command wrappers -> readable form
    s = re.sub(r"<command-name>(.*?)</command-name>", r"\1", s, flags=re.S)
    s = re.sub(r"<command-args>(.*?)</command-args>", r" \1", s, flags=re.S)
    s = re.sub(r"<local-command-stdout>(.*?)</local-command-stdout>", r"\n\1", s, flags=re.S)
    return s.strip()


def summarise_tool_input(name, inp):
    """Short one-line description of a tool call, plus optional body."""
    if not isinstance(inp, dict):
        return name, json.dumps(inp, indent=2)
    for key in ("command", "description", "file_path", "path", "pattern", "query", "url", "prompt"):
        if key in inp and isinstance(inp[key], str):
            head = inp[key].strip().splitlines()[0] if inp[key].strip() else ""
            break
    else:
        head = ""
    body = inp.get("command") if isinstance(inp.get("command"), str) else None
    if body is None:
        if name in ("Write",) and "content" in inp:
            body = inp["content"]
        elif name in ("Edit",) and "new_string" in inp:
            body = "--- old\n%s\n+++ new\n%s" % (inp.get("old_string", ""), inp["new_string"])
        else:
            body = json.dumps(inp, indent=2, ensure_ascii=False)
    return ("%s: %s" % (name, head[:120])) if head else name, body


class Session:
    def __init__(self, sid):
        self.sid = sid
        self.files = set()
        self.records = {}
        self.title = None
        self.summary = None
        self.cwd = None
        self.branch = None
        self.entrypoint = None

    def add(self, path, recs):
        self.files.add(str(path))
        for i, r in enumerate(recs):
            t = r.get("type")
            if t == "ai-title" and r.get("aiTitle"):
                self.title = r["aiTitle"]
            elif t == "summary" and r.get("summary"):
                self.summary = r["summary"]
            if t not in ("user", "assistant"):
                continue
            key = r.get("uuid") or "%s:%d" % (path, i)
            self.records.setdefault(key, r)
            self.cwd = self.cwd or r.get("cwd")
            self.branch = self.branch or r.get("gitBranch")
            self.entrypoint = self.entrypoint or r.get("entrypoint")

    def ordered(self):
        return sorted(self.records.values(), key=lambda r: r.get("timestamp") or "")

    @property
    def start(self):
        ts = [r.get("timestamp") for r in self.records.values() if r.get("timestamp")]
        return min(ts) if ts else ""

    @property
    def end(self):
        ts = [r.get("timestamp") for r in self.records.values() if r.get("timestamp")]
        return max(ts) if ts else ""

    def searchable_text(self):
        parts = [self.cwd or "", self.title or "", self.summary or ""]
        for r in self.records.values():
            for b in blocks_of(r.get("message", {})):
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif b.get("type") == "tool_use":
                    parts.append(json.dumps(b.get("input", {})))
        return "\n".join(parts).lower()


def build_events(sess, opts, redact):
    """Turn raw records into a flat list of display events."""
    events = []
    tool_names = {}
    for r in sess.ordered():
        if r.get("isSidechain") and not opts.include_subagents:
            continue
        if r.get("isMeta"):
            continue
        ts = r.get("timestamp", "")
        msg = r.get("message", {}) or {}
        side = bool(r.get("isSidechain"))
        if r["type"] == "user":
            for b in blocks_of(msg):
                bt = b.get("type")
                if bt == "text":
                    txt = clean_user_text(b.get("text", ""))
                    if txt:
                        events.append(dict(kind="user", ts=ts, text=redact(txt), side=side))
                elif bt == "tool_result" and opts.tools:
                    out = tool_result_text(b)
                    events.append(dict(kind="tool_result", ts=ts, side=side,
                                       name=tool_names.get(b.get("tool_use_id"), "tool"),
                                       error=bool(b.get("is_error")),
                                       text=redact(truncate(out, opts.max_tool_lines))))
                elif bt == "image":
                    events.append(dict(kind="user", ts=ts, text="[image attached]", side=side))
        else:  # assistant
            for b in blocks_of(msg):
                bt = b.get("type")
                if bt == "text" and b.get("text", "").strip():
                    events.append(dict(kind="assistant", ts=ts, text=redact(b["text"].strip()), side=side))
                elif bt == "thinking" and opts.thinking and b.get("thinking", "").strip():
                    events.append(dict(kind="thinking", ts=ts, text=redact(b["thinking"].strip()), side=side))
                elif bt == "tool_use":
                    tool_names[b.get("id")] = b.get("name", "tool")
                    if opts.tools:
                        head, body = summarise_tool_input(b.get("name", "tool"), b.get("input"))
                        events.append(dict(kind="tool_use", ts=ts, side=side, name=redact(head),
                                           text=redact(truncate(body, opts.max_tool_lines))))
    return events


def truncate(s, n):
    if not s or n <= 0:
        return s
    lines = s.splitlines()
    if len(lines) <= n:
        return s
    keep_head = max(1, int(n * 0.7))
    keep_tail = n - keep_head
    return "\n".join(lines[:keep_head] + ["... [%d lines omitted] ..." % (len(lines) - n)]
                     + (lines[-keep_tail:] if keep_tail else []))


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def fmt_ts(ts, tz):
    if not ts:
        return ""
    try:
        t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return t.astimezone(tz).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ts


def session_title(sess):
    if sess.title:
        return sess.title
    if sess.summary:
        return sess.summary
    for r in sess.ordered():
        if r["type"] == "user":
            for b in blocks_of(r.get("message", {})):
                if b.get("type") == "text":
                    t = clean_user_text(b.get("text", ""))
                    if t:
                        return t.splitlines()[0][:80]
    return sess.sid


def fence(s):
    ticks = "```"
    while ticks in (s or ""):
        ticks += "`"
    return ticks


def render_md(sess, events, opts, redact, tz):
    title = redact(session_title(sess))
    out = ["# %s" % title, ""]
    meta = [("Session", sess.sid), ("Started", fmt_ts(sess.start, tz)),
            ("Ended", fmt_ts(sess.end, tz)), ("Working dir", redact(sess.cwd or "")),
            ("Git branch", sess.branch or ""), ("Source", sess.entrypoint or "")]
    out += ["- **%s:** %s" % (k, v) for k, v in meta if v] + ["", "---", ""]
    for e in events:
        tag = " *(subagent)*" if e["side"] else ""
        when = fmt_ts(e["ts"], tz)
        if e["kind"] == "user":
            out += ["### User%s — %s" % (tag, when), "", e["text"], ""]
        elif e["kind"] == "assistant":
            out += ["### Claude%s — %s" % (tag, when), "", e["text"], ""]
        elif e["kind"] == "thinking":
            out += ["<details><summary>Claude's reasoning</summary>", "", e["text"], "", "</details>", ""]
        elif e["kind"] == "tool_use":
            f = fence(e["text"])
            out += ["<details><summary>Tool call — %s</summary>" % html.escape(e["name"]), "",
                    f, e["text"] or "", f, "", "</details>", ""]
        elif e["kind"] == "tool_result":
            f = fence(e["text"])
            label = "Tool error" if e["error"] else "Tool output"
            out += ["<details><summary>%s — %s</summary>" % (label, html.escape(e["name"])), "",
                    f, e["text"] or "(empty)", f, "", "</details>", ""]
    return "\n".join(out)


CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#6e6e73;--user:#eef3ff;--claude:#f6f6f4;--tool:#fafafa;--border:#e2e2e2;--err:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#ececec;--muted:#9a9a9a;--user:#1f2a3d;--claude:#222;--tool:#1c1c1c;--border:#333;--err:#ff8a80}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:860px;margin:0 auto;padding:24px 16px}
h1{font-size:1.5em;margin-bottom:.3em} .meta{color:var(--muted);font-size:.9em;margin-bottom:1.5em}
.msg{border:1px solid var(--border);border-radius:10px;padding:10px 14px;margin:12px 0}
.user{background:var(--user)} .assistant{background:var(--claude)}
.who{font-weight:600;font-size:.85em;color:var(--muted);margin-bottom:4px}
.text{white-space:pre-wrap;word-wrap:break-word}
details{margin:6px 0 6px 16px;font-size:.88em} summary{cursor:pointer;color:var(--muted)}
details.err summary{color:var(--err)}
pre{background:var(--tool);border:1px solid var(--border);border-radius:6px;padding:8px;overflow-x:auto;white-space:pre-wrap;word-wrap:break-word}
a{color:inherit} table{border-collapse:collapse;width:100%} td,th{border-bottom:1px solid var(--border);padding:6px;text-align:left;vertical-align:top}
"""


def render_html(sess, events, opts, redact, tz):
    title = redact(session_title(sess))
    h = ["<!doctype html><html><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width,initial-scale=1'>",
         "<title>%s</title><style>%s</style></head><body>" % (html.escape(title), CSS),
         "<h1>%s</h1>" % html.escape(title),
         "<div class='meta'>%s → %s · %s</div>" % (
             fmt_ts(sess.start, tz), fmt_ts(sess.end, tz), html.escape(redact(sess.cwd or "")))]
    for e in events:
        tag = " (subagent)" if e["side"] else ""
        when = fmt_ts(e["ts"], tz)
        if e["kind"] in ("user", "assistant"):
            who = "User" if e["kind"] == "user" else "Claude"
            h.append("<div class='msg %s'><div class='who'>%s%s · %s</div><div class='text'>%s</div></div>"
                     % (e["kind"], who, tag, when, html.escape(e["text"])))
        elif e["kind"] == "thinking":
            h.append("<details><summary>Claude's reasoning</summary><pre>%s</pre></details>"
                     % html.escape(e["text"]))
        elif e["kind"] == "tool_use":
            h.append("<details><summary>Tool call — %s</summary><pre>%s</pre></details>"
                     % (html.escape(e["name"]), html.escape(e["text"] or "")))
        elif e["kind"] == "tool_result":
            cls = " class='err'" if e["error"] else ""
            label = "Tool error" if e["error"] else "Tool output"
            h.append("<details%s><summary>%s — %s</summary><pre>%s</pre></details>"
                     % (cls, label, html.escape(e["name"]), html.escape(e["text"] or "(empty)")))
    h.append("</body></html>")
    return "\n".join(h)


def slug(s, n=50):
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s[:n] or "session"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", "--keywords", nargs="*", default=[],
                    help="keep sessions mentioning any of these (case-insensitive). Omit to keep all.")
    ap.add_argument("-o", "--out", default="claude_chat_export", help="output directory")
    ap.add_argument("--root", action="append", default=[], help="extra directory to search (repeatable)")
    ap.add_argument("--no-default-roots", action="store_true", help="search only --root directories")
    ap.add_argument("--since", help="only sessions starting on/after YYYY-MM-DD")
    ap.add_argument("--until", help="only sessions starting on/before YYYY-MM-DD")
    ap.add_argument("--format", choices=["md", "html", "both"], default="both")
    ap.add_argument("--list", action="store_true", help="list matching sessions and exit")
    ap.add_argument("--no-tools", dest="tools", action="store_false", help="omit tool calls and outputs")
    ap.add_argument("--max-tool-lines", type=int, default=40,
                    help="truncate each tool call/output to N lines (0 = no limit; default 40)")
    ap.add_argument("--thinking", action="store_true", help="include Claude's reasoning blocks")
    ap.add_argument("--include-subagents", action="store_true", help="include subagent (sidechain) messages")
    ap.add_argument("--no-redact-paths", dest="redact_paths", action="store_false",
                    help="keep full home-directory paths")
    ap.add_argument("--no-redact-secrets", dest="redact_secrets", action="store_false",
                    help="keep strings that look like API keys/tokens")
    ap.add_argument("--redact-emails", action="store_true", help="replace email addresses")
    ap.add_argument("--redact", action="append", default=[], metavar="REGEX",
                    help="extra regex to replace with [REDACTED] (repeatable)")
    ap.add_argument("--min-messages", type=int, default=1, help="skip sessions with fewer user prompts")
    opts = ap.parse_args()

    roots = ([] if opts.no_default_roots else DEFAULT_ROOTS) + [Path(r) for r in opts.root]
    redact = Redactor(opts.redact_paths, opts.redact_secrets, opts.redact_emails, opts.redact)
    tz = dt.datetime.now().astimezone().tzinfo
    kws = [k.lower() for k in opts.keywords]

    sessions = {}
    nfiles = 0
    for p in iter_jsonl_files(roots):
        recs = load_records(p)
        if not looks_like_transcript(recs):
            continue
        nfiles += 1
        by_sid = defaultdict(list)
        for r in recs:
            by_sid[r.get("sessionId") or p.stem].append(r)
        for sid, rs in by_sid.items():
            sessions.setdefault(sid, Session(sid)).add(p, rs)

    def keep(s):
        if not s.records:
            return False
        nprompts = sum(1 for r in s.records.values() if r["type"] == "user"
                       and any(b.get("type") == "text" for b in blocks_of(r.get("message", {}))))
        if nprompts < opts.min_messages:
            return False
        day = s.start[:10]
        if opts.since and day < opts.since:
            return False
        if opts.until and day > opts.until:
            return False
        if kws:
            text = s.searchable_text()
            return any(k in text for k in kws)
        return True

    chosen = sorted((s for s in sessions.values() if keep(s)), key=lambda s: s.start)
    print("Scanned %d transcript files, %d sessions; %d match." % (nfiles, len(sessions), len(chosen)),
          file=sys.stderr)

    if opts.list or not chosen:
        for s in chosen:
            print("%s  %-36s  %s" % (fmt_ts(s.start, tz), s.sid, redact(session_title(s))[:70]))
        return

    out = Path(opts.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for i, s in enumerate(chosen, 1):
        events = build_events(s, opts, redact)
        if not any(e["kind"] in ("user", "assistant") for e in events):
            continue
        base = "%03d_%s_%s" % (i, s.start[:10], slug(redact(session_title(s))))
        files = []
        if opts.format in ("md", "both"):
            (out / (base + ".md")).write_text(render_md(s, events, opts, redact, tz), encoding="utf-8")
            files.append(base + ".md")
        if opts.format in ("html", "both"):
            (out / (base + ".html")).write_text(render_html(s, events, opts, redact, tz), encoding="utf-8")
            files.append(base + ".html")
        nuser = sum(e["kind"] == "user" for e in events)
        index.append((s, files, nuser))

    # index files
    md = ["# Claude session export", "",
          "Keywords: %s  " % (", ".join(opts.keywords) or "(all)"),
          "Generated: %s" % dt.datetime.now(tz).strftime("%Y-%m-%d %H:%M"), "",
          "| # | Started | Title | Prompts | Files |", "|---|---|---|---|---|"]
    rows = []
    for n, (s, files, nuser) in enumerate(index, 1):
        t = redact(session_title(s)).replace("|", "/")
        md.append("| %d | %s | %s | %d | %s |" % (n, fmt_ts(s.start, tz), t, nuser,
                                                  " · ".join("[%s](%s)" % (f.rsplit(".", 1)[1], f) for f in files)))
        link = next((f for f in files if f.endswith(".html")), files[0])
        rows.append("<tr><td>%d</td><td>%s</td><td><a href='%s'>%s</a></td><td>%d</td></tr>"
                    % (n, fmt_ts(s.start, tz), link, html.escape(t), nuser))
    (out / "index.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if opts.format in ("html", "both"):
        (out / "index.html").write_text(
            "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
            "content='width=device-width,initial-scale=1'><title>Claude session export</title>"
            "<style>%s</style></head><body><h1>Claude session export</h1>"
            "<div class='meta'>Keywords: %s</div><table><tr><th>#</th><th>Started</th><th>Title</th>"
            "<th>Prompts</th></tr>%s</table></body></html>"
            % (CSS, html.escape(", ".join(opts.keywords) or "(all)"), "\n".join(rows)),
            encoding="utf-8")
    print("Wrote %d sessions to %s" % (len(index), out), file=sys.stderr)
    print("Review the output before sharing: redaction is pattern-based and not exhaustive.", file=sys.stderr)


if __name__ == "__main__":
    main()