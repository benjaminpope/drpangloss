# Development transcripts

This package was developed with Claude Code. To keep the workflow transparent,
`export_claude_chats.py` collates local Claude Code session transcripts into
readable Markdown/HTML (Python 3.8+, no dependencies).

```bash
python3 docs/misc/export_claude_chats.py -k virgil pangloss -o ~/Desktop/virgil_chats
```

Use `--list` to preview matching sessions, `--help` for all options. Home paths
and likely secrets are redacted by default, but redaction is pattern-based:
review the output before sharing.
