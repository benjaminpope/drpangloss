# Changelog fragments

Never edit `CHANGELOG.md` in a PR; add one file here per change instead, named `<PR>.<type>.md` (`<PR>.<type>.1.md` for a second entry), or `+slug.<type>.md` with no PR.
Each file is one terse bullet (no leading `- `); towncrier appends the PR link.
Types: `default` (changes to default behaviour), `added`, `changed`, `fixed`, `deprecated`, `removed`, `docs`.
The release PR runs `towncrier build --yes --version X.Y.Z`.
