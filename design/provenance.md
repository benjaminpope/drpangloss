# Provenance of virgil's methods

The public narrative (origin of the package, who contributed what) and the
tables of sources by module now live on the public page `docs/references.md`.
That page is generated from `references/provenance.toml`, the single list of
sources, by `scripts/sync_references_page.py`; edit the TOML (and
`references/references_intro.md` for the prose), never the page.

- Adding an idea or port: add or extend its TOML entry (`used_in`, `how =
  "ported"` when code was adapted) in the same PR, run
  `python scripts/sync_references_page.py`, and cite it in the docstring in one
  line: `Source: Kammerer et al. (2020), [references](../references.md#kammerer2020).`
- `tests/lint/test_provenance.py` fails on any arXiv ID, GitHub repository or
  "Author et al. (YYYY)" citation in `src/` or `docs/` with no entry, and on a
  `used_in` symbol that does not import.
- Census for a paper: `python scripts/sync_references_page.py --by-module`
  and `--unverified`; `scripts/fetch_references.py` resolves entries against
  NASA ADS and writes `references/virgil.bib`.
- Public docs never link `design/*.md`.

## Open points

- `docs/contributors.md` credits Skilling 1989 with DOI `…_4`; Crossref gives
  that DOI to Gull 1989 ("Developments in Maximum Entropy Data Analysis") and
  `…_3` to Skilling's "Classic Maximum Entropy".
- `virgil.ensemble` cited Drevon et al. as 2025; arXiv:2609.15365 is dated
  2026.
- fouriever has no paper; the author attribution (Kammerer) is from the
  GitHub repository owner.
- jaxoplanet, exoplanet and AMICAL are in `virgil.bib` for the paper
  (`role = "bibliography"`) but are imports or separate repositories, not
  provenance. eht-imaging is a comparison only; nothing was taken from it.
  Entries with `role = "compared"` were studied for comparison, not ported.
- ADS resolved 88 of 100 entries (bibcodes in the TOML; BibTeX in
  `references/virgil.bib`, keyed by our bibkeys). ADS does not index Gull 1989,
  Skilling 1989, MacKay 1992, Wilks 1938 or Efron 1979 under their DOIs (their
  BibTeX is written from the TOML), nor Thiele 1883 or Starck et al. 2010.
- Package-only entries (CANDID, fouriever, AMICAL, ImPlaneIA, nifty, zodiax,
  GDit, frito, jaxbessel, CEPHES) have a URL and no paper, so ADS cannot
  resolve them.
