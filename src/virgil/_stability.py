"""Stability tiers: the policy text, and the admonition provisional code shows.

Imports nothing, so ``scripts/sync_tutorial_docs.py`` can load it by path.

Each module with provisional public names lists them in a module-level
``__provisional__`` tuple (``("*",)`` for the whole module). A test checks
that every listed name exists and that its docstring carries
:data:`ADMONITION`. Nothing warns at runtime.
"""

POLICY = """\
Stable: a change needs a deprecation cycle (a release that still accepts the
old form and warns). Provisional: the name, signature, defaults and output
may change in any 0.4.x release without deprecation. New public features
start provisional; promotion to stable is decided at a release."""

TITLE = "Provisional"

SUMMARY = "May change in any 0.4.x release without a deprecation cycle."

ADMONITION = f'!!! warning "{TITLE}"\n    {SUMMARY}'
