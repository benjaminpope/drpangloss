"""Evidence for virgil-validation.

Tests can declare what they validate, and against which root of trust, with

    @pytest.mark.validates("virgil.models.UniformDisk", roots=["mathematics"])

(same marker and roots as https://github.com/benjaminpope/virgil-validation,
docs/design.md). Each claim is attached to the test as a JUnit property
``validates``, so `pytest --junitxml=... -o junit_family=xunit1` carries it;
CI uploads that file from main and virgil-validation reads it as evidence.
Tests without the marker are unaffected.
"""

import json

import pytest

ROOTS = (
    "mathematics",
    "standards",
    "dlux",
    "pmoired",
    "candid",
    "fouriever",
    "statistics",
    "self-consistency",
)
KINDS = ("check", "control", "finding", "upstream", "reference", "guard")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "validates(obj, *more, roots, kind='check', tier='A'): virgil objects "
        "this test validates and the roots of trust it checks them against "
        "(see virgil-validation's docs/design.md)",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        for mark in item.iter_markers("validates"):
            roots = list(mark.kwargs.get("roots", ()))
            kind = mark.kwargs.get("kind", "check")
            bad = [
                r
                for r in roots
                if r not in ROOTS and not r.startswith("golden:")
            ]
            if not mark.args or not roots or bad or kind not in KINDS:
                raise pytest.UsageError(
                    f"{item.nodeid}: malformed validates marker"
                )
            claim = {
                "objects": list(mark.args),
                "roots": roots,
                "kind": kind,
                "tier": mark.kwargs.get("tier", "A"),
            }
            item.user_properties.append(("validates", json.dumps(claim)))
