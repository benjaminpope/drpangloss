#!/usr/bin/env python3
"""Refresh the test timings CI uses to shard the suite and police its budget.

Reads one or more JUnit reports from CI (the ``virgil-evidence-*`` artifacts
of a run on main, or the nightly's ``junit-*`` artifacts) and writes

- ``tests/test_durations.json``: seconds per test file, the largest over the
  reports given. ``--shard K/N`` (tests/conftest.py) packs whole files into
  shards with these, so module-scoped fixtures run once per job.
- with ``--over-budget SECONDS``, ``tests/over_budget.txt``: the tests that
  are not marked slow and took longer than SECONDS in any report. They are
  exempt from the per-test budget until they are shrunk or marked slow; the
  list should only ever get shorter.

    gh run download RUN_ID -R benjaminpope/virgil -D /tmp/junit
    python scripts/update_test_durations.py /tmp/junit/*/*.xml

Only the standard library is used.
"""

import argparse
import collections
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DURATIONS = ROOT / "tests" / "test_durations.json"
OVER_BUDGET = ROOT / "tests" / "over_budget.txt"


def node_id(classname, name):
    """The pytest node id of a JUnit test case (xdist group suffix dropped)."""
    parts = classname.split(".")
    # tests.test_x.TestClass -> tests/test_x.py::TestClass
    module = next(i for i, p in enumerate(parts) if p.startswith("test_"))
    path = "/".join(parts[: module + 1]) + ".py"
    name = re.sub(r"@[^\]]*$", "", name)
    return "::".join([path, *parts[module + 1 :], name])


def read(reports):
    tests = collections.defaultdict(float)
    for report in reports:
        for case in ET.parse(report).getroot().iter("testcase"):
            if case.find("skipped") is not None:
                continue
            nid = node_id(case.get("classname"), case.get("name"))
            tests[nid] = max(tests[nid], float(case.get("time", 0)))
    return tests


def per_file(reports):
    files = collections.Counter()
    for report in reports:
        one = collections.Counter()
        for nid, seconds in read([report]).items():
            one[nid.split("::")[0]] += seconds
        for path, seconds in one.items():
            files[path] = max(files[path], seconds)
    return files


def slow_tests():
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "-m", "slow"]
        + ["--collect-only", "-q", "-p", "no:cacheprovider", "-W", "ignore"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout
    return {line for line in out.splitlines() if "::" in line}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--over-budget", type=float, metavar="SECONDS")
    args = parser.parse_args()

    files = per_file(args.reports)
    DURATIONS.write_text(
        json.dumps(
            {k: round(v, 1) for k, v in sorted(files.items())}, indent=1
        )
        + "\n"
    )
    print(f"{len(files)} files -> {DURATIONS.relative_to(ROOT)}")

    if args.over_budget is not None:
        slow = slow_tests()
        over = sorted(
            nid
            for nid, seconds in read(args.reports).items()
            if seconds > args.over_budget and nid not in slow
        )
        header = (
            "# Tests over the per-test budget (tests/conftest.py), exempt "
            "until they are\n# shrunk or marked slow. Remove lines; never "
            "add them. Written by\n# scripts/update_test_durations.py "
            f"--over-budget {args.over_budget:g}.\n"
        )
        OVER_BUDGET.write_text(header + "".join(n + "\n" for n in over))
        print(f"{len(over)} tests -> {OVER_BUDGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
