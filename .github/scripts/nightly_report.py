"""Open, update or close the one nightly-failure issue (tests.yml).

    RESULT=failure RUN_URL=... python3 .github/scripts/nightly_report.py DIR

DIR holds the nightly's JUnit reports (one ``junit-*`` artifact per job).
On failure, the failing test ids, the run and the commits since the last
green nightly go into the open issue labelled ``nightly-failure``, or a new
one. On success, an open issue is closed. Either way it then deletes the JAX
caches of earlier weeks, keeping the store under GitHub's 10 GB. Uses the
``gh`` CLI and the standard library only.
"""

import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

LABEL = "nightly-failure"


def gh(*args):
    return subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def failures(directory):
    """{job: [failing test ids]} from the JUnit reports."""
    out = {}
    for report in sorted(Path(directory).glob("*/*.xml")):
        failed = [
            f"{case.get('classname')}::{case.get('name')}"
            for case in ET.parse(report).getroot().iter("testcase")
            if case.find("failure") is not None
            or case.find("error") is not None
        ]
        if failed:
            out[report.parent.name.removeprefix("junit-")] = failed
    return out


def last_green():
    runs = json.loads(
        gh(
            "run",
            "list",
            "--workflow",
            "tests.yml",
            "--event",
            "schedule",
            "--status",
            "success",
            "--limit",
            "1",
            "--json",
            "headSha",
        )
    )
    return runs[0]["headSha"] if runs else None


def body(directory):
    repo = os.environ["GITHUB_REPOSITORY"]
    sha = os.environ["GITHUB_SHA"]
    lines = [f"Nightly run {os.environ['RUN_URL']} failed at {sha[:12]}.", ""]
    found = failures(directory)
    if not found:
        lines.append(
            "No failing tests in the JUnit reports: a job failed outside "
            "pytest (environment, timeout, packaging). See the run."
        )
    for job, ids in found.items():
        lines.append(f"**{job}** ({len(ids)})")
        lines += [f"- `{i}`" for i in ids[:30]]
        if len(ids) > 30:
            lines.append(f"- ... and {len(ids) - 30} more")
        lines.append("")
    green = last_green()
    if green and green != sha:
        lines.append(
            f"Commits since the last green nightly: "
            f"https://github.com/{repo}/compare/{green[:12]}...{sha[:12]}"
        )
    elif not green:
        lines.append("No green nightly on record yet.")
    return "\n".join(lines)


def clean_caches():
    """Delete every ``jax-`` cache not of the current ISO week."""
    week = time.strftime("%G-W%V", time.gmtime())
    caches = json.loads(
        gh("cache", "list", "--key", "jax-", "--limit", "500",
           "--json", "id,key")
    )
    for cache in caches:
        if not cache["key"].endswith(f"-{week}"):
            gh("cache", "delete", str(cache["id"]))


def report(directory):
    open_issue = gh(
        "issue",
        "list",
        "--label",
        LABEL,
        "--state",
        "open",
        "--json",
        "number",
        "--jq",
        ".[0].number // empty",
    )
    if os.environ["RESULT"] == "success":
        if open_issue:
            gh(
                "issue",
                "close",
                open_issue,
                "--comment",
                f"Nightly run {os.environ['RUN_URL']} is green.",
            )
        return
    text = body(directory)
    print(text)
    if open_issue:
        gh("issue", "comment", open_issue, "--body", text)
    else:
        gh(
            "label",
            "create",
            LABEL,
            "--force",
            "--color",
            "B60205",
            "--description",
            "The nightly CI run is failing",
        )
        gh(
            "issue",
            "create",
            "--title",
            "Nightly CI is failing",
            "--label",
            LABEL,
            "--body",
            text,
        )


def main():
    try:
        report(sys.argv[1])
    finally:
        clean_caches()


if __name__ == "__main__":
    main()
