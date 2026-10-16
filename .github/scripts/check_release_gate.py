"""Refuse a release whose commit has no green full test run (publish.yml).

    GH_REPO=owner/repo SHA=<tagged commit> python3 .github/scripts/check_release_gate.py

Passes when tests.yml completed successfully on exactly SHA in the full
tier: a dispatched or scheduled run, which adds macOS py3.11, py3.13 and
the dependency floors to the gate (see ci_matrix.py). A push run lacks
those jobs, so it does not count. Uses the ``gh`` CLI and the standard library only.
"""

import json
import os
import subprocess
import sys

# Events whose run is in the full tier (ci_matrix.py).
FULL_EVENTS = ("workflow_dispatch", "schedule")
# Substrings of the job names the full tier adds to the gate.
FULL_JOBS = ("macos-latest py3.11", "py3.13", "py3.11 lowest")


def gh_api(path):
    out = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", path],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out)


def problems(run, jobs):
    """Why ``run`` (with its ``jobs``) does not clear the gate; [] if it does."""
    if run["conclusion"] != "success":
        return [f"concluded {run['conclusion']}"]
    names = [j["name"] for j in jobs]
    missing = [m for m in FULL_JOBS if not any(m in n for n in names)]
    bad = [
        j["name"]
        for j in jobs
        if j["conclusion"] not in ("success", "skipped")
    ]
    return [f"missing job {m}" for m in missing] + [f"{n} failed" for n in bad]


def main():
    repo, sha = os.environ["GH_REPO"], os.environ["SHA"]
    pages = gh_api(
        f"repos/{repo}/actions/workflows/tests.yml/runs"
        f"?head_sha={sha}&status=completed&per_page=100"
    )
    runs = [r for page in pages for r in page["workflow_runs"]]
    full = [
        r for r in runs if r["event"] in FULL_EVENTS and r["head_sha"] == sha
    ]
    notes = []
    for run in full:
        jobs = [
            j
            for page in gh_api(
                f"repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100"
            )
            for j in page["jobs"]
        ]
        why = problems(run, jobs)
        if not why:
            print(f"{sha}: green full run {run['html_url']}")
            return 0
        notes.append(f"  {run['html_url']} ({run['event']}): {'; '.join(why)}")
    print(
        f"::error::No green full test run on {sha}. Tag only a commit on "
        "which tests.yml succeeded in the full tier (macOS, py3.13 and the "
        "dependency floors). Run it with\n"
        f"  gh workflow run tests.yml --ref <branch or tag at {sha[:12]}>\n"
        "wait for it to pass, then publish the release again (re-run this "
        "workflow)."
    )
    print(
        "\n".join(notes)
        or f"  {len(runs)} completed tests.yml runs on this commit, none "
        "dispatched or scheduled."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
