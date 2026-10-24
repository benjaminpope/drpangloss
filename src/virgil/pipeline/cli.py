"""The ``virgil-pipeline`` command.

    virgil-pipeline binary DATA... --output DIR [--through S] [--fresh]
                                   [--set key=value ...]
    virgil-pipeline star DATA... --output DIR [--model uniform|limb_darkened]
                                 [--through S] [--fresh] [--set key=value ...]
    virgil-pipeline disk DATA... --output DIR [--model rings|gp]
                                 [--through S] [--fresh] [--set key=value ...]
    virgil-pipeline image DATA... --output DIR [--through S] [--fresh]
                                  [--set key=value ...]
    virgil-pipeline orbit DATA... --output DIR [--metadata FILE.json]
                                  [--through S] [--fresh] [--set key=value ...]
    virgil-pipeline info DIR
    virgil-pipeline quicklook DIR [--no-execute]
    virgil-pipeline validate DIR

``info`` and ``validate`` read only JSON and file hashes: they import
neither the pipeline classes nor the notebook tools, and do no JAX work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._io import SCHEMA, read_json, sha256_file

_PIPELINES = {
    "binary": ("virgil.pipeline.binary", "BinaryPipeline"),
    "disk": ("virgil.pipeline.disk", "DiskPipeline"),
    "image": ("virgil.pipeline.image", "ImagingPipeline"),
    "orbit": ("virgil.pipeline.orbit", "OrbitPipeline"),
    "star": ("virgil.pipeline.star", "StarPipeline"),
}


def _run(args):  # deliberate: argparse callback
    import importlib

    module, name = _PIPELINES[args.command]
    cls = getattr(importlib.import_module(module), name)
    second = getattr(args, "model", None)
    if getattr(args, "metadata", None):
        second = read_json(Path(args.metadata))
    settings = {}
    for pair in args.set or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value; got {pair!r}.")
        try:
            settings[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            settings[key.strip()] = value
    pipeline = cls.from_oifits(
        args.data, second, output=args.output, **settings
    )
    res = pipeline.run(through=args.through, resume=not args.fresh)
    print(res.describe())
    return 0


def _info(args):  # deliberate: argparse callback
    folder = Path(args.dir)
    run = read_json(folder / "run.json")
    print(
        f"{run.get('class', '?')} ({run.get('stability', '?')}), "
        f"schema {run.get('schema')}"
    )
    print(f"status: {run.get('status')}")
    if run.get("error"):
        error = run["error"]
        print(
            f"error in {error['stage']}: {error['type']}: {error['message']}"
        )
    for name, entry in run.get("stages", {}).items():
        print(
            f"  {name:<10} {entry['status']:<9} {entry.get('seconds', 0):8.1f} s"
        )
    summary_file = folder / "summary.json"
    if summary_file.exists():
        summary = read_json(summary_file)
        for check in summary.get("checks", []):
            print(f"  [{check['status']}] {check['name']}: {check['message']}")
    return 0


# Keys every run.json must hold.
_REQUIRED_KEYS = (
    "pipeline",
    "class",
    "config",
    "inputs",
    "provenance",
    "stages",
)
_STATUSES = ("running", "partial", "complete", "failed")


def _validate(args):  # deliberate: argparse callback
    folder = Path(args.dir)
    try:
        run = read_json(folder / "run.json")
        if not isinstance(run, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as err:
        print(f"invalid: cannot read run.json ({err})")
        return 1
    problems = []  # the contents of run.json itself
    if run.get("schema") != SCHEMA:
        problems.append(f"unknown schema {run.get('schema')!r}")
    if run.get("status") not in _STATUSES:
        problems.append(f"unknown status {run.get('status')!r}")
    problems += [
        f"run.json lacks {key!r}" for key in _REQUIRED_KEYS if key not in run
    ]
    if run.get("status") == "running":
        problems.append(
            "the run did not finish (status 'running'): it is still running "
            "or was interrupted"
        )
    if run.get("status") == "failed":
        error = run.get("error") or {}
        problems.append(
            f"the run failed in stage {error.get('stage')!r}: "
            f"{error.get('type')}: {error.get('message')}"
        )
    try:
        summary = read_json(folder / "summary.json")
    except (OSError, ValueError) as err:
        problems.append(f"cannot read summary.json ({err})")
    else:
        if summary.get("schema") != run.get("schema"):
            problems.append("summary.json schema differs from run.json")
    # Missing stage outputs, and files that changed since the run.
    for name, entry in run.get("stages", {}).items():
        if entry.get("status") != "complete":
            continue
        problems += [
            f"stage {name}: missing {rel}"
            for rel in entry.get("outputs", [])
            if not (folder / rel).exists()
        ]
    for rel, digest in (run.get("files") or {}).items():
        path = folder / rel
        if not path.exists():
            problems.append(f"missing {rel}")
        elif sha256_file(path) != digest:
            problems.append(f"{rel} changed since the run")
    if problems:
        print("invalid:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"valid: {run['class']} run, status {run['status']}")
    return 0


def _quicklook(args):  # deliberate: argparse callback
    from ._quicklook import write_quicklook

    path = write_quicklook(args.dir, execute=not args.no_execute)
    print(path)
    return 0


def build_parser():
    """The argument parser of ``virgil-pipeline``."""
    parser = argparse.ArgumentParser(
        prog="virgil-pipeline",
        description="Run and inspect virgil pipelines.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in _PIPELINES:
        p = sub.add_parser(name, help=f"run the {name} pipeline")
        p.add_argument("data", nargs="+", help="OIFITS files, read together")
        p.add_argument("--output", "-o", default="run", help="run folder")
        p.add_argument("--through", help="last stage to run")
        p.add_argument(
            "--fresh",
            action="store_true",
            help="start over instead of resuming",
        )
        p.add_argument(
            "--set",
            action="append",
            metavar="KEY=VALUE",
            help="override a setting (VALUE is parsed as JSON if it can be)",
        )
        if name == "star":
            p.add_argument(
                "--model",
                help='fit one model, e.g. "uniform" or "limb_darkened" '
                "(default: both, compared)",
            )
        if name == "orbit":
            p.add_argument(
                "--metadata",
                help="JSON file {file: {epoch, mjd}} (see "
                "OrbitPipeline.from_oifits)",
            )
        if name == "disk":
            p.add_argument(
                "--model",
                choices=("rings", "gp"),
                help='what to fit: "rings" or "gp" (default: gp)',
            )
        p.set_defaults(func=_run)
    p = sub.add_parser("info", help="print a run's status and checks")
    p.add_argument("dir")
    p.set_defaults(func=_info)
    p = sub.add_parser("validate", help="check a run folder is complete")
    p.add_argument("dir")
    p.set_defaults(func=_validate)
    p = sub.add_parser("quicklook", help="rebuild a run's quicklook notebook")
    p.add_argument("dir")
    p.add_argument("--no-execute", action="store_true")
    p.set_defaults(func=_quicklook)
    return parser


def main(argv=None):
    """Entry point of ``virgil-pipeline``."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
