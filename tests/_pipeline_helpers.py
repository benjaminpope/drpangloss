"""Helpers shared by the pipeline robustness tests."""

import json


def checks_by_name(res):
    return {c.name: c for c in res.checks}


def stage_outputs(pipeline, stage):
    return {s.name: s.outputs for s in pipeline._stages()}[stage]


def started_times(pipeline):
    run = json.loads((pipeline.output / "run.json").read_text())
    return {name: s.get("started") for name, s in run["stages"].items()}
