# Generate, save, load, and analyse synthetic OIFITS data

This tutorial shows a complete roundtrip using DrPangloss utilities:

1. Generate synthetic binary interferometry observables with realistic OIFITS headers.
2. Save them to an OIFITS file.
3. Load the file back into the analysis pipeline.
4. Briefly analyse recovery with grid search and HMC.

The implementation lives in:

- `examples/synthetic_binary_workflow.py`

## Run the workflow

```python
from examples.synthetic_binary_workflow import (
    run_synthetic_binary_demo,
    within_two_sigma,
)

summary = run_synthetic_binary_demo("docs/generated/synthetic_binary.oifits")
checks = within_two_sigma(summary)

print("Saved:", summary.output_file)
print("Truth:", summary.truth)
print("Grid:", summary.grid_estimate)
print("HMC median:", summary.hmc_median)
print("HMC std:", summary.hmc_std)
print("Within 2σ:", checks)
```

## What this gives you

- A concrete saved OIFITS file with required tables and headers.
- A documented pattern to create reproducible synthetic datasets for testing and examples.
- A compact recovery analysis that can be used in CI checks.

## CI policy for synthetic docs workflows

In CI, this synthetic workflow is executed as a test and fails if either condition is not met:

- Any runtime error occurs in generation, save/load, or recovery.
- Recovered binary parameters `(dra, ddec, flux)` are inconsistent with truth at more than `2σ` under HMC posterior uncertainty.
