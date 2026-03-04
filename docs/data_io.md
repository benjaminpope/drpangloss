<!-- AUTO-GENERATED FROM /home/runner/work/drpangloss/drpangloss/notebooks/data_io.ipynb by scripts/sync_tutorial_docs.py. -->
<!-- Edit the notebook, then re-run the sync script. -->

# Data I/O

`drpangloss` has I/O tools for reading and writing `.oifits` files, which are the data standard in interferometry. These are largely cribbed from [`ImPlaneIA`](https://github.com/anand0xff/ImPlaneIA).

```python
import numpy as np
import pyoifits as oifits

from pathlib import Path
import sys

notebook_dir = (
    (Path.cwd() / "notebooks")
    if (Path.cwd() / "notebooks").exists()
    else Path.cwd()
)
if str(notebook_dir) not in sys.path:
    sys.path.insert(0, str(notebook_dir))

from tutorial_helpers import find_repo_root, load_synthetic_workflow_module

repo_root = find_repo_root()
module = load_synthetic_workflow_module(repo_root)
```

## Simulate Data
Let's simulate some synthetic data and save it to an `.oifits` file:

```python
out = (
    repo_root / "docs" / "generated" / "synthetic_binary_from_notebook.oifits"
).resolve()

synth_dict, truth, noise_settings = module._build_synthetic_oifits_dict(seed=4)
module.save_oifits_dict(
    synth_dict,
    filename=out.name,
    datadir=str(out.parent),
    verbose=True,
)
```

```text


### Init creation of OI_FITS (synthetic_binary_from_notebook.oifits) :
-> Including OI Wavelength table...
-> Including OI Target table...
-> Including OI Array table...
-> Including OI Vis table...
-> Including OI Vis2 table...
-> Including OI T3 table...


### OIFITS CREATED (synthetic_binary_from_notebook.oifits).
```

# Reading Data

Let's read the data - this is easy!

```python
loaded = oifits.open(str(out))
oidata = module.OIData(loaded)
```

## OIData Object

OIData knows automatically whether you're using visibilities or squared visibilities, which are just saved as `oidata.vis` with uncertainty `oidata.d_vis` and toggled with `oidata.v2_flag`. 

Likewise OIData can tell if you're using closure phases or absolute phases, which are saved just as `oidata.phi` with uncertainty `oidata.d_phi` and toggled with `oidata.cp_flag`.

$u,v$ information is saved in `oidata.u` and `oidata.v`, with closure phase indices in `oidata.i_cps1`, `oidata.i_cps2`, `oidata.i_cps3`.

When you're using this, you will pass it a model object, which will automatically evaluate it at the appropriate arguments.

```python
# let's have a tour of the oidata object
print(oidata)
```

```text
OIData(u=[ 3.2  1.4 -1.1 -1.8 -4.3 -2.5], v=[ 0.2  2.6  1.8  2.4  1.6 -0.8], CP=[-0.05890521 -0.45451784  0.01607962  0.40036282], d_CP=[0.02 0.02 0.02 0.02], V2=[0.99050045 1.000406   0.9921805  0.9915316  0.9863897  0.995878  ], d_V2=[0.00099192 0.00099192 0.00099192 0.00099192 0.00099192 0.00099192], i_cps1=[0 0 1 3], i_cps2=[3 4 5 5], i_cps3=[1 2 2 4])
```

```python
print("OIData keys:", list(oidata.__dict__.keys()))
```

```text
OIData keys: ['wavel', 'vis', 'd_vis', 'u', 'v', 'v2_flag', 'phi', 'd_phi', 'i_cps1', 'i_cps2', 'i_cps3', 'cp_flag']
```

## Verification
And just to verify, let's make sure all the keys are saved and loaded correctly:

```python
# Compare arrays written to OIFITS with arrays reloaded via OIData.
# We test all core OIData keys in one cell.
expected = {
    "u": np.asarray(synth_dict["OI_VIS2"]["UCOORD"]),
    "v": np.asarray(synth_dict["OI_VIS2"]["VCOORD"]),
    "vis": np.asarray(synth_dict["OI_VIS2"]["VIS2DATA"]),
    "d_vis": np.asarray(synth_dict["OI_VIS2"]["VIS2ERR"]),
    "phi": np.asarray(synth_dict["OI_T3"]["T3PHI"]),
    "d_phi": np.asarray(synth_dict["OI_T3"]["T3PHIERR"]),
    "wavel": np.atleast_1d(
        np.asarray(synth_dict["OI_WAVELENGTH"]["EFF_WAVE"])
    ),
}

reloaded = {
    "u": np.asarray(oidata.u),
    "v": np.asarray(oidata.v),
    "vis": np.asarray(oidata.vis),
    "d_vis": np.asarray(oidata.d_vis),
    "phi": np.asarray(oidata.phi),
    "d_phi": np.asarray(oidata.d_phi),
    "wavel": np.atleast_1d(np.asarray(oidata.wavel)),
}

equality = {
    key: bool(np.allclose(reloaded[key], expected[key], equal_nan=True))
    for key in expected
}

print("All arrays equal:", all(equality.values()))
for key, equal in equality.items():
    print(f"{key}: {'equal' if equal else 'not equal'}")
```

```text
All arrays equal: True
u: equal
v: equal
vis: equal
d_vis: equal
phi: equal
d_phi: equal
wavel: equal
```
