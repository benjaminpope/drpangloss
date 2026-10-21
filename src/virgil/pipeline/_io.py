"""Reading and writing pipeline outputs, and the [`Result`][virgil.pipeline.Result] of a run.

Every file is written atomically (to a temporary file in the same folder,
then ``os.replace``), so an interrupted run never leaves a half-written
output behind. Nothing is pickled: arrays go to HDF5 or ``.npz`` (read with
``allow_pickle=False``), everything else to JSON, and models are stored as
a recursive spec of public virgil classes (see :func:`model_spec`).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib
import json
import os
import tempfile
from pathlib import Path

import numpy as onp

from .. import _utils
from .._utils import clean_json  # noqa: F401 - re-exported
from ._checks import Check

SCHEMA = "virgil-pipeline-run-v1"


def require_extra(module):
    """Import a dependency of the ``pipeline`` extra, or explain how to get it."""
    try:
        return importlib.import_module(module)
    except ImportError as err:
        raise ImportError(
            f"virgil.pipeline needs {module.split('.')[0]}, which is not "
            "installed. Install it with: pip install 'virgil-astro[pipeline]'"
        ) from err


# === ATOMIC WRITES ===


@contextlib.contextmanager
def atomic_path(path):
    """Yield a temporary path that replaces ``path`` on success.

    The temporary file keeps ``path``'s suffix, so writers that infer the
    format from it (matplotlib, numpy) work.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix
    )
    os.close(fd)
    try:
        yield Path(tmp)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp)
        raise


def _json_default(value):
    if isinstance(value, (onp.integer,)):
        return int(value)
    if isinstance(value, (onp.floating,)):
        return _utils._clean_float(float(value))
    if isinstance(value, onp.bool_):
        return bool(value)
    if hasattr(value, "tolist"):
        return clean_json(onp.asarray(value).tolist())
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable.")


def write_json(path, record):
    """Write ``record`` as indented, strict JSON, atomically."""
    text = json.dumps(
        clean_json(record),
        indent=2,
        default=_json_default,
        ensure_ascii=False,
        allow_nan=False,
    )
    with atomic_path(path) as tmp:
        tmp.write_text(text + "\n", encoding="utf-8")


def read_json(path):
    """Read a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    """SHA-256 of a file's bytes, as hex."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def data_fingerprint(data):
    """Hash of every field of an ``OIData``, for the resume check.

    This is ``detection._fingerprint`` (arrays by dtype, shape and bytes,
    every static field, the class of every node), so a change to the
    stations, ``t_ref``, ``dt``, gains, closure offsets, extra
    observables or uv grid is noticed as well as one to the observables.
    """
    from ..detection import _fingerprint  # lazy: pulls in JAX

    digest = _fingerprint(data)
    if digest is None:
        raise ValueError(
            "The data cannot be fingerprinted (a field holds an object "
            "that cannot be hashed reproducibly), so a pipeline run "
            "cannot check that it resumes on the same data."
        )
    return digest


# === HDF5 ===


def write_h5(path, groups, attrs=None):
    """Write nested dicts of arrays to HDF5, atomically.

    ``groups`` maps group name to ``{dataset: array}``; a dataset may be
    given as ``(array, {attr: value})``. ``attrs`` maps group name (``"/"``
    for the file) to attributes; values that are not scalars or strings
    are stored as JSON strings.
    """
    h5py = require_extra("h5py")
    with atomic_path(path) as tmp:
        with h5py.File(tmp, "w") as f:
            for group_name, datasets in groups.items():
                group = f.require_group(group_name)
                for name, value in datasets.items():
                    meta = {}
                    if isinstance(value, tuple):
                        value, meta = value
                    ds = group.create_dataset(name, data=onp.asarray(value))
                    for key, item in meta.items():
                        ds.attrs[key] = _h5_attr(item)
            for group_name, items in (attrs or {}).items():
                target = (
                    f if group_name == "/" else f.require_group(group_name)
                )
                for key, item in items.items():
                    target.attrs[key] = _h5_attr(item)


def _h5_attr(value):
    if isinstance(value, (str, int, float, bool, onp.number)):
        return value
    return json.dumps(clean_json(value), default=_json_default)


def read_h5(path, group):
    """Read every dataset of ``group`` into a dict of numpy arrays."""
    h5py = require_extra("h5py")
    with h5py.File(path, "r") as f:
        if group not in f:
            return {}
        return {name: onp.asarray(ds) for name, ds in f[group].items()}


def read_h5_attrs(path, group="/"):
    """Attributes of ``group``, with JSON strings decoded."""
    h5py = require_extra("h5py")
    with h5py.File(path, "r") as f:
        target = f if group == "/" else f[group]
        out = {}
        for key, value in target.attrs.items():
            if isinstance(value, bytes):
                value = value.decode()
            if isinstance(value, str):
                with contextlib.suppress(ValueError):
                    value = json.loads(value)
            elif isinstance(value, onp.generic):
                value = value.item()
            out[key] = value
        return out


# === MODELS: RECURSIVE SPECS OF PUBLIC CLASSES, NO PICKLES ===


class ModelSpecError(ValueError):
    """A model cannot be stored as, or rebuilt from, a spec of public classes."""


def _public_class(obj):
    cls = type(obj)
    module = cls.__module__
    if not module.startswith("virgil.") or cls.__name__.startswith("_"):
        raise ModelSpecError(
            f"{module}.{cls.__qualname__} is not a public virgil class."
        )
    if getattr(importlib.import_module(module), cls.__name__, None) is not cls:
        raise ModelSpecError(
            f"{module}.{cls.__qualname__} is not importable by name."
        )
    return {"module": module, "class": cls.__name__}


def model_spec(model):
    """A JSON spec and arrays that rebuild ``model`` without pickling.

    The spec is recursive: each equinox module becomes its public class
    (module and name) with its dataclass fields, nested modules become
    nested specs, and arrays are stored by reference into the returned
    ``arrays`` dict (written to ``values.npz``). Only public virgil classes
    and plain values (numbers, strings, booleans, ``None``, tuples, lists,
    dicts of these) are accepted; anything else raises
    :class:`ModelSpecError`.

    Returns
    -------
    tuple
        ``(spec, arrays)``.
    """
    import dataclasses

    import equinox as eqx

    arrays = {}

    def encode(value):
        if isinstance(value, eqx.Module):
            node = _public_class(value)
            node["fields"] = {
                f.name: encode(getattr(value, f.name))
                for f in dataclasses.fields(value)
            }
            return {"$module": node}
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            return {"$float": repr(value)}
        if isinstance(value, (tuple, list)):
            key = "$tuple" if isinstance(value, tuple) else "$list"
            return {key: [encode(v) for v in value]}
        if isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise ModelSpecError("only dicts with string keys are stored.")
            return {"$dict": {k: encode(v) for k, v in value.items()}}
        if (
            isinstance(value, (onp.ndarray, onp.generic))
            or hasattr(value, "__jax_array__")
            or type(value).__module__.startswith("jax")
        ):
            name = f"leaf_{len(arrays)}"
            kind = (
                "numpy"
                if isinstance(value, (onp.ndarray, onp.generic))
                else "jax"
            )
            arrays[name] = onp.asarray(value)
            return {"$array": name, "kind": kind}
        raise ModelSpecError(
            f"cannot store a {type(value).__name__} in a model spec."
        )

    return encode(model), arrays


def model_from_spec(spec, arrays):
    """Rebuild a model from :func:`model_spec` output."""
    import dataclasses

    import jax.numpy as np

    def decode(node):
        if not isinstance(node, dict):
            return node
        if "$module" in node:
            info = node["$module"]
            cls = getattr(
                importlib.import_module(info["module"]), info["class"]
            )
            fields = {f.name for f in dataclasses.fields(cls)}
            stored = set(info["fields"])
            if fields != stored:
                raise ModelSpecError(
                    f"{info['class']} now has fields {sorted(fields)} but the "
                    f"stored model has {sorted(stored)}."
                )
            obj = object.__new__(cls)
            for name, value in info["fields"].items():
                object.__setattr__(obj, name, decode(value))
            return obj
        if "$array" in node:
            array = onp.asarray(arrays[node["$array"]])
            return np.asarray(array) if node.get("kind") == "jax" else array
        if "$float" in node:
            return float(node["$float"])
        if "$tuple" in node:
            return tuple(decode(v) for v in node["$tuple"])
        if "$list" in node:
            return [decode(v) for v in node["$list"]]
        if "$dict" in node:
            return {k: decode(v) for k, v in node["$dict"].items()}
        raise ModelSpecError(f"unknown spec node {sorted(node)}.")

    return decode(spec)


def spec_digest(spec, arrays):
    """A short hash of a model spec and its arrays, for resume checks."""
    h = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
    for name in sorted(arrays):
        array = onp.ascontiguousarray(arrays[name])
        h.update(f"{name}|{array.dtype.str}|{array.shape}|".encode())
        h.update(array.tobytes())
    return h.hexdigest()


def save_model(folder, model, values, params):
    """Write ``manifest.json`` and ``values.npz`` for a model.

    ``values`` maps parameter path to value (the fitted parameters); they
    are stored under ``param/<path>`` beside the spec's arrays.
    """
    folder = Path(folder)
    try:
        spec, arrays = model_spec(model)
        error = None
    except ModelSpecError as err:
        spec, arrays, error = None, {}, str(err)
    manifest = {
        "schema": SCHEMA,
        "class": type(model).__name__,
        "module": type(model).__module__,
        "params": list(params),
        "value_keys": list(values),
        "spec": spec,
        "rebuildable": spec is not None,
        "error": error,
    }
    store = {f"spec/{k}": v for k, v in arrays.items()}
    store.update({f"param/{k}": onp.asarray(v) for k, v in values.items()})
    with atomic_path(folder / "values.npz") as tmp:
        onp.savez(tmp, **store)
    write_json(folder / "manifest.json", manifest)


def load_model_values(folder):
    """The stored parameter values of a saved model, as path → array."""
    with onp.load(Path(folder) / "values.npz", allow_pickle=False) as f:
        return {
            k[len("param/") :]: onp.asarray(f[k])
            for k in f.files
            if k.startswith("param/")
        }


def load_model(folder):
    """Rebuild a model saved with :func:`save_model`, values set."""
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    if not manifest["rebuildable"]:
        raise ModelSpecError(
            f"The {manifest['class']} model was not stored as a spec "
            f"({manifest['error']}); its parameter values are available "
            "from Result.model_values()."
        )
    with onp.load(folder / "values.npz", allow_pickle=False) as f:
        arrays = {
            k[len("spec/") :]: onp.asarray(f[k])
            for k in f.files
            if k.startswith("spec/")
        }
    return model_from_spec(manifest["spec"], arrays)


# === SKY IMAGES AS FITS ===

_MAS_PER_DEG = 3.6e6


@dataclasses.dataclass(frozen=True)
class SkyImage:
    """An image read back from a run's FITS file by [`Result.image`][virgil.pipeline.Result.image].

    Attributes
    ----------
    pixels : numpy.ndarray, shape (ny, nx)
        Relative flux per pixel, in virgil's orientation: row 0 is North and
        column 0 is East.
    pixel_scale_mas : float
        Pixel size.
    dra, ddec : float
        Offset of the image centre from the phase centre, in mas (East,
        North).
    beam : dict or None
        ``major_mas``, ``minor_mas`` and ``pa_deg`` of the data's beam.
    header : dict
        Every header keyword of the primary HDU (``STARFRAC``, the star's
        share of the total flux, ``NMEMBERS`` and so on).
    """

    pixels: onp.ndarray
    pixel_scale_mas: float
    dra: float
    ddec: float
    beam: dict | None
    header: dict


def write_fits_image(
    path,
    pixels,
    pixel_scale_mas,
    *,
    dra=0.0,
    ddec=0.0,
    beam=None,
    header=None,
    extensions=None,
):
    """Write a sky image as FITS with a celestial WCS, atomically.

    ``pixels`` follows virgil's orientation (row 0 North, column 0 East) and
    is stored flipped so that, as usual in FITS, row 0 is the southern edge
    and the first axis runs towards the West: ``CDELT1`` is negative. The
    coordinates are ``RA---SIN`` and ``DEC--SIN`` offsets about the phase
    centre, which sits at (0, 0) deg, so the reference pixel (the image
    centre) is at ``CRVAL = (dra, ddec)`` in degrees. The offsets are
    small; the east–west offset is not scaled by cos(δ), since the phase
    centre is placed at δ = 0.

    Parameters
    ----------
    pixels : array, shape (ny, nx)
        The image.
    pixel_scale_mas : float
        Pixel size.
    dra, ddec : float, optional
        Offset of the image centre from the phase centre (East, North), mas.
    beam : virgil.imaging.Beam or dict, optional
        Written as ``BMAJ``, ``BMIN`` (deg) and ``BPA``.
    header : dict, optional
        Extra keywords of the primary HDU.
    extensions : dict, optional
        Name to array of the same shape: further image extensions with the
        same WCS.
    """
    fits = require_extra("astropy.io.fits")
    pixels = onp.asarray(pixels, dtype=float)
    ny, nx = pixels.shape
    scale = float(pixel_scale_mas) / _MAS_PER_DEG

    def hdu(array, cls):
        h = cls(onp.ascontiguousarray(onp.asarray(array, float)[::-1]))
        for key, value in {
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CRPIX1": (nx + 1) / 2.0,
            "CRPIX2": (ny + 1) / 2.0,
            "CRVAL1": float(dra) / _MAS_PER_DEG,
            "CRVAL2": float(ddec) / _MAS_PER_DEG,
            "CDELT1": -scale,
            "CDELT2": scale,
            "RADESYS": "ICRS",
        }.items():
            h.header[key] = value
        return h

    primary = hdu(pixels, fits.PrimaryHDU)
    primary.header["BUNIT"] = "relative flux"
    if beam is not None:
        b = beam if isinstance(beam, dict) else dataclasses.asdict(beam)
        primary.header["BMAJ"] = b["major_mas"] / _MAS_PER_DEG
        primary.header["BMIN"] = b["minor_mas"] / _MAS_PER_DEG
        primary.header["BPA"] = b["pa_deg"]
    for key, value in (header or {}).items():
        primary.header[key] = value
    hdus = [primary]
    for name, array in (extensions or {}).items():
        extra = hdu(array, fits.ImageHDU)
        extra.name = name
        hdus.append(extra)
    with atomic_path(path) as tmp:
        fits.HDUList(hdus).writeto(tmp, overwrite=True)


def read_fits_image(path, extension=None):
    """Read an image written by :func:`write_fits_image` as a ``SkyImage``."""
    fits = require_extra("astropy.io.fits")
    with fits.open(path) as hdul:
        primary = hdul[0].header
        data = hdul[0 if extension is None else extension].data
        header = {
            k: primary[k]
            for k in primary
            if k and k not in ("COMMENT", "HISTORY")
        }
    beam = None
    if "BMAJ" in header:
        beam = {
            "major_mas": float(header["BMAJ"]) * _MAS_PER_DEG,
            "minor_mas": float(header["BMIN"]) * _MAS_PER_DEG,
            "pa_deg": float(header["BPA"]),
        }
    return SkyImage(
        pixels=onp.asarray(data, dtype=float)[::-1],
        pixel_scale_mas=float(header["CDELT2"]) * _MAS_PER_DEG,
        dra=float(header["CRVAL1"]) * _MAS_PER_DEG,
        ddec=float(header["CRVAL2"]) * _MAS_PER_DEG,
        beam=beam,
        header=header,
    )


# === PROCESSED DATA AS OIFITS ===


_IRREGULAR_CP = "irregular closure-phase layout"
_IRREGULAR_WAVEL = "irregular wavelength layout"


def oidata_tables(data):
    """OIFITS tables for :func:`virgil.oifits.write_oifits` from ``OIData``.

    Supports the common layout: squared visibilities with closure phases
    (needing station indices) or absolute phases, one or more wavelength
    channels per baseline, nothing flagged or projected and no extra
    observables. Returns ``(tables, None)``, or ``(None, reason)`` when the
    data cannot be written faithfully.
    """
    reason = _unsupported_reason(data)
    if reason:
        return None, reason
    layout, reason = _baseline_layout(data)
    if reason:
        return None, reason
    tables = _vis2_tables(data, layout)
    phi = onp.asarray(data.phi, dtype=float)
    if phi.size == 0:
        return tables, None
    phases = _closure_tables if data.cp_flag else _absolute_phase_tables
    more, reason = phases(data, layout)
    if reason:
        return None, reason
    tables.update(more)
    return tables, None


def _unsupported_reason(data):
    """Why ``data`` has more than the supported observables, or ``None``."""
    if data.vis_mat is not None or data.phi_mat is not None:
        return "projected observables (vis_mat/phi_mat)"
    if data.uv_grid is not None:
        return "uv-grid (AMIGO DISCO) data"
    if data.extras:
        return "extra observables"
    if data.gains is not None or data.phase_offsets is not None:
        return "calibration gains or closure offsets"
    if data.vis_index is not None or data.phi_index is not None:
        return "flagged samples"
    if not data.v2_flag or data.vis_mode != "v2":
        return "visibility amplitudes rather than V²"
    return None


def _baseline_layout(data):
    """Samples as ``n_b`` baselines by ``n_w`` channels: ``(layout, reason)``.

    ``layout`` holds ``n_b``, ``n_w``, the channel wavelengths ``waves``,
    the baseline coordinates ``ub``, ``vb``, the ``stations`` of every
    baseline and, if the data carry times, the ``mjd`` of every baseline.
    """
    u = onp.asarray(data.u, dtype=float)
    v = onp.asarray(data.v, dtype=float)
    wavel = onp.broadcast_to(onp.asarray(data.wavel, dtype=float), u.shape)
    waves = onp.asarray(list(dict.fromkeys(wavel.tolist())))
    n_w = waves.size
    if u.size % n_w:
        return None, _IRREGULAR_WAVEL
    n_b = u.size // n_w
    u, v = u.reshape(n_b, n_w), v.reshape(n_b, n_w)
    if not (
        onp.all(wavel.reshape(n_b, n_w) == waves[None, :])
        and onp.all(u == u[:, :1])
        and onp.all(v == v[:, :1])
    ):
        return None, _IRREGULAR_WAVEL
    if onp.asarray(data.vis).size != u.size:
        return None, "visibilities do not match the samples"
    stations, reason = _stations(data, n_b, n_w)
    if reason:
        return None, reason
    mjd = None
    if data.dt is not None and data.t_ref is not None:
        dt = onp.asarray(data.dt, dtype=float).reshape(n_b, n_w)
        mjd = (dt + data.t_ref)[:, 0]
    return {
        "n_b": n_b,
        "n_w": n_w,
        "waves": waves,
        "ub": u[:, 0],
        "vb": v[:, 0],
        "stations": stations,
        "mjd": mjd,
    }, None


def _stations(data, n_b, n_w):
    """Station pair of every baseline (invented if there are no closures)."""
    if data.stations is not None:
        stations = onp.asarray(data.stations, dtype=int)
        return stations.reshape(n_b, n_w, 2)[:, 0], None
    if data.cp_flag:
        return None, "closure phases without station indices"
    index = onp.arange(n_b)
    return onp.stack([2 * index + 1, 2 * index + 2], axis=1), None


def _vis2_tables(data, layout):
    n_b, n_w = layout["n_b"], layout["n_w"]
    tables = {
        "info": {"TARGET": "PIPELINE", "INSTRUME": "VIRGIL"},
        "OI_WAVELENGTH": {"EFF_WAVE": layout["waves"], "EFF_BAND": 0.0},
        "OI_VIS2": {
            "VIS2DATA": onp.asarray(data.vis, float).reshape(n_b, n_w),
            "VIS2ERR": onp.asarray(data.d_vis, float).reshape(n_b, n_w),
            "UCOORD": layout["ub"],
            "VCOORD": layout["vb"],
            "STA_INDEX": layout["stations"],
        },
    }
    if layout["mjd"] is not None:
        tables["OI_VIS2"]["MJD"] = layout["mjd"]
        tables["info"]["MJD"] = float(layout["mjd"][0])
    return tables


def _absolute_phase_tables(data, layout):
    n_b, n_w = layout["n_b"], layout["n_w"]
    phi = onp.rad2deg(onp.asarray(data.phi, dtype=float))
    d_phi = onp.rad2deg(onp.asarray(data.d_phi, dtype=float))
    if phi.size != n_b * n_w:
        return None, "absolute phases do not match the samples"
    table = {
        "VISPHI": phi.reshape(n_b, n_w),
        "VISPHIERR": d_phi.reshape(n_b, n_w),
        "VISAMP": onp.full((n_b, n_w), onp.nan),
        "VISAMPERR": onp.full((n_b, n_w), onp.nan),
        "UCOORD": layout["ub"],
        "VCOORD": layout["vb"],
        "STA_INDEX": layout["stations"],
    }
    if layout["mjd"] is not None:
        table["MJD"] = layout["mjd"]
    return {"OI_VIS": table}, None


def _closure_baselines(data, n_w, n_t):
    """Baseline of each leg of every triangle, or ``None`` if irregular.

    Each leg indexes the flat samples as ``baseline * n_w + channel`` and
    must run through the channels in order within one baseline.
    """
    legs = [
        onp.asarray(i, dtype=int).reshape(n_t, n_w)
        for i in (data.i_cps1, data.i_cps2, data.i_cps3)
    ]
    rows, cols = [leg // n_w for leg in legs], [leg % n_w for leg in legs]
    if not all(
        onp.all(r == r[:, :1]) and onp.all(c == onp.arange(n_w)[None, :])
        for r, c in zip(rows, cols)
    ):
        return None
    return [r[:, 0] for r in rows]


def _closure_tables(data, layout):
    n_w, stations = layout["n_w"], layout["stations"]
    phi = onp.rad2deg(onp.asarray(data.phi, dtype=float))
    d_phi = onp.rad2deg(onp.asarray(data.d_phi, dtype=float))
    if phi.size % n_w:
        return None, _IRREGULAR_CP
    n_t = phi.size // n_w
    baselines = _closure_baselines(data, n_w, n_t)
    if baselines is None:
        return None, _IRREGULAR_CP
    b1, b2, b3 = baselines
    # virgil's closure phase is φ[i_cps1] + φ[i_cps2] - φ[i_cps3], i.e. the
    # baselines AB, BC and AC of the triangle ABC that OI_T3 stores.
    if not (
        onp.all(stations[b2, 0] == stations[b1, 1])
        and onp.all(stations[b3, 0] == stations[b1, 0])
        and onp.all(stations[b3, 1] == stations[b2, 1])
    ):
        return None, _IRREGULAR_CP
    ub, vb = layout["ub"], layout["vb"]
    table = {
        "T3PHI": phi.reshape(n_t, n_w),
        "T3PHIERR": d_phi.reshape(n_t, n_w),
        "U1COORD": ub[b1],
        "V1COORD": vb[b1],
        "U2COORD": ub[b2],
        "V2COORD": vb[b2],
        "STA_INDEX": onp.stack(
            [stations[b1, 0], stations[b1, 1], stations[b2, 1]], axis=1
        ),
    }
    if layout["mjd"] is not None:
        table["MJD"] = layout["mjd"][b1]
    return {"OI_T3": table}, None


# === RESULTS ===


def lobe_text(summary):
    """The preferred model's diameter lobes in one line, or ``None``.

    Each lobe is its fitted diameter and its Δχ² over the best lobe.
    """
    fit = summary.get("fit") or {}
    rows = (fit.get("models", {}).get(fit.get("best"), {})).get("lobes")
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["delta_chi2"])
    return "; ".join(
        f"{r['diam_mas']:.4g} mas "
        + ("(best)" if i == 0 else f"(Δχ² {r['delta_chi2']:.3g})")
        for i, r in enumerate(rows)
    )


class Result:
    """The outputs of a pipeline run, reloaded from its folder.

    Nothing is recomputed: every accessor reads the files the run wrote,
    and returns plain numbers, numpy arrays or real virgil objects that can
    be refined by hand.

    Attributes
    ----------
    path : pathlib.Path
        The run folder.
    run : dict
        The contents of ``run.json``: schema, status, resolved config,
        inputs, provenance and per-stage timings.
    summary : dict
        The contents of ``summary.json``: key numbers by stage, and the
        checks as dicts.
    """

    def __init__(self, path):
        self.path = Path(path)
        run_file = self.path / "run.json"
        if not run_file.exists():
            raise FileNotFoundError(f"{run_file} does not exist.")
        self.run = read_json(run_file)
        if self.run.get("schema") != SCHEMA:
            raise ValueError(
                f"{run_file} has schema {self.run.get('schema')!r}; this "
                f"version of virgil reads {SCHEMA!r}."
            )
        summary_file = self.path / "summary.json"
        self.summary = read_json(summary_file) if summary_file.exists() else {}

    def __repr__(self):
        return (
            f"Result({str(self.path)!r}, pipeline={self.run['pipeline']!r}, "
            f"status={self.status!r})"
        )

    @property
    def status(self):
        """Run status: ``"running"``, ``"partial"``, ``"complete"`` or ``"failed"``."""
        return self.run["status"]

    @property
    def checks(self):
        """The quality checks, as a list of [`Check`][virgil.pipeline.Check]."""
        return [Check.from_dict(c) for c in self.summary.get("checks", [])]

    def describe(self):
        """The key numbers of the summary as short text, one per line."""
        lines = [f"{self.run['class']} run in {self.path}: {self.status}"]
        for section, values in self.summary.items():
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                if isinstance(value, bool) or value is None:
                    continue
                if isinstance(value, (int, float)):
                    lines.append(f"  {section}.{key} = {value:.6g}")
        lobes = lobe_text(self.summary)
        if lobes:
            lines.append(f"  lobes: {lobes}")
        for check in self.checks:
            lines.append(f"  [{check.status}] {check.name}: {check.message}")
        caught = self.summary.get("warnings", [])
        if caught:
            stages = ", ".join(sorted({w["stage"] for w in caught}))
            lines.append(
                f"  {len(caught)} warning(s) recorded in the {stages} "
                "stage(s) (see the checks, or Result.summary['warnings'])"
            )
        return "\n".join(lines)

    @property
    def plots(self):
        """Paths of the plots the run wrote, sorted by name."""
        return sorted((self.path / "plots").glob("*.png"))

    def data(self):
        """The data actually fitted, as an ``OIData``, from ``data/processed.oifits``."""
        from ..oidata import OIData

        path = self.path / "data" / "processed.oifits"
        if not path.exists():
            report = self.path / "stages" / "load" / "report.json"
            reason = (
                read_json(report).get("processed_oifits_skipped")
                if report.exists()
                else None
            )
            raise FileNotFoundError(
                f"{path} was not written"
                + (f" ({reason})" if reason else "")
                + "; rebuild the data from the inputs listed in "
                "Result.run['inputs']."
            )
        return OIData(path)

    def image(self, name="mean"):
        """An image of an imaging run, read from ``images/<name>.fits``.

        Parameters
        ----------
        name : {"mean", "clean"}, optional
            ``"mean"``: the ensemble's mean image (with one member, that
            member's); ``"clean"``: CLEAN's restored image, whose
            components are the ``COMPONENTS`` extension of the file.

        Returns
        -------
        SkyImage
            The pixels (row 0 North, column 0 East), pixel scale, offset
            of the image centre and beam.
        """
        stage = {"mean": "combine", "clean": "start"}.get(name)
        if stage is None:
            raise ValueError(f'name must be "mean" or "clean"; got {name!r}.')
        return read_fits_image(self._stage_file(f"images/{name}.fits", stage))

    def model(self, name=None):
        """A fitted model, a real virgil object with its values set.

        Parameters
        ----------
        name : str, optional
            For a run that fitted several models (``StarPipeline``), the
            model to rebuild; by default the preferred one, ``"best"``.
        """
        return load_model(self._model_folder(name))

    def model_values(self, name=None):
        """The fitted parameter values, as path → numpy array (see ``model``)."""
        return load_model_values(self._model_folder(name))

    def samples(self, group_by_chain=True, name=None):
        """Posterior samples by site, shaped ``(chain, draw)`` (or flat).

        ``name`` selects one of several fitted models, as in ``model``.
        """
        samples = read_h5(
            self._stage_file("samples.h5", "posterior"),
            self._group("posterior", name),
        )
        if group_by_chain:
            return samples
        return {k: v.reshape((-1,) + v.shape[2:]) for k, v in samples.items()}

    def aliases(self):
        """The alias search of an orbit run, as an
        [`AliasResult`][virgil.aliases.AliasResult] (every band's modes,
        evidences and best orbit), from ``stages/search/aliases.json``."""
        from ..aliases import AliasResult

        return AliasResult.load(
            self._stage_file("stages/search/aliases.json", "search")
        )

    def sample_stats(self, name=None):
        """NUTS diagnostics per transition, shaped ``(chain, draw)``."""
        return read_h5(
            self._stage_file("samples.h5", "posterior"),
            self._group("sample_stats", name),
        )

    def grid(self):
        """Grid results: ``"axes"`` (usable as ``grid=``) plus one map per name."""
        stage = {"star": "fit", "disk": "geometry"}.get(
            self.run["pipeline"], "search"
        )
        path = self._stage_file("grids.h5", stage)
        out = {"axes": read_h5(path, "axes")}
        for group in ("search", "limits", "scan", "geometry"):
            # Each group is written by the stage of its name (the star's
            # scan by its fit); a group left by an unfinished stage is not
            # this run's.
            if self._complete("fit" if group == "scan" else group):
                out.update(read_h5(path, group))
        order = read_h5_attrs(path).get("axis_order")
        if order:
            out["axes"] = {k: out["axes"][k] for k in order}
        return out

    def fit_result(self, name=None):
        """The MAP fit as a virgil ``FitResult``, e.g. to continue with ``fit``.

        ``name`` selects one of several fitted models, as in ``model``.
        """
        from ..fitting import FitResult

        folder = self._model_folder(name)
        info = read_json(Path(folder) / "info.json")
        return FitResult(
            model=load_model(folder),
            values=load_model_values(folder),
            info=info,
        )

    def _model_folder(self, name):
        stage = "fit"
        if self.run["pipeline"] == "image":
            # The imaging pipeline's models: the evidence grid's winner, the
            # CLEAN start, and the preferred one the combine stage writes.
            stage = {
                None: "combine",
                "best": "combine",
                "clean": "start",
                "hyper_best": "hyperparameters",
            }.get(name, "combine")
        elif self.run["pipeline"] == "disk":
            # The disk pipeline's models come from three stages: the
            # single-ring geometry fit, the candidates of the profile
            # stage, and the preferred one chosen by compare.
            stage = {
                "geometry": "geometry",
                None: "compare",
                "best": "compare",
            }.get(name, "profile")
        return self._stage_file(
            "models/best" if name is None else f"models/{name}", stage
        )

    @staticmethod
    def _group(group, name):
        return group if name is None else f"{group}_{name}"

    def _status(self, stage):
        return self.run.get("stages", {}).get(stage, {}).get("status")

    def _complete(self, stage):
        return self._status(stage) == "complete"

    def _stage_file(self, relative, stage):
        """``relative`` in the run folder, if ``stage`` completed and wrote it."""
        path = self.path / relative
        status = self._status(stage)
        if status != "complete" or not path.exists():
            reason = (
                f"{path} does not exist"
                if not path.exists()
                else f"{path} is left over from an earlier run"
            )
            raise FileNotFoundError(
                f"{reason}: the {stage!r} stage has status "
                f"{status or 'not run'}; run the pipeline through it first."
            )
        return path


def load(path):
    """Reload a pipeline run from its folder, without recomputing anything.

    Parameters
    ----------
    path : str or os.PathLike
        The run folder (the pipeline's ``output``).

    Returns
    -------
    Result
    """
    return Result(path)
