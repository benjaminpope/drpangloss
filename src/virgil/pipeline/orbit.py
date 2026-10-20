"""`OrbitPipeline`: a blind period-alias search, modes and posterior of a
multi-epoch binary orbit."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
import warnings
from pathlib import Path

import numpy as onp

from . import _checks, _io
from ._common import (
    _bound_samples,
    _chain_health,
    _corner_trace,
    _diagnostics,
    _draws,
    _geometry,
    _integer,
    _pair,
    _pick,
    _positive,
    _quicklook,
    _run_chains,
    _sampler_checks,
    _sampler_report,
    _save,
    _site_summary,
    _validate_sampler,
    _write_samples,
)
from ._core import Stage, _Pipeline
from .._deprecate import external_stacklevel
from .._precision import run_in

# Grid step of the per-night position search, as a fraction of the finest
# fringe spacing λ/B_max (Nyquist is one half).
_STEP_FRACTION = 0.25
# The search box is widened by this factor, up to _MAX_WIDEN times, while a
# night's best peak sits on its edge.
_WIDEN = 1.5
_MAX_WIDEN = 5
# Engine coordinates of the sampled orbit, in the order of ``aliases._physical``.
_SITES = (
    "log_period",
    "tau",
    "ecc",
    "cos_inc",
    "omega",
    "Omega",
    "log_a",
    "log_flux",
    "log_diam",
)
# The star alone: a companion this faint (log flux ratio) is invisible.
_NO_COMPANION = math.log(1e-12)
# Diameters tried for the star alone when the orbit fit a diameter.
_N_DIAM_NULL = 100
# Fraction of a prior's range, at an end, counted as "at the bound".
_AT_BOUND = 0.01
_BAND_ROWS = 10


def _engine_defaults(*names):
    """The defaults of ``fit_orbit_aliases`` for ``names``: the search
    effort lives there only."""
    from ..aliases import fit_orbit_aliases

    params = inspect.signature(fit_orbit_aliases).parameters
    return {k: params[k].default for k in names}


def _x64(fn):
    """Run a stage in float64, as the engine's entry points do."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with run_in("float64"):
            return fn(*args, **kwargs)

    return wrapped


class OrbitPipeline(_Pipeline):
    """Fit the orbit of a binary seen on several nights, blind to its period.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Stages: ``load → epochs → search → modes → posterior → quicklook``.

    * ``load`` groups the data into nights as given (the data are expected
      to be prepared already, with the instrument's own channel, band and
      noise handling) and records every file's night.
    * ``epochs`` finds the companion's position in each night
      ([`epoch_positions`][virgil.epochs.epoch_positions]) on a grid widened
      until no best peak is at its edge, and flags the nights that cannot
      seed an orbit (aliased peaks, a low gap, a high χ², a separation below
      the resolution, a flux at its bound).
    * ``search`` runs [`fit_orbit_aliases`][virgil.aliases.fit_orbit_aliases]
      with no published period, window, start or prior: the period window
      comes from the data, the bands are checkpointed in the run folder, and
      the label swap, flux flip, V² amplitude, primary diameter and
      tempering are the engine's defaults.
    * ``modes`` ranks the alias bands, gives the winner's margin over the
      runner-up and the band probabilities, and compares the best orbit with
      a single star (``no-orbit``).
    * ``posterior`` samples each significant band with NUTS on the engine's
      own posterior (tempered, V² amplitude profiled).
    * ``quicklook`` writes and executes ``quicklook.ipynb``.

    The raw χ²/N on the quoted errors is always reported; the error scale
    of every night and block is marginalized inside the likelihood.

    Parameters
    ----------
    epochs : Epochs or dict
        The nights: an [`Epochs`][virgil.epochs.Epochs], or
        ``{night: OIData | [OIData] | [paths]}``. Use
        ``OrbitPipeline.from_oifits`` for files.
    output : str or os.PathLike, optional
        Run folder (default ``"run"``).
    **settings
        Overrides of the defaults (see ``OrbitPipeline.defaults()``):

        - ``gap_days`` (0.5): frames further apart than this are in
          different nights (``from_oifits``).
        - ``exclude`` ({}): ``{file, night or dataset: reason}`` left out,
          a ruling recorded in ``run.json``.
        - ``flag_nights`` (True): flag the nights that cannot seed an orbit;
          ``flag_sep_blocks`` (False): whether a separation below λ/2B also
          keeps a night from seeding (it is reported either way).
        - ``outlier_eps`` (0.02): the prior outlier probability of each
          night in the orbit fit, whose likelihood is
          ``(1 - eps) L + eps L_broad`` with ``L_broad`` the night's mean
          likelihood over the position grid, so that one bad night (a
          sidelobe) cannot move the orbit; 0 turns it off. The posterior
          outlier probability of each night is in the search report.
        - ``n_peaks`` (5), ``max_sep_mas`` (None), ``grid_step_mas``
          (None), ``max_grid`` (201): the position search; by default the
          step is a quarter of the finest λ/B (coarsened to ``max_grid``
          points) and the half-width the coarsest λ/B, widened to at most
          ``a_range[1] (1 + ecc_max)``.
        - ``period_range`` (None): ``[min, max]`` days; by default from the
          data (see ``fit_orbit_aliases``).
        - ``a_range`` ([1, 30]), ``flux_range`` ([0.01, 1]), ``ecc_max``
          (0.9), ``s_max`` (None): the engine's prior ranges (mas; flux
          ratio at most 1; eccentricity; error-scale bound).
        - ``primary_diam`` ("auto"), ``vis_amp`` (True), ``sign_free``
          (True), ``flip_flux`` (True), ``temper`` (True): as for the
          engine.
        - ``n_bands_max``, ``n_candidates``, ``n_refine``, ``max_steps``,
          ``n_is``: search effort; the defaults are those of
          ``fit_orbit_aliases``, so the engine is the one place to raise
          them (the ``mode_converged`` check says when to).
        - ``alias_nats`` (10), ``no_orbit_nats`` (10): thresholds of the
          ``alias_ambiguity`` and ``no_orbit`` checks.
        - ``posterior_p_min`` (0.05), ``max_modes`` (3): bands sampled by
          NUTS besides the winner.
        - ``num_warmup``, ``num_samples``, ``num_chains`` (1000, 1000, 4),
          ``chain_method`` ("vectorized") and ``seed`` (0): NUTS.
        - ``quicklook_execute`` (True): execute the quicklook notebook.

    Examples
    --------
    >>> from virgil.pipeline import OrbitPipeline  # doctest: +SKIP
    >>> res = OrbitPipeline.from_oifits(files, output="runs/hd1234").run()  # doctest: +SKIP
    >>> res.summary["modes"]["winner"]["period"]  # doctest: +SKIP
    """

    NAME = "orbit"
    STABILITY = "provisional"
    STAGES = ("load", "epochs", "search", "modes", "posterior", "quicklook")
    _SECOND_KEY = "metadata"
    _DEFAULTS = {
        "gap_days": 0.5,
        "exclude": {},
        "flag_nights": True,
        "flag_sep_blocks": False,
        "outlier_eps": 0.02,
        "n_peaks": 5,
        "max_sep_mas": None,
        "grid_step_mas": None,
        "max_grid": 201,
        "period_range": None,
        "a_range": [1.0, 30.0],
        "flux_range": [0.01, 1.0],
        "ecc_max": 0.9,
        "s_max": None,
        "primary_diam": "auto",
        "vis_amp": True,
        "sign_free": True,
        "flip_flux": True,
        "temper": True,
        **_engine_defaults(
            "n_bands_max", "n_candidates", "n_refine", "max_steps", "n_is"
        ),
        "alias_nats": 10.0,
        "no_orbit_nats": 10.0,
        "posterior_p_min": 0.05,
        "max_modes": 3,
        "num_warmup": 1000,
        "num_samples": 1000,
        "num_chains": 4,
        "chain_method": "vectorized",
        "seed": 0,
        "quicklook_execute": True,
    }

    def __init__(self, epochs, output="run", **settings):
        notes = getattr(self, "_notes", {})
        super().__init__(_as_mapping(epochs), None, output=output, **settings)
        mapping, times = self.data
        mapping, self._excluded = _apply_exclude(
            mapping, self.settings["exclude"], notes
        )
        self._notes = notes
        self.data = _build_epochs(mapping, times)
        if len(self.data.names) < 2 or len(self.data.data) < 2:
            raise ValueError(
                "An orbit needs at least two nights; got "
                f"{len(self.data.names)} after the exclusions."
            )

    @classmethod
    def from_oifits(cls, paths, metadata=None, *, output="run", **settings):
        """Build a pipeline from OIFITS files, grouped into nights.

        Parameters
        ----------
        paths : str, os.PathLike or list of them
            OIFITS files, each split into nights at gaps of ``gap_days``.
        metadata : dict, optional
            ``{file (path or name): {...}}``, each entry giving any of:
            ``epoch`` (a night label for the file), ``mjd`` (the time of the
            night the file holds, when its stamp is wrong or absent). Rulings that leave data
            out go in the ``exclude`` setting, with their reason.
        output : str or os.PathLike, optional
            Run folder.
        **settings
            Settings overriding the defaults.
        """
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        if not paths:
            raise ValueError(
                f"{cls.__name__}.from_oifits needs at least one OIFITS file; "
                "got an empty list."
            )
        paths = [str(p) for p in paths]
        gap = settings.get("gap_days", cls._DEFAULTS["gap_days"])
        mapping, times, notes = _read_nights(paths, metadata or {}, gap)
        pipeline = cls.__new__(cls)
        pipeline._input_paths = [str(Path(p).resolve()) for p in paths]
        pipeline._notes = notes
        cls.__init__(pipeline, (mapping, times), output=output, **settings)
        return pipeline

    def _default_model(self):
        return None

    @classmethod
    def _validate(cls, s):
        _positive(s, "gap_days")
        for name in ("max_sep_mas", "grid_step_mas", "s_max"):
            if s[name] is not None:
                _positive(s, name)
        for name in ("alias_nats", "no_orbit_nats"):
            _positive(s, name)
        for name, low in (
            ("n_peaks", 1),
            ("n_bands_max", 1),
            ("n_candidates", 1),
            ("n_refine", 1),
            ("max_steps", 1),
            ("n_is", 1),
            ("max_modes", 1),
        ):
            _integer(s, name, low)
        _integer(s, "max_grid", 3)
        _pair(s, "a_range")
        _pair(s, "flux_range")
        if s["flux_range"][1] > 1.0:
            raise ValueError(
                "flux_range must be at most 1: the flux flip makes the "
                "brighter star the primary, so a larger ratio would count "
                "the label-swapped mode twice."
            )
        if s["period_range"] is not None:
            _pair(s, "period_range")
        if not 0 < s["ecc_max"] < 1:
            raise ValueError("ecc_max must be between 0 and 1.")
        if not 0 <= s["posterior_p_min"] <= 1:
            raise ValueError("posterior_p_min must be in [0, 1].")
        diam = s["primary_diam"]
        if not (
            diam is None
            or diam in ("auto", "fit")
            or (isinstance(diam, (int, float)) and diam >= 0)
        ):
            raise ValueError(
                'primary_diam must be "auto", "fit", None or a diameter >= 0.'
            )
        temper = s["temper"]
        if not (
            isinstance(temper, bool)
            or (isinstance(temper, (int, float)) and temper >= 1.0)
        ):
            raise ValueError("temper must be a bool or a factor f >= 1.")
        vis_amp = s["vis_amp"]
        if not isinstance(vis_amp, bool) and not (
            len(vis_amp) == 2 and 0 < vis_amp[0] < vis_amp[1]
        ):
            raise ValueError(
                "vis_amp must be a bool or (lo, hi), 0 < lo < hi."
            )
        _validate_sampler(s)
        if not isinstance(s["exclude"], dict):
            raise ValueError("exclude must be a {name: reason} dict.")

    def _data_fingerprint(self):
        epochs = self.data
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                [
                    list(epochs.dataset_names),
                    list(epochs.epoch_of),
                    [float(t) for t in epochs.times],
                ]
            ).encode()
        )
        for d in epochs.data:
            digest.update(_io.data_fingerprint(d).encode())
        return digest.hexdigest()

    def _input_files(self):
        files = super()._input_files()
        for entry in files:
            note = self._notes.get(_resolved(self._notes, entry["path"]), {})
            entry.update(
                {k: v for k, v in note.items() if k != "path"},
            )
        return files

    def _model_digest(self):
        return {"class": "OrbitalBinary", "digest": None}

    def _stages(self):
        return (
            Stage("load", _load, ("stages/load/report.json",)),
            Stage("epochs", _epochs, ("stages/epochs/positions.json",)),
            Stage(
                "search",
                _search,
                ("stages/search/aliases.json", "plots/search_bands.png"),
            ),
            Stage("modes", _modes, ("plots/modes_sky.png",)),
            Stage(
                "posterior",
                _posterior,
                (
                    "samples.h5",
                    "plots/posterior_corner.png",
                    "plots/posterior_trace.png",
                ),
            ),
            Stage("quicklook", _quicklook, ("quicklook.ipynb",)),
        )

    def _summarize(self, reports):
        return _summarize(self.settings, reports)

    def __repr__(self):
        return (
            f"{type(self).__name__}(output={str(self.output)!r}, "
            f"nights={len(self.data.names)})"
        )


# === INPUTS ===


def _resolved(notes, path):
    """The key of ``notes`` for the file ``path`` (as given or resolved)."""
    for key in notes:
        if str(Path(key).resolve()) == path:
            return key
    return path


def _as_mapping(epochs):
    """``(mapping, times)`` of whatever the user gave: ``{night: {dataset:
    OIData}}`` and ``{dataset: MJD or None}``."""
    from ..epochs import Epochs
    from ..oidata import OIData

    if isinstance(epochs, tuple):  # from_oifits
        return epochs
    if isinstance(epochs, Epochs):
        mapping = {}
        for name, night, d in zip(
            epochs.dataset_names, epochs.epoch_of, epochs.data
        ):
            mapping.setdefault(night, {})[name] = d
        return mapping, dict(zip(epochs.dataset_names, epochs.times))
    if not isinstance(epochs, dict) or not epochs:
        raise TypeError(
            "epochs must be an Epochs or a non-empty {night: data} dict; "
            f"got {type(epochs).__name__}."
        )

    def one(value):
        return OIData(value) if isinstance(value, (str, Path)) else value

    mapping = {}
    for night, group in epochs.items():
        if isinstance(group, dict):
            mapping[night] = {k: one(v) for k, v in group.items()}
        elif isinstance(group, (list, tuple)) and not all(
            isinstance(g, (str, Path)) for g in group
        ):
            mapping[night] = _named(night, [one(g) for g in group])
        elif isinstance(group, (list, tuple)):  # paths: one night, one OIData
            mapping[night] = {night: OIData([str(g) for g in group])}
        else:
            mapping[night] = {night: one(group)}
    return mapping, {}


def _named(night, datasets):
    """``{name: OIData}`` for the datasets of a night, named as ``Epochs``
    names a list."""
    if len(datasets) == 1:
        return {night: datasets[0]}
    return {f"{night}[{k}]": d for k, d in enumerate(datasets)}


def _build_epochs(mapping, times):
    """An ``Epochs`` of ``{night: {dataset: OIData}}`` with the given
    snapshot times (``{dataset: MJD}``, entries None to use the data's)."""
    from ..epochs import Epochs

    names = {name for group in mapping.values() for name in group}
    keep = {
        k: float(v) for k, v in times.items() if v is not None and k in names
    }
    return Epochs(mapping, times=keep or None)


def _match_metadata(metadata, paths):
    """``{path: entry}`` of the metadata whose key is a path, its
    resolved path or its file name."""
    out, used = {}, set()
    for path in paths:
        for key, entry in metadata.items():
            if key in (path, str(Path(path).resolve()), Path(path).name):
                out[path] = entry
                used.add(key)
    unknown = sorted(set(metadata) - used)
    if unknown:
        raise ValueError(f"metadata keys {unknown} match no input file.")
    allowed = {"epoch", "mjd"}
    for path, entry in out.items():
        bad = sorted(set(entry) - allowed)
        if bad:
            raise ValueError(
                f"metadata of {path}: unknown keys {bad}; allowed are "
                f"{sorted(allowed)} (rulings go in the exclude setting)."
            )
    return out


def _read_nights(paths, metadata, gap_days):
    """``(mapping, times, notes)``: the files read, each split into nights
    at ``gap_days``, as ``{night: {dataset: OIData}}`` in time order, the
    dataset times, and per file the nights and datasets it fed."""
    from ..oidata import OIData

    meta = _match_metadata(metadata, paths)
    parts, notes = [], {}
    for path in paths:
        entry = meta.get(path, {})
        try:
            data = OIData(path)
        except ValueError as error:
            if not str(error).startswith("No unflagged data"):
                raise ValueError(f"{path}: {error}") from error
            warnings.warn(f"{path} is skipped: {error}", stacklevel=2)
            notes[path] = {
                "path": path,
                **entry,
                "skipped": str(error),
                "nights": [],
                "datasets": [],
            }
            continue
        notes[path] = {"path": path, **entry, "nights": [], "datasets": []}
        for piece in data.split_by_epoch(gap_days):
            if "mjd" in entry:
                mean = float(entry["mjd"])
            elif data.mjd is None:
                raise ValueError(f"{path} has no times: give metadata 'mjd'.")
            else:
                mean = float(onp.mean(piece.mjd))
            parts.append((mean, entry.get("epoch"), piece, path))
    groups, last = {}, None  # last: (label, mean) of the open automatic night
    for mean, epoch, piece, path in sorted(parts, key=lambda t: t[0]):
        if epoch:
            label = epoch
        elif last is not None and mean - last[1] <= gap_days:
            label = last[0]
        else:
            label = f"mjd{math.floor(mean)}"
            while label in groups:  # two nights in one day, gap_days small
                label += "b"
        if not epoch:
            last = (label, mean)
        groups.setdefault(label, []).append((piece, path, mean))
    mapping, times = {}, {}
    for label, group in groups.items():
        names = _named(label, [piece for piece, _, _ in group])
        mapping[label] = names
        for name, (_, path, mean) in zip(names, group):
            notes[path]["nights"].append(label)
            notes[path]["datasets"].append(name)
            if "mjd" in notes[path]:
                times[name] = mean
    return mapping, times, notes


def _apply_exclude(mapping, exclude, notes):
    """``(mapping, record)`` without the files, nights and datasets named in
    ``exclude``; ``record`` is ``{key: {"reason", "datasets"}}``."""
    record, dropped = {}, set()
    for key, reason in exclude.items():
        hit = set()
        for path, note in notes.items():
            if key in (path, str(Path(path).resolve()), Path(path).name):
                hit |= set(note["datasets"])
        for night, group in mapping.items():
            if key == night:
                hit |= set(group)
            hit |= {name for name in group if name == key}
        if not hit:
            raise ValueError(
                f"exclude key {key!r} matches no file, night or dataset."
            )
        record[key] = {"reason": reason, "datasets": sorted(hit)}
        dropped |= hit
    kept = {
        night: {n: d for n, d in group.items() if n not in dropped}
        for night, group in mapping.items()
    }
    return {k: v for k, v in kept.items() if v}, record


# === STAGES ===


# --- load ---


@_x64
def _load(p):
    epochs = p.data
    data = list(epochs.data)
    rows = [
        {
            "dataset": name,
            "night": night,
            "mjd": float(t),
            "spread_days": float(spread),
            "n_vis": int(onp.asarray(d.vis).size),
            "n_phi": int(onp.asarray(d.phi).size),
        }
        for name, night, t, spread, d in zip(
            epochs.dataset_names,
            epochs.epoch_of,
            epochs.times,
            epochs.spread_days,
            data,
        )
    ]
    return {
        "n_nights": len(epochs.names),
        "n_datasets": len(data),
        "n_vis": sum(r["n_vis"] for r in rows),
        "n_phi": sum(r["n_phi"] for r in rows),
        "n_independent": int(sum(d.n_independent for d in data)),
        "closure_phases": all(d.cp_flag for d in data if d.phi.size),
        "span_days": float(onp.ptp(epochs.times)),
        **_geometry(data),
        "nights": rows,
        "excluded": p._excluded,
        "data_sha256": p._data_fingerprint(),
        "processed_oifits_skipped": "one OIFITS per night is not written",
    }


# --- epochs ---


def _position_grid(p, half):
    """The ``(grid, n, step)`` of the position search of half-width ``half``
    (mas); ``max_grid`` coarsens the step (the report says so)."""
    s = p.settings
    finest = _geometry(list(p.data.data))["lambda_over_b_mas"]
    wanted = s["grid_step_mas"] or _STEP_FRACTION * finest
    most = s["max_grid"] - (1 - s["max_grid"] % 2)  # the odd number
    n = min(2 * math.ceil(half / wanted) + 1, most)
    axis = onp.linspace(-half, half, n)
    lo, hi = s["flux_range"]
    flux = onp.geomspace(lo, hi, 13)
    return (
        {"dra": axis, "ddec": axis, "flux": flux},
        n,
        2 * half / (n - 1),
        wanted,
    )


def _night_positions(p):
    """``(EpochPositions, info)``: each night's peaks on a grid widened by
    ``_WIDEN`` until no best peak is on its edge."""
    from ..epochs import epoch_positions

    # The grid widens until no best peak is on its edge (AGENTS.md, "Binary
    # orbits: solved problems"; test_pipeline_orbit.py::
    # test_the_position_grid_widens_until_no_peak_is_on_its_edge).
    s = p.settings
    coarsest = _geometry(list(p.data.data))["fov_mas"]
    cap = s["a_range"][1] * (1.0 + s["ecc_max"])  # the largest separation
    half = s["max_sep_mas"] or min(coarsest, cap)
    halves = []
    for _ in range(_MAX_WIDEN + 1):
        grid, n, step, wanted = _position_grid(p, half)
        # Only the last pass's warnings (an edge, say) describe the result.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            found = epoch_positions(
                p.data,
                grid,
                n_peaks=s["n_peaks"],
                s_max=s["s_max"],
                flag_nights=s["flag_nights"],
                flag_sep_blocks=s["flag_sep_blocks"],
            )
        halves.append(float(half))
        if not found.edge.any() or s["max_sep_mas"] or half >= cap:
            break
        half = min(_WIDEN * half, cap)  # a companion may lie beyond the box
    for w in caught:
        warnings.warn(w.message, w.category, stacklevel=external_stacklevel())
    if step > wanted * (1 + 1e-9):
        warnings.warn(
            f"max_grid={s['max_grid']} coarsens the position grid step "
            f"from {wanted:.3g} to {step:.3g} mas; raise max_grid (or "
            "grid_step_mas) for a finer search.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
    info = {
        "half_widths_mas": halves,
        "grid_points": int(n),
        "grid_step_mas": float(step),
        "grid_step_wanted_mas": float(wanted),
        "edge_nights": [n for n, e in zip(found.names, found.edge) if e],
    }
    return found, info


def _positions_json(found):
    """``EpochPositions`` as a dict that ``json`` writes exactly (NaN and
    infinities as the ``json`` module spells them)."""
    return {
        "names": list(found.names),
        "mjd": found.mjd.tolist(),
        "dra": found.dra.tolist(),
        "ddec": found.ddec.tolist(),
        "cov": found.cov.tolist(),
        "flux": found.flux.tolist(),
        "gap_marginal": found.gap_marginal.tolist(),
        "chi2_raw": [dict(c) for c in found.chi2_raw],
        "scale": [dict(c) for c in found.scale],
        "peaks": [
            {
                k: getattr(peak, k).tolist()
                for k in (
                    "dra",
                    "ddec",
                    "flux",
                    "loglike",
                    "marginal",
                    "weight",
                    "edge",
                    "laplace",
                )
            }
            for peak in found.peaks
        ],
        "flags": [
            [
                {"code": f.code, "reason": f.reason, "blocking": f.blocking}
                for f in night
            ]
            for night in found.flags
        ],
        "primary_diam": found.primary_diam,
        "grid": None
        if found.grid is None
        else [a.tolist() for a in found.grid],
    }


def _positions_from_json(d):
    """The ``EpochPositions`` that ``_positions_json`` wrote."""
    from ..epochs import EpochPeaks, EpochPositions, NightFlag

    def peak(row):
        return EpochPeaks(
            **{
                k: onp.asarray(
                    v, dtype=bool if k in ("edge", "laplace") else float
                )
                for k, v in row.items()
            }
        )

    return EpochPositions(
        names=tuple(d["names"]),
        mjd=onp.asarray(d["mjd"], float),
        dra=onp.asarray(d["dra"], float),
        ddec=onp.asarray(d["ddec"], float),
        cov=onp.asarray(d["cov"], float),
        flux=onp.asarray(d["flux"], float),
        gap_marginal=onp.asarray(d["gap_marginal"], float),
        chi2_raw=tuple(d["chi2_raw"]),
        scale=tuple(d["scale"]),
        peaks=tuple(peak(r) for r in d["peaks"]),
        flags=tuple(
            tuple(
                NightFlag(f["code"], f["reason"], f.get("blocking", True))
                for f in night
            )
            for night in d["flags"]
        ),
        primary_diam=d["primary_diam"],
        grid=None
        if d.get("grid") is None
        else tuple(onp.asarray(a, float) for a in d["grid"]),
    )


def _read_positions(p):
    path = p.output / "stages" / "epochs" / "positions.json"
    return _positions_from_json(json.loads(path.read_text()))


@_x64
def _epochs(p):
    found, info = _night_positions(p)
    path = p.output / "stages" / "epochs" / "positions.json"
    with _io.atomic_path(path) as tmp:
        Path(tmp).write_text(json.dumps(_positions_json(found)))
    edge = {n: bool(e) for n, e in zip(found.names, found.edge)}
    nights = [
        {
            "dataset": name,
            "mjd": float(found.mjd[k]),
            "dra_mas": float(found.dra[k]),
            "ddec_mas": float(found.ddec[k]),
            "sep_mas": float(math.hypot(found.dra[k], found.ddec[k])),
            "flux": float(found.flux[k]),
            "gap_marginal": float(found.gap_marginal[k]),
            "chi2_raw": float(found.chi2_raw[k].get("all", math.nan)),
            "edge": edge[name],
            "flags": [
                {"code": f.code, "reason": f.reason, "blocking": f.blocking}
                for f in found.flags[k]
            ],
        }
        for k, name in enumerate(found.names)
    ]
    return {
        **info,
        "nights": nights,
        "flagged": found.flag_codes(),
    }


# --- search ---


def _band_checkpoints(p):
    # Outside the stage's declared outputs, so a re-run keeps the finished
    # bands (test_band_checkpoints_survive_a_rerun_but_not_a_fresh_run).
    return p.output / "stages" / "search" / "bands"


@_x64
def _search(p):
    from ..aliases import fit_orbit_aliases

    # The default run is blind: no period window, start or paper prior
    # (test_blind_search_resolves_the_two_season_gap_alias).
    s = p.settings
    epochs = p.data
    result = fit_orbit_aliases(
        list(epochs.data),
        None if s["period_range"] is None else tuple(s["period_range"]),
        positions=_read_positions(p),
        times=epochs.times,
        a_range=tuple(s["a_range"]),
        flux_range=tuple(s["flux_range"]),
        ecc_max=s["ecc_max"],
        s_max=s["s_max"],
        n_candidates=s["n_candidates"],
        n_refine=s["n_refine"],
        max_steps=s["max_steps"],
        sign_free=s["sign_free"],
        n_is=s["n_is"],
        seed=s["seed"],
        checkpoint_dir=_band_checkpoints(p),
        primary_diam=s["primary_diam"],
        vis_amp=s["vis_amp"]
        if isinstance(s["vis_amp"], bool)
        else tuple(s["vis_amp"]),
        flip_flux=s["flip_flux"],
        n_bands_max=s["n_bands_max"],
        temper=s["temper"],
        outlier_eps=s["outlier_eps"],
    )
    with _io.atomic_path(
        p.output / "stages" / "search" / "aliases.json"
    ) as tmp:
        result.save(tmp)
    _plot_bands(p.output, result)
    return _search_report(result, list(epochs.data), epochs.dataset_names)


def _search_report(result, data, names):
    nights = result.chi2_per_night(data)
    primary = result.primary_fit or {}
    return {
        "p_range": list(result.p_range),
        "window_capped": result.window_capped,
        "n_bands": len(result.bands),
        "failed_bands": [
            int(b.n) for b in result.bands if "band-failed" in b.flags
        ],
        "t_ref": result.t_ref,
        "span_days": result.span,
        "primary_diam": result.primary_diam,
        "primary_resolved": primary.get("resolved"),
        "primary_diam_estimate": primary.get("diam"),
        "primary_at_grid_edge": bool(primary.get("at_grid_edge", False)),
        "tempering": {
            "f": result.temper_f,
            "f_raw": result.temper_f_raw,
            "blocks": result.temper_blocks,
            "out_of_range": bool(result.tempering),
        },
        "outliers": None
        if result.outlier is None
        else {
            "eps": result.outlier["eps"],
            "probability": dict(
                zip(result.outlier["names"], result.outlier["probability"])
            ),
        },
        "night_flags": dict(result.night_flags or {}),
        "labels_free": bool(result.labels_free),
        "period_bound": bool(result.period_bound),
        "period_at_lower_bound": bool(result.period_at_lower_bound),
        "unseeded": [dict(u) for u in result.unseeded],
        "short_arc": None
        if result.short_arc is None
        else {
            k: result.short_arc[k]
            for k in ("kept", "excluded", "readmitted", "velocities")
        },
        "chi2": {
            "nights": {
                name: {
                    "chi2": row["chi2"],
                    "n": row["n"],
                    "chi2_per_n": row["chi2_per_n"],
                    "vis_amp": row["vis_amp"],
                }
                for name, row in zip(names, nights)
            },
            "chi2": sum(r["chi2"] for r in nights),
            "n": sum(r["n"] for r in nights),
        },
        "bands": _band_rows(result)[:_BAND_ROWS],
    }


def _band_rows(result):
    """The band table: rank, orbit count, period, evidence and flags."""
    top = result.best.log_z
    rows = []
    for rank, b in enumerate(result.bands):
        rows.append(
            {
                "rank": rank,
                "n": int(b.n),
                "period": b.best.get("period"),
                "p_lo": b.p_lo,
                "p_hi": b.p_hi,
                "log_z": b.log_z,
                "delta_log_z": b.log_z - top,
                "p": b.p,
                "chi2_red": b.chi2_red,
                "labels_free": bool(b.labels_free),
                "flags": list(b.flags),
            }
        )
    return rows


def _plot_bands(output, result):
    import matplotlib.pyplot as plt

    rows = [r for r in _band_rows(result) if onp.isfinite(r["log_z"])]
    period = onp.array([0.5 * (r["p_lo"] + r["p_hi"]) for r in rows])
    delta = onp.array([r["delta_log_z"] for r in rows])
    chi2 = onp.array([r["chi2_red"] for r in rows])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4))
    ax.semilogx(period, delta, ".", color="0.4")
    ax.semilogx(period[:1], delta[:1], "o", color="C3", label="best band")
    ax.set_xlabel("period (d)")
    ax.set_ylabel("log Z − log Z$_{best}$ (nats)")
    ax.set_ylim(max(delta.min(), -200.0), 5.0)
    ax.legend()
    bx.loglog(period, chi2, ".", color="0.4")
    bx.loglog(period[:1], chi2[:1], "o", color="C3")
    bx.set_xlabel("period (d)")
    bx.set_ylabel("raw χ²/ν")
    fig.tight_layout()
    _save(fig, output / "plots" / "search_bands.png")


def _load_result(p):
    from ..aliases import AliasResult

    return AliasResult.load(p.output / "stages" / "search" / "aliases.json")


# --- modes ---


def _single_star(problem, x, diam_range):
    # test_a_single_star_is_no_orbit pins the verdict.
    """The best log likelihood of the star alone: the engine's likelihood at
    ``x`` with an invisible companion (and, if the orbit fitted the primary's
    diameter, the best diameter: a log grid over its prior range, then a
    finer one about the best cell, so the null is not under-fitted)."""
    import jax.numpy as np

    x = onp.array(x, float)
    x[7] = _NO_COMPANION

    def best(logd):
        xs = onp.tile(x, (len(logd), 1))
        xs[:, 8] = logd
        return onp.asarray(problem.batch(np.asarray(xs), mixed=False))

    if len(x) <= 8:
        return float(
            onp.max(
                onp.asarray(problem.batch(np.asarray(x[None]), mixed=False))
            )
        )
    grid = onp.log(onp.geomspace(*diam_range, _N_DIAM_NULL))
    ll = best(grid)
    k = int(onp.argmax(ll))
    near = onp.linspace(
        grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)], _N_DIAM_NULL
    )
    return float(max(onp.max(ll), onp.max(best(near))))


@_x64
def _modes(p):
    from ..aliases import _band_target

    s = p.settings
    result = _load_result(p)
    epochs = p.data
    data = list(epochs.data)
    best, bands = result.best, result.bands
    runner = next((b for b in bands[1:] if onp.isfinite(b.log_z)), None)
    delta = None if runner is None else float(best.log_z - runner.log_z)
    problem, _, x, settings = _band_target(result, data)
    null = _single_star(problem, x, settings.diam_range)
    delta_null = float(best.log_z - null)
    sampled = [best.n] + [
        b.n
        for b in bands[1 : 1 + s["max_modes"]]
        if b.p >= s["posterior_p_min"] and onp.isfinite(b.log_z)
    ]
    verdict = "orbit" if delta_null >= s["no_orbit_nats"] else "no-orbit"
    _plot_sky(p, result, epochs)
    return {
        "winner": {
            "n": int(best.n),
            "p": best.p,
            "log_z": best.log_z,
            "elements": dict(best.best),
            "flipped": bool(best.flipped),
            "flags": list(best.flags),
        },
        "runner_up": None
        if runner is None
        else {"n": int(runner.n), "p": runner.p, "log_z": runner.log_z},
        "delta_log_z": delta,
        "labels_free": bool(result.labels_free),
        "no_orbit": {
            "delta_log_z": delta_null,
            "log_like_single_star": null,
            "threshold": s["no_orbit_nats"],
            "verdict": verdict,
        },
        "sampled": [int(n) for n in sampled],
        "table": _band_rows(result)[: max(_BAND_ROWS, len(sampled))],
    }


def _plot_sky(p, result, epochs):
    """The best orbit's track on the sky with each night's position, its
    error ellipse (flagged nights hollow) and the orbit at the night."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    from ..orbits import KeplerOrbit

    best = result.best.best
    orbit = KeplerOrbit(
        best["period"],
        best["dt_peri"],
        best["ecc"],
        best["inc"],
        best["omega"],
        best["Omega"],
        best["a_mas"],
        t_ref=result.t_ref,
    )
    track = onp.asarray(
        orbit.relative(result.t_ref + onp.linspace(0, best["period"], 400))[:2]
    )
    found = _read_positions(p)
    model = onp.asarray(orbit.relative(found.mjd)[:2])
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(track[0], track[1], color="0.6", lw=1, label="best orbit")
    flagged = found.flagged
    for k in range(len(found.names)):
        w, v = onp.linalg.eigh(found.cov[k])
        angle = math.degrees(math.atan2(v[1, 1], v[0, 1]))
        ax.add_patch(
            Ellipse(
                (found.dra[k], found.ddec[k]),
                *(2 * onp.sqrt(onp.maximum(w[::-1], 0))),
                angle=angle,
                fill=not flagged[k],
                alpha=0.4,
                color="C0",
            )
        )
    ax.plot(
        found.dra,
        found.ddec,
        "o",
        mfc="none",
        color="C0",
        label="night positions",
    )
    ax.plot(model[0], model[1], "x", color="C3", label="orbit at the nights")
    ax.plot(0, 0, "*", color="k")
    ax.set_xlabel("ΔRA (mas)")
    ax.set_ylabel("ΔDec (mas)")
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.legend()
    _save(fig, p.output / "plots" / "modes_sky.png")


# --- posterior ---


def _box(bounds, x):
    """The sampling box of one mode: the band's finite prior bounds, and for
    the periodic coordinates one fundamental domain about the mode (τ one
    cycle; ω, Ω folded by the (Ω + 180°, ω + 180°) mirror)."""
    lo, hi = onp.array(bounds.lower, float), onp.array(bounds.upper, float)
    for k, half in ((1, 0.5), (4, 180.0), (5, 90.0)):
        lo[k], hi[k] = x[k] - half, x[k] + half
    return lo, hi


def _orbit_model(problem, priors):
    """The numpyro model of the engine's posterior: uniform priors in the
    engine's coordinates and the tempered, scale-marginalized likelihood."""
    import jax.numpy as np
    import numpyro

    from ..aliases import _loglike

    def model():
        x = np.stack([numpyro.sample(k, d) for k, d in priors.items()])
        numpyro.factor(
            "loglike",
            _loglike(
                problem.surfaces,
                x,
                problem.data,
                problem.cfg,
                problem.dof,
                problem.mix,
            ),
        )

    return model


def _sample_band(p, result, data, n):
    """NUTS over the engine's posterior of band ``n`` from its best mode.

    Returns ``(samples, stats, summary)``: draws by physical name and by
    sampler site (each ``(chain, draw)``), the NUTS statistics and the
    report.
    """
    import numpyro.distributions as dist

    from ..aliases import _band_target, _physical

    s = p.settings
    problem, bounds, x0, _ = _band_target(result, data, n)
    sites = _SITES[: len(x0)]
    lo, hi = _box(bounds, x0)
    priors = {
        name: dist.Uniform(float(a), float(b))
        for name, a, b in zip(sites, lo, hi)
    }
    start = {k: onp.asarray(v) for k, v in zip(sites, x0)}
    mcmc = _run_chains(
        _orbit_model(problem, priors),
        s,
        start,
        dense_mass=True,
        priors=priors,
    )
    drawn, stats = _draws(mcmc)
    shape = drawn[sites[0]].shape
    flat = onp.stack([drawn[k].reshape(-1) for k in sites], axis=-1)
    physical = {
        k: onp.asarray(v).reshape(shape) for k, v in _physical(flat).items()
    }
    summary = _band_summary(n, physical, drawn, stats, priors, bounds)
    summary["optimum_gap"] = _optimum_gap(problem, flat, x0)
    return physical, stats, summary


def _optimum_gap(problem, draws, x0, n_max=500):
    """How far (nats) the best sampled point lies above the start, the
    search's best mode: positive means the search stopped short of the
    optimum (a draw cannot beat a true maximum). At most ``n_max`` draws,
    evenly spread, are scored."""
    import jax.numpy as np

    keep = draws[:: max(1, len(draws) // n_max)]
    scored = onp.asarray(problem.batch(np.asarray(keep)))
    start = float(onp.asarray(problem.batch(np.asarray(x0[None])))[0])
    gap = float(onp.nanmax(scored)) - start
    return max(0.0, gap) if math.isfinite(gap) else math.nan


def _band_summary(n, physical, drawn, stats, priors, bounds):
    """The report of one sampled band: intervals, diagnostics, bound fractions."""
    diagnostics = {k: _diagnostics([x]) for k, x in drawn.items()}
    angles = {"omega": 360.0, "Omega": 180.0}
    names = {
        "log_period": "period",
        "tau": "dt_peri",
        "cos_inc": "inc",
        "log_a": "a_mas",
        "log_flux": "flux",
        "log_diam": "diam",
    }
    by_name = {names.get(site, site): d for site, d in diagnostics.items()}
    params = {
        name: _site_summary(
            x,
            by_name.get(name),
            std=False,
            angle=name in angles,
            period=angles.get(name, 360.0),
        )
        for name, x in physical.items()
    }
    lows = {
        k: float(v) for k, v in zip(_SITES, onp.asarray(bounds.lower, float))
    }
    return {
        "n": int(n),
        "params": params,
        **_chain_health(diagnostics, stats),
        "prior_bound_fraction": _bound_fractions(
            drawn, priors, bounds.upper[2]
        ),
        "log_period_lower": lows["log_period"],
    }


def _bound_fractions(drawn, priors, ecc_top):
    """Fraction of draws at a bound of the box: the band's period edges
    (the posterior cut by its band), the size and flux (both ends) and the top
    of the eccentricity (zero is no bound: circular orbits sit there)."""
    out = {}
    for site in ("log_period", "log_a", "log_flux", "log_diam"):
        if site in drawn:
            out[site] = _bound_samples(priors[site], drawn[site])
    prior = priors["ecc"]
    width = float(prior.high - prior.low)
    out["ecc"] = float(
        onp.mean(drawn["ecc"] > float(ecc_top) - _AT_BOUND * width)
    )
    return {k: v for k, v in out.items() if v is not None}


@_x64
def _posterior(p):
    s = p.settings
    modes = p._report("modes")
    if modes["no_orbit"]["verdict"] == "no-orbit":
        # No orbit to sample: a single star fits as well (see the checks).
        _write_samples(p, {}, skipped="no-orbit")
        for plot in ("corner", "trace"):
            _empty_plot(
                p.output / "plots" / f"posterior_{plot}.png", "no orbit"
            )
        return {"skipped": "no-orbit", "modes": []}
    result = _load_result(p)
    data = list(p.data.data)
    groups, summaries = {}, []
    for n in modes["sampled"]:
        physical, stats, summary = _sample_band(p, result, data, n)
        groups[f"posterior_n{n}"] = physical
        groups[f"sample_stats_n{n}"] = stats
        summaries.append(summary)
        if n == modes["winner"]["n"]:
            groups["posterior"], groups["sample_stats"] = physical, stats
            _corner_trace(
                p.output, physical, [k for k in physical if k != "dt_peri"]
            )
    _write_samples(
        p, groups, winner=modes["winner"]["n"], bands=modes["sampled"]
    )
    return {
        "modes": summaries,
        **_sampler_report(s, summaries),
        # NaN-propagating: a band that could not be scored must show.
        "optimum_gap_max": float(
            onp.max([m["optimum_gap"] for m in summaries])
        ),
        "prior_bound_fraction": _worst_fractions(summaries),
    }


def _worst_fractions(summaries):
    out = {}
    for m in summaries:
        for k, f in m["prior_bound_fraction"].items():
            out[k] = float(onp.max([out.get(k, 0.0), f]))
    return out


def _empty_plot(path, text):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.axis("off")
    _save(fig, path)


# === SUMMARY AND CHECKS ===

_DATA_KEYS = (
    "n_nights",
    "n_datasets",
    "n_vis",
    "n_phi",
    "n_independent",
    "closure_phases",
    "span_days",
    "wavel_min_m",
    "wavel_max_m",
    "baseline_min_m",
    "baseline_max_m",
    "resolution_mas",
    "fov_mas",
    "excluded",
)
_SEARCH_KEYS = (
    "p_range",
    "window_capped",
    "n_bands",
    "failed_bands",
    "t_ref",
    "span_days",
    "primary_diam",
    "primary_resolved",
    "primary_diam_estimate",
    "tempering",
    "labels_free",
    "period_bound",
    "period_at_lower_bound",
    "unseeded",
    "short_arc",
)
_MODES_KEYS = (
    "winner",
    "runner_up",
    "delta_log_z",
    "labels_free",
    "no_orbit",
    "sampled",
    "table",
)
_POSTERIOR_KEYS = (
    "modes",
    "num_chains",
    "num_samples",
    "num_warmup",
    "r_hat_max",
    "ess_bulk_min",
    "divergence_fraction",
    "optimum_gap_max",
)


def _nights_section(epochs, search):
    """Per night: position, flags with reasons and the raw χ²/N at the best
    orbit."""
    rows = []
    chi2 = search["chi2"]["nights"] if search else {}
    for night in epochs["nights"]:
        row = _pick(
            night,
            ("dataset", "mjd", "sep_mas", "flux", "gap_marginal", "edge"),
            strict=False,
        )
        row["flags"] = night["flags"]
        row["chi2_per_n"] = chi2.get(night["dataset"], {}).get("chi2_per_n")
        rows.append(row)
    return rows


def _orbit_section(modes, post):
    """The best band's elements: from the posterior (median, 1σ interval)
    when sampled, else the maximum a posteriori."""
    winner = next(
        (
            m
            for m in (post or {}).get("modes", [])
            if m["n"] == modes["winner"]["n"]
        ),
        None,
    )
    return {
        "n": modes["winner"]["n"],
        "verdict": modes["no_orbit"]["verdict"],
        "source": "posterior" if winner else "map",
        "elements": winner["params"]
        if winner
        else modes["winner"]["elements"],
    }


def _summarize(settings, reports):
    """``(sections, checks)`` of the summary from the stage reports so far."""
    load, epochs, search, modes, post = (
        reports.get(k)
        for k in ("load", "epochs", "search", "modes", "posterior")
    )
    sections, checks = {}, []
    if load:
        sections["data"] = _pick(load, _DATA_KEYS, strict=False)
    if epochs:
        sections["nights"] = _nights_section(epochs, search)
    if search:
        pooled = search["chi2"]
        sections["chi2"] = {
            "n_independent": pooled["n"],
            "best_orbit_reduced": pooled["chi2"] / pooled["n"],
            "nights": pooled["nights"],
        }
        sections["search"] = _pick(search, _SEARCH_KEYS, strict=False)
        checks.append(
            _checks.chi2_reduced(
                "chi2",
                pooled["chi2"] / pooled["n"],
                label="the best-fit orbit",
            )
        )
    if epochs:
        checks.append(
            _checks.night_flags(epochs["flagged"], len(epochs["nights"]))
        )
    if search:
        checks.append(_checks.labels_free(search["labels_free"]))
        checks.append(_checks.bands_failed(search["failed_bands"]))
    if modes:
        sections["modes"] = _pick(modes, _MODES_KEYS, strict=False)
        runner = modes["runner_up"]
        checks.append(
            _checks.alias_ambiguity(
                modes["delta_log_z"],
                None if runner is None else runner["n"],
                warn=settings["alias_nats"],
            )
        )
        checks.append(_checks.winner_flags(modes["winner"]["flags"]))
    if search:
        t = search["tempering"]
        checks.append(_checks.tempering(t["out_of_range"], t["f"], t["f_raw"]))
        checks.append(
            _checks.primary_resolved(
                search["primary_resolved"],
                search["primary_diam_estimate"],
                search["primary_at_grid_edge"],
            )
        )
    if modes:
        checks.append(
            _checks.no_orbit(
                modes["no_orbit"]["delta_log_z"],
                threshold=settings["no_orbit_nats"],
            )
        )
    if search:
        checks.append(
            _checks.period_bound(
                search["period_bound"],
                search["period_at_lower_bound"],
                search["p_range"],
            )
        )
    if modes:
        sections["orbit"] = _orbit_section(modes, post)
    if post:
        sections["posterior"] = _pick(post, _POSTERIOR_KEYS, strict=False)
        if "r_hat_max" in post:
            checks += _sampler_checks(post)
            checks.append(_checks.mode_converged(post["optimum_gap_max"]))
    return sections, checks
