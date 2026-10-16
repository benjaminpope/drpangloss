# OIFITS files

How [`read_oifits`][virgil.oifits.read_oifits] and [`write_oifits`][virgil.oifits.write_oifits] treat the tables of an OIFITS file. The format is the OIFITS standard: version 1 of Pauls et al. (2005, PASP 117, 1255) and version 2 of Duvert et al. (2017, A&A 597, A8). For the sign and unit conventions of $(u, v)$ and phases see [conventions](conventions.md#baselines-and-the-fourier-sign).

Each (baseline, wavelength) sample becomes one element of the flat `u`, `v` and `wavel` arrays of the record, ordered by table, then row, then wavelength channel. Closure phases index into these samples, so a multi-wavelength closure phase is formed from visibilities at its own wavelength.

## Which tables are read {#tables}

Squared visibilities (`OI_VIS2`) are preferred over amplitudes (`OI_VIS`), and closure phases (`OI_T3`) over absolute phases (`OI_VIS` `VISPHI`).

- `VISAMP` is read only when its table's `AMPTYP` is `'absolute'`, and `VISPHI` only when its `PHITYP` is `'absolute'`; a missing keyword counts as `'absolute'`, as in OIFITS1. Differential amplitudes and phases, and correlated fluxes, raise a `ValueError`.
- Nothing in the standard marks `OI_VIS2` `VIS2DATA` that holds squared correlated flux rather than squared visibility, as in MATISSE products reduced with `corrFlux=TRUE`. Such data are read as visibilities, so calibrate them (e.g. with virgil-vlti) before fitting.
- A file with only `OI_T3` gives closure phases alone: its baselines come from the triangle coordinates, every sample is flagged, and `vis` is empty. A file with neither `OI_T3` nor `VISPHI` gives visibilities alone: `phi` is empty. Files with and without phases cannot be read together, and all files in a list must hold the same kinds of observable (squared visibilities or amplitudes; closure or absolute phases; the same extras).
- Samples are flagged when their `FLAG` is set or their value or uncertainty is not finite.
- Phases are converted to radians from the column's `TUNIT` (degrees if it is missing).

### One instrument at a time {#insname}

A table is selected by its `INSNAME`. A GRAVITY product holds fringe-tracker tables (`GRAVITY_FT`, a few low-resolution channels) beside the science channel (`GRAVITY_SC`, or `GRAVITY_SC_P1`/`_P2` in split polarization). These are different measurements of the same baselines and must not be merged, so reading such a file without `insname` raises an error. The two polarizations of the science channel are independent measurements and may be read together.

### Several files {#several-files}

A list of files is read file by file and concatenated in the order given. `TARGET_ID` numbering is per file, so with several files the target must be chosen by name. Frames are numbered within the record, so different files never share one.

## Extra observables {#extras}

`extras=` adds observables beside the visibilities and phases (none by default); see [spectro-interferometric observables](spectro_observables.md).

- `"flux"`: `OI_FLUX` as a spectrum known up to a grey scale (`FLUXDATA`, or GRAVITY's `FLUX`).
- `"nflux"`: `OI_FLUX` as a spectrum normalized to its continuum (choose one of `"flux"` and `"nflux"`).
- `"t3amp"`: the triple amplitudes `T3AMP` of `OI_T3`.
- `"visamp"`: `OI_VIS` `VISAMP` beside `OI_VIS2`, as the table's `AMPTYP` declares it (`'absolute'`: $|V|$; `'correlated flux'`: $|V|$ times the total flux, up to a grey scale).
- `"visphi"`: `OI_VIS` `VISPHI` as a differential phase, whatever its `PHITYP`, beside any closure phases.

A `VISPHI` or `VISAMP` row is matched to the visibility sample of its baseline and time as closure-phase legs are (below); a reversed baseline negates the phase. With `extras`, `VISPHI` is never read as an absolute phase, so without `OI_T3` the phase block is empty. A file whose `OI_VIS` holds correlated fluxes and no `OI_VIS2` gives flagged visibility samples, with the amplitudes in `"visamp"`.

With `bandwidth=True` the record also holds `"bandwidth"`, the `EFF_BAND` of each sample's channel (matched by wavelength within the selected `insname`; absent if the file gives only zeros). It is off by default, so that data are not smeared unless asked.

## Closure-phase triangles {#closure-phases}

Each triangle $(a, b, c)$ is matched to the visibility baselines $(a, b)$, $(b, c)$ and $(a, c)$ with the same `INSNAME` of the same array (or, failing that, any `INSNAME` of that array with identical wavelengths) and the nearest `MJD` and `TIME`, within its own file. Station numbers belong to an array, so tables of another `ARRNAME` are never used.

The `MJD`s (and `TIME`s) must agree to within twice the longest `INT_TIME` in the visibility table, or about 9 seconds if there is none. Pipelines such as GRAVITY's average different frames of one exposure for each table: 8 DITs of 30 s with 120 s of valid frames can give MJDs 131 s apart. The closure phase then uses the visibility rows' $(u, v)$, which have rotated slightly.

A leg with no such visibility row, or stored only reversed as $(b, a)$, gets an extra flagged sample at its own $(u, v)$ from the `OI_T3` row (`U1COORD` to `V2COORD`: $(a, b)$ at $(U1, V1)$, $(b, c)$ at $(U2, V2)$ and $(a, c)$ at their sum), as the standard makes `OI_T3` self-contained. For a reversed leg the model's visibility there is the conjugate of the stored one. A file with legs that have no visibility row in either orientation (V² coverage incomplete, as in some MIRC-X and ESO phase-3 files) gives one warning with their number.

Those coordinates must be finite and non-zero (older AMI and legacy writers leave them NaN or 0), and must agree to within 10% with the legs of the triangle that do have a visibility row; a table written with the opposite baseline direction would otherwise fit a mirrored closure phase. Otherwise the reader raises.

## Frames and times {#frames}

A frame is one exposure of one instrument: the baselines that closure phases tie together, together with any rows of the same `INSNAME` at the same `MJD` and `TIME` (each within about 9 seconds). Both are compared because some OIFITS v1 writers (e.g. OYSTER) give a night one `MJD` and put each snapshot in `TIME`, and closure phases are correlated only within a snapshot.

With `frame_mjd="mean"` (the default) every sample of a frame is given the mean `MJD` of the frame's rows; `"row"` keeps each row's own `MJD`. The time of a sample is `mjd` (days, float64).

## Writing {#writing}

`write_oifits(tables, filename)` writes an OIFITS2 file from a dictionary of tables in the layout of the legacy `virgil.legacy.oifits_implaneia.save`:

- `"OI_WAVELENGTH"` (required): `EFF_WAVE` and `EFF_BAND` in metres, one value per channel.
- At least one of `"OI_VIS2"`, `"OI_VIS"` and `"OI_T3"`, and optionally `"OI_FLUX"` (`FLUXDATA`, `FLUXERR` and a `STA_INDEX` per row). Data columns (e.g. `VIS2DATA`, `T3PHI`) have shape `(nrow,)` or `(nrow, nwave)`; **phases are in degrees**. `T3AMP` and `T3AMPERR` may be omitted (they are then NaN and flagged). The keywords `AMPTYP` and `PHITYP` of `OI_VIS` (default `'absolute'`) and `CALSTAT` of `OI_FLUX` (default `'C'`) may be given as entries of their table. `TARGET_ID`, `TIME`, `MJD` and `INT_TIME` may be scalars; `FLAG` is optional (default: nothing flagged).
- `"OI_TARGET"` and `"OI_ARRAY"` (optional): columns of those tables. When omitted, a single target named from `info` and an array built from `info["STAXY"]` (or placeholder stations) are written.
- `"info"` (optional): primary-header keywords (`OBJECT`, `TELESCOP`, `INSTRUME`, `DATE-OBS`) plus `INSNAME`, `ARRNAME`, `TARGET`, `MJD` and `STAXY` defaults.

Unlike the legacy writer, it does not query SIMBAD: target coordinates come from `tables["OI_TARGET"]` or `info` (`RA`/`DEC` in degrees) and are zero otherwise. [`build_hdulist`][virgil.oifits.build_hdulist] builds the `HDUList` without writing it, to add non-standard columns or keywords first.
