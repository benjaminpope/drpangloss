# `drpangloss.oifits`

Read and write OIFITS files with `astropy.io.fits` alone. This is the
maintained OIFITS path; [`OIData`](oidata.md) uses `read_oifits` whenever it
is given a file path or an opened `HDUList` (including `pyoifits` objects).

The reader supports:

- several wavelength channels (every baseline × channel becomes one sample);
- several `OI_VIS2`/`OI_VIS`/`OI_T3` tables and epochs, matched by
  `INSNAME`, station indices and `MJD`;
- `FLAG` columns and non-finite values, which are left out of the observables;
- several targets, chosen with `target=`;
- absolute phases from `OI_VIS` `VISPHI` when there is no `OI_T3`.

Closure-phase triangles `(a, b, c)` must find their baselines stored as
`(a, b)`, `(b, c)` and `(a, c)`; reversed legs raise a clear error.

## Functions

::: drpangloss.oifits
    options:
      members:
        - read_oifits
        - write_oifits
        - build_hdulist
