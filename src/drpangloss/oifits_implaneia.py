"""Legacy OIFITS helpers from ImPlaneIA (reading, plotting, writing AMI data).

These functions work on the ImPlaneIA dictionary layout (``info``,
``OI_VIS``, ``OI_VIS2``, ``OI_T3``, ...). They are kept for existing scripts,
but new code should prefer [`drpangloss.oifits`][drpangloss.oifits], which reads and writes
OIFITS with ``astropy.io.fits`` alone:

* [`drpangloss.oifits.write_oifits`][drpangloss.oifits.write_oifits] accepts the same dictionary layout
  as [`save`][drpangloss.oifits_implaneia.save], without querying SIMBAD.
* [`drpangloss.oidata.OIData`][drpangloss.oidata.OIData] reads OIFITS files directly (including
  several channels and flags), replacing [`load_oifits`][drpangloss.oifits_implaneia.load_oifits].

TODO: once downstream scripts have moved, reduce this module to thin
wrappers around [`drpangloss.oifits`][drpangloss.oifits] (keeping [`load`][drpangloss.oifits_implaneia.load] and
[`show`][drpangloss.oifits_implaneia.show] for the ImPlaneIA dictionary layout) and drop the astroquery and
termcolor dependencies.

Phases in these dictionaries are in **degrees**, as in OIFITS.
"""

import copy
import datetime
import os

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

import jax.numpy as jnp

# cp_indices is re-exported here for existing imports.
from .oidata import OIData, cp_indices  # noqa: F401


# astroquery is imported only when save() queries SIMBAD; tests may replace
# this with a stand-in.
Simbad = None


def _simbad():
    """The astroquery ``Simbad`` class, imported on first use."""
    if Simbad is not None:
        return Simbad
    from astroquery.simbad import Simbad as simbad_class

    return simbad_class


def _cprint(text, *args, **kwargs):
    """Coloured print if termcolor is installed, plain print otherwise."""
    try:
        from termcolor import cprint
    except ImportError:
        print(text)
    else:
        cprint(text, *args, **kwargs)


def _scalar(value):
    """First element of a scalar, list or array (per-table metadata)."""
    return np.ravel(np.asarray(value))[0]


list_color = ["#00a7b5", "#afd1de", "#055c63", "#ce0058", "#8a8d8f", "#f1b2dc"]


def rad2mas(rad):
    """Convert radians to milliarcseconds."""
    return rad / u.milliarcsec.to(u.rad)


def GetWavelength(ins, filt):
    """Return ``(wavelength, bandwidth)`` in metres for an instrument filter.

    Only JWST NIRISS AMI filters (``ins="JWST"``) are tabulated.
    """
    dic_filt = {
        "JWST": {
            "F277W": [2.776, 0.715],
            "F380M": [3.828, 0.205],
            "F430M": [4.286, 0.202],
            "F480M": [4.817, 0.298],
        }
    }

    wl = dic_filt[ins][filt][0] * 1e-6
    e_wl = dic_filt[ins][filt][1] * 1e-6

    return wl, e_wl


def Format_STAINDEX_V2(tab):
    """Return baseline station pairs as 1-based ``(n, 2)`` integers.

    Zero-based indices (any 0 present) are shifted up by one.
    """
    tab = np.asarray(tab, dtype=int).reshape(-1, 2)
    return tab + 1 if tab.min() == 0 else tab  # RAC 2/2021


def Format_STAINDEX_T3(tab):
    """Return triangle station triplets as 1-based ``(n, 3)`` integers.

    Zero-based indices (any 0 present) are shifted up by one.
    """
    tab = np.asarray(tab, dtype=int).reshape(-1, 3)
    return tab + 1 if tab.min() == 0 else tab


def ApplyFlag(data, unit="arcsec"):
    """Remove flagged points and convert baselines to the requested units.

    Parameters
    ----------
    data : dict
        ImPlaneIA dictionary with ``OI_WAVELENGTH``, ``OI_VIS2`` (with a
        ``BL`` baseline-length column), ``OI_T3`` and ``info``.
    unit : {"m", "rad", "arcsec", "lambda"}, optional
        Units of the returned baselines: metres, cycles per radian, cycles
        per arcsecond, or millions of wavelengths.

    Returns
    -------
    tuple
        ``(U, V, bmax, V2, e_V2, cp, e_cp, sp_freq_vis, sp_freq_cp, wl,
        filter)``, with the flagged V² points removed from ``U``/``V`` too.
    """

    wl = data["OI_WAVELENGTH"]["EFF_WAVE"]
    uv_scale = {
        "m": 1,
        "rad": 1 / wl,
        "arcsec": 1 / wl / rad2mas(1e-3),
        "lambda": 1 / wl / 1e6,
    }

    flag_v2 = ~np.asarray(data["OI_VIS2"]["FLAG"], dtype=bool)
    U = np.asarray(data["OI_VIS2"]["UCOORD"])[flag_v2] * uv_scale[unit]
    V = np.asarray(data["OI_VIS2"]["VCOORD"])[flag_v2] * uv_scale[unit]

    V2 = data["OI_VIS2"]["VIS2DATA"][flag_v2]
    e_V2 = data["OI_VIS2"]["VIS2ERR"][flag_v2] * 1
    sp_freq_vis = data["OI_VIS2"]["BL"][flag_v2] * uv_scale[unit]
    flag_cp = ~np.asarray(data["OI_T3"]["FLAG"], dtype=bool)
    cp = data["OI_T3"]["T3PHI"][flag_cp]
    e_cp = data["OI_T3"]["T3PHIERR"][flag_cp]
    sp_freq_cp = data["OI_T3"]["BL"][flag_cp] * uv_scale[unit]
    bmax = 1.2 * np.max(np.sqrt(U**2 + V**2))

    return (
        U,
        V,
        bmax,
        V2,
        e_V2,
        cp,
        e_cp,
        sp_freq_vis,
        sp_freq_cp,
        wl,
        data["info"]["FILT"],
    )


def save(dic, filename=None, datadir=None, verbose=False):
    """
    Save dictionary formatted data into a proper OIFITS (version 2) format file.

    New code should prefer [`drpangloss.oifits.write_oifits`][drpangloss.oifits.write_oifits], which
    takes the same dictionary, supports flags, and makes no network calls.

    Parameters
    ----------
    dic : dict
        ImPlaneIA dictionary with the tables ``OI_WAVELENGTH``, ``OI_VIS``,
        ``OI_VIS2`` and ``OI_T3`` and an ``info`` dict. ``info`` must hold
        ``TARGET``, ``OBJECT``, ``INSTRUME``, ``MASK``, ``ARRNAME``, ``FILT``,
        ``DATE-OBS``, ``TELESCOP``, ``OBSERVER``, ``INSMODE``, ``PA``,
        ``MJD``, ``PSCALE`` (mas/pixel), ``ISZ`` and ``STAXY``/``CTRS_EQT``
        (or an ``OI_ARRAY`` table holding them). Phases (``VISPHI``,
        ``T3PHI``) are in degrees. ``dic`` is not modified.
    filename : str or os.PathLike, optional
        Output filename. If omitted, the name is built from ``TARGET``,
        ``INSTRUME``, ``MASK``, ``FILT`` and ``MJD`` in ``dic["info"]``.
    datadir : str or os.PathLike, optional
        Destination directory (default ``"Saveoifits/"``), created if
        needed.
    verbose : bool, optional
        If ``True``, print progress while writing tables.

    Returns
    -------
    None
        Writes an OIFITS file to disk.

    Notes
    -----
    Unless ``info["TARGET"]`` is ``"UNKNOWN"``, the target's coordinates,
    proper motion, parallax and spectral type are queried from SIMBAD over
    the network (this needs ``astroquery``).
    """
    if dic is None:
        raise ValueError("save(): dic is None; nothing to write.")
    # Work on a copy: the per-table metadata below is normalized in place.
    dic = copy.deepcopy(dic)

    datadir = "Saveoifits/" if datadir is None else os.fspath(datadir)
    os.makedirs(datadir, exist_ok=True)

    if not isinstance(filename, (str, os.PathLike)):
        info = dic["info"]
        filename = "%s_%s_%s_%s_%s.oifits" % (
            str(info["TARGET"]).replace(" ", ""),
            info["INSTRUME"],
            info["MASK"],
            info["FILT"],
            _scalar(info["MJD"]),
        )

    # ------------------------------
    #       Creation OIFITS
    # ------------------------------
    if verbose:
        print("\n\n### Init creation of OI_FITS (%s) :" % (filename))

    hdulist = fits.HDUList()
    hdu = fits.PrimaryHDU()
    hdu.header["DATE"] = datetime.date.today().isoformat()  # Creation date
    hdu.header["ORIGIN"] = "STScI"
    hdu.header["DATE-OBS"] = dic["info"]["DATE-OBS"]
    hdu.header["CONTENT"] = "OIFITS2"
    hdu.header["TELESCOP"] = dic["info"]["TELESCOP"]
    hdu.header["INSTRUME"] = dic["info"]["INSTRUME"]
    hdu.header["OBSERVER"] = dic["info"]["OBSERVER"]
    hdu.header["OBJECT"] = dic["info"]["OBJECT"]
    hdu.header["INSMODE"] = dic["info"]["INSMODE"]
    hdu.header["FILT"] = dic["info"]["FILT"]
    hdu.header["ARRNAME"] = dic["info"]["ARRNAME"]  # Anand 9/2020
    hdu.header["MASK"] = dic["info"]["MASK"]  # Anand 9/2020
    hdu.header["PA"] = dic["info"]["PA"]  # RC 1/2021
    # name of calibrator if applicable. RC 1/2021
    try:
        hdu.header["CALIB"] = dic["info"]["CALIB"]
    except KeyError:
        pass

    hdulist.append(hdu)
    # ------------------------------
    #        OI Wavelength
    # ------------------------------

    if verbose:
        print("-> Including OI Wavelength table...")
    data = dic["OI_WAVELENGTH"]

    # One row per wavelength channel.
    eff_wave = np.atleast_1d(np.asarray(data["EFF_WAVE"], dtype=float))
    eff_band = np.broadcast_to(
        np.asarray(data["EFF_BAND"], dtype=float), eff_wave.shape
    )
    col1 = fits.Column(
        name="EFF_WAVE", format="1E", unit="METERS", array=eff_wave
    )
    col2 = fits.Column(
        name="EFF_BAND", format="1E", unit="METERS", array=eff_band
    )

    coldefs = fits.ColDefs([col1, col2])
    hdu = fits.BinTableHDU.from_columns(coldefs)

    # Header
    hdu.header["EXTNAME"] = "OI_WAVELENGTH"
    hdu.header["OI_REVN"] = 2  # , 'Revision number of the table definition'
    # 'Name of detector, for cross-referencing'
    hdu.header["INSNAME"] = dic["info"]["INSTRUME"]
    hdulist.append(hdu)  # Add current HDU to the final fits file.

    # ------------------------------
    #          OI Target
    # ------------------------------
    if verbose:
        print("-> Including OI Target table...")

    name_star = dic["info"]["TARGET"]

    # Add informations from Simbad:
    if name_star == "UNKNOWN":
        ra, dec, spectyp = [0], [0], ["unknown"]
        pmra, pmdec, plx = [0], [0], [0]
    else:
        # TODO: let callers pass coordinates instead of querying SIMBAD here
        # (drpangloss.oifits.write_oifits already does).
        ra, dec, spectyp, pmra, pmdec, plx = _query_simbad(name_star)

    col1 = fits.Column(name="TARGET_ID", format="1I", array=[1])
    col2 = fits.Column(name="TARGET", format="16A", array=[name_star])
    col3 = fits.Column(name="RAEP0", format="1D", unit="DEGREES", array=ra)
    col4 = fits.Column(name="DECEP0", format="1D", unit="DEGREES", array=dec)
    col5 = fits.Column(name="EQUINOX", format="1E", unit="YEARS", array=[2000])
    col6 = fits.Column(name="RA_ERR", format="1D", unit="DEGREES", array=[0])
    col7 = fits.Column(name="DEC_ERR", format="1D", unit="DEGREES", array=[0])
    col8 = fits.Column(name="SYSVEL", format="1D", unit="M/S", array=[0])
    col9 = fits.Column(name="VELTYP", format="8A", array=["UNKNOWN"])
    col10 = fits.Column(name="VELDEF", format="8A", array=["OPTICAL"])
    col11 = fits.Column(name="PMRA", format="1D", unit="DEG/YR", array=pmra)
    col12 = fits.Column(name="PMDEC", format="1D", unit="DEG/YR", array=pmdec)
    col13 = fits.Column(name="PMRA_ERR", format="1D", unit="DEG/YR", array=[0])
    col14 = fits.Column(
        name="PMDEC_ERR", format="1D", unit="DEG/YR", array=[0]
    )
    col15 = fits.Column(
        name="PARALLAX", format="1E", unit="DEGREES", array=plx
    )
    col16 = fits.Column(
        name="PARA_ERR", format="1E", unit="DEGREES", array=[0]
    )
    col17 = fits.Column(name="SPECTYP", format="16A", array=spectyp)

    coldefs = fits.ColDefs(
        [
            col1,
            col2,
            col3,
            col4,
            col5,
            col6,
            col7,
            col8,
            col9,
            col10,
            col11,
            col12,
            col13,
            col14,
            col15,
            col16,
            col17,
        ]
    )
    hdu = fits.BinTableHDU.from_columns(coldefs)

    hdu.header["EXTNAME"] = "OI_TARGET"
    hdu.header["OI_REVN"] = 2, "Revision number of the table definition"
    hdulist.append(hdu)

    # ------------------------------
    #           OI Array
    # ------------------------------

    if verbose:
        print("-> Including OI Array table...")
    try:
        staxy = dic["info"][
            "STAXY"
        ]  # these are the mask hole xy-coords as built (ctrs_inst)
    except KeyError:
        staxy = dic["OI_ARRAY"]["STAXY"]
    try:
        ctrs_eqt = dic["info"]["CTRS_EQT"]
    except KeyError:
        ctrs_eqt = dic["OI_ARRAY"]["CTRS_EQT"]

    N_ap = len(staxy)

    tel_name = ["A%i" % x for x in np.arange(N_ap) + 1]
    sta_name = tel_name
    diameter = [0] * N_ap

    staxyz = []
    for x in staxy:
        a = list(x)
        line = [a[0], a[1], 0]
        staxyz.append(line)

    sta_index = np.arange(N_ap) + 1

    pscale = dic["info"]["PSCALE"] / 1000.0  # arcsec
    isz = dic["info"]["ISZ"]  # Size of the image to extract NRM data
    fov = [pscale * isz / 2.0] * N_ap  # FOVTYPE RADIUS: half the width
    fovtype = ["RADIUS"] * N_ap

    col1 = fits.Column(name="TEL_NAME", format="16A", array=tel_name)
    col2 = fits.Column(name="STA_NAME", format="16A", array=sta_name)
    col3 = fits.Column(name="STA_INDEX", format="1I", array=sta_index)
    col4 = fits.Column(
        name="DIAMETER", unit="METERS", format="1E", array=diameter
    )
    col5 = fits.Column(name="STAXYZ", unit="METERS", format="3D", array=staxyz)
    col6 = fits.Column(name="FOV", unit="ARCSEC", format="1D", array=fov)
    col7 = fits.Column(name="FOVTYPE", format="6A", array=fovtype)
    col8 = fits.Column(
        name="CTRS_EQT", unit="METERS", format="2D", array=ctrs_eqt
    )  # for debugging

    coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8])
    hdu = fits.BinTableHDU.from_columns(coldefs)

    hdu.header["EXTNAME"] = "OI_ARRAY"
    hdu.header["ARRAYX"] = float(0)
    hdu.header["ARRAYY"] = float(0)
    hdu.header["ARRAYZ"] = float(0)
    hdu.header["ARRNAME"] = dic["info"]["MASK"]
    hdu.header["FRAME"] = "SKY"
    hdu.header["PSCALE"] = pscale * 1000.0  # [mas] RAC 9/2020
    hdu.header["ISZ"] = isz  # RAC 9/2020
    hdu.header["OI_REVN"] = 2, "Revision number of the table definition"

    hdulist.append(hdu)

    # ------------------------------
    #           OI VIS
    # ------------------------------

    if verbose:
        print("-> Including OI Vis table...")

    data = dic["OI_VIS"]
    npts = len(dic["OI_VIS"]["VISAMP"])

    sta_index = Format_STAINDEX_V2(data["STA_INDEX"])
    _normalize_row_metadata(data)
    nslice = _n_channels(data["VISAMP"])
    col1 = fits.Column(
        name="TARGET_ID", format="1I", array=[data["TARGET_ID"]] * npts
    )
    col2 = fits.Column(
        name="TIME", format="1D", unit="SECONDS", array=[data["TIME"]] * npts
    )
    col3 = fits.Column(
        name="MJD", unit="DAY", format="1D", array=[data["MJD"]] * npts
    )
    col4 = fits.Column(
        name="INT_TIME",
        format="1D",
        unit="SECONDS",
        array=[data["INT_TIME"]] * npts,
    )
    col5 = fits.Column(
        name="VISAMP", format="%dD" % nslice, array=data["VISAMP"]
    )
    col6 = fits.Column(
        name="VISAMPERR", format="%dD" % nslice, array=data["VISAMPERR"]
    )
    col7 = fits.Column(
        name="VISPHI",
        format="%dD" % nslice,
        unit="DEGREES",
        array=data["VISPHI"],
    )
    col8 = fits.Column(
        name="VISPHIERR",
        format="%dD" % nslice,
        unit="DEGREES",
        array=data["VISPHIERR"],
    )
    col9 = fits.Column(
        name="UCOORD", format="1D", unit="METERS", array=data["UCOORD"]
    )
    col10 = fits.Column(
        name="VCOORD", format="1D", unit="METERS", array=data["VCOORD"]
    )
    col11 = fits.Column(name="STA_INDEX", format="2I", array=sta_index)
    col12 = fits.Column(
        name="FLAG", format="%dL" % nslice, array=_flags(data, npts, nslice)
    )

    coldefs = fits.ColDefs(
        [
            col1,
            col2,
            col3,
            col4,
            col5,
            col6,
            col7,
            col8,
            col9,
            col10,
            col11,
            col12,
        ]
    )
    hdu = fits.BinTableHDU.from_columns(coldefs)

    hdu.header["OI_REVN"] = 2, "Revision number of the table definition"
    hdu.header["EXTNAME"] = "OI_VIS"
    hdu.header["INSNAME"] = dic["info"]["INSTRUME"]
    hdu.header["ARRNAME"] = dic["info"]["MASK"]
    hdu.header["DATE-OBS"] = (
        dic["info"]["DATE-OBS"],
        "Zero-point for table (UTC)",
    )
    hdulist.append(hdu)

    # ------------------------------
    #           OI VIS2
    # ------------------------------
    if verbose:
        print("-> Including OI Vis2 table...")

    data = dic["OI_VIS2"]
    npts = len(dic["OI_VIS2"]["VIS2DATA"])

    # OI_VIS2 may order its baselines differently from OI_VIS.
    sta_index = Format_STAINDEX_V2(data["STA_INDEX"])
    _normalize_row_metadata(data)
    nslice = _n_channels(data["VIS2DATA"])
    col1 = fits.Column(
        name="TARGET_ID", format="1I", array=[data["TARGET_ID"]] * npts
    )
    col2 = fits.Column(
        name="TIME", format="1D", unit="SECONDS", array=[data["TIME"]] * npts
    )
    col3 = fits.Column(
        name="MJD", unit="DAY", format="1D", array=[data["MJD"]] * npts
    )
    col4 = fits.Column(
        name="INT_TIME",
        format="1D",
        unit="SECONDS",
        array=[data["INT_TIME"]] * npts,
    )
    col5 = fits.Column(
        name="VIS2DATA", format="%dD" % nslice, array=data["VIS2DATA"]
    )
    col6 = fits.Column(
        name="VIS2ERR", format="%dD" % nslice, array=data["VIS2ERR"]
    )
    col7 = fits.Column(
        name="UCOORD", format="1D", unit="METERS", array=data["UCOORD"]
    )
    col8 = fits.Column(
        name="VCOORD", format="1D", unit="METERS", array=data["VCOORD"]
    )
    col9 = fits.Column(name="STA_INDEX", format="2I", array=sta_index)
    col10 = fits.Column(
        name="FLAG", format="%dL" % nslice, array=_flags(data, npts, nslice)
    )

    coldefs = fits.ColDefs(
        [col1, col2, col3, col4, col5, col6, col7, col8, col9, col10]
    )
    hdu = fits.BinTableHDU.from_columns(coldefs)

    hdu.header["EXTNAME"] = "OI_VIS2"
    hdu.header["INSNAME"] = dic["info"]["INSTRUME"]
    hdu.header["ARRNAME"] = dic["info"]["MASK"]
    hdu.header["OI_REVN"] = 2, "Revision number of the table definition"
    hdu.header["DATE-OBS"] = (
        dic["info"]["DATE-OBS"],
        "Zero-point for table (UTC)",
    )
    hdulist.append(hdu)

    # ------------------------------
    #           OI T3
    # ------------------------------
    if verbose:
        print("-> Including OI T3 table...")

    data = dic["OI_T3"]
    npts = len(dic["OI_T3"]["T3PHI"])

    sta_index = Format_STAINDEX_T3(data["STA_INDEX"])
    _normalize_row_metadata(data)
    nslice = _n_channels(data["T3PHI"])

    col1 = fits.Column(
        name="TARGET_ID", format="1I", array=[data["TARGET_ID"]] * npts
    )
    col2 = fits.Column(
        name="TIME", format="1D", unit="SECONDS", array=[data["TIME"]] * npts
    )
    col3 = fits.Column(
        name="MJD", format="1D", unit="DAY", array=[data["MJD"]] * npts
    )
    col4 = fits.Column(
        name="INT_TIME",
        format="1D",
        unit="SECONDS",
        array=[data["INT_TIME"]] * npts,
    )
    col5 = fits.Column(
        name="T3AMP", format="%dD" % nslice, array=data["T3AMP"]
    )
    col6 = fits.Column(
        name="T3AMPERR", format="%dD" % nslice, array=data["T3AMPERR"]
    )
    col7 = fits.Column(
        name="T3PHI",
        format="%dD" % nslice,
        unit="DEGREES",
        array=data["T3PHI"],
    )
    col8 = fits.Column(
        name="T3PHIERR",
        format="%dD" % nslice,
        unit="DEGREES",
        array=data["T3PHIERR"],
    )
    col9 = fits.Column(
        name="U1COORD", format="1D", unit="METERS", array=data["U1COORD"]
    )
    col10 = fits.Column(
        name="V1COORD", format="1D", unit="METERS", array=data["V1COORD"]
    )
    col11 = fits.Column(
        name="U2COORD", format="1D", unit="METERS", array=data["U2COORD"]
    )
    col12 = fits.Column(
        name="V2COORD", format="1D", unit="METERS", array=data["V2COORD"]
    )
    col13 = fits.Column(name="STA_INDEX", format="3I", array=sta_index)
    col14 = fits.Column(
        name="FLAG", format="%dL" % nslice, array=_flags(data, npts, nslice)
    )

    coldefs = fits.ColDefs(
        [
            col1,
            col2,
            col3,
            col4,
            col5,
            col6,
            col7,
            col8,
            col9,
            col10,
            col11,
            col12,
            col13,
            col14,
        ]
    )
    hdu = fits.BinTableHDU.from_columns(coldefs)

    hdu.header["EXTNAME"] = "OI_T3"
    hdu.header["INSNAME"] = dic["info"]["INSTRUME"]
    hdu.header["ARRNAME"] = dic["info"]["MASK"]  # Anand 9/2020
    hdu.header["OI_REVN"] = 2, "Revision number of the table definition"
    hdu.header["DATE-OBS"] = (
        dic["info"]["DATE-OBS"],
        "Zero-point for table (UTC)",
    )
    hdulist.append(hdu)

    # ------------------------------
    #          Save file
    # ------------------------------
    hdulist.writeto(os.path.join(datadir, filename), overwrite=True)
    if verbose:
        _cprint("\n\n### OIFITS CREATED (%s)." % filename, "cyan")


def _normalize_row_metadata(data):
    """Reduce per-table TARGET_ID/TIME/MJD/INT_TIME to scalars.

    TODO: write the per-row values instead, so that tables spanning several
    epochs keep their times (drpangloss.oifits.write_oifits does).
    """
    for key in ("TARGET_ID", "TIME", "MJD", "INT_TIME"):
        if key in data:
            data[key] = _scalar(data[key])


def _n_channels(values):
    """Number of wavelength channels of a data column."""
    values = np.asarray(values)
    return 1 if values.ndim == 1 else values.shape[1]


def _flags(data, npts, nslice):
    """FLAG column of shape ``(npts, nslice)`` (default: nothing flagged)."""
    if "FLAG" not in data:
        return np.zeros((npts, nslice), dtype=bool)
    return np.asarray(data["FLAG"], dtype=bool).reshape(npts, nslice)


def _query_simbad(name):
    """Return SIMBAD ``(ra, dec, spectyp, pmra, pmdec, plx)`` for OI_TARGET.

    Coordinates are in degrees, proper motions in deg/yr and the parallax in
    degrees, as OIFITS requires (SIMBAD gives mas/yr and mas). Any failure,
    including a network error or an unknown target, gives zeros.
    """
    unknown = [0], [0], ["unknown"], [0], [0], [0]
    mas_to_deg = 1.0 / 3.6e6
    try:
        custom_simbad = _simbad()()
        custom_simbad.add_votable_fields(
            "propermotions", "sp_type", "parallax"
        )
        query = custom_simbad.query_object(name)
        names = {col.lower(): col for col in query.colnames}

        def column(*candidates):
            for candidate in candidates:
                if candidate in names:
                    return query[names[candidate]][0]
            raise KeyError(candidates)

        ra, dec = column("ra"), column("dec")
        if isinstance(ra, str):
            # astroquery < 0.4.8 returns sexagesimal strings.
            coord = SkyCoord(ra + " " + dec, unit=(u.hourangle, u.deg))
            ra, dec = coord.ra.deg, coord.dec.deg
        return (
            [float(ra)],
            [float(dec)],
            [str(column("sp_type"))],
            [float(column("pmra")) * mas_to_deg],
            [float(column("pmdec")) * mas_to_deg],
            [float(column("plx_value")) * mas_to_deg],
        )
    except Exception:
        return unknown


def load(filename, target=None, ins=None, mask=None, include_vis=True):
    """Load an OIFITS file into an internal dictionary representation.

    Parameters
    ----------
    filename : str
        Name of the OIFITS file.
    target : str, optional
        Fallback target name if not present in headers.
    ins : str, optional
        Fallback instrument name if not present in headers.
    mask : str, optional
        Fallback mask name if not present in headers.
    include_vis : bool, optional
        If ``True`` (default), include the ``OI_VIS`` (visibility amplitude
        and phase) table when available.

    Returns
    -------
    dict
        ImPlaneIA dictionary of the tables and metadata. Only the last table
        of each type is kept, and phases stay in degrees. To fit the data,
        use [`drpangloss.oidata.OIData`][drpangloss.oidata.OIData] on the file instead.
    """
    with fits.open(filename, mode="readonly", memmap=False) as hdulist:
        fitsHandler = copy.deepcopy(hdulist)

        hdr = fitsHandler[0].header

        dic = {}
        dic["info"] = {}
        try:
            dic["info"]["TARGET"] = hdr["OBJECT"]
        except KeyError:
            dic["info"]["TARGET"] = target
        try:
            dic["info"]["OBJECT"] = hdr["OBJECT"]
        except KeyError:
            dic["info"]["OBJECT"] = None
        try:
            dic["info"]["INSTRUME"] = hdr["INSTRUME"]
        except KeyError:
            dic["info"]["INSTRUME"] = ins
        try:
            dic["info"]["MASK"] = hdr["MASK"]
        except KeyError:
            dic["info"]["MASK"] = mask
        try:
            dic["info"]["FILT"] = hdr["FILT"]
        except KeyError:
            dic["info"]["FILT"] = None
        try:
            dic["info"]["DATE-OBS"] = hdr["DATE-OBS"]
        except KeyError:
            dic["info"]["DATE-OBS"] = None
        try:
            dic["info"]["TELESCOP"] = hdr["TELESCOP"]
        except KeyError:
            dic["info"]["TELESCOP"] = (
                None  # try to get it from somewhere else?
            )
        try:
            dic["info"]["OBSERVER"] = hdr["OBSERVER"]
        except KeyError:
            dic["info"]["OBSERVER"] = None
        try:
            dic["info"]["INSMODE"] = hdr["INSMODE"]
        except KeyError:
            dic["info"]["INSMODE"] = None
        try:
            dic["info"]["PA"] = hdr["PA"]
        except KeyError:
            dic["info"]["PA"] = None

        for hdu in fitsHandler[1:]:
            extname = str(hdu.header.get("EXTNAME", "")).strip().upper()
            # RAC 9/2020
            # try to read in info from the OI_ARRAY required for re-saving
            if extname == "OI_ARRAY":
                # PSCALE, ISZ and CTRS_EQT are ImPlaneIA extensions.
                for key in ("PSCALE", "ISZ"):
                    if key in hdu.header:
                        dic["info"][key] = hdu.header[key]

                # make staxy from staxyz array (remove last column)
                staxyz = hdu.data["STAXYZ"]
                staxy = np.delete(staxyz, -1, 1)
                dic["OI_ARRAY"] = {"STAXYZ": staxyz, "STAXY": staxy}
                if "CTRS_EQT" in hdu.columns.names:
                    dic["OI_ARRAY"]["CTRS_EQT"] = hdu.data["CTRS_EQT"]

            if extname == "OI_WAVELENGTH":
                dic["OI_WAVELENGTH"] = {
                    "EFF_WAVE": hdu.data["EFF_WAVE"],
                    "EFF_BAND": hdu.data["EFF_BAND"],
                }

            if extname == "OI_VIS2":
                dic["OI_VIS2"] = {
                    "VIS2DATA": hdu.data["VIS2DATA"],
                    "VIS2ERR": hdu.data["VIS2ERR"],
                    "UCOORD": hdu.data["UCOORD"],
                    "VCOORD": hdu.data["VCOORD"],
                    "STA_INDEX": hdu.data["STA_INDEX"],
                    "MJD": hdu.data["MJD"],
                    "INT_TIME": hdu.data["INT_TIME"],
                    "TIME": hdu.data["TIME"],
                    "TARGET_ID": hdu.data["TARGET_ID"],
                    "FLAG": np.array(hdu.data["FLAG"]),
                }
                # these are in every extension, but take them from here
                dic["info"]["MJD"] = hdu.data["MJD"][0]
                dic["info"]["ARRNAME"] = hdu.header.get("ARRNAME")
                try:
                    dic["OI_VIS2"]["BL"] = hdu.data["BL"]
                except KeyError:
                    dic["OI_VIS2"]["BL"] = (
                        hdu.data["UCOORD"] ** 2 + hdu.data["VCOORD"] ** 2
                    ) ** 0.5

            if extname == "OI_VIS" and include_vis:
                dic["OI_VIS"] = {
                    "TARGET_ID": hdu.data["TARGET_ID"],
                    "TIME": hdu.data["TIME"],
                    "MJD": hdu.data["MJD"],
                    "INT_TIME": hdu.data["INT_TIME"],
                    "VISAMP": hdu.data["VISAMP"],
                    "VISAMPERR": hdu.data["VISAMPERR"],
                    "VISPHI": hdu.data["VISPHI"],
                    "VISPHIERR": hdu.data["VISPHIERR"],
                    "UCOORD": hdu.data["UCOORD"],
                    "VCOORD": hdu.data["VCOORD"],
                    "STA_INDEX": hdu.data["STA_INDEX"],
                    "FLAG": hdu.data["FLAG"],
                }
                try:
                    dic["OI_VIS"]["BL"] = hdu.data["BL"]
                except KeyError:
                    dic["OI_VIS"]["BL"] = (
                        hdu.data["UCOORD"] ** 2 + hdu.data["VCOORD"] ** 2
                    ) ** 0.5

            if extname == "OI_T3":
                u1 = hdu.data["U1COORD"]
                u2 = hdu.data["U2COORD"]
                v1 = hdu.data["V1COORD"]
                v2 = hdu.data["V2COORD"]
                # Longest baseline of each triangle, in metres.
                bl_cp = np.max(
                    np.hypot([u1, u2, u1 + u2], [v1, v2, v1 + v2]), axis=0
                )

                dic["OI_T3"] = {
                    "T3PHI": hdu.data["T3PHI"],
                    "T3PHIERR": hdu.data["T3PHIERR"],
                    "T3AMP": hdu.data["T3AMP"],
                    "T3AMPERR": hdu.data["T3AMPERR"],
                    "U1COORD": hdu.data["U1COORD"],
                    "V1COORD": hdu.data["V1COORD"],
                    "U2COORD": hdu.data["U2COORD"],
                    "V2COORD": hdu.data["V2COORD"],
                    "STA_INDEX": hdu.data["STA_INDEX"],
                    "MJD": hdu.data["MJD"],
                    "FLAG": hdu.data["FLAG"],
                    "TARGET_ID": hdu.data["TARGET_ID"],
                    "TIME": hdu.data["TIME"],
                    "INT_TIME": hdu.data["INT_TIME"],
                }
                dic["OI_T3"]["BL"] = bl_cp
    del fitsHandler

    return dic


def show(
    inputList,
    diffWl=False,
    vmin=0,
    vmax=1.05,
    cmax=180,
    setlog=False,
    unit="arcsec",
    unit_cp="deg",
):
    """Display visibility and closure-phase diagnostics for one or more datasets.

    Parameters
    ----------
    inputList : list or str or dict
        Single input or list of inputs, where each item is either an OIFITS
        filename or a dictionary produced by ``load``.
    diffWl : bool, optional
        If ``True``, color-code points by wavelength/filter.
    vmin : float, optional
        Lower y-axis limit for visibility panel.
    vmax : float, optional
        Upper y-axis limit for visibility panel.
    cmax : float, optional
        Maximum absolute closure phase for plotting.
    setlog : bool, optional
        If ``True``, use logarithmic scaling for visibility values.
    unit : str, optional
        Unit for spatial frequencies.
    unit_cp : str, optional
        Unit label for closure phases.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing UV, visibility, and closure-phase panels.
    """
    from matplotlib import pyplot as plt

    if not isinstance(inputList, list):
        inputList = [inputList]

    if isinstance(inputList[0], (str, os.PathLike)):
        l_dic = [load(x) for x in inputList]
    elif isinstance(inputList[0], dict):
        l_dic = inputList
    else:
        raise TypeError(
            "show() expects OIFITS filenames or dictionaries from load()."
        )

    dic_color = {}
    for dic in l_dic:
        filt = dic["info"]["FILT"]
        if filt not in dic_color:
            dic_color[filt] = list_color[len(dic_color) % len(list_color)]

    fig = plt.figure(figsize=(16, 5.5))
    ax1 = plt.subplot2grid((2, 6), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=4)
    ax3 = plt.subplot2grid((2, 6), (1, 2), colspan=4)

    # Plot plan UV
    # -------
    l_bmax, l_band_al = [], []
    for dic in l_dic:
        tmp = ApplyFlag(dic)
        U = tmp[0]
        V = tmp[1]
        band = tmp[10]
        wl = tmp[9]
        label = "%2.2f $\\mu m$ (%s)" % (_scalar(wl) * 1e6, band)
        if diffWl:
            c1, c2 = dic_color[band], dic_color[band]
            if band in l_band_al:
                label = None  # one legend entry per filter
        else:
            c1, c2 = "#00adb5", "#fc5185"
        l_bmax.append(tmp[2])
        l_band_al.append(band)

        ax1.scatter(
            U,
            V,
            s=50,
            c=c1,
            label=label,
            edgecolors="#364f6b",
            marker="o",
            alpha=1,
        )
        ax1.scatter(
            -1 * np.array(U),
            -1 * np.array(V),
            s=50,
            c=c2,
            edgecolors="#364f6b",
            marker="o",
            alpha=1,
        )

    Bmax = np.max(l_bmax)
    ax1.axis([Bmax, -Bmax, -Bmax, Bmax])
    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.patch.set_facecolor("#f7f9fc")
    ax1.patch.set_alpha(1)
    ax1.xaxis.set_ticks_position("none")
    ax1.yaxis.set_ticks_position("none")
    if diffWl:
        handles, labels = ax1.get_legend_handles_labels()
        labels, handles = zip(
            *sorted(zip(labels, handles), key=lambda t: t[0])
        )
        ax1.legend(handles, labels, loc="best", fontsize=9)
        # ax1.legend(loc='best')

    unitlabel = {
        "m": "m",
        "rad": "rad$^{-1}$",
        "arcsec": "arcsec$^{-1}$",
        "lambda": "M$\\lambda$",
    }

    ax1.set_xlabel(r"U [%s]" % unitlabel[unit])
    ax1.set_ylabel(r"V [%s]" % unitlabel[unit])
    ax1.grid(alpha=0.2)

    # Plot V2
    # -------
    max_f_vis = []
    for dic in l_dic:
        tmp = ApplyFlag(dic, unit="arcsec")
        V2 = tmp[3]
        e_V2 = tmp[4]
        sp_freq_vis = tmp[7]
        max_f_vis.append(np.max(sp_freq_vis))
        band = tmp[10]
        if diffWl:
            mfc = dic_color[band]
        else:
            mfc = "#00adb5"

        ax2.errorbar(
            sp_freq_vis,
            V2,
            yerr=e_V2,
            linestyle="None",
            capsize=1,
            mfc=mfc,
            ecolor="#364f6b",
            mec="#364f6b",
            marker=".",
            elinewidth=0.5,
            alpha=1,
            ms=9,
        )

    ax2.hlines(
        1, 0, 1.2 * np.max(max_f_vis), lw=1, color="k", alpha=0.2, ls="--"
    )

    ax2.set_ylim([vmin, vmax])
    ax2.set_xlim([0, 1.2 * np.max(max_f_vis)])
    ax2.set_ylabel(r"$V^2$")
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.patch.set_facecolor("#f7f9fc")
    ax2.patch.set_alpha(1)
    ax2.xaxis.set_ticks_position("none")
    ax2.yaxis.set_ticks_position("none")
    # ax2.set_xticklabels([])

    if setlog:
        ax2.set_yscale("log")
    ax2.grid(which="both", alpha=0.2)

    # Plot CP
    # -------

    if unit_cp == "rad":
        conv_cp = np.pi / 180.0
        h1 = np.pi
    else:
        conv_cp = 1
        h1 = np.rad2deg(np.pi)

    cmin = -cmax

    max_f_cp = []
    for dic in l_dic:
        tmp = ApplyFlag(dic, unit="arcsec")
        cp = tmp[5] * conv_cp
        e_cp = tmp[6] * conv_cp
        sp_freq_cp = tmp[8]
        max_f_cp.append(np.max(sp_freq_cp))
        band = tmp[10]
        if diffWl:
            mfc = dic_color[band]
        else:
            mfc = "#00adb5"

        ax3.errorbar(
            sp_freq_cp,
            cp,
            yerr=e_cp,
            linestyle="None",
            capsize=1,
            mfc=mfc,
            ecolor="#364f6b",
            mec="#364f6b",
            marker=".",
            elinewidth=0.5,
            alpha=1,
            ms=9,
        )
    ax3.hlines(
        h1, 0, 1.2 * np.max(max_f_cp), lw=1, color="k", alpha=0.2, ls="--"
    )
    ax3.hlines(
        -h1, 0, 1.2 * np.max(max_f_cp), lw=1, color="k", alpha=0.2, ls="--"
    )
    ax3.spines["left"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["bottom"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.patch.set_facecolor("#f7f9fc")
    ax3.patch.set_alpha(1)
    ax3.xaxis.set_ticks_position("none")
    ax3.yaxis.set_ticks_position("none")
    ax3.set_xlabel("Spatial frequency [cycle/arcsec]")
    ax3.set_ylabel("Clos. $\\phi$ [%s]" % unit_cp)
    ax3.axis([0, 1.2 * np.max(max_f_cp), cmin * conv_cp, cmax * conv_cp])
    ax3.grid(which="both", alpha=0.2)

    plt.subplots_adjust(
        top=0.974,
        bottom=0.091,
        left=0.04,
        right=0.99,
        hspace=0.127,
        wspace=0.35,
    )

    plt.show(block=False)
    return fig


def load_oifits(filename, directory):
    """Load a single OIFITS file and return flattened AMI-ready observables.

    Prefer [`drpangloss.oidata.OIData`][drpangloss.oidata.OIData], which reads the same file with
    phases in radians, several wavelength channels and flags.

    Parameters
    ----------
    filename : str
        OIFITS file name.
    directory : str
        Directory containing the file.

    Returns
    -------
    tuple
        ``(u, v, cp, cp_err, vis2, vis2_err, i_cps1, i_cps2, i_cps3)``:
        spatial frequencies in cycles per radian, closure phases and their
        errors in **degrees**, squared visibilities and their errors, and the
        closure-phase baseline indices. Flagged points are removed, except
        for spatial frequencies needed by closure phases.
    """
    data = OIData(os.path.join(directory, filename))
    uu, vv, cp, cp_err, vis2, vis2_err, i1, i2, i3 = data.unpack_all()
    return (
        jnp.asarray(uu),
        jnp.asarray(vv),
        jnp.rad2deg(cp),
        jnp.rad2deg(cp_err),
        vis2,
        vis2_err,
        i1,
        i2,
        i3,
    )
