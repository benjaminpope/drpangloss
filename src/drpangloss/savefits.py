"""Legacy OIFITS writer, kept for backwards compatibility.

This module used to hold a stale copy of the writer in
[`drpangloss.oifits_implaneia`][drpangloss.oifits_implaneia], which could no longer write a file (its
OI_ARRAY table referenced undefined columns). It now re-exports the
maintained functions under their old names.

New code should use [`drpangloss.oifits.write_oifits`][drpangloss.oifits.write_oifits] to write OIFITS
files and [`drpangloss.oifits.read_oifits`][drpangloss.oifits.read_oifits] (or
[`drpangloss.oidata.OIData`][drpangloss.oidata.OIData]) to read them; both depend only on
``astropy.io.fits``.

TODO: deprecate this module, and then remove it, once downstream scripts
have moved to [`drpangloss.oifits`][drpangloss.oifits].
"""

from .oidata import cp_indices
from .oifits_implaneia import (
    ApplyFlag,
    Format_STAINDEX_T3,
    Format_STAINDEX_V2,
    GetWavelength,
    rad2mas,
    save,
)


__all__ = [
    "ApplyFlag",
    "Format_STAINDEX_T3",
    "Format_STAINDEX_V2",
    "GetWavelength",
    "cp_indices",
    "rad2mas",
    "save",
]
