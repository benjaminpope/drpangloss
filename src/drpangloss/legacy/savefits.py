"""Old name for the legacy ImPlaneIA writer.

Re-exports [`save`][drpangloss.legacy.oifits_implaneia.save] and its helpers
from [`drpangloss.legacy.oifits_implaneia`][drpangloss.legacy.oifits_implaneia]
under their old names. New code should use
[`drpangloss.oifits.write_oifits`][drpangloss.oifits.write_oifits].

TODO: remove once downstream scripts no longer import it.
"""

from ..oidata import cp_indices
from .oifits_implaneia import (
    Format_STAINDEX_T3,
    Format_STAINDEX_V2,
    GetWavelength,
    rad2mas,
    save,
)


__all__ = [
    "Format_STAINDEX_T3",
    "Format_STAINDEX_V2",
    "GetWavelength",
    "cp_indices",
    "rad2mas",
    "save",
]
