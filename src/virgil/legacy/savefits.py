"""Old name for the legacy ImPlaneIA writer.

Re-exports [`save`][virgil.legacy.oifits_implaneia.save] and its helpers
from [`virgil.legacy.oifits_implaneia`][virgil.legacy.oifits_implaneia]
under their old names. New code should use
[`virgil.oifits.write_oifits`][virgil.oifits.write_oifits].

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
