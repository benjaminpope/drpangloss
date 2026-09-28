"""drpangloss: the best of all possible interferometry models."""

name = "drpangloss"

# The legacy writers (drpangloss.savefits, drpangloss.oifits_implaneia) are
# not imported here, so that importing drpangloss does not need astroquery;
# import them explicitly if you need them.
from . import grid_fit, inference, models, oidata, oifits, plotting

__all__ = []
