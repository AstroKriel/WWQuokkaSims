## { MODULE

##
## === DEPENDENCIES
##

## personal
from jormi.ww_plots import latex_labels

##
## === FIELD LABELS
##

DENSITY = latex_labels.LatexLabel(content=r"\rho")
VELOCITY = latex_labels.LatexLabel(content=r"\vec{v}")
VELOCITY_MAGNITUDE = latex_labels.LatexLabel(content=r"|\vec{v}|")
MAGNETIC = latex_labels.LatexLabel(content=r"\vec{b}")
TOTAL_ENERGY = latex_labels.LatexLabel(content=r"E_\mathrm{tot}")
INTERNAL_ENERGY = latex_labels.LatexLabel(content=r"E_\mathrm{int}")
KINETIC_ENERGY = latex_labels.LatexLabel(content=r"E_\mathrm{kin}")
KINETIC_ENERGY_COMPRESSIVE = latex_labels.LatexLabel(content=r"E_{\mathrm{kin}, \parallel}")
KINETIC_ENERGY_SOLENOIDAL = latex_labels.LatexLabel(content=r"E_{\mathrm{kin}, \perp}")
KINETIC_ENERGY_BULK = latex_labels.LatexLabel(content=r"E_{\mathrm{kin}, \mathrm{bulk}}")
MAGNETIC_ENERGY = latex_labels.LatexLabel(content=r"E_\mathrm{mag}")
ENERGY_RATIO = latex_labels.LatexLabel(content=r"E_\mathrm{mag} / E_\mathrm{kin}")
PLASMA_BETA = latex_labels.LatexLabel(content=r"\beta")
PRESSURE = latex_labels.LatexLabel(content=r"p")
VELOCITY_DIVERGENCE = latex_labels.LatexLabel(content=r"\nabla\cdot\vec{v}")
VELOCITY_GRADIENT = latex_labels.LatexLabel(content=r"\nabla\vec{v}")
VORTICITY = latex_labels.LatexLabel(content=r"\nabla\times\vec{v}")
VORTICITY_MAGNITUDE = latex_labels.LatexLabel(content=r"|\nabla\times\vec{v}|")
MAGNETIC_DIVERGENCE = latex_labels.LatexLabel(content=r"\nabla\cdot\vec{b}")
CURRENT_DENSITY = latex_labels.LatexLabel(content=r"\nabla\times\vec{b}")
CURRENT_DENSITY_MAGNITUDE = latex_labels.LatexLabel(content=r"|\nabla\times\vec{b}|")

## } MODULE
