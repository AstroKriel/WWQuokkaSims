## { MODULE

##
## === DEPENDENCIES
##

##
## === LOOKUP TABLES
## Shorthand scheme letters, as used in dataset directory/combo names (e.g. `q26-b25-ppm_ep`),
## mapped to the values Quokka's MHD implementation actually expects. Shared across every
## problem-type profile in `sim_types/`, since these mappings are universal, not per-problem.
##

RECONSTRUCTION_ORDER_BY_INTERPOLATION = {
    "pcm": 1,
    "plm": 2,
    "ppm": 3,
    "ppm_ep": 5,
}

EMF_COMPUTE_SCHEME_BY_LETTER = {
    "q26": "Quokka2026",
    "fs17": "FelkerStone2017",
    "b25": "Balsara2025",
}

EMF_AVERAGING_SCHEME_BY_LETTER = {
    "b25": "Balsara2025",
    "ld04": "LondrilloDelZanna2004",
}


## } MODULE
