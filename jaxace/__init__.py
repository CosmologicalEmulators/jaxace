"""
JAX AbstractCosmologicalEmulators (jaxace)

A JAX/Flax implementation of the AbstractCosmologicalEmulators.jl interface,
providing foundational neural network emulator infrastructure for cosmological
computations.
"""

from .core import (
    AbstractTrainedEmulator,
    FlaxEmulator,
    run_emulator,
    get_emulator_description
)

from .initialization import (
    init_emulator,
    MLP
)

from .utils import (
    maximin,
    inv_maximin,
    validate_nn_dict_structure,
    validate_parameter_ranges,
    validate_layer_structure,
    safe_dict_access
)

# Import background cosmology functions
from .background import (
    W0WaCDMCosmology,
    a_z, E_a, E_z, dlogEdloga, Ωm_a,
    D_z, f_z, D_f_z,
    r_z,
    dA_z,
    dL_z,
    ρc_z, Ωtot_z
)

__version__ = "0.2.0"

__all__ = [
    # Core types and functions
    "AbstractTrainedEmulator",
    "FlaxEmulator", 
    "run_emulator",
    "get_emulator_description",
    
    # Initialization
    "init_emulator",
    "MLP",
    
    # Utilities
    "maximin",
    "inv_maximin",
    "validate_nn_dict_structure",
    "validate_parameter_ranges",
    "validate_layer_structure",
    "safe_dict_access",
    
    # Background cosmology
    "W0WaCDMCosmology",
    "a_z", "E_a", "E_z", "dlogEdloga", "Ωm_a",
    "D_z", "f_z", "D_f_z",
    "r_z",
    "dA_z",
    "dL_z",
    "ρc_z", "Ωtot_z"
]