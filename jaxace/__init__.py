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
    a_z, E_a, E_z, dlogEdloga, Ωma,
    D_z, f_z, D_f_z,
    D_z_from_cosmo, f_z_from_cosmo, D_f_z_from_cosmo,
    r_z, r_z_from_cosmo,
    dA_z, dA_z_from_cosmo,
    dL_z, dL_z_from_cosmo,
    ρc_z, Ωtot_z
)

__version__ = "0.1.0"

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
    "a_z", "E_a", "E_z", "dlogEdloga", "Ωma",
    "D_z", "f_z", "D_f_z",
    "D_z_from_cosmo", "f_z_from_cosmo", "D_f_z_from_cosmo",
    "r_z", "r_z_from_cosmo",
    "dA_z", "dA_z_from_cosmo", 
    "dL_z", "dL_z_from_cosmo",
    "ρc_z", "Ωtot_z"
]