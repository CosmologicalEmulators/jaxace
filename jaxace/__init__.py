"""
JAX AbstractCosmologicalEmulators (jaxace)

A JAX/Flax implementation of the AbstractCosmologicalEmulators.jl interface,
providing foundational neural network emulator infrastructure for cosmological
computations.
"""

from .core import (
    AbstractTrainedEmulator,
    FlaxEmulator,
    GenericEmulator,
    run_emulator,
    get_emulator_description
)

from .initialization import (
    init_emulator,
    load_trained_emulator,
    load_trained_emulator_from_artifact,
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
    w0waCDMCosmology,
    a_z, E_a, E_z, dlogEdloga, Ωm_a,
    D_z, f_z, D_f_z,
    r̃_z, d̃M_z, d̃A_z,
    r_z, dM_z,
    dA_z,
    dL_z,
    ρc_z, Ωtot_z,
    S_of_K
)

# Import artifact management for auto-loading emulators
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional
from fetch_artifacts import load_artifacts

__version__ = "0.5.0"

# Initialize the trained_emulators dictionary
trained_emulators: Dict[str, GenericEmulator] = {}

# Path to Artifacts.toml (in package directory)
_ARTIFACTS_TOML = Path(__file__).parent.parent / "Artifacts.toml"

# Global artifact manager
_artifact_manager = None


def _get_artifact_manager():
    """Get or create the artifact manager singleton."""
    global _artifact_manager
    if _artifact_manager is None:
        if _ARTIFACTS_TOML.exists():
            _artifact_manager = load_artifacts(_ARTIFACTS_TOML)
        else:
            warnings.warn(f"Artifacts.toml not found at {_ARTIFACTS_TOML}")
    return _artifact_manager


def list_emulators() -> List[str]:
    """
    List all available emulators defined in Artifacts.toml.

    Returns
    -------
    list of str
        Names of available emulators.

    Examples
    --------
    >>> import jaxace
    >>> jaxace.list_emulators()
    ['ACE_mnuw0wacdm_sigma8_basis']
    """
    manager = _get_artifact_manager()
    if manager is None:
        return []
    return list(manager.artifacts.keys())


def get_emulator(name: str, download: bool = True) -> GenericEmulator:
    """
    Get a trained emulator by name.

    Parameters
    ----------
    name : str
        Name of the emulator as defined in Artifacts.toml
    download : bool
        Whether to download if not already loaded (default: True)

    Returns
    -------
    GenericEmulator
        The loaded emulator

    Examples
    --------
    >>> import jaxace
    >>> emu = jaxace.get_emulator('ACE_mnuw0wacdm_sigma8_basis')
    >>> output = emu.run_emulator(input_params)
    """
    if name in trained_emulators:
        return trained_emulators[name]

    if download:
        # Load and cache the emulator
        emu = load_trained_emulator_from_artifact(name)
        trained_emulators[name] = emu
        return emu
    else:
        raise ValueError(f"Emulator '{name}' not loaded and download=False")


# Auto-load emulators on import (unless disabled)
if not os.environ.get("JAXACE_NO_AUTO_DOWNLOAD"):
    manager = _get_artifact_manager()
    if manager is not None:
        # Pre-load all emulators defined in Artifacts.toml
        for emulator_name in manager.artifacts:
            try:
                trained_emulators[emulator_name] = load_trained_emulator_from_artifact(
                    emulator_name
                )
            except Exception:
                # Silently skip emulators that fail to load
                pass

__all__ = [
    # Core types and functions
    "AbstractTrainedEmulator",
    "FlaxEmulator",
    "GenericEmulator",
    "run_emulator",
    "get_emulator_description",

    # Initialization
    "init_emulator",
    "load_trained_emulator",
    "load_trained_emulator_from_artifact",
    "MLP",

    # Artifact management
    "trained_emulators",
    "list_emulators",
    "get_emulator",

    # Utilities
    "maximin",
    "inv_maximin",
    "validate_nn_dict_structure",
    "validate_parameter_ranges",
    "validate_layer_structure",
    "safe_dict_access",

    # Background cosmology
    "w0waCDMCosmology",
    "a_z", "E_a", "E_z", "dlogEdloga", "Ωm_a",
    "D_z", "f_z", "D_f_z",
    "r̃_z", "d̃M_z", "d̃A_z",
    "r_z", "dM_z",
    "dA_z",
    "dL_z",
    "ρc_z", "Ωtot_z",
    "S_of_K"
]