"""
Core emulator types and functions with automatic JIT compilation
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass, field
import os

# Allow user to configure precision via environment variable
if os.environ.get('JAXACE_ENABLE_X64', 'true').lower() == 'true':
    try:
        jax.config.update("jax_enable_x64", True)
    except RuntimeError:
        # Config already set, that's fine
        pass


class AbstractTrainedEmulator(ABC):
    """Abstract base class for trained emulators."""
    
    @abstractmethod
    def run_emulator(self, input_data: jnp.ndarray) -> jnp.ndarray:
        """Run the emulator on input data."""
        pass
    
    @abstractmethod
    def get_emulator_description(self) -> Dict[str, Any]:
        """Get the emulator description dictionary."""
        pass


@dataclass
class FlaxEmulator(AbstractTrainedEmulator):
    """
    Flax-based emulator with automatic JIT compilation.
    
    Key features:
    1. Automatic JIT compilation on first use
    2. Automatic batch detection and vmap application
    3. Cached compiled functions for performance
    
    Attributes:
        model: Flax model (nn.Module)
        parameters: Model parameters dictionary
        states: Model states (usually empty for standard feedforward networks)
        description: Emulator description dictionary
    """
    model: nn.Module
    parameters: Dict[str, Any]
    states: Optional[Dict[str, Any]] = None
    description: Dict[str, Any] = None
    
    # Private cached JIT-compiled functions
    _jit_single: Optional[Any] = field(default=None, init=False, repr=False)
    _jit_batch: Optional[Any] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        if self.states is None:
            self.states = {}
        if self.description is None:
            self.description = {}
        
        # Pre-compile functions
        self._ensure_jit_compiled()
    
    def _ensure_jit_compiled(self):
        """Lazily compile JIT functions on first use."""
        if self._jit_single is None:
            # JIT compile single evaluation
            self._jit_single = jax.jit(self._run_single)
        
        if self._jit_batch is None:
            # JIT compile batch evaluation with vmap
            self._jit_batch = jax.jit(jax.vmap(self._run_single, in_axes=0))
    
    def _run_single(self, input_data: jnp.ndarray) -> jnp.ndarray:
        """Internal method for single sample evaluation."""
        return self.model.apply(self.parameters, input_data)
    
    def run_emulator(self, input_data: Union[jnp.ndarray, np.ndarray]) -> jnp.ndarray:
        """
        Run the emulator with automatic JIT compilation and batch detection.
        
        This method automatically:
        1. Converts numpy arrays to JAX arrays
        2. Detects if input is a batch or single sample
        3. Applies JIT compilation
        4. Uses vmap for batch processing
        
        Args:
            input_data: Input array (single sample or batch)
                       Shape: (n_features,) for single or (n_samples, n_features) for batch
            
        Returns:
            Output array from the neural network
        """
        # Convert to JAX array if needed
        if not isinstance(input_data, jnp.ndarray):
            input_data = jnp.asarray(input_data)
        
        # Ensure JIT functions are compiled
        self._ensure_jit_compiled()
        
        # Check if this is a batch (2D) or single sample (1D)
        is_batch = input_data.ndim == 2
        
        # Use appropriate JIT-compiled function
        if is_batch:
            return self._jit_batch(input_data)
        else:
            return self._jit_single(input_data)
    
    def get_emulator_description(self) -> Dict[str, Any]:
        """Get the emulator description dictionary."""
        return self.description.get("emulator_description", {})
    
    def __call__(self, input_data: Union[jnp.ndarray, np.ndarray]) -> jnp.ndarray:
        """Allow the emulator to be called directly as a function."""
        return self.run_emulator(input_data)


def run_emulator(input_data: jnp.ndarray, emulator: AbstractTrainedEmulator) -> jnp.ndarray:
    """
    Generic function to run any emulator type.
    Maintained for backward compatibility.
    
    Args:
        input_data: Input array
        emulator: The emulator instance
        
    Returns:
        Output array from the neural network
    """
    return emulator.run_emulator(input_data)


def get_emulator_description(emulator: AbstractTrainedEmulator) -> Dict[str, Any]:
    """
    Get emulator description from any emulator type.
    
    Args:
        emulator: AbstractTrainedEmulator instance
        
    Returns:
        Description dictionary
    """
    return emulator.get_emulator_description()