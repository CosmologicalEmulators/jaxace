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


@dataclass
class GenericEmulator(AbstractTrainedEmulator):
    """
    Generic emulator that wraps a trained neural network with normalization and postprocessing.

    This class provides a complete emulator interface that:
    1. Normalizes inputs using min-max scaling
    2. Runs the underlying neural network
    3. Denormalizes outputs
    4. Applies optional postprocessing

    This matches the Julia AbstractCosmologicalEmulators.jl GenericEmulator struct.

    Attributes:
        trained_emulator: The underlying trained neural network (FlaxEmulator)
        in_minmax: Input normalization parameters, shape (n_input_features, 2)
                   Column 0 is min, column 1 is max
        out_minmax: Output normalization parameters, shape (n_output_features, 2)
                    Column 0 is min, column 1 is max
        postprocessing: Optional postprocessing function with signature
                       (input_params, output, auxiliary_params, emulator) -> processed_output
    """
    trained_emulator: AbstractTrainedEmulator
    in_minmax: np.ndarray
    out_minmax: np.ndarray
    postprocessing: callable = None

    # Cached JIT-compiled functions for the full pipeline
    _jit_run: Optional[Any] = field(default=None, init=False, repr=False)
    _jit_run_batch: Optional[Any] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Convert to JAX arrays for efficient computation
        self._in_minmax_jax = jnp.asarray(self.in_minmax)
        self._out_minmax_jax = jnp.asarray(self.out_minmax)

        # Set default identity postprocessing if None
        if self.postprocessing is None:
            self.postprocessing = lambda input_params, output, aux, emu: output

        # Pre-compile JIT functions
        self._ensure_jit_compiled()

    def _maximin(self, input_data: jnp.ndarray) -> jnp.ndarray:
        """Normalize input data using min-max scaling to [0, 1]."""
        if input_data.ndim == 1:
            return (input_data - self._in_minmax_jax[:, 0]) / (self._in_minmax_jax[:, 1] - self._in_minmax_jax[:, 0])
        else:
            # Batch case: (n_samples, n_features)
            return (input_data - self._in_minmax_jax[:, 0]) / (self._in_minmax_jax[:, 1] - self._in_minmax_jax[:, 0])

    def _inv_maximin(self, output_data: jnp.ndarray) -> jnp.ndarray:
        """Denormalize output data from [0, 1] to original scale."""
        if output_data.ndim == 1:
            return output_data * (self._out_minmax_jax[:, 1] - self._out_minmax_jax[:, 0]) + self._out_minmax_jax[:, 0]
        else:
            # Batch case
            return output_data * (self._out_minmax_jax[:, 1] - self._out_minmax_jax[:, 0]) + self._out_minmax_jax[:, 0]

    def _run_pipeline_single(self, input_data: jnp.ndarray) -> jnp.ndarray:
        """Internal method: normalize -> run NN -> denormalize (single sample)."""
        normalized = self._maximin(input_data)
        nn_output = self.trained_emulator._run_single(normalized)
        denormalized = self._inv_maximin(nn_output)
        return denormalized

    def _ensure_jit_compiled(self):
        """Lazily compile JIT functions."""
        if self._jit_run is None:
            self._jit_run = jax.jit(self._run_pipeline_single)
        if self._jit_run_batch is None:
            self._jit_run_batch = jax.jit(jax.vmap(self._run_pipeline_single, in_axes=0))

    def run_emulator(
        self,
        input_params: Union[jnp.ndarray, np.ndarray],
        auxiliary_params: Union[jnp.ndarray, np.ndarray, None] = None
    ) -> jnp.ndarray:
        """
        Run the complete emulator pipeline.

        Steps:
        1. Normalize inputs using in_minmax
        2. Run the neural network
        3. Denormalize outputs using out_minmax
        4. Apply postprocessing function

        Args:
            input_params: Input parameters, shape (n_features,) or (n_samples, n_features)
            auxiliary_params: Optional auxiliary parameters passed to postprocessing

        Returns:
            Processed output array
        """
        # Convert to JAX array if needed
        if not isinstance(input_params, jnp.ndarray):
            input_params = jnp.asarray(input_params)

        if auxiliary_params is None:
            auxiliary_params = jnp.array([])
        elif not isinstance(auxiliary_params, jnp.ndarray):
            auxiliary_params = jnp.asarray(auxiliary_params)

        # Ensure JIT functions are compiled
        self._ensure_jit_compiled()

        # Detect batch vs single
        is_batch = input_params.ndim == 2

        # Run the normalize -> NN -> denormalize pipeline
        if is_batch:
            output = self._jit_run_batch(input_params)
        else:
            output = self._jit_run(input_params)

        # Apply postprocessing
        return self.postprocessing(input_params, output, auxiliary_params, self)

    def get_emulator_description(self) -> Dict[str, Any]:
        """Get description from the underlying trained emulator."""
        return self.trained_emulator.get_emulator_description()

    def __call__(
        self,
        input_params: Union[jnp.ndarray, np.ndarray],
        auxiliary_params: Union[jnp.ndarray, np.ndarray, None] = None
    ) -> jnp.ndarray:
        """Allow the emulator to be called directly as a function."""
        return self.run_emulator(input_params, auxiliary_params)