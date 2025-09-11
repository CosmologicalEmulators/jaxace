"""
Core emulator types and functions matching AbstractCosmologicalEmulators.jl
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass
import os

# Allow user to configure precision via environment variable
if os.environ.get('JAXACE_ENABLE_X64', 'true').lower() == 'true':
    try:
        jax.config.update("jax_enable_x64", True)
    except RuntimeError:
        # Config already set, that's fine
        pass


class AbstractTrainedEmulator(ABC):
    """Abstract base class for trained emulators, matching AbstractTrainedEmulators in Julia."""
    
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
    Flax-based emulator matching LuxEmulator in Julia.
    
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
    
    def __post_init__(self):
        if self.states is None:
            self.states = {}
        if self.description is None:
            self.description = {}
    
    def run_emulator(self, input_data: jnp.ndarray) -> jnp.ndarray:
        """
        Run the emulator on input data.
        
        Args:
            input_data: Input array
            
        Returns:
            Output array from the neural network
        """
        # Apply the model with parameters
        output = self.model.apply(self.parameters, input_data)
        return output
    
    def get_emulator_description(self) -> Dict[str, Any]:
        """Get the emulator description dictionary."""
        return self.description.get("emulator_description", {})


def run_emulator(input_data: jnp.ndarray, emulator: AbstractTrainedEmulator) -> jnp.ndarray:
    """
    Generic function to run any emulator type, matching Julia's multiple dispatch.
    
    Args:
        input_data: Input array
        emulator: AbstractTrainedEmulator instance
        
    Returns:
        Output from the emulator
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