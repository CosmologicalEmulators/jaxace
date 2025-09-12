"""
Neural network initialization functions matching AbstractCosmologicalEmulators.jl
"""
from typing import Dict, Any, List, Tuple, Type, Optional
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from .core import FlaxEmulator
from .utils import validate_nn_dict_structure, validate_trained_weights


class MLP(nn.Module):
    """Multi-layer perceptron matching the structure from jaxeffort."""
    features: List[int]
    activations: List[str]

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features[:-1]):
            if self.activations[i] == "tanh":
                x = nn.tanh(nn.Dense(feat)(x))
            elif self.activations[i] == "relu":
                x = nn.relu(nn.Dense(feat)(x))
            else:
                raise ValueError(f"Unknown activation function: {self.activations[i]}")
        x = nn.Dense(self.features[-1])(x)
        return x


def _get_in_out_arrays(nn_dict: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get input and output dimensions for each layer.
    Matches Julia's _get_in_out_arrays function.
    """
    n = nn_dict["n_hidden_layers"]
    in_array = np.zeros(n + 1, dtype=int)
    out_array = np.zeros(n + 1, dtype=int)

    in_array[0] = nn_dict["n_input_features"]
    out_array[-1] = nn_dict["n_output_features"]

    for i in range(n):
        layer_key = f"layer_{i+1}"
        in_array[i+1] = nn_dict["layers"][layer_key]["n_neurons"]
        out_array[i] = nn_dict["layers"][layer_key]["n_neurons"]

    return in_array, out_array


def _get_i_array(in_array: np.ndarray, out_array: np.ndarray) -> np.ndarray:
    """
    Get starting indices for weights of each layer.
    Matches Julia's _get_i_array function.
    """
    i_array = np.empty_like(in_array)
    i_array[0] = 0

    for i in range(1, len(i_array)):
        i_array[i] = i_array[i-1] + in_array[i-1] * out_array[i-1] + out_array[i-1]

    return i_array


def _get_weight_bias(i: int, n_in: int, n_out: int,
                     weight_bias: np.ndarray) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Extract weight and bias for a single layer.

    Uses row-major (C) order for compatibility with Python-trained models.
    This matches the original jaxcapse implementation.

    Note: If you need Julia compatibility (column-major), use order='F' in reshape.
    """
    weight = np.reshape(weight_bias[i:i+n_out*n_in], (n_in, n_out))  # Row-major (C order)
    bias = weight_bias[i+n_out*n_in:i+n_out*n_in+n_out]

    return {'kernel': weight, 'bias': bias}, i + n_out*n_in + n_out


def _get_flax_params(nn_dict: Dict[str, Any], weights: np.ndarray) -> Dict[str, Any]:
    """
    Convert weights to Flax parameter format.
    Matches Julia's _get_lux_params function.
    """
    in_array, out_array = _get_in_out_arrays(nn_dict)
    i_array = _get_i_array(in_array, out_array)

    params = {'params': {}}

    for j in range(nn_dict["n_hidden_layers"] + 1):
        layer_params, _ = _get_weight_bias(
            i_array[j], in_array[j], out_array[j], weights
        )
        params['params'][f"Dense_{j}"] = layer_params

    return params


def _get_flax_states(nn_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get initial states for Flax model.
    Matches Julia's _get_lux_states function.
    For standard feedforward networks, this is typically empty.
    """
    # Flax doesn't typically use states for standard Dense layers
    return {}


def _get_activation_functions(nn_dict: Dict[str, Any]) -> List[str]:
    """Extract activation functions for each hidden layer."""
    activations = []
    for j in range(nn_dict["n_hidden_layers"]):
        layer_key = f"layer_{j+1}"
        activations.append(nn_dict["layers"][layer_key]["activation_function"])
    return activations


def _get_layer_sizes(nn_dict: Dict[str, Any]) -> List[int]:
    """Extract layer sizes (number of neurons)."""
    sizes = []
    for j in range(nn_dict["n_hidden_layers"]):
        layer_key = f"layer_{j+1}"
        sizes.append(nn_dict["layers"][layer_key]["n_neurons"])
    sizes.append(nn_dict["n_output_features"])
    return sizes


def _get_nn_flax(nn_dict: Dict[str, Any]) -> nn.Module:
    """
    Create Flax neural network model from dictionary specification.
    Matches Julia's _get_nn_lux function.
    """
    activations = _get_activation_functions(nn_dict)
    features = _get_layer_sizes(nn_dict)

    # Validate activation functions before creating the model
    for i, activation in enumerate(activations):
        if activation not in ["tanh", "relu"]:
            raise ValueError(f"Unknown activation function: {activation}")

    return MLP(features=features, activations=activations)


def _get_emulator_description_dict(nn_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract emulator description from nn_dict.
    Matches Julia's _get_emulator_description_dict function.
    """
    return nn_dict.get("emulator_description", {})


def _init_flaxemulator(nn_dict: Dict[str, Any], weight: np.ndarray) -> FlaxEmulator:
    """
    Initialize a FlaxEmulator from nn_dict and weights.
    Matches Julia's _init_luxemulator function.
    """
    params = _get_flax_params(nn_dict, weight)
    states = _get_flax_states(nn_dict)
    model = _get_nn_flax(nn_dict)
    description = {"emulator_description": _get_emulator_description_dict(nn_dict)}

    return FlaxEmulator(
        model=model,
        parameters=params,
        states=states,
        description=description
    )


def init_emulator(nn_dict: Dict[str, Any],
                  weight: np.ndarray,
                  emulator_type: Type[FlaxEmulator] = FlaxEmulator,
                  validate: bool = True,
                  validate_weights: Optional[bool] = None) -> FlaxEmulator:
    """
    Initialize an emulator from neural network dictionary and weights.

    Args:
        nn_dict: Neural network specification dictionary
        weight: Flattened weight array
        emulator_type: Type of emulator (currently only FlaxEmulator)
        validate: Whether to validate nn_dict structure
        validate_weights: Whether to validate weight dimensions

    Returns:
        Initialized FlaxEmulator instance
    """
    if validate_weights is None:
        validate_weights = validate

    if validate:
        validate_nn_dict_structure(nn_dict)

    if validate_weights:
        validate_trained_weights(weight, nn_dict)

    if emulator_type != FlaxEmulator:
        raise ValueError(f"Only FlaxEmulator is supported, got {emulator_type}")

    return _init_flaxemulator(nn_dict, weight)
