"""
Utility functions matching AbstractCosmologicalEmulators.jl
"""
from typing import Dict, Any, Union
import numpy as np
import jax.numpy as jnp


def maximin(input_data: Union[np.ndarray, jnp.ndarray], 
            minmax: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
    """
    Normalize input data using min-max scaling.
    Matches Julia's maximin function.
    
    Args:
        input_data: Input array to normalize (shape: (n_features,) or (n_features, n_samples))
        minmax: Array of shape (n_features, 2) where column 0 is min, column 1 is max
        
    Returns:
        Normalized array in range [0, 1]
    """
    # Handle both 1D and 2D cases
    if input_data.ndim == 1:
        return (input_data - minmax[:, 0]) / (minmax[:, 1] - minmax[:, 0])
    else:
        # For 2D arrays, broadcast correctly
        return (input_data - minmax[:, 0:1]) / (minmax[:, 1:2] - minmax[:, 0:1])


def inv_maximin(output_data: Union[np.ndarray, jnp.ndarray], 
                minmax: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
    """
    Denormalize output data from min-max scaling.
    Matches Julia's inv_maximin function.
    
    Args:
        output_data: Normalized array (shape: (n_features,) or (n_features, n_samples))
        minmax: Array of shape (n_features, 2) where column 0 is min, column 1 is max
        
    Returns:
        Denormalized array
    """
    # Handle both 1D and 2D cases
    if output_data.ndim == 1:
        return output_data * (minmax[:, 1] - minmax[:, 0]) + minmax[:, 0]
    else:
        # For 2D arrays, broadcast correctly
        return output_data * (minmax[:, 1:2] - minmax[:, 0:1]) + minmax[:, 0:1]


def validate_nn_dict_structure(nn_dict: Dict[str, Any]) -> None:
    """
    Validate the structure of the neural network dictionary.
    Matches Julia's validate_nn_dict_structure function.
    
    Args:
        nn_dict: Neural network specification dictionary
        
    Raises:
        ValueError: If the dictionary structure is invalid
    """
    required_keys = ["n_input_features", "n_output_features", "n_hidden_layers", "layers"]
    
    for key in required_keys:
        if key not in nn_dict:
            raise ValueError(f"Missing required key: {key}")
    
    n_hidden = nn_dict["n_hidden_layers"]
    
    if not isinstance(n_hidden, int) or n_hidden < 0:
        raise ValueError(f"n_hidden_layers must be a non-negative integer, got {n_hidden}")
    
    if "layers" not in nn_dict or not isinstance(nn_dict["layers"], dict):
        raise ValueError("Missing or invalid 'layers' dictionary")
    
    for i in range(1, n_hidden + 1):
        layer_key = f"layer_{i}"
        if layer_key not in nn_dict["layers"]:
            raise ValueError(f"Missing layer definition: {layer_key}")
        
        layer = nn_dict["layers"][layer_key]
        validate_layer_structure(layer, layer_key)


def validate_layer_structure(layer: Dict[str, Any], layer_name: str) -> None:
    """
    Validate the structure of a single layer dictionary.
    Matches Julia's validate_layer_structure function.
    
    Args:
        layer: Layer specification dictionary
        layer_name: Name of the layer for error messages
        
    Raises:
        ValueError: If the layer structure is invalid
    """
    required_keys = ["n_neurons", "activation_function"]
    
    for key in required_keys:
        if key not in layer:
            raise ValueError(f"Missing required key '{key}' in {layer_name}")
    
    if not isinstance(layer["n_neurons"], int) or layer["n_neurons"] <= 0:
        raise ValueError(f"n_neurons must be a positive integer in {layer_name}")
    
    validate_activation_function(layer["activation_function"], layer_name)


def validate_activation_function(activation: str, context: str) -> None:
    """
    Validate activation function name.
    
    Args:
        activation: Activation function name
        context: Context for error message
        
    Raises:
        ValueError: If activation function is not supported
    """
    supported = ["tanh", "relu", "identity"]
    if activation not in supported:
        raise ValueError(
            f"Unsupported activation function '{activation}' in {context}. "
            f"Supported: {supported}"
        )


def validate_parameter_ranges(params: Dict[str, Any]) -> None:
    """
    Validate parameter ranges for emulator inputs.
    Matches Julia's validate_parameter_ranges function.
    
    Args:
        params: Parameter dictionary to validate
        
    Raises:
        ValueError: If parameters are out of valid ranges
    """
    # This would contain domain-specific validation
    # For now, just check that parameters are numeric
    for key, value in params.items():
        if not isinstance(value, (int, float, np.ndarray, jnp.ndarray)):
            raise ValueError(f"Parameter {key} must be numeric, got {type(value)}")


def validate_trained_weights(weights: np.ndarray, nn_dict: Dict[str, Any]) -> None:
    """
    Validate that weight dimensions match the neural network specification.
    
    Args:
        weights: Flattened weight array
        nn_dict: Neural network specification dictionary
        
    Raises:
        ValueError: If weight dimensions don't match
    """
    # Calculate expected number of weights
    expected_size = calculate_weight_size(nn_dict)
    
    if len(weights) != expected_size:
        raise ValueError(
            f"Weight array size mismatch. Expected {expected_size}, got {len(weights)}"
        )


def calculate_weight_size(nn_dict: Dict[str, Any]) -> int:
    """
    Calculate the expected size of the flattened weight array.
    
    Args:
        nn_dict: Neural network specification dictionary
        
    Returns:
        Expected number of weight parameters
    """
    n_hidden = nn_dict["n_hidden_layers"]
    total_size = 0
    
    # Input layer to first hidden layer
    n_in = nn_dict["n_input_features"]
    n_out = nn_dict["layers"]["layer_1"]["n_neurons"]
    total_size += n_in * n_out + n_out  # weights + bias
    
    # Hidden layers
    for i in range(1, n_hidden):
        n_in = nn_dict["layers"][f"layer_{i}"]["n_neurons"]
        n_out = nn_dict["layers"][f"layer_{i+1}"]["n_neurons"]
        total_size += n_in * n_out + n_out
    
    # Last hidden layer to output
    n_in = nn_dict["layers"][f"layer_{n_hidden}"]["n_neurons"]
    n_out = nn_dict["n_output_features"]
    total_size += n_in * n_out + n_out
    
    return total_size


def safe_dict_access(dictionary: Dict[str, Any], *keys, default=None) -> Any:
    """
    Safely access nested dictionary values.
    Matches Julia's safe_dict_access function.
    
    Args:
        dictionary: Dictionary to access
        *keys: Sequence of keys to traverse
        default: Default value if key not found
        
    Returns:
        Value at the specified path or default
    """
    current = dictionary
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def get_emulator_description(description: Dict[str, Any]) -> None:
    """
    Print emulator description information.
    Matches Julia's get_emulator_description function.
    
    Args:
        description: Emulator description dictionary
    """
    # Print author information
    if "author" in description:
        print(f"Author: {description['author']}")
    
    if "author_email" in description:
        print(f"Email: {description['author_email']}")
    
    # Print emulator details
    if "emulator_type" in description:
        print(f"Emulator Type: {description['emulator_type']}")
    
    if "description" in description:
        print(f"Description: {description['description']}")
    
    # Print input parameters
    if "input_parameters" in description:
        print(f"Input Parameters: {description['input_parameters']}")
    else:
        print("We do not know which parameters are the inputs of the emulator")
    
    # Print output parameters  
    if "output_parameters" in description:
        print(f"Output Parameters: {description['output_parameters']}")
    else:
        print("We do not know which parameters are the outputs of the emulator")
    
    # Print version information
    if "version" in description:
        print(f"Version: {description['version']}")
    
    # Print any additional metadata
    for key, value in description.items():
        if key not in ["author", "author_email", "emulator_type", "description", 
                       "input_parameters", "output_parameters", "version"]:
            print(f"{key}: {value}")