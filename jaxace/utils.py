"""
Utility functions matching AbstractCosmologicalEmulators.jl
"""
from typing import Dict, Any, Union, NamedTuple
import numpy as np
import jax
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

    validate_normalization_ranges(nn_dict)
    validate_architecture_numerical_stability(nn_dict)


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

    # Check for NaNs and Infs
    is_finite = np.isfinite(weights)
    if not np.all(is_finite):
        nan_count = np.sum(np.isnan(weights))
        inf_count = np.sum(np.isinf(weights))
        raise ValueError(
            f"Invalid trained weights detected: NaN values: {nan_count}, Inf values: {inf_count}, "
            f"Total invalid: {nan_count + inf_count} out of {len(weights)}. "
            "This indicates the emulator was not properly trained or the weights are corrupted."
        )

    # Check for excessively large weights
    max_weight = np.max(np.abs(weights))
    if max_weight > 1e6:
        import warnings
        warnings.warn(
            f"Large weight magnitudes detected (max absolute value: {max_weight}). "
            "This may indicate training instability, poor normalization, or gradient explosion."
        )

    # Check for all-zero or very small weights
    if np.all(np.abs(weights) < 1e-10):
        import warnings
        warnings.warn(
            "All weights are very small (< 1e-10). "
            "This may indicate the emulator was not properly trained."
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
    elif "parameters" in description:
        print(f"The parameters the model has been trained are, in the following order: {description['parameters']}.")
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
    if "miscellanea" in description:
        print(description["miscellanea"])

    for key, value in description.items():
        if key not in ["author", "author_email", "emulator_type", "description",
                       "input_parameters", "parameters", "output_parameters", "version", "miscellanea"]:
            print(f"{key}: {value}")


def validate_normalization_ranges(nn_dict: Dict[str, Any]) -> None:
    if "emulator_description" in nn_dict:
        desc = nn_dict["emulator_description"]
        range_keys = ["input_ranges", "minmax", "parameter_ranges", "normalization_ranges"]
        minmax_data = None

        for key in range_keys:
            if key in desc:
                minmax_data = desc[key]
                break

        if minmax_data is not None:
            validate_minmax_data(minmax_data)


def validate_minmax_data(minmax_data: Any) -> None:
    ranges = convert_minmax_format(minmax_data)

    range_widths = ranges[:, 1] - ranges[:, 0]
    degenerate_indices = np.where(np.abs(range_widths) < 1e-15)[0]

    if len(degenerate_indices) > 0:
        raise ValueError(
            f"Degenerate normalization ranges detected at parameter indices: {degenerate_indices.tolist()}. "
            f"Range widths: {range_widths[degenerate_indices].tolist()}. "
            "This will cause division by zero in maximin normalization. "
            "Please ensure min ≠ max for all parameters."
        )

    validate_cosmological_ranges(ranges)


def convert_minmax_format(minmax_data: Any) -> np.ndarray:
    if isinstance(minmax_data, (list, tuple)) and len(minmax_data) > 0:
        if isinstance(minmax_data[0], (list, tuple)):
            ranges = np.zeros((len(minmax_data), 2))
            for i, range_pair in enumerate(minmax_data):
                if len(range_pair) != 2:
                    raise ValueError("Each range must have exactly 2 elements [min, max]")
                ranges[i, 0] = float(range_pair[0])
                ranges[i, 1] = float(range_pair[1])
            return ranges
        else:
            raise ValueError("1D list minmax format not recognized. Expected List[List[float]]")
    elif isinstance(minmax_data, dict):
        if "min" in minmax_data and "max" in minmax_data:
            min_vals = minmax_data["min"]
            max_vals = minmax_data["max"]
            if len(min_vals) != len(max_vals):
                raise ValueError("min and max arrays must have same length")
            return np.column_stack((np.array(min_vals, dtype=float), np.array(max_vals, dtype=float)))
        else:
            raise ValueError("Dictionary minmax format must have 'min' and 'max' keys")
    elif isinstance(minmax_data, (np.ndarray, jnp.ndarray)):
        arr = np.array(minmax_data)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("Matrix minmax data must have exactly 2 columns [min, max]")
        return arr
    else:
        raise ValueError(f"Unsupported minmax data format: {type(minmax_data)}")


def validate_cosmological_ranges(ranges: np.ndarray) -> None:
    n_params = ranges.shape[0]
    for i in range(n_params):
        min_val, max_val = ranges[i, 0], ranges[i, 1]
        if min_val >= max_val:
            raise ValueError(f"Invalid range for parameter {i}: min ({min_val}) >= max ({max_val})")


def validate_architecture_numerical_stability(nn_dict: Dict[str, Any]) -> None:
    n_input = nn_dict["n_input_features"]
    n_output = nn_dict["n_output_features"]
    n_hidden = nn_dict["n_hidden_layers"]

    layer_sizes = [n_input]
    for i in range(1, n_hidden + 1):
        layer_sizes.append(nn_dict["layers"][f"layer_{i}"]["n_neurons"])
    layer_sizes.append(n_output)

    import warnings
    for i in range(1, len(layer_sizes)):
        ratio = layer_sizes[i] / layer_sizes[i-1]
        if ratio > 100:
            warnings.warn(
                f"Large layer size expansion detected: Layer {i-1} ({layer_sizes[i-1]}) → Layer {i} ({layer_sizes[i]}) "
                f"(ratio: {ratio:.2f}). This may cause increased memory usage, potential overfitting, or training instability."
            )
        elif ratio < 0.01:
            warnings.warn(
                f"Severe layer size reduction detected: Layer {i-1} ({layer_sizes[i-1]}) → Layer {i} ({layer_sizes[i]}) "
                f"(ratio: {ratio:.4f}). This may cause information bottlenecks, underfitting, or loss of representational capacity."
            )

    if n_hidden > 20:
        warnings.warn(
            f"Very deep network detected ({n_hidden} hidden layers). "
            "Consider using residual connections, batch normalization, or gradient clipping."
        )

    for i in range(1, n_hidden + 1):
        activation = nn_dict["layers"][f"layer_{i}"]["activation_function"]
        if activation == "tanh" and n_hidden > 10:
            warnings.warn(
                f"Deep network ({n_hidden} layers) using tanh activation may suffer from vanishing gradients. "
                "Consider using ReLU or other activations for deep networks."
            )
            break


# =============================================================================
# Akima Spline Interpolation
# =============================================================================

def _akima_slopes(u, t):
    """
    Compute Akima slopes with boundary extrapolation.

    This is a direct translation of AbstractCosmologicalEmulators.jl's _akima_slopes
    function for both 1D and 2D arrays.

    Args:
        u: Ordinates (function values) at data nodes
           - 1D: shape (n,)
           - 2D: shape (n, n_cols)
        t: Abscissae (x-coordinates) at data nodes, shape (n,)

    Returns:
        Slopes array with extrapolated boundary slopes
        - 1D input: shape (n+3,)
        - 2D input: shape (n+3, n_cols)
    """
    # Check if u is 1D or 2D
    is_1d = jnp.ndim(u) == 1

    if is_1d:
        n = len(u)
        dt = jnp.diff(t)

        # Create slopes array with proper dtype promotion
        dtype = jnp.result_type(u.dtype, t.dtype)
        m = jnp.zeros(n + 3, dtype=dtype)

        # Interior slopes
        m = m.at[2:n+1].set(jnp.diff(u) / dt)

        # Boundary extrapolation
        m = m.at[1].set(2 * m[2] - m[3])
        m = m.at[0].set(2 * m[1] - m[2])
        m = m.at[n+1].set(2 * m[n] - m[n-1])
        m = m.at[n+2].set(2 * m[n+1] - m[n])

        return m
    else:
        # 2D case: u has shape (n, n_cols)
        n, n_cols = u.shape
        dt = jnp.diff(t)

        # Create slopes array with proper dtype promotion
        dtype = jnp.result_type(u.dtype, t.dtype)
        m = jnp.zeros((n + 3, n_cols), dtype=dtype)

        # Interior slopes for all columns at once
        # diff(u) gives shape (n-1, n_cols), dt has shape (n-1,)
        # Need to broadcast: diff(u) / dt[:, None]
        m = m.at[2:n+1, :].set(jnp.diff(u, axis=0) / dt[:, jnp.newaxis])

        # Boundary extrapolation for all columns
        m = m.at[1, :].set(2 * m[2, :] - m[3, :])
        m = m.at[0, :].set(2 * m[1, :] - m[2, :])
        m = m.at[n+1, :].set(2 * m[n, :] - m[n-1, :])
        m = m.at[n+2, :].set(2 * m[n+1, :] - m[n, :])

        return m


def _akima_coefficients(t, m):
    """
    Compute Akima cubic polynomial coefficients.

    This is a direct translation of AbstractCosmologicalEmulators.jl's _akima_coefficients
    function for both 1D and 2D arrays.

    Args:
        t: Abscissae (x-coordinates), shape (n,)
        m: Slopes from _akima_slopes
           - 1D: shape (n+3,)
           - 2D: shape (n+3, n_cols)

    Returns:
        Tuple (b, c, d) where:
        - 1D input:
            b: shape (n,)
            c: shape (n-1,)
            d: shape (n-1,)
        - 2D input:
            b: shape (n, n_cols)
            c: shape (n-1, n_cols)
            d: shape (n-1, n_cols)
    """
    # Check if m is 1D or 2D
    is_1d = jnp.ndim(m) == 1

    if is_1d:
        n = len(t)
        dt = jnp.diff(t)

        # Initial b values (average of surrounding slopes)
        b = (m[3:n+3] + m[0:n]) / 2

        # Compute weighted slopes where appropriate
        dm = jnp.abs(jnp.diff(m))
        f1 = dm[2:n+2]
        f2 = dm[0:n]
        f12 = f1 + f2

        # Epsilon threshold for numerical stability
        eps_akima = jnp.finfo(f12.dtype).eps * 100

        # Weighted average where slopes vary significantly
        b_weighted = (f1 * m[1:n+1] + f2 * m[2:n+2]) / f12

        # Use weighted average where f12 > eps_akima, otherwise keep simple average
        b = jnp.where(f12 > eps_akima, b_weighted, b)

        # Compute cubic coefficients c and d
        c = (3 * m[2:n+1] - 2 * b[0:n-1] - b[1:n]) / dt
        d = (b[0:n-1] + b[1:n] - 2 * m[2:n+1]) / dt ** 2

        return b, c, d
    else:
        # 2D case: m has shape (n+3, n_cols)
        n = len(t)
        n_cols = m.shape[1]
        dt = jnp.diff(t)

        # Initial b values (average of surrounding slopes) for all columns
        # Shape: (n, n_cols)
        b = (m[3:n+3, :] + m[0:n, :]) / 2

        # Compute weighted slopes where appropriate
        # dm has shape (n+2, n_cols)
        dm = jnp.abs(jnp.diff(m, axis=0))
        # f1, f2 have shape (n, n_cols)
        f1 = dm[2:n+2, :]
        f2 = dm[0:n, :]
        f12 = f1 + f2

        # Epsilon threshold for numerical stability
        eps_akima = jnp.finfo(f12.dtype).eps * 100

        # Weighted average where slopes vary significantly
        # All shapes: (n, n_cols)
        b_weighted = (f1 * m[1:n+1, :] + f2 * m[2:n+2, :]) / f12

        # Use weighted average where f12 > eps_akima
        b = jnp.where(f12 > eps_akima, b_weighted, b)

        # Compute cubic coefficients c and d
        # Need to broadcast dt (shape n-1) with b slices (shape n-1, n_cols)
        # c and d will have shape (n-1, n_cols)
        c = (3 * m[2:n+1, :] - 2 * b[0:n-1, :] - b[1:n, :]) / dt[:, jnp.newaxis]
        d = (b[0:n-1, :] + b[1:n, :] - 2 * m[2:n+1, :]) / (dt ** 2)[:, jnp.newaxis]

        return b, c, d


def _akima_find_interval(t, tq):
    """
    Find the interval index for a query point.

    This is a direct translation of AbstractCosmologicalEmulators.jl's _akima_find_interval
    function.

    Args:
        t: Abscissae array, shape (n,)
        tq: Query point (scalar)

    Returns:
        Interval index in range [0, n-2]
    """
    n = len(t)

    # Julia's searchsortedlast(t, tq) finds the last index i where t[i] <= tq
    # JAX's searchsorted with side='right' finds the first index where t[i] > tq
    # So searchsorted(t, tq, side='right') - 1 gives us the equivalent of searchsortedlast
    idx = jnp.searchsorted(t, tq, side='right') - 1

    # Clamp to valid interval range [0, n-2]
    # Julia returns 1 for tq <= t[1] and n-1 for tq >= t[end]
    # Python returns 0 for tq <= t[0] and n-2 for tq >= t[-1]
    idx = jnp.clip(idx, 0, n - 2)

    return idx


def _akima_eval(u, t, b, c, d, tq):
    """
    Evaluate Akima spline at query points.

    This is a direct translation of AbstractCosmologicalEmulators.jl's _akima_eval
    function. Handles both scalar and array inputs, 1D and 2D data.

    Args:
        u: Ordinates at data nodes
           - 1D: shape (n,)
           - 2D: shape (n, n_cols)
        t: Abscissae at data nodes, shape (n,)
        b, c, d: Polynomial coefficients from _akima_coefficients
           - 1D: b shape (n,), c and d shape (n-1,)
           - 2D: b shape (n, n_cols), c and d shape (n-1, n_cols)
        tq: Query point(s), scalar or array

    Returns:
        Interpolated value(s) at tq
        - 1D input + scalar tq: scalar
        - 1D input + array tq: array of shape (len(tq),)
        - 2D input + scalar tq: array of shape (n_cols,)
        - 2D input + array tq: array of shape (len(tq), n_cols)
    """
    # Check if u is 1D or 2D
    is_1d = jnp.ndim(u) == 1

    # Check if tq is scalar
    is_scalar_tq = jnp.ndim(tq) == 0
    tq_arr = jnp.atleast_1d(tq)

    n = len(t)

    if is_1d:
        # 1D case
        # Vectorized interval finding
        idx = jnp.searchsorted(t, tq_arr, side='right') - 1
        idx = jnp.clip(idx, 0, n - 2)

        # Evaluate polynomial using Horner's method
        wj = tq_arr - t[idx]
        result = ((d[idx] * wj + c[idx]) * wj + b[idx]) * wj + u[idx]

        # Return scalar if input was scalar
        return result[0] if is_scalar_tq else result
    else:
        # 2D case: u has shape (n, n_cols)
        n_cols = u.shape[1]
        n_query = len(tq_arr)

        # Vectorized interval finding (same for all columns)
        idx = jnp.searchsorted(t, tq_arr, side='right') - 1
        idx = jnp.clip(idx, 0, n - 2)

        # Evaluate polynomial for all columns at once
        # wj has shape (n_query,)
        wj = tq_arr - t[idx]

        # Broadcast evaluation: result will have shape (n_query, n_cols)
        # idx has shape (n_query,), we need to index into (n, n_cols) arrays
        # Use advanced indexing: u[idx, :] gives shape (n_query, n_cols)
        result = (
            ((d[idx, :] * wj[:, jnp.newaxis] + c[idx, :]) * wj[:, jnp.newaxis] + b[idx, :])
            * wj[:, jnp.newaxis] + u[idx, :]
        )

        # Return appropriate shape
        if is_scalar_tq:
            # Return shape (n_cols,)
            return result[0, :]
        else:
            # Return shape (n_query, n_cols)
            return result


def akima_interpolation(u, t, t_new):
    """
    Akima spline interpolation for 1D or 2D data.

    This is a direct translation of AbstractCosmologicalEmulators.jl's akima_interpolation
    function. Evaluates the Akima spline that interpolates the data points (t_i, u_i)
    at new abscissae t_new.

    The Akima spline is a piecewise cubic polynomial that uses weighted averaging of
    local slopes to determine derivatives at each node. This dampens oscillations
    without explicit shape constraints. The spline is C¹ continuous but generally
    not C².

    This implementation is fully compatible with JAX's jit and automatic differentiation.

    Args:
        u: Ordinates (function values) at data nodes.
           - 1D case: shape (n,)
           - 2D case: shape (n, n_cols) where each column is interpolated independently
        t: Strictly increasing abscissae (x-coordinates), shape (n,)
        t_new: Query point(s) where spline is evaluated, scalar or array

    Returns:
        Interpolated value(s) at t_new.
        - 1D input: Scalar if t_new is scalar, array if t_new is array
        - 2D input: Matrix of shape (len(t_new), n_cols)

    Example (1D):
        >>> import jax.numpy as jnp
        >>> t = jnp.linspace(0, 1, 10)
        >>> u = jnp.sin(2 * jnp.pi * t)
        >>> t_new = jnp.linspace(0, 1, 50)
        >>> u_new = akima_interpolation(u, t, t_new)

    Example (2D - multiple columns):
        >>> # Interpolate Jacobian with 11 parameter columns
        >>> k_in = jnp.linspace(0.01, 0.3, 50)
        >>> jacobian = jnp.randn(50, 11)  # 11 parameters
        >>> k_out = jnp.linspace(0.01, 0.3, 100)
        >>> result = akima_interpolation(jacobian, k_in, k_out)  # (100, 11)

        # Works with jit
        >>> akima_jit = jax.jit(akima_interpolation)
        >>> u_new = akima_jit(u, t, t_new)

        # Works with autodiff
        >>> grad_fn = jax.grad(lambda u: jnp.sum(akima_interpolation(u, t, t_new)))
        >>> grad_u = grad_fn(u)
    """
    m = _akima_slopes(u, t)
    b, c, d = _akima_coefficients(t, m)
    return _akima_eval(u, t, b, c, d, t_new)


from typing import NamedTuple

class AkimaSpline(NamedTuple):
    u: jnp.ndarray
    t: jnp.ndarray
    b: jnp.ndarray
    c: jnp.ndarray
    d: jnp.ndarray

    def __call__(self, t_new):
        """Evaluate the Akima spline at new points."""
        return _akima_eval(self.u, self.t, self.b, self.c, self.d, t_new)


def prepare_akima_spline(u: jnp.ndarray, t: jnp.ndarray) -> AkimaSpline:
    """
    Precompute the Akima spline coefficients for repeated evaluation.
    This structure acts as a valid JAX PyTree for JIT and grad.
    """
    m = _akima_slopes(u, t)
    b, c, d = _akima_coefficients(t, m)
    return AkimaSpline(u=u, t=t, b=b, c=c, d=d)


def evaluate_akima_spline(spline: AkimaSpline, t_new: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate a precomputed Akima spline at new points.
    """
    return _akima_eval(spline.u, spline.t, spline.b, spline.c, spline.d, t_new)


# =============================================================================
# Cubic Spline Interpolation
# =============================================================================

def _cubic_spline_coefficients(u, t):
    """
    Compute Natural Cubic Spline coefficients.

    This matches the `_cubic_spline_coefficients` from AbstractCosmologicalEmulators.jl.
    It constructs and solves a tridiagonal system for the second derivatives `z`.

    Args:
        u: Ordinates at data nodes
           - 1D: shape (n,)
           - 2D: shape (n, n_cols)
        t: Abscissae at data nodes, shape (n,)

    Returns:
        Tuple (h, z) where:
        - h is the interval widths, shape (n+1,) with 0 padding
        - z is the second derivatives
          - 1D: shape (n,)
          - 2D: shape (n, n_cols)
    """
    n = len(t)
    dt = jnp.diff(t)

    # We need an array h (intervals) for evaluation, padded with 0 at boundaries
    dtype = jnp.result_type(u.dtype, t.dtype)
    h = jnp.zeros(n + 1, dtype=dtype)
    h = h.at[1:n].set(dt)

    # Construct full Tridiagonal matrix `A` for solver since JAX solve_banded is removed
    A = jnp.zeros((n, n), dtype=dtype)

    # Fill main diagonal (Natural Spline boundaries: z[0]=0, z[N-1]=0 -> A[0,0]=A[N-1,N-1]=1)
    i_idx = jnp.arange(n)
    A_diag = jnp.zeros(n, dtype=dtype)
    # The intermediate nodes are 2*(h[i] + h[i+1])
    A_diag = A_diag.at[1:n-1].set(2 * (h[1:n-1] + h[2:n]))
    A_diag = A_diag.at[0].set(1.0)
    A_diag = A_diag.at[-1].set(1.0)

    A = A.at[i_idx, i_idx].set(A_diag)

    # Fill superdiagonal and subdiagonal
    # We only want off-diagonals for the interior nodes (1 to N-2)
    # So A[i, i+1] = h[i+1] for i in 1..N-2 (0-indexed: 1 to n-2)
    i_interior = jnp.arange(1, n - 1)
    # Superdiagonal values: A[i, i+1] = h[i+1] (which is dt[i])
    A = A.at[i_interior, i_interior + 1].set(dt[1:])
    # Subdiagonal values: A[i, i-1] = h[i] (which is dt[i-1])
    A = A.at[i_interior, i_interior - 1].set(dt[:-1])

    is_1d = jnp.ndim(u) == 1

    if is_1d:
        # RHS d
        d = jnp.zeros(n, dtype=dtype)

        # d[i] = 6 * (u[i+1] - u[i]) / h[i+1] - 6 * (u[i] - u[i-1]) / h[i] for i in 1..n-2 (0-indexed: 1..n-2)
        d_inner = 6 * jnp.diff(u)[1:] / h[2:n] - 6 * jnp.diff(u)[:-1] / h[1:n-1]
        d = d.at[1:n-1].set(d_inner)

        # Solve Tridiagonal system
        z = jax.scipy.linalg.solve(A, d)

        return h, z
    else:
        # 2D case
        n_cols = u.shape[1]

        # RHS d (Matrix)
        d = jnp.zeros((n, n_cols), dtype=dtype)

        # Broadcasting math over columns
        h_next = h[2:n][:, jnp.newaxis]
        h_prev = h[1:n-1][:, jnp.newaxis]

        d_inner = 6 * jnp.diff(u, axis=0)[1:, :] / h_next - 6 * jnp.diff(u, axis=0)[:-1, :] / h_prev
        d = d.at[1:n-1, :].set(d_inner)

        # Solve Tridiagonal system for multiple RHS
        z = jax.scipy.linalg.solve(A, d)

        return h, z

def _cubic_spline_eval(u, t, h, z, tq):
    """
    Evaluate Natural Cubic Spline.

    Matches AbstractCosmologicalEmulators.jl `_cubic_spline_eval`.
    """
    is_1d = jnp.ndim(u) == 1
    is_scalar_tq = jnp.ndim(tq) == 0
    tq_arr = jnp.atleast_1d(tq)

    n = len(t)

    # We use akima's interval finder logic
    idx = jnp.searchsorted(t, tq_arr, side='right') - 1

    # We create masks for extrapolation
    mask_left = tq_arr < t[0]
    mask_right = tq_arr > t[-1]
    mask_inside = ~(mask_left | mask_right)

    idx = jnp.clip(idx, 0, n - 2)

    if is_1d:
        dt = tq_arr - t[idx]
        dt_next = t[idx+1] - tq_arr
        h_i = h[idx+1]

        term1 = (z[idx] * dt_next**3 + z[idx+1] * dt**3) / (6 * h_i)
        term2 = (u[idx+1] / h_i - z[idx+1] * h_i / 6) * dt
        term3 = (u[idx] / h_i - z[idx] * h_i / 6) * dt_next

        val_inside = term1 + term2 + term3

        return val_inside[0] if is_scalar_tq else val_inside
    else:
        n_cols = u.shape[1]

        dt = tq_arr - t[idx]
        dt_next = t[idx+1] - tq_arr
        h_i = h[idx+1]

        # Broadcast terms
        dt = dt[:, jnp.newaxis]
        dt_next = dt_next[:, jnp.newaxis]
        h_i = h_i[:, jnp.newaxis]

        term1 = (z[idx, :] * dt_next**3 + z[idx+1, :] * dt**3) / (6 * h_i)
        term2 = (u[idx+1, :] / h_i - z[idx+1, :] * h_i / 6) * dt
        term3 = (u[idx, :] / h_i - z[idx, :] * h_i / 6) * dt_next

        val_inside = term1 + term2 + term3

        return val_inside[0, :] if is_scalar_tq else val_inside

def cubic_spline_interpolation(u, t, t_new):
    """
    Natural Cubic Spline interpolation for 1D or 2D data.

    This is a direct translation of AbstractCosmologicalEmulators.jl's cubic_spline_interpolation
    function. Evaluates the Natural Cubic Spline that interpolates the data points (t_i, u_i)
    at new abscissae t_new.

    This implementation is fully compatible with JAX's jit and automatic differentiation.

    Args:
        u: Ordinates (function values) at data nodes.
           - 1D case: shape (n,)
           - 2D case: shape (n, n_cols) where each column is interpolated independently
        t: Strictly increasing abscissae (x-coordinates), shape (n,)
        t_new: Query point(s) where spline is evaluated, scalar or array

    Returns:
        Interpolated value(s) at t_new.
        - 1D input: Scalar if t_new is scalar, array if t_new is array
        - 2D input: Matrix of shape (len(t_new), n_cols)
    """
    h, z = _cubic_spline_coefficients(u, t)
    return _cubic_spline_eval(u, t, h, z, t_new)
