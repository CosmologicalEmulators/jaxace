"""
Test core functionality of jaxace neural network emulators.

These tests match AbstractCosmologicalEmulators.jl/test/test_core_functionality.jl
"""

import json
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from jaxace import (
    FlaxEmulator,
    init_emulator,
    run_emulator,
    get_emulator_description,
    maximin,
    inv_maximin,
    validate_nn_dict_structure,
    validate_parameter_ranges,
    validate_layer_structure,
    safe_dict_access
)

# Configure JAX for 64-bit precision
jax.config.update('jax_enable_x64', True)


class TestNormalization:
    """Test maximin and inv_maximin normalization functions."""
    
    def test_maximin_scalar(self):
        """Test normalization with scalar input."""
        m = 100
        x = np.random.rand(m)
        minmax = np.column_stack((np.zeros(m), np.ones(m)))
        
        # Store original
        X = x.copy()
        
        # Normalize
        norm_x = maximin(x, minmax)
        
        # Check bounds
        assert np.all(norm_x >= 0)
        assert np.all(norm_x <= 1)
        
        # Inverse normalization
        x_recovered = inv_maximin(norm_x, minmax)
        
        # Check recovery
        assert np.allclose(x_recovered, X)
    
    def test_maximin_matrix(self):
        """Test normalization with matrix input."""
        m = 100
        n = 300
        y = np.random.rand(m, n)
        minmax = np.column_stack((np.zeros(m), np.ones(m)))
        
        # Store original
        Y = y.copy()
        
        # Normalize
        norm_y = maximin(y, minmax)
        
        # Check bounds
        assert np.all(norm_y >= 0)
        assert np.all(norm_y <= 1)
        
        # Inverse normalization
        y_recovered = inv_maximin(norm_y, minmax)
        
        # Check recovery
        assert np.allclose(y_recovered, Y)
    
    def test_gradient_compatibility(self):
        """Test that normalization functions work with JAX gradients."""
        n_grad = 1024
        A = jnp.array(np.random.randn(n_grad))
        B = jnp.ones((n_grad, 2))
        B = B.at[:, 0].set(0.0)
        
        def test_sum(A):
            return jnp.sum(maximin(A, B) ** 2)
        
        def test_suminv(A):
            return jnp.sum(inv_maximin(A, B) ** 2)
        
        # Compute gradients
        grad_sum = jax.grad(test_sum)(A)
        grad_suminv = jax.grad(test_suminv)(A)
        
        # Check that gradients are finite
        assert jnp.all(jnp.isfinite(grad_sum))
        assert jnp.all(jnp.isfinite(grad_suminv))


class TestEmulatorInitialization:
    """Test emulator initialization and weight loading."""
    
    @pytest.fixture
    def nn_dict(self):
        """Load test neural network configuration."""
        test_file = Path(__file__).parent / "testNN.json"
        with open(test_file, 'r') as f:
            return json.load(f)
    
    @pytest.fixture
    def weights(self, nn_dict):
        """Create random weights matching the network architecture."""
        # Calculate total weight size
        total_size = 0
        
        # Input to first hidden layer
        n_in = nn_dict["n_input_features"]
        n_out = nn_dict["layers"]["layer_1"]["n_neurons"]
        total_size += n_in * n_out + n_out
        
        # Hidden layers
        for i in range(1, nn_dict["n_hidden_layers"]):
            n_in = nn_dict["layers"][f"layer_{i}"]["n_neurons"]
            n_out = nn_dict["layers"][f"layer_{i+1}"]["n_neurons"]
            total_size += n_in * n_out + n_out
        
        # Last hidden to output
        n_in = nn_dict["layers"][f"layer_{nn_dict['n_hidden_layers']}"]["n_neurons"]
        n_out = nn_dict["n_output_features"]
        total_size += n_in * n_out + n_out
        
        # Return random weights
        return np.random.randn(total_size) * 0.1
    
    def test_init_emulator(self, nn_dict, weights):
        """Test emulator initialization."""
        # Initialize emulator
        emulator = init_emulator(nn_dict, weights, FlaxEmulator, validate=True)
        
        # Check type
        assert isinstance(emulator, FlaxEmulator)
        
        # Check components
        assert emulator.model is not None
        assert emulator.parameters is not None
        assert emulator.description is not None
        
        # Check description
        desc = emulator.get_emulator_description()
        assert desc["author"] == "Marco Bonici"
        assert desc["author_email"] == "bonici.marco@gmail.com"
    
    def test_run_emulator(self, nn_dict, weights):
        """Test running the emulator."""
        # Initialize emulator
        emulator = init_emulator(nn_dict, weights, FlaxEmulator, validate=False)
        
        # Create input
        input_data = np.random.randn(6)
        
        # Run emulator
        output = run_emulator(input_data, emulator)
        
        # Check output shape
        assert output.shape == (40,)
        assert np.all(np.isfinite(output))
        
        # Test batched input
        batch_input = np.column_stack((input_data, input_data))
        batch_output = run_emulator(batch_input.T, emulator)
        
        # Check batch output
        assert batch_output.shape == (2, 40)
        assert np.allclose(batch_output[0], output)
        assert np.allclose(batch_output[1], output)
    
    def test_invalid_activation(self, nn_dict, weights):
        """Test error handling for invalid activation function."""
        # Modify nn_dict with invalid activation
        nn_dict_invalid = nn_dict.copy()
        nn_dict_invalid["layers"] = nn_dict["layers"].copy()
        nn_dict_invalid["layers"]["layer_1"] = nn_dict["layers"]["layer_1"].copy()
        nn_dict_invalid["layers"]["layer_1"]["activation_function"] = "invalid_activation"
        
        # Should raise error
        with pytest.raises(ValueError, match="Unknown activation function"):
            init_emulator(nn_dict_invalid, weights, FlaxEmulator, validate=False)


class TestValidation:
    """Test validation functions."""
    
    def test_validate_nn_dict_structure(self):
        """Test neural network dictionary validation."""
        # Valid structure
        valid_dict = {
            "n_input_features": 10,
            "n_output_features": 100,
            "n_hidden_layers": 3,
            "layers": {
                "layer_1": {"n_neurons": 64, "activation_function": "tanh"},
                "layer_2": {"n_neurons": 64, "activation_function": "relu"},
                "layer_3": {"n_neurons": 32, "activation_function": "tanh"}
            }
        }
        
        # Should not raise
        validate_nn_dict_structure(valid_dict)
        
        # Missing required key
        invalid_dict = valid_dict.copy()
        del invalid_dict["n_input_features"]
        
        with pytest.raises(ValueError, match="Missing required key"):
            validate_nn_dict_structure(invalid_dict)
        
        # Invalid n_hidden_layers
        invalid_dict = valid_dict.copy()
        invalid_dict["n_hidden_layers"] = -1
        
        with pytest.raises(ValueError):
            validate_nn_dict_structure(invalid_dict)
        
        # Missing layer
        invalid_dict = valid_dict.copy()
        del invalid_dict["layers"]["layer_2"]
        
        with pytest.raises(ValueError, match="Missing layer definition"):
            validate_nn_dict_structure(invalid_dict)
    
    def test_validate_layer_structure(self):
        """Test layer structure validation."""
        # Valid layer
        valid_layer = {
            "n_neurons": 64,
            "activation_function": "tanh"
        }
        
        # Should not raise
        validate_layer_structure(valid_layer, "test_layer")
        
        # Missing n_neurons
        invalid_layer = {"activation_function": "tanh"}
        
        with pytest.raises(ValueError, match="Missing required key"):
            validate_layer_structure(invalid_layer, "test_layer")
        
        # Invalid activation
        invalid_layer = {
            "n_neurons": 64,
            "activation_function": "invalid"
        }
        
        with pytest.raises(ValueError, match="Unsupported activation"):
            validate_layer_structure(invalid_layer, "test_layer")
        
        # Invalid n_neurons
        invalid_layer = {
            "n_neurons": -1,
            "activation_function": "tanh"
        }
        
        with pytest.raises(ValueError):
            validate_layer_structure(invalid_layer, "test_layer")
    
    def test_safe_dict_access(self):
        """Test safe dictionary access."""
        test_dict = {
            "level1": {
                "level2": {
                    "level3": "value"
                }
            }
        }
        
        # Valid access
        result = safe_dict_access(test_dict, "level1", "level2", "level3")
        assert result == "value"
        
        # Invalid access with default
        result = safe_dict_access(test_dict, "level1", "invalid", default="default")
        assert result == "default"
        
        # No default returns None
        result = safe_dict_access(test_dict, "invalid")
        assert result is None


class TestEmulatorDescription:
    """Test emulator description functionality."""
    
    @pytest.fixture
    def nn_dict(self):
        """Load test neural network configuration."""
        test_file = Path(__file__).parent / "testNN.json"
        with open(test_file, 'r') as f:
            return json.load(f)
    
    def test_get_emulator_description(self, nn_dict, capsys):
        """Test getting emulator description."""
        from jaxace.utils import get_emulator_description as get_desc_util
        
        # Test with valid description
        desc = nn_dict["emulator_description"]
        get_desc_util(desc)
        
        captured = capsys.readouterr()
        assert "Marco Bonici" in captured.out
        assert "bonici.marco@gmail.com" in captured.out
        assert "ln10As, ns, H0, ωb, ωc, τ" in captured.out
        
        # Test with incomplete description
        incomplete_desc = {"author": "Test Author"}
        get_desc_util(incomplete_desc)
        
        captured = capsys.readouterr()
        assert "Test Author" in captured.out
        assert "We do not know which parameters" in captured.out


class TestJAXIntegration:
    """Test JAX-specific features."""
    
    @pytest.fixture
    def emulator(self):
        """Create a test emulator."""
        test_file = Path(__file__).parent / "testNN.json"
        with open(test_file, 'r') as f:
            nn_dict = json.load(f)
        
        # Calculate weight size and create random weights
        total_size = 0
        n_in = nn_dict["n_input_features"]
        
        for i in range(1, nn_dict["n_hidden_layers"] + 1):
            n_out = nn_dict["layers"][f"layer_{i}"]["n_neurons"]
            total_size += n_in * n_out + n_out
            n_in = n_out
        
        n_out = nn_dict["n_output_features"]
        total_size += n_in * n_out + n_out
        
        weights = np.random.randn(total_size) * 0.1
        
        return init_emulator(nn_dict, weights, FlaxEmulator, validate=False)
    
    def test_jit_compilation(self, emulator):
        """Test that emulator runs with JIT."""
        input_data = jnp.array(np.random.randn(6))
        
        # JIT compile the run function
        @jax.jit
        def run_jit(x):
            return emulator.run_emulator(x)
        
        # Run twice (first compiles, second uses compiled)
        output1 = run_jit(input_data)
        output2 = run_jit(input_data)
        
        assert jnp.allclose(output1, output2)
        assert output1.shape == (40,)
    
    def test_vmap(self, emulator):
        """Test vectorization with vmap."""
        batch_size = 10
        batch_input = np.random.randn(batch_size, 6)
        
        # Vectorize run_emulator
        vmap_run = jax.vmap(lambda x: emulator.run_emulator(x))
        
        # Run on batch
        batch_output = vmap_run(batch_input)
        
        assert batch_output.shape == (batch_size, 40)
        assert jnp.all(jnp.isfinite(batch_output))
        
        # Check that individual results match
        single_output = emulator.run_emulator(batch_input[0])
        assert jnp.allclose(batch_output[0], single_output)
    
    def test_gradient(self, emulator):
        """Test gradient computation through emulator."""
        input_data = jnp.array(np.random.randn(6))
        
        # Define loss function
        def loss_fn(x):
            output = emulator.run_emulator(x)
            return jnp.sum(output ** 2)
        
        # Compute gradient
        grad = jax.grad(loss_fn)(input_data)
        
        assert grad.shape == (6,)
        assert jnp.all(jnp.isfinite(grad))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])