"""
Test that verifies the NN produces expected output with specific architecture and weights.
This test replicates the comparison_NN setup to ensure consistency.
"""
import numpy as np
import jax.numpy as jnp
import pytest
from jaxace.initialization import init_emulator
from jaxace.core import FlaxEmulator


class TestNNComparison:
    """Test neural network with explicit structure matching comparison_NN setup."""
    
    def test_nn_with_ones_weights(self):
        """
        Test that a 2→3→3→2 network with all weights=1 produces expected output.
        This matches the setup in comparison_NN scripts.
        """
        # Create the NN dictionary matching the structure used in comparison_NN
        nn_dict = {
            "n_input_features": 2,
            "layers": {
                "layer_1": {
                    "n_neurons": 3,
                    "activation_function": "tanh"
                },
                "layer_2": {
                    "n_neurons": 3,
                    "activation_function": "tanh"
                }
            },
            "n_output_features": 2,
            "n_hidden_layers": 2,
            "emulator_description": {
                "author": "Test Suite",
                "miscellanea": "Test emulator with known weights"
            }
        }
        
        # Create weights array of ones
        # Architecture: 2→3→3→2
        # Layer 1: 2*3 + 3 = 9 weights
        # Layer 2: 3*3 + 3 = 12 weights  
        # Output: 3*2 + 2 = 8 weights
        # Total: 29 weights
        weights = np.ones(29, dtype=np.float32)
        
        # Initialize the emulator
        emulator = init_emulator(nn_dict, weights, emulator_type=FlaxEmulator)
        
        # Test input
        test_input = jnp.array([0.5, 0.5])
        
        # Run the emulator
        output = emulator.run_emulator(test_input)
        
        # Expected output (from both Julia and Python scripts)
        expected_output = jnp.array([3.9975033, 3.9975033])
        
        # Check that outputs match
        np.testing.assert_allclose(
            output, 
            expected_output, 
            rtol=1e-5,  # Relative tolerance
            atol=1e-6,  # Absolute tolerance
            err_msg="NN output doesn't match expected value from comparison scripts"
        )
    
    def test_nn_batch_processing(self):
        """Test that the same network handles batches correctly."""
        # Same NN structure
        nn_dict = {
            "n_input_features": 2,
            "layers": {
                "layer_1": {
                    "n_neurons": 3,
                    "activation_function": "tanh"
                },
                "layer_2": {
                    "n_neurons": 3,
                    "activation_function": "tanh"
                }
            },
            "n_output_features": 2,
            "n_hidden_layers": 2,
        }
        
        # Same weights
        weights = np.ones(29, dtype=np.float32)
        
        # Initialize emulator
        emulator = init_emulator(nn_dict, weights, emulator_type=FlaxEmulator)
        
        # Create batch input (3 samples of [0.5, 0.5])
        batch_input = jnp.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        
        # Run on batch
        batch_output = emulator.run_emulator(batch_input)
        
        # Expected: same output for each sample
        expected_single = jnp.array([3.9975033, 3.9975033])
        
        # Check batch shape
        assert batch_output.shape == (3, 2), "Batch output shape is incorrect"
        
        # Check each output in batch
        for i in range(3):
            np.testing.assert_allclose(
                batch_output[i], 
                expected_single,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Batch sample {i} doesn't match expected output"
            )
    
    def test_nn_auto_jit_enabled(self):
        """Verify that auto-JIT is working by checking for cached functions."""
        nn_dict = {
            "n_input_features": 2,
            "layers": {
                "layer_1": {"n_neurons": 3, "activation_function": "tanh"},
                "layer_2": {"n_neurons": 3, "activation_function": "tanh"}
            },
            "n_output_features": 2,
            "n_hidden_layers": 2,
        }
        
        weights = np.ones(29, dtype=np.float32)
        emulator = init_emulator(nn_dict, weights, emulator_type=FlaxEmulator)
        
        # Initially, JIT functions should be None or get compiled on first use
        test_input = jnp.array([0.5, 0.5])
        
        # First call - triggers JIT compilation
        _ = emulator.run_emulator(test_input)
        
        # After first call, JIT functions should be cached
        assert emulator._jit_single is not None, "Single JIT function not cached"
        
        # Test batch to trigger batch JIT
        batch_input = jnp.array([[0.5, 0.5], [0.5, 0.5]])
        _ = emulator.run_emulator(batch_input)
        
        assert emulator._jit_batch is not None, "Batch JIT function not cached"
    
    def test_nn_callable_interface(self):
        """Test that emulator can be called directly as emulator(input)."""
        nn_dict = {
            "n_input_features": 2,
            "layers": {
                "layer_1": {"n_neurons": 3, "activation_function": "tanh"},
                "layer_2": {"n_neurons": 3, "activation_function": "tanh"}
            },
            "n_output_features": 2,
            "n_hidden_layers": 2,
        }
        
        weights = np.ones(29, dtype=np.float32)
        emulator = init_emulator(nn_dict, weights, emulator_type=FlaxEmulator)
        
        test_input = jnp.array([0.5, 0.5])
        
        # Test both interfaces produce same result
        output1 = emulator.run_emulator(test_input)
        output2 = emulator(test_input)  # Direct call
        
        np.testing.assert_array_equal(
            output1, 
            output2,
            err_msg="Direct call emulator(input) doesn't match run_emulator(input)"
        )


class TestWeightOrdering:
    """Test that weights are correctly mapped to network parameters."""
    
    def test_weight_count_calculation(self):
        """Verify weight count for 2→3→3→2 architecture."""
        # Layer 1: input_size * hidden_size + bias
        layer1_weights = 2 * 3  # weights
        layer1_bias = 3         # bias
        
        # Layer 2: hidden_size * hidden_size + bias  
        layer2_weights = 3 * 3  # weights
        layer2_bias = 3         # bias
        
        # Output layer: hidden_size * output_size + bias
        output_weights = 3 * 2  # weights
        output_bias = 2         # bias
        
        total = (layer1_weights + layer1_bias + 
                layer2_weights + layer2_bias + 
                output_weights + output_bias)
        
        assert total == 29, f"Expected 29 weights, calculated {total}"
    
    def test_different_weight_values(self):
        """Test network with non-uniform weights to verify correct mapping."""
        nn_dict = {
            "n_input_features": 2,
            "layers": {
                "layer_1": {"n_neurons": 3, "activation_function": "relu"},
                "layer_2": {"n_neurons": 3, "activation_function": "relu"}
            },
            "n_output_features": 2,
            "n_hidden_layers": 2,
        }
        
        # Create weights with a pattern to test mapping
        weights = np.arange(29, dtype=np.float32) * 0.1
        
        emulator = init_emulator(nn_dict, weights, emulator_type=FlaxEmulator)
        
        # Just verify it runs without error
        test_input = jnp.array([1.0, 1.0])
        output = emulator.run_emulator(test_input)
        
        # Output should be 2D
        assert output.shape == (2,), "Output shape incorrect"
        
        # With relu and these weights, output should not be all the same
        assert not np.allclose(output[0], output[1]), \
            "Output values shouldn't be identical with non-uniform weights"