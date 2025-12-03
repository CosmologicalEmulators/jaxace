"""
Test GenericEmulator and load_trained_emulator functionality.

These tests match AbstractCosmologicalEmulators.jl/test/test_generic_emulator.jl
"""

import json
import tempfile
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from jaxace import (
    FlaxEmulator,
    GenericEmulator,
    init_emulator,
    load_trained_emulator
)

# Configure JAX for 64-bit precision
jax.config.update('jax_enable_x64', True)


class TestGenericEmulatorConstruction:
    """Test GenericEmulator construction and basic properties."""

    @pytest.fixture
    def basic_emulator(self):
        """Create a basic FlaxEmulator for testing."""
        nn_dict = {
            "n_input_features": 3,
            "n_output_features": 10,
            "n_hidden_layers": 1,
            "layers": {
                "layer_1": {"n_neurons": 8, "activation_function": "tanh"}
            },
            "emulator_description": {
                "author": "Test Author",
                "parameters": "param1, param2, param3"
            }
        }

        # Calculate weight size: (3 × 8) + 8 + (8 × 10) + 10 = 32 + 80 + 10 = 122
        # Actually: (3*8 + 8) + (8*10 + 10) = 32 + 90 = 122
        weights = np.random.randn(122) * 0.1

        return init_emulator(nn_dict, weights, FlaxEmulator, validate=True)

    @pytest.fixture
    def in_minmax(self):
        """Create input normalization matrix."""
        return np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    @pytest.fixture
    def out_minmax(self):
        """Create output normalization matrix."""
        return np.column_stack([np.zeros(10), np.ones(10)])

    def test_basic_construction(self, basic_emulator, in_minmax, out_minmax):
        """Test basic GenericEmulator construction."""
        gen_emu = GenericEmulator(
            trained_emulator=basic_emulator,
            in_minmax=in_minmax,
            out_minmax=out_minmax,
            postprocessing=lambda params, output, aux, emu: output
        )

        assert gen_emu.trained_emulator is basic_emulator
        assert np.allclose(gen_emu.in_minmax, in_minmax)
        assert np.allclose(gen_emu.out_minmax, out_minmax)

    def test_default_postprocessing(self, basic_emulator, in_minmax, out_minmax):
        """Test GenericEmulator with default (None) postprocessing."""
        gen_emu = GenericEmulator(
            trained_emulator=basic_emulator,
            in_minmax=in_minmax,
            out_minmax=out_minmax,
            postprocessing=None  # Should default to identity
        )

        # Run emulator - should work without error
        input_params = np.array([0.5, 0.5, 0.5])
        result = gen_emu.run_emulator(input_params)

        assert result.shape == (10,)
        assert np.all(np.isfinite(result))


class TestGenericEmulatorEvaluation:
    """Test GenericEmulator evaluation methods."""

    @pytest.fixture
    def gen_emulator(self):
        """Create a GenericEmulator for testing."""
        nn_dict = {
            "n_input_features": 3,
            "n_output_features": 10,
            "n_hidden_layers": 1,
            "layers": {
                "layer_1": {"n_neurons": 8, "activation_function": "tanh"}
            }
        }
        weights = np.random.randn(122) * 0.1
        flax_emu = init_emulator(nn_dict, weights, FlaxEmulator, validate=True)

        in_minmax = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        out_minmax = np.column_stack([np.zeros(10), np.ones(10)])

        return GenericEmulator(
            trained_emulator=flax_emu,
            in_minmax=in_minmax,
            out_minmax=out_minmax,
            postprocessing=lambda params, output, aux, emu: output
        )

    def test_single_sample_evaluation(self, gen_emulator):
        """Test evaluation with a single sample."""
        input_params = np.array([0.5, 0.5, 0.5])
        result = gen_emulator.run_emulator(input_params)

        assert result.shape == (10,)
        assert np.all(np.isfinite(result))

    def test_batch_evaluation(self, gen_emulator):
        """Test evaluation with a batch of samples."""
        batch_input = np.array([
            [0.5, 0.5, 0.5],
            [0.3, 0.7, 0.2],
            [0.9, 0.1, 0.4]
        ])
        result = gen_emulator.run_emulator(batch_input)

        assert result.shape == (3, 10)
        assert np.all(np.isfinite(result))

    def test_auxiliary_params(self, gen_emulator):
        """Test evaluation with auxiliary parameters."""
        input_params = np.array([0.5, 0.5, 0.5])
        aux_params = np.array([1.0, 2.0])

        result = gen_emulator.run_emulator(input_params, aux_params)

        assert result.shape == (10,)
        assert np.all(np.isfinite(result))

    def test_callable_interface(self, gen_emulator):
        """Test that GenericEmulator can be called directly."""
        input_params = np.array([0.5, 0.5, 0.5])

        # Should work as a callable
        result = gen_emulator(input_params)

        assert result.shape == (10,)
        assert np.all(np.isfinite(result))


class TestCustomPostprocessing:
    """Test GenericEmulator with custom postprocessing functions."""

    @pytest.fixture
    def base_emulator(self):
        """Create base components for GenericEmulator."""
        nn_dict = {
            "n_input_features": 3,
            "n_output_features": 10,
            "n_hidden_layers": 1,
            "layers": {
                "layer_1": {"n_neurons": 8, "activation_function": "tanh"}
            }
        }
        weights = np.random.randn(122) * 0.1
        flax_emu = init_emulator(nn_dict, weights, FlaxEmulator, validate=True)

        in_minmax = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        out_minmax = np.column_stack([np.zeros(10), np.ones(10)])

        return flax_emu, in_minmax, out_minmax

    def test_scaling_postprocessing(self, base_emulator):
        """Test postprocessing that scales outputs."""
        flax_emu, in_minmax, out_minmax = base_emulator

        def scale_postprocess(params, output, aux, emu):
            # Scale by growth factor squared if aux is provided
            if aux is not None and len(aux) > 0:
                growth_factor = aux[0]
                return output * growth_factor ** 2
            return output

        gen_emu = GenericEmulator(
            trained_emulator=flax_emu,
            in_minmax=in_minmax,
            out_minmax=out_minmax,
            postprocessing=scale_postprocess
        )

        input_params = np.array([0.5, 0.5, 0.5])
        aux_params = np.array([2.0])  # Growth factor = 2

        result_scaled = gen_emu.run_emulator(input_params, aux_params)
        result_baseline = gen_emu.run_emulator(input_params, np.array([1.0]))

        # Verify scaling: result_scaled should be 4x baseline (2^2 = 4)
        assert np.allclose(result_scaled, result_baseline * 4.0)


class TestLoadTrainedEmulator:
    """Test load_trained_emulator functionality."""

    @pytest.fixture
    def temp_emulator_dir(self):
        """Create a temporary directory with emulator files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nn_setup.json
            nn_setup = {
                "n_input_features": 3,
                "n_output_features": 5,
                "n_hidden_layers": 1,
                "layers": {
                    "layer_1": {"n_neurons": 8, "activation_function": "tanh"}
                },
                "emulator_description": {
                    "author": "Test Author",
                    "version": "1.0"
                }
            }
            with open(Path(tmpdir) / "nn_setup.json", "w") as f:
                json.dump(nn_setup, f)

            # Create weights: (3*8 + 8) + (8*5 + 5) = 32 + 45 = 77
            weights = np.random.randn(77).astype(np.float64)
            np.save(Path(tmpdir) / "weights.npy", weights)

            # Create normalization matrices
            in_minmax = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
            out_minmax = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
            np.save(Path(tmpdir) / "inminmax.npy", in_minmax)
            np.save(Path(tmpdir) / "outminmax.npy", out_minmax)

            # Create postprocessing.py
            with open(Path(tmpdir) / "postprocessing.py", "w") as f:
                f.write("def postprocessing(input_params, output, auxiliary_params, emulator):\n")
                f.write("    return output\n")

            yield tmpdir

    def test_load_trained_emulator(self, temp_emulator_dir):
        """Test loading a trained emulator from disk."""
        loaded_emu = load_trained_emulator(temp_emulator_dir)

        assert isinstance(loaded_emu, GenericEmulator)
        assert loaded_emu.in_minmax.shape == (3, 2)
        assert loaded_emu.out_minmax.shape == (5, 2)

        # Test that it runs
        input_params = np.array([0.5, 0.5, 0.5])
        result = loaded_emu.run_emulator(input_params)

        assert result.shape == (5,)
        assert np.all(np.isfinite(result))

    def test_emulator_description(self, temp_emulator_dir):
        """Test that emulator description is accessible."""
        loaded_emu = load_trained_emulator(temp_emulator_dir)

        desc = loaded_emu.get_emulator_description()
        assert desc["author"] == "Test Author"
        assert desc["version"] == "1.0"


class TestRealTrainedEmulators:
    """Test loading real trained emulators using fetch-artifacts."""

    @pytest.fixture
    def sigma8_emulator(self):
        """Load sigma8 basis emulator from artifact."""
        from jaxace import load_trained_emulator_from_artifact
        return load_trained_emulator_from_artifact('ACE_mnuw0wacdm_sigma8_basis')

    def test_load_sigma8_emulator_from_artifact(self, sigma8_emulator):
        """Test loading the sigma8 basis emulator from artifact."""
        assert isinstance(sigma8_emulator, GenericEmulator)

        # Check dimensions (9 inputs, 7 outputs)
        assert sigma8_emulator.in_minmax.shape[0] == 9
        assert sigma8_emulator.out_minmax.shape[0] == 7

    def test_evaluate_sigma8_emulator(self, sigma8_emulator):
        """Test evaluation with sigma8 emulator."""
        # Parameters: z, sigma8, ns, H0, ombh2, omch2, Mν, w0, wa
        input_params = np.array([0.5, 0.8, 0.96, 67.0, 0.022, 0.12, 0.06, -1.0, 0.0])
        result = sigma8_emulator.run_emulator(input_params)

        assert result.shape == (7,)
        assert np.all(np.isfinite(result))


class TestJAXIntegrationGenericEmulator:
    """Test JAX-specific features with GenericEmulator."""

    @pytest.fixture
    def gen_emulator(self):
        """Create a GenericEmulator for JAX integration tests."""
        nn_dict = {
            "n_input_features": 3,
            "n_output_features": 10,
            "n_hidden_layers": 1,
            "layers": {
                "layer_1": {"n_neurons": 8, "activation_function": "tanh"}
            }
        }
        weights = np.random.randn(122) * 0.1
        flax_emu = init_emulator(nn_dict, weights, FlaxEmulator, validate=True)

        in_minmax = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        out_minmax = np.column_stack([np.zeros(10), np.ones(10)])

        return GenericEmulator(
            trained_emulator=flax_emu,
            in_minmax=in_minmax,
            out_minmax=out_minmax,
            postprocessing=lambda params, output, aux, emu: output
        )

    def test_gradient_computation(self, gen_emulator):
        """Test gradient computation through GenericEmulator using jax.grad."""
        input_params = jnp.array([0.5, 0.5, 0.5])

        def loss_fn(x):
            output = gen_emulator.run_emulator(x)
            return jnp.sum(output ** 2)

        grad = jax.grad(loss_fn)(input_params)

        assert grad.shape == (3,)
        assert jnp.all(jnp.isfinite(grad))

    def test_jacobian_forward_mode(self, gen_emulator):
        """Test Jacobian computation using jax.jacfwd (forward-mode AD)."""
        input_params = jnp.array([0.5, 0.5, 0.5])

        def emulator_fn(x):
            return gen_emulator.run_emulator(x)

        # Compute Jacobian using forward-mode AD
        jacobian = jax.jacfwd(emulator_fn)(input_params)

        # Jacobian shape should be (n_outputs, n_inputs) = (10, 3)
        assert jacobian.shape == (10, 3)
        assert jnp.all(jnp.isfinite(jacobian))

    def test_jacobian_reverse_mode(self, gen_emulator):
        """Test Jacobian computation using jax.jacrev (reverse-mode AD)."""
        input_params = jnp.array([0.5, 0.5, 0.5])

        def emulator_fn(x):
            return gen_emulator.run_emulator(x)

        # Compute Jacobian using reverse-mode AD
        jacobian = jax.jacrev(emulator_fn)(input_params)

        # Jacobian shape should be (n_outputs, n_inputs) = (10, 3)
        assert jacobian.shape == (10, 3)
        assert jnp.all(jnp.isfinite(jacobian))

    def test_jacobian_consistency(self, gen_emulator):
        """Test that forward and reverse mode Jacobians are consistent."""
        input_params = jnp.array([0.5, 0.5, 0.5])

        def emulator_fn(x):
            return gen_emulator.run_emulator(x)

        # Compute Jacobian using both modes
        jac_fwd = jax.jacfwd(emulator_fn)(input_params)
        jac_rev = jax.jacrev(emulator_fn)(input_params)

        # They should be equal (within numerical tolerance)
        assert jnp.allclose(jac_fwd, jac_rev, rtol=1e-5)

    def test_jit_compilation(self, gen_emulator):
        """Test JIT compilation of GenericEmulator."""
        input_params = jnp.array([0.5, 0.5, 0.5])

        @jax.jit
        def run_jit(x):
            return gen_emulator.run_emulator(x)

        output1 = run_jit(input_params)
        output2 = run_jit(input_params)

        assert jnp.allclose(output1, output2)
        assert output1.shape == (10,)

    def test_jit_gradient(self, gen_emulator):
        """Test JIT-compiled gradient computation."""
        input_params = jnp.array([0.5, 0.5, 0.5])

        def loss_fn(x):
            output = gen_emulator.run_emulator(x)
            return jnp.sum(output ** 2)

        # JIT compile the gradient function
        grad_fn = jax.jit(jax.grad(loss_fn))

        grad = grad_fn(input_params)

        assert grad.shape == (3,)
        assert jnp.all(jnp.isfinite(grad))

    def test_jit_jacobian(self, gen_emulator):
        """Test JIT-compiled Jacobian computation."""
        input_params = jnp.array([0.5, 0.5, 0.5])

        def emulator_fn(x):
            return gen_emulator.run_emulator(x)

        # JIT compile the Jacobian function
        jac_fn = jax.jit(jax.jacfwd(emulator_fn))

        jacobian = jac_fn(input_params)

        assert jacobian.shape == (10, 3)
        assert jnp.all(jnp.isfinite(jacobian))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
