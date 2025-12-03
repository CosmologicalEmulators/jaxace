"""
Tests for artifact loading functionality.

These tests verify that emulators can be loaded from artifacts
and that they work correctly with JAX autodiff.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

import jaxace


class TestArtifactLoading:
    """Test loading emulators from artifacts."""

    def test_function_exists(self):
        """Test that load_trained_emulator_from_artifact exists."""
        assert hasattr(jaxace, 'load_trained_emulator_from_artifact')
        assert callable(jaxace.load_trained_emulator_from_artifact)

    def test_artifacts_toml_exists(self):
        """Test that Artifacts.toml exists in package root."""
        package_root = Path(jaxace.__file__).parent.parent
        artifacts_toml = package_root / "Artifacts.toml"
        assert artifacts_toml.exists(), "Artifacts.toml should exist in package root"

    def test_load_ace_emulator(self):
        """Test loading ACE_mnuw0wacdm_sigma8_basis emulator from artifact."""
        emu = jaxace.load_trained_emulator_from_artifact('ACE_mnuw0wacdm_sigma8_basis')

        # Verify it's a GenericEmulator
        assert isinstance(emu, jaxace.GenericEmulator)

        # Verify it has the expected structure
        assert hasattr(emu, 'trained_emulator')
        assert hasattr(emu, 'in_minmax')
        assert hasattr(emu, 'out_minmax')
        assert hasattr(emu, 'postprocessing')

    def test_emulator_dimensions(self):
        """Test that emulator has correct input/output dimensions."""
        emu = jaxace.load_trained_emulator_from_artifact('ACE_mnuw0wacdm_sigma8_basis')

        # Should have 9 inputs and 7 outputs
        assert emu.in_minmax.shape == (9, 2), "Should have 9 input parameters"
        assert emu.out_minmax.shape == (7, 2), "Should have 7 output quantities"


class TestEmulatorEvaluation:
    """Test emulator evaluation with specific inputs."""

    @pytest.fixture
    def emulator(self):
        """Load emulator for testing."""
        return jaxace.load_trained_emulator_from_artifact('ACE_mnuw0wacdm_sigma8_basis')

    @pytest.fixture
    def test_input(self):
        """
        Test input parameters.

        Parameters: z, sigma8, ns, H0, ombh2, omch2, Mν, w0, wa
        """
        return np.array([
            0.7765,                      # z
            0.8686735341759542,          # sigma8
            1.05795,                     # ns
            81.84666666666666,           # H0
            0.020955833333333333,        # ombh2
            0.08851666666666666,         # omch2
            0.14325,                     # Mν
            -0.8807500000000004,         # w0
            -2.1341666666666668          # wa
        ])

    def test_single_evaluation(self, emulator, test_input):
        """Test single forward pass with specific test input."""
        output = emulator.run_emulator(test_input)

        # Output should have 7 elements: ln10As, sigma8_z, r_drag, H_z, r_z, D_z, f_z
        assert output.shape == (7,), f"Expected shape (7,), got {output.shape}"

    def test_output_is_finite(self, emulator, test_input):
        """Test that output is finite (no NaN or Inf)."""
        output = emulator.run_emulator(test_input)

        # Check that all outputs are finite
        assert jnp.all(jnp.isfinite(output)), (
            f"Output contains non-finite values: {output}"
        )

        # Individual checks for debugging
        output_names = ['ln10As', 'sigma8_z', 'r_drag', 'H_z', 'r_z', 'D_z', 'f_z']
        for i, (name, value) in enumerate(zip(output_names, output)):
            assert jnp.isfinite(value), (
                f"Output {i} ({name}) is not finite: {value}"
            )

    def test_output_values_reasonable(self, emulator, test_input):
        """Test that output values are in reasonable ranges."""
        output = emulator.run_emulator(test_input)

        # ln10As should be around 3
        assert 2.0 < output[0] < 4.0, f"ln10As = {output[0]} out of range"

        # sigma8_z should be positive and less than 1
        assert 0.0 < output[1] < 1.5, f"sigma8_z = {output[1]} out of range"

        # r_drag should be positive (around 100-200)
        assert output[2] > 0, f"r_drag = {output[2]} should be positive"

        # H_z should be positive
        assert output[3] > 0, f"H_z = {output[3]} should be positive"

        # r_z should be positive
        assert output[4] > 0, f"r_z = {output[4]} should be positive"

        # D_z should be positive and less than 1
        assert 0.0 < output[5] < 1.0, f"D_z = {output[5]} out of range"

        # f_z should be positive and reasonable
        assert 0.0 < output[6] < 2.0, f"f_z = {output[6]} out of range"

    def test_batch_evaluation(self, emulator, test_input):
        """Test batch evaluation."""
        # Create batch of 5 identical inputs
        batch_input = np.tile(test_input, (5, 1))
        batch_output = emulator.run_emulator(batch_input)

        assert batch_output.shape == (5, 7), (
            f"Expected shape (5, 7), got {batch_output.shape}"
        )

        # All outputs should be identical
        for i in range(1, 5):
            np.testing.assert_allclose(
                batch_output[i],
                batch_output[0],
                rtol=1e-6,
                err_msg="Batch outputs should be identical for identical inputs"
            )


class TestJAXAutodiff:
    """Test JAX autodiff compatibility."""

    @pytest.fixture
    def emulator(self):
        """Load emulator for testing."""
        return jaxace.load_trained_emulator_from_artifact('ACE_mnuw0wacdm_sigma8_basis')

    @pytest.fixture
    def test_input(self):
        """Test input as JAX array."""
        return jnp.array([
            0.7765,                      # z
            0.8686735341759542,          # sigma8
            1.05795,                     # ns
            81.84666666666666,           # H0
            0.020955833333333333,        # ombh2
            0.08851666666666666,         # omch2
            0.14325,                     # Mν
            -0.8807500000000004,         # w0
            -2.1341666666666668          # wa
        ])

    def test_grad_wrt_all_inputs(self, emulator, test_input):
        """Test gradient computation with respect to all input parameters."""

        def loss_fn(params):
            """Simple loss: sum of all outputs."""
            output = emulator.run_emulator(params)
            return jnp.sum(output)

        # Compute gradient
        grad = jax.grad(loss_fn)(test_input)

        # Gradient should have same shape as input
        assert grad.shape == test_input.shape, (
            f"Expected gradient shape {test_input.shape}, got {grad.shape}"
        )

        # Gradient should be finite
        assert jnp.all(jnp.isfinite(grad)), (
            f"Gradient contains non-finite values: {grad}"
        )

    def test_grad_each_output(self, emulator, test_input):
        """Test gradient for each output separately."""
        output_names = ['ln10As', 'sigma8_z', 'r_drag', 'H_z', 'r_z', 'D_z', 'f_z']

        for i, name in enumerate(output_names):
            def select_output(params):
                """Select specific output."""
                output = emulator.run_emulator(params)
                return output[i]

            # Compute gradient
            grad = jax.grad(select_output)(test_input)

            # Check gradient is finite
            assert jnp.all(jnp.isfinite(grad)), (
                f"Gradient for {name} contains non-finite values: {grad}"
            )

    def test_jacobian(self, emulator, test_input):
        """Test Jacobian computation (gradients of all outputs wrt all inputs)."""

        def forward(params):
            """Forward pass."""
            return emulator.run_emulator(params)

        # Compute Jacobian
        jacobian = jax.jacfwd(forward)(test_input)

        # Jacobian should be (7, 9) - 7 outputs, 9 inputs
        assert jacobian.shape == (7, 9), (
            f"Expected Jacobian shape (7, 9), got {jacobian.shape}"
        )

        # All Jacobian entries should be finite
        assert jnp.all(jnp.isfinite(jacobian)), (
            f"Jacobian contains non-finite values"
        )

    def test_hessian_diagonal(self, emulator, test_input):
        """Test Hessian diagonal (second derivatives) for one output."""

        def select_first_output(params):
            """Select first output (ln10As)."""
            output = emulator.run_emulator(params)
            return output[0]

        # Compute Hessian diagonal
        hessian_diag = jax.hessian(select_first_output)(test_input)

        # Hessian should be (9, 9)
        assert hessian_diag.shape == (9, 9), (
            f"Expected Hessian shape (9, 9), got {hessian_diag.shape}"
        )

        # Diagonal should be finite
        diag = jnp.diag(hessian_diag)
        assert jnp.all(jnp.isfinite(diag)), (
            f"Hessian diagonal contains non-finite values: {diag}"
        )

    def test_jit_compilation(self, emulator, test_input):
        """Test that emulator works with JAX JIT compilation."""

        @jax.jit
        def jitted_forward(params):
            return emulator.run_emulator(params)

        # First call (compilation)
        output1 = jitted_forward(test_input)

        # Second call (should use cached)
        output2 = jitted_forward(test_input)

        # Outputs should be identical
        np.testing.assert_allclose(output1, output2, rtol=1e-10)

        # Should be finite
        assert jnp.all(jnp.isfinite(output1))

    def test_vmap_compatibility(self, emulator, test_input):
        """Test that emulator works with JAX vmap (vectorization)."""

        # Create batch using vmap instead of manual batching
        batch_size = 10
        batch_input = jnp.tile(test_input, (batch_size, 1))

        # Vectorize over batch dimension
        vmapped_forward = jax.vmap(lambda x: emulator.run_emulator(x))
        batch_output = vmapped_forward(batch_input)

        assert batch_output.shape == (batch_size, 7)
        assert jnp.all(jnp.isfinite(batch_output))

    def test_grad_and_value(self, emulator, test_input):
        """Test simultaneous value and gradient computation."""

        def loss_fn(params):
            output = emulator.run_emulator(params)
            return jnp.sum(output)

        # Get both value and gradient
        value, grad = jax.value_and_grad(loss_fn)(test_input)

        # Both should be finite
        assert jnp.isfinite(value), f"Value is not finite: {value}"
        assert jnp.all(jnp.isfinite(grad)), f"Gradient contains non-finite values"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_artifact_name(self):
        """Test loading with invalid artifact name."""
        with pytest.raises(Exception):  # Should raise some error
            jaxace.load_trained_emulator_from_artifact('nonexistent_artifact')

    def test_wrong_input_shape(self):
        """Test with wrong input shape."""
        emu = jaxace.load_trained_emulator_from_artifact('ACE_mnuw0wacdm_sigma8_basis')

        # Try with wrong number of parameters
        wrong_input = np.array([1.0, 0.8, 0.96])  # Only 3 instead of 9

        with pytest.raises(Exception):  # Should raise dimension mismatch
            emu.run_emulator(wrong_input)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
